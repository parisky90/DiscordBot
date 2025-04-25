import discord
from discord.ext import tasks, commands
import asyncio
import logging
from datetime import datetime, timezone
from collections import deque
import aiohttp
import re # Import regex for cleaning titles

# Import fuzzy matching and sentiment analysis libraries
from thefuzz import fuzz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config  # Import configuration variables (loads .env)
import scraper # Import scraping functions
import llm_handler # Import LLM functions (Ollama for Eval, DeepL for Translate)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, # Set default level (DEBUG is very verbose)
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Silence less important logs from libraries
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("discord").setLevel(logging.WARNING)
# Optional: Keep ollama logs less verbose if desired
# logging.getLogger("ollama").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Bot Setup ---
intents = discord.Intents.default()
# Remove message content intent if not using prefix commands heavily
# intents.message_content = True # Uncomment if you have prefix commands that need content
bot = commands.Bot(command_prefix="!", intents=intents)

# --- State Management ---
MAX_SEEN_HEADLINES = 1000 # Store more history for deduplication
seen_headlines = deque(maxlen=MAX_SEEN_HEADLINES)
# Deque for normalized titles for cross-source deduplication
MAX_SEEN_TITLES = 200 # How many recent titles to check for fuzzy match
seen_normalized_titles = deque(maxlen=MAX_SEEN_TITLES)

# Initialize Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# --- Bot Events ---
@bot.event
async def on_ready():
    """Called when the bot successfully connects to Discord."""
    logger.info(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    logger.info('------')
    logger.info(f'Using Ollama model for evaluation: {config.OLLAMA_MODEL}')
    # Adjusted message: Removed glossary mention
    logger.info(f"Using DeepL API for translation: {'Enabled' if config.DEEPL_API_KEY else 'DISABLED (No API Key)'}")
    logger.info(f"User Keywords: {len(config.USER_KEYWORDS)} | No-Translate Terms: {len(config.NO_TRANSLATE_TERMS)}")
    logger.info(f"Fuzzy Match Threshold: {config.FUZZY_MATCH_THRESHOLD}")

    channel = bot.get_channel(config.DISCORD_CHANNEL_ID)
    if channel:
        logger.info(f"Target channel found: {channel.name} (ID: {config.DISCORD_CHANNEL_ID})")
        if not heartbeat_task.is_running():
            heartbeat_task.start(channel)
        if not check_news_task.is_running():
            check_news_task.start(channel)
        try:
            # Adjusted startup message
            await channel.send(f"✅ **Bot V2.1 Started!** Monitoring news. Eval: `{config.OLLAMA_MODEL}`, Translate: `DeepL` (Free Tier)")
            logger.info("Startup message sent.")
        except discord.Forbidden:
            logger.error(f"Permission error: Cannot send messages to channel ID {config.DISCORD_CHANNEL_ID}. Check bot permissions.")
        except Exception as e:
            logger.error(f"Error sending startup message: {e}")
    else:
        logger.error(f"Could not find channel with ID {config.DISCORD_CHANNEL_ID}. Check the ID and bot's presence in the server.")
        logger.error("Bot tasks will not start. Shutting down.")
        await bot.close()

# --- Helper Function: Normalize Title ---
def normalize_title(title: str) -> str:
    """Converts title to lowercase and removes punctuation for comparison."""
    if not title: return ""
    text = title.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = ' '.join(text.split()) # Normalize whitespace
    return text

# --- Background Tasks ---
@tasks.loop(minutes=config.HEARTBEAT_INTERVAL_MINUTES)
async def heartbeat_task(channel: discord.TextChannel):
    """Sends a periodic message to confirm the bot is running."""
    now_utc = datetime.now(timezone.utc)
    message = f"🟢 Bot operational. Status check at {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}."
    try:
        await channel.send(message)
        logger.info("Heartbeat message sent.")
    except discord.errors.ConnectionClosed as e:
        logger.warning(f"Heartbeat failed: Discord connection closed (Code: {e.code}). Will retry.")
    except Exception as e:
        logger.error(f"Error sending heartbeat message: {e}")

@tasks.loop(seconds=config.SCRAPE_INTERVAL_SECONDS)
async def check_news_task(channel: discord.TextChannel):
    """Periodically scrapes news, evaluates, deduplicates, translates, and posts."""
    logger.info(f"Starting news check cycle (Interval: {config.SCRAPE_INTERVAL_SECONDS}s)...")
    try:
        all_new_headlines = await scraper.scrape_all(config.NEWS_SOURCES)
        posted_count = 0
        processed_count = 0
        # Set for deduplication within this specific cycle run
        cycle_processed_ids = set()

        for source, headlines in all_new_headlines.items():
            if not headlines: continue
            logger.info(f"Processing {len(headlines)} headlines from {source}...")

            # Process headlines from oldest to newest within the batch
            for title, url, detected_time in reversed(headlines):
                processed_count += 1

                # --- Basic Info & Normalization ---
                unique_id = f"{source}:{url}"
                normalized_title_current = normalize_title(title)

                # --- Deduplication Step 1: Within this cycle ---
                if unique_id in cycle_processed_ids:
                    logger.debug(f"Skipped (duplicate within this cycle): '{unique_id}'")
                    continue
                cycle_processed_ids.add(unique_id)

                # --- Deduplication Step 2: Fuzzy Title Match against recent history ---
                is_title_duplicate = False
                if normalized_title_current: # Only check if title is valid
                    for seen_title in seen_normalized_titles:
                        similarity = fuzz.token_set_ratio(normalized_title_current, seen_title)
                        if similarity >= config.FUZZY_MATCH_THRESHOLD:
                            logger.info(f"Skipped (Fuzzy Title Match >= {config.FUZZY_MATCH_THRESHOLD}%): '{title[:60]}...' similar to '{seen_title[:60]}...'")
                            is_title_duplicate = True
                            # Add the exact unique_id to main seen list anyway to prevent exact re-post later
                            if unique_id not in seen_headlines: seen_headlines.append(unique_id)
                            break # Stop checking once a duplicate is found
                if is_title_duplicate:
                    continue # Skip to next headline

                # --- Deduplication Step 3: Exact ID Match against longer history ---
                logger.debug(f"Checking unique_id: '{unique_id}' | Title: '{title[:50]}...'")
                if unique_id in seen_headlines:
                    logger.info(f"Skipped (already seen exact ID): '{unique_id}'")
                    # Add normalized title if this is the first time we see it, even if URL was seen
                    if normalized_title_current and normalized_title_current not in seen_normalized_titles:
                        seen_normalized_titles.append(normalized_title_current)
                    continue # Skip to next headline

                # --- If it's a new headline (passed all checks) ---
                logger.info(f"NEW Headline: '{unique_id}'")
                seen_headlines.append(unique_id)
                if normalized_title_current:
                    seen_normalized_titles.append(normalized_title_current) # Add to title deque for future fuzzy checks


                # --- User Keyword Check ---
                boost_by_keyword = False
                for keyword in config.USER_KEYWORDS:
                     if keyword in title.lower():
                         logger.info(f"Headline significance boosted by user keyword: '{keyword}'")
                         boost_by_keyword = True
                         break

                # --- Evaluate Significance and Category using LLM ---
                evaluation_result = await llm_handler.evaluate_significance_and_category(title)

                if evaluation_result:
                    is_significant_llm = evaluation_result["significant"]
                    category = evaluation_result["category"] # Already normalized in llm_handler
                    reason = evaluation_result["reason"]

                    # Apply user keyword boost
                    is_significant_final = is_significant_llm or boost_by_keyword
                    if boost_by_keyword and not is_significant_llm:
                        logger.info(f"Overriding LLM: Marking as significant due to user keyword for '{title[:60]}...'")
                        # Ensure a valid category if LLM ignored it
                        if not category or category == "Unknown": category = "General"


                    # --- If deemed significant (by LLM or Keyword), proceed ---
                    if is_significant_final:
                        logger.info(f"Attempting translation for significant headline ({category}).")
                        # Call translation function (which handles placeholder logic)
                        greek_translation = await llm_handler.translate_en_to_el(title)

                        if greek_translation:
                            # --- Sentiment Analysis ---
                            sentiment_score = sentiment_analyzer.polarity_scores(title)['compound']
                            sentiment_label = "Neutral"
                            embed_color = config.CATEGORIES.get(category, config.DEFAULT_CATEGORY_COLOR)

                            if sentiment_score >= config.SENTIMENT_POSITIVE_THRESHOLD:
                                sentiment_label = "Positive"
                                # Optional: Override color based on sentiment for specific categories
                                # if category == "Stocks": embed_color = discord.Color.green()
                            elif sentiment_score <= config.SENTIMENT_NEGATIVE_THRESHOLD:
                                sentiment_label = "Negative"
                                # if category == "Stocks": embed_color = discord.Color.red()

                            logger.info(f"Posting translated headline. Sentiment: {sentiment_label} ({sentiment_score:.2f})")

                            # --- Create Embed ---
                            embed_description = f"{greek_translation}\n\n*Category: {category} | Sentiment: {sentiment_label}*"
                            # Optionally add LLM's reason if available and not just generic
                            if reason and reason.lower() not in ["significant news", "meets criteria"]:
                                embed_description += f" | _{reason}_"

                            embed = discord.Embed(
                                title=f"📰 {title}", # Original English Title
                                url=url, # Use the original url variable
                                description=embed_description,
                                color=embed_color
                            )
                            # Footer and Timestamp are intentionally omitted

                            # --- Send to Discord ---
                            try:
                                await channel.send(embed=embed)
                                posted_count += 1
                                await asyncio.sleep(2) # Throttle posting slightly
                            except discord.Forbidden:
                                logger.error(f"Permission error sending to channel {channel.name}. Check bot permissions.")
                                # Consider stopping the loop for this source or the whole task if permissions are wrong
                                break # Stop processing this source for now
                            except discord.RateLimited as e:
                                logger.warning(f"Discord rate limit hit. Sleeping for {e.retry_after:.2f} seconds...")
                                await asyncio.sleep(e.retry_after)
                                try: # Retry sending the same message once after rate limit
                                    await channel.send(embed=embed)
                                    posted_count += 1
                                    await asyncio.sleep(2)
                                except Exception as retry_e:
                                    logger.error(f"Error sending Discord message after retry for '{title[:60]}': {retry_e}")
                            except Exception as e:
                                logger.error(f"Error sending Discord message for '{title[:60]}': {e}")

                        else:
                            # Translation failed
                            logger.warning(f"Translation failed for significant headline: '{title[:80]}...'")
                            await asyncio.sleep(0.5) # Small delay even on failure
                    else:
                        # Ignored by LLM (and not boosted by keyword)
                        logger.info(f"Skipping post for non-significant headline ({category}): '{title[:80]}...'")
                        await asyncio.sleep(0.1) # Very small delay
                else:
                    # LLM evaluation itself failed
                    logger.warning(f"LLM evaluation failed for headline: '{title[:80]}...'")
                    await asyncio.sleep(0.5)

            # Small delay between processing different sources
            await asyncio.sleep(1)

        # --- End of cycle logging ---
        if posted_count > 0:
             logger.info(f"Posted {posted_count} translated headlines out of {processed_count} processed this cycle.")
        elif processed_count > 0:
             logger.info(f"Processed {processed_count} headlines, none met criteria or passed translation.")
        # else: logger.debug("No new headlines found across all sources in this cycle.")


    except aiohttp.ClientError as e:
        logger.error(f"Network error during scraping cycle: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in news check cycle: {e}", exc_info=True) # Log traceback for unexpected errors

# --- Task setup and error handling ---
@check_news_task.before_loop
async def before_check_news():
    await bot.wait_until_ready()
    logger.info("Bot ready, starting news check task.")

@heartbeat_task.before_loop
async def before_heartbeat():
    await bot.wait_until_ready()
    logger.info("Bot ready, starting heartbeat task.")

async def restart_task_on_error(task, error, task_name):
    """Logs critical task loop errors."""
    logger.error(f"Unhandled error in {task_name} task loop: {error}", exc_info=True) # Log traceback
    logger.info(f"Task {task_name} stopped due to error. Check logs for details. It might restart on the next interval.")
    # Consider adding notification logic here if needed (e.g., send DM to bot owner)

@heartbeat_task.error
async def heartbeat_error(error):
    """Handles errors specifically in the heartbeat task loop."""
    logger.error(f"Error in heartbeat task loop: {error}", exc_info=False) # Don't need full traceback usually

@check_news_task.error
async def check_news_error(error):
     """Handles errors specifically in the check_news_task loop."""
     await restart_task_on_error(check_news_task, error, "check_news")

# --- Run the Bot ---
if __name__ == "__main__":
    # Validate essential config before starting
    if not config.DISCORD_BOT_TOKEN:
         logger.critical("CRITICAL: DISCORD_BOT_TOKEN is missing in the environment/config.")
    elif not config.DISCORD_CHANNEL_ID:
         logger.critical("CRITICAL: DISCORD_CHANNEL_ID is missing in the environment/config.")
    elif not config.DEEPL_API_KEY:
         logger.warning("WARNING: DEEPL_API_KEY is missing. Translation will be disabled.")
         # Decide if you want the bot to run without translation or exit
         # exit() # Uncomment to force exit if DeepL key is missing

    # Proceed only if critical tokens are present
    if config.DISCORD_BOT_TOKEN and config.DISCORD_CHANNEL_ID:
        try:
            logger.info(f"Attempting to log in with token...")
            # Suppress default discord.py logging if using basicConfig
            bot.run(config.DISCORD_BOT_TOKEN, log_handler=None)
        except discord.LoginFailure:
            logger.critical("Login Failed: Invalid Discord Bot Token.")
        except discord.PrivilegedIntentsRequired:
             logger.critical("Privileged Intents (like Message Content) might be required and are not enabled in the Discord Developer Portal.")
        except Exception as e:
            logger.critical(f"Critical error running the bot: {e}", exc_info=True)
    else:
        logger.critical("Bot cannot start due to missing critical configuration (Token or Channel ID).")