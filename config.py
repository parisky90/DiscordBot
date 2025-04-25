import os
from dotenv import load_dotenv
import discord # Import discord for Color object
import logging # Import logging

# Initialize logger for config file messages
logger = logging.getLogger(__name__)

load_dotenv() # Load variables from .env file

# --- Core Settings ---
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID")) if os.getenv("DISCORD_CHANNEL_ID") else None
HEARTBEAT_INTERVAL_MINUTES = int(os.getenv("HEARTBEAT_INTERVAL_MINUTES", 10))
SCRAPE_INTERVAL_SECONDS = int(os.getenv("SCRAPE_INTERVAL_SECONDS", 60))

# --- DeepL Settings ---
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
# DEEPL_GLOSSARY_ID REMOVED

# --- Ollama Settings ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# --- Behaviour Settings ---
raw_keywords = os.getenv("USER_KEYWORDS", "")
USER_KEYWORDS = set(k.strip().lower() for k in raw_keywords.split(',') if k.strip())
FUZZY_MATCH_THRESHOLD = 88
SENTIMENT_POSITIVE_THRESHOLD = 0.1
SENTIMENT_NEGATIVE_THRESHOLD = -0.1

# **NEW**: Define terms/patterns to keep untranslated (case-insensitive matching)
# Use lowercase for easier matching. Order longer terms before shorter ones if they overlap (e.g., 'Dow Jones' before 'Dow')
# Regex patterns can be used too, but start simple.
NO_TRANSLATE_TERMS = sorted([
    # Acronyms & Symbols
    "fomc", "fed", "ecb", "boj", "boe", "opec", "sec", "esma",
    "cpi", "nfp", "gdp", "pmi", "ism",
    "eur/usd", "usd/jpy", "gbp/usd", "usd/chf", "aud/usd", "usd/cad", # Major FX pairs
    "btc", "eth", "xrp", "sol", "ada", # Major Crypto symbols
    "aapl", "msft", "goog", "googl", "amzn", "nvda", "tsla", "meta", # Major Tickers
    "jpm", "bac", "wfc", "gs", # Banks
    "xom", "cvx", # Energy
    # Specific Names (will be matched case-insensitively)
    "bitcoin", "ethereum", "nvidia", "tesla", "microsoft", "apple", "amazon", "google",
    "federal reserve", "european central bank",
    # Financial Jargon (add cautiously - might interfere with translation)
    # "quantitative easing", "bull market", "bear market" # Example: Maybe translate these? Decide case-by-case.
], key=len, reverse=True) # Sort by length descending to match longer terms first

CATEGORIES = {
    "Stocks": discord.Color.blue(),
    "Economy": discord.Color.gold(),
    "Forex": discord.Color.purple(),
    "Crypto": discord.Color.orange(),
    "Geopolitics": discord.Color.red(),
    "Commodities": discord.Color.dark_green(),
    "General": discord.Color.light_grey(),
    "Unknown": discord.Color.dark_grey()
}
DEFAULT_CATEGORY_COLOR = discord.Color.blurple()

# --- News Sources Configuration ---
NEWS_SOURCES = {
    "MarketWatch": {"url": "https://www.marketwatch.com/latest-news", "base_url": "https://www.marketwatch.com"},
    "CNBC": {"url": "https://www.cnbc.com/world/?region=world", "base_url": "https://www.cnbc.com"},
    "Yahoo Finance": {"url": "https://finance.yahoo.com/topic/stock-market-news/", "base_url": "https://finance.yahoo.com"},
    "Finviz": {"url": "https://finviz.com/news.ashx", "base_url": "https://finviz.com/"},
    "Seeking Alpha": {"url": "https://seekingalpha.com/market-news", "base_url": "https://seekingalpha.com"}
}

# --- Essential Variable Checks ---
if not DISCORD_BOT_TOKEN: raise ValueError("Missing DISCORD_BOT_TOKEN")
if not DISCORD_CHANNEL_ID: raise ValueError("Missing DISCORD_CHANNEL_ID")
if not DEEPL_API_KEY: logger.warning("Missing DEEPL_API_KEY. DeepL Translation disabled.")

print("Configuration loaded successfully.")
print(f"Target Discord Channel ID: {DISCORD_CHANNEL_ID}")
print(f"DeepL API Key loaded: {'Yes' if DEEPL_API_KEY else 'NO'}")
# print(f"DeepL Glossary ID: REMOVED") # No longer relevant
print(f"Ollama Model for Eval: {OLLAMA_MODEL}")
print(f"User Keywords loaded: {len(USER_KEYWORDS)}")
print(f"No-Translate Terms loaded: {len(NO_TRANSLATE_TERMS)}")
print(f"Fuzzy Match Threshold: {FUZZY_MATCH_THRESHOLD}")