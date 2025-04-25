import ollama
import logging
import config
import asyncio
import deepl
from deepl import DeepLException
import json
import re # Import regex for placeholder logic

logger = logging.getLogger(__name__)

# --- DeepL Translator Initialization (No Glossary Check) ---
deepl_translator = None
if config.DEEPL_API_KEY:
    try:
        deepl_translator = deepl.Translator(config.DEEPL_API_KEY)
        logger.info("DeepL Translator initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize DeepL Translator: {e}", exc_info=True)
        deepl_translator = None
else:
    logger.warning("DEEPL_API_KEY not found. DeepL Translation disabled.")

# --- Ollama Significance & Category Evaluation Function ---
async def evaluate_significance_and_category(text: str) -> dict | None:
    """
    Uses Ollama to evaluate headline significance AND determine its category.
    Outputs JSON: {"significant": bool, "category": str, "reason": str}
    """
    if not text:
        logger.warning("Received empty text for evaluation.")
        return None

    # --- Enhanced Prompt V5 (Focus on Ignoring & More Examples) ---
    prompt = f"""
    **TASK:** Analyze the financial news headline below. Determine if it is significant market news based ONLY on the provided criteria. Output a JSON object containing your decision, the primary category, and a brief reason.

    **HEADLINE:** "{text}"

    **CRITERIA FOR SIGNIFICANCE (Set "significant": true ONLY if headline CLEARLY meets one or more criteria below):**
    *   **IMPACT:** News highly likely to cause **noticeable (>1-2% for major assets)** short-term movement or change sentiment in broad markets (US stocks, major indices), specific sectors (Tech, Energy), major currencies (EUR/USD, USD/JPY), major cryptos (BTC, ETH > +/- 5%), or key commodities (Oil, Gold).
    *   **SOURCE:** **Direct, official policy announcements** (Fed/FOMC rate decisions/statements, ECB/BOJ policy shifts, White House statements on major economy/trade actions, OPEC decisions, major international summit outcomes like G7/G20 agreements).
    *   **DATA RELEASES:** **Key, market-moving** economic indicators (**CPI, NFP/Jobs Report, Core PCE, GDP growth/revision, ISM PMI** - especially if significantly deviating from consensus expectations).
    *   **MAJOR COMPANY NEWS:** Earnings reports/guidance **significant surprises** (large beat/miss) for **VERY large cap companies** (FAANG+, NVDA, TSLA, MSFT, JPM, etc.). Major M&A (> $10B or involving large caps). Major product launches with clear, broad market implications (e.g., new iPhone generation, major AI chip).
    *   **GEOPOLITICAL EVENTS:** **Major escalations/de-escalations** in conflicts DIRECTLY impacting markets (e.g., oil supply disruptions, major new sanctions on large economies, key trade deal breakthroughs/collapses).
    *   **CRYPTO:** **Major regulatory actions** with broad implications (SEC lawsuits against major exchanges, spot ETF approvals/rejections), major exchange solvency crises/hacks, >10% moves in BTC/ETH driven by verifiable, specific news.
    *   **FOREX:** **Direct central bank interventions** targeting currency rates, unexpected major policy shifts clearly impacting currency pairs (>1% move).

    **CRITERIA FOR IGNORING (Set "significant": false if headline fits ANY of these, EVEN IF IT SEEMS VAGUELY RELATED TO A SIGNIFICANT TOPIC):**
    *   **Routine Summaries:** Daily/weekly market open/close reports, futures/pre-market action *without* a specific major news driver mentioned. General "market update" articles.
    *   **Minor News:** Small/mid-cap company news (personnel, partnerships, minor product updates, local events). Most news about companies outside the top ~50 by market cap unless M&A related.
    *   **Analyst Ratings:** **IGNORE almost all individual analyst upgrades/downgrades/price target changes.** (Exception: Maybe a coordinated downgrade wave on a mega-cap causing >3% pre-market drop - very rare).
    *   **Opinion/Commentary:** Opinion pieces, interviews *unless* revealing new, concrete policy or data. General commentary, predictions, forecasts without specific triggers. "Expert says X..." articles. Listicles ("Top 5 Stocks..."). Educational content ("What is Forex?").
    *   **Vague Geopolitics:** Routine political commentary, minor diplomatic events, ongoing situations *without* a clear, new market-moving development reported in the headline.
    *   **Standard Volatility:** Normal price fluctuations in stocks, crypto, forex without a specific, major news catalyst mentioned in the headline. Headlines like "Bitcoin dips slightly" or "Stocks edge lower".
    *   **Promotional Content:** Content clearly marked as sponsored, PR releases.

    **CATEGORIES (Choose ONE primary category):**
    "Stocks", "Economy", "Forex", "Crypto", "Geopolitics", "Commodities", "General"

    **OUTPUT FORMAT:**
    CRITICAL: Output *ONLY* a single, valid JSON object adhering precisely to the format below. Do NOT include ANY introductory text, explanations, apologies, markdown formatting, or anything else before or after the JSON object. Your entire response MUST be the JSON object itself.
    ```json
    {{
      "significant": boolean,
      "category": "ChosenCategoryString",
      "reason": "Brief justification (2-5 words)"
    }}
    ```

    **EXAMPLES (Pay close attention to the 'significant: false' examples):**
    *   Headline: "Fed holds rates steady, signals potential cuts later this year."
        ```json
        {{
          "significant": true,
          "category": "Economy",
          "reason": "FOMC rate decision/outlook"
        }}
        ```
    *   Headline: "Apple shares fall 5% after reporting weaker iPhone sales."
        ```json
        {{
          "significant": true,
          "category": "Stocks",
          "reason": "AAPL earnings miss"
        }}
        ```
    *   Headline: "US Job Growth Slows Sharply in April, Unemployment Rate Ticks Up"
        ```json
        {{
          "significant": true,
          "category": "Economy",
          "reason": "NFP/Jobs data miss"
        }}
        ```
    *   Headline: "Oil prices jump 3% after OPEC+ announces surprise production cut"
        ```json
        {{
          "significant": true,
          "category": "Commodities",
          "reason": "OPEC+ production cut"
        }}
        ```
    *   **IGNORE EXAMPLE 1:** Headline: "Stock Market Today: Dow Closes Slightly Lower Ahead of Fed Meeting"
        ```json
        {{
          "significant": false,
          "category": "Stocks",
          "reason": "Routine market summary"
        }}
        ```
    *   **IGNORE EXAMPLE 2:** Headline: "Analyst upgrades MicroTech Inc. (MCTK) to 'Outperform'"
        ```json
        {{
          "significant": false,
          "category": "Stocks",
          "reason": "Minor analyst rating"
        }}
        ```
    *   **IGNORE EXAMPLE 3:** Headline: "CEO of MidCap Corp Discusses Future Strategy in Interview"
        ```json
        {{
          "significant": false,
          "category": "Stocks",
          "reason": "Interview/Commentary"
        }}
        ```
    *   **IGNORE EXAMPLE 4:** Headline: "Bitcoin Hovers Around $65,000 as Traders Await Next Catalyst"
        ```json
        {{
          "significant": false,
          "category": "Crypto",
          "reason": "Standard price movement"
        }}
        ```
    *   **IGNORE EXAMPLE 5:** Headline: "Understanding the Impact of Interest Rates on Bonds"
        ```json
        {{
          "significant": false,
          "category": "General",
          "reason": "Educational content"
        }}
        ```
    *   **IGNORE EXAMPLE 6:** Headline: "European Leaders Meet to Discuss Regional Cooperation"
        ```json
        {{
          "significant": false,
          "category": "Geopolitics",
          "reason": "Routine meeting/Vague"
        }}
        ```

    **FINAL JSON OUTPUT (ONLY the JSON object):**""" # End of prompt string

    try:
        client = ollama.AsyncClient(timeout=60)
        response = await client.chat(
            model=config.OLLAMA_MODEL,
            messages=[
                # System prompt reinforcing JSON-only output
                {'role': 'system', 'content': 'You are a financial news evaluator. Analyze the headline based ONLY on the user\'s criteria. Your entire response MUST be ONLY a single valid JSON object in the specified format: {"significant": boolean, "category": "CategoryString", "reason": "Brief ReasonString"}. Do not include explanations, apologies, markdown, or any text outside the JSON structure.'},
                {'role': 'user', 'content': prompt}
            ],
            options={
                 "temperature": 0.1, # Keep low for consistency
                 "num_predict": 100  # Allow reasonable length for JSON
            },
            # Keep format='json' from previous step
            format='json'
        )
        # When format='json', the response content should already be just the JSON string
        result_text = response['message']['content'].strip()

        # --- Attempt to parse the JSON response ---
        try:
            # Now directly load the result text, hoping it's valid JSON
            data = json.loads(result_text)

            # Validate expected keys and types
            if isinstance(data.get("significant"), bool) and isinstance(data.get("category"), str):
                # Normalize category
                category = data.get("category", "Unknown").strip().capitalize()
                if category not in config.CATEGORIES:
                    logger.warning(f"LLM returned unknown category '{category}'. Using 'General'.")
                    category = "General" # Or maybe Unknown

                data["category"] = category # Ensure normalized category is stored
                data["reason"] = data.get("reason", "").strip() # Clean reason

                log_level = logging.INFO if data["significant"] else logging.DEBUG # Log ignored ones as debug
                logger.log(log_level, f"{'SIGNIFICANT' if data['significant'] else 'IGNORE'} ({data['category']}): '{text[:60]}...' | Reason: {data.get('reason','')}")
                return data # Return the parsed data
            else:
                logger.error(f"LLM JSON missing required keys ('significant', 'category') or wrong types (format='json' used): '{result_text}'")
                return None

        except json.JSONDecodeError as json_e:
            logger.error(f"Failed to decode JSON from LLM response (format='json' used): {json_e}")
            logger.error(f"LLM Raw Text: '{result_text}'")
            return None

    except ollama.ResponseError as e:
         logger.error(f"Ollama API Error during evaluation: {e.status_code} - {e.error}")
         return None
    except asyncio.TimeoutError:
        logger.error(f"Ollama evaluation timed out for: '{text[:60]}...'")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during Ollama evaluation: {type(e).__name__} - {e}", exc_info=True)
        return None


# --- DeepL Translation Function (Using Placeholder Workaround - Unchanged) ---
async def translate_en_to_el(text: str) -> str | None:
    """
    Translates English text to Modern Greek using DeepL API,
    preserving specific terms defined in config.NO_TRANSLATE_TERMS.
    """
    if not deepl_translator:
        logger.error("DeepL translator not available.")
        return None
    if not text:
        logger.warning("Received empty text for DeepL translation.")
        return None

    placeholders = {}
    current_text = text
    placeholder_index = 0

    # 1. Pre-process: Replace terms with placeholders
    for term in config.NO_TRANSLATE_TERMS:
        if re.search(r'\W', term): # If term has non-word chars (like EUR/USD)
             pattern = re.compile(re.escape(term), re.IGNORECASE)
        else: # Otherwise, use word boundaries
             pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)

        matches = list(pattern.finditer(current_text))
        for match in reversed(matches):
            original_term = match.group(0) # Get the exact matched text (with original casing)
            placeholder = f"__PLACEHOLDER_{placeholder_index}__"
            placeholders[placeholder] = original_term
            current_text = current_text[:match.start()] + placeholder + current_text[match.end():]
            placeholder_index += 1
            logger.debug(f"Replaced '{original_term}' with '{placeholder}'")

    if not placeholders:
         logger.debug("No terms found to replace with placeholders.")

    processed_text = current_text # Text with placeholders ready for translation
    logger.debug(f"Text sent to DeepL: '{processed_text[:100]}...'")

    # 2. Translate the processed text
    try:
        loop = asyncio.get_running_loop()
        def sync_translate():
            # No glossary argument needed now
            return deepl_translator.translate_text(processed_text, target_lang="EL")

        result = await loop.run_in_executor(None, sync_translate)

        if result and result.text:
            translated_with_placeholders = result.text
            logger.debug(f"Received from DeepL: '{translated_with_placeholders[:100]}...'")

            # 3. Post-process: Replace placeholders back with original terms
            final_translation = translated_with_placeholders
            for placeholder, original_term in sorted(placeholders.items(), key=lambda item: len(item[0]), reverse=True):
                final_translation = final_translation.replace(placeholder, original_term)
                logger.debug(f"Reverted '{placeholder}' back to '{original_term}'")

            logger.info(f"DeepL Final Translation: '{final_translation[:60]}...'")
            return final_translation
        else:
            logger.warning(f"DeepL translation returned empty result for processed text: '{processed_text[:60]}...'")
            return None

    except DeepLException as e:
        logger.error(f"DeepL API error: {e}")
        if "Quota" in str(e) or "limit" in str(e): logger.error(">>> DeepL Quota likely exceeded! <<<")
        elif "Authorization" in str(e) or "AuthKey" in str(e): logger.error(">>> DeepL API Key seems invalid! <<<")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during DeepL translation: {type(e).__name__} - {e}", exc_info=True)
        return None


# --- Example usage (for testing this file directly - Unchanged) ---
if __name__ == "__main__":
    # You might want to add test cases here using the new prompt logic
    logging.basicConfig(level=logging.DEBUG) # Set to DEBUG to see more logs when testing directly
    logger.info("Testing llm_handler directly...")

    async def run_test():
        test_headlines = [
            "Fed holds rates steady, signals potential cuts later this year.", # SIG: Economy
            "Apple shares fall 5% after reporting weaker iPhone sales.", # SIG: Stocks
            "Stock Market Today: Dow Closes Slightly Lower Ahead of Fed Meeting", # IGNORE: Stocks
            "Analyst upgrades MicroTech Inc. (MCTK) to 'Outperform'", # IGNORE: Stocks
            "Bitcoin Hovers Around $65,000 as Traders Await Next Catalyst", # IGNORE: Crypto
            "Understanding the Impact of Interest Rates on Bonds", # IGNORE: General
            "Oil prices jump 3% after OPEC+ announces surprise production cut" # SIG: Commodities
        ]
        for headline in test_headlines:
            print(f"\n--- Testing Headline: {headline} ---")
            eval_result = await evaluate_significance_and_category(headline)
            if eval_result:
                print(f"Evaluation Result: {eval_result}")
                if eval_result["significant"]:
                    print("Attempting translation...")
                    translation = await translate_en_to_el(headline)
                    print(f"Translation: {translation if translation else 'Failed'}")
            else:
                print("Evaluation Failed (Check logs)")

    asyncio.run(run_test())
    logger.info("Direct test finished.")