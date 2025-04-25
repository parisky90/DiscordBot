import aiohttp
from bs4 import BeautifulSoup
import logging
from typing import List, Tuple, Dict, Any
from urllib.parse import urljoin
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

# --- Helper Function ---
async def fetch_html(session: aiohttp.ClientSession, url: str) -> str | None:
    """Fetches HTML content from a URL asynchronously."""
    # Use headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache',
    }
    try:
        # Added timeout and disable ssl verification for flexibility (though use carefully)
        async with session.get(url, headers=headers, timeout=20, ssl=False) as response:
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            logger.info(f"Successfully fetched {url} with status {response.status}")
            return await response.text()
    except aiohttp.ClientResponseError as e:
        logger.error(f"HTTP Error fetching {url}: Status {e.status}, Message: {e.message}")
    except aiohttp.ClientConnectionError as e:
        logger.error(f"Connection Error fetching {url}: {e}")
    except asyncio.TimeoutError:
        logger.error(f"Timeout error fetching {url}")
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {type(e).__name__} - {e}")
    return None

# --- Website Specific Scrapers ---
# IMPORTANT: These selectors are examples and WILL LIKELY BREAK as websites update their structure.
# They need regular inspection and maintenance.

async def scrape_marketwatch(session: aiohttp.ClientSession, config: Dict[str, Any]) -> List[Tuple[str, str, datetime]]:
    """Scrapes headlines from MarketWatch."""
    url = config['url']
    base_url = config['base_url']
    html = await fetch_html(session, url)
    if not html: return []
    soup = BeautifulSoup(html, 'html.parser')
    headlines = []
    detection_time = datetime.now()
    # Selector needs constant checking - this is a common pattern
    articles = soup.select('div.element--article, div.article__content')
    for article in articles:
        headline_tag = article.select_one('h3.article__headline > a.link, h3.article__headline > a.article__link, a.link, a[href]')
        if headline_tag and headline_tag.text:
            headline = ' '.join(headline_tag.text.split()) # Clean whitespace
            link = headline_tag.get('href')
            absolute_link = urljoin(base_url, link) if link else url
            if headline and absolute_link:
                headlines.append((headline, absolute_link, detection_time))
                if len(headlines) >= 15: break
    logger.info(f"MarketWatch: Found {len(headlines)} potential headlines.")
    return headlines

async def scrape_cnbc(session: aiohttp.ClientSession, config: Dict[str, Any]) -> List[Tuple[str, str, datetime]]:
    """Scrapes headlines from CNBC."""
    url = config['url']
    base_url = config['base_url']
    html = await fetch_html(session, url)
    if not html: return []
    soup = BeautifulSoup(html, 'html.parser')
    headlines = []
    detection_time = datetime.now()
    # Selectors for CNBC vary a lot
    articles = soup.select('div.Card-standardBreakerCard, div.Card-titleContainer, li.LatestNews-item, div[class*="RiverCard-container"]')
    for article in articles:
        headline_tag = article.select_one('a[href]') # More generic link selector
        if headline_tag:
             title_tag = article.select_one('.Card-title, .LatestNews-headline, [class*="RiverHeadline-headline"]') # Find title specifically
             headline = ' '.join(title_tag.text.split()) if title_tag else ' '.join(headline_tag.text.split())
             link = headline_tag.get('href')
             absolute_link = urljoin(base_url, link) if link else url
             if headline and absolute_link:
                 headlines.append((headline, absolute_link, detection_time))
                 if len(headlines) >= 15: break
    logger.info(f"CNBC: Found {len(headlines)} potential headlines.")
    return headlines


async def scrape_yahoo_finance(session: aiohttp.ClientSession, config: Dict[str, Any]) -> List[Tuple[str, str, datetime]]:
    """Scrapes headlines from Yahoo Finance."""
    url = config['url']
    base_url = config['base_url']
    html = await fetch_html(session, url)
    if not html: return []
    soup = BeautifulSoup(html, 'html.parser')
    headlines = []
    detection_time = datetime.now()
    # Yahoo's structure is often complex and JS-reliant
    articles = soup.select('li.js-stream-content h3 a[href]') # Select links directly
    if not articles: # Fallback selector
         articles = soup.select('div[class*="stream-item"] a[href]')

    for headline_tag in articles:
        headline = ' '.join(headline_tag.text.split())
        link = headline_tag.get('href')
        # Yahoo links are often absolute but sometimes relative
        if link and link.startswith('/'):
            absolute_link = urljoin(base_url, link)
        elif link and link.startswith('http'):
            absolute_link = link
        else:
             absolute_link = url # Fallback

        if headline and absolute_link and not headline.lower().startswith("yahoo finance"): # Filter out generic links
            headlines.append((headline, absolute_link, detection_time))
            if len(headlines) >= 15: break
    logger.info(f"Yahoo Finance: Found {len(headlines)} potential headlines.")
    return headlines


async def scrape_finviz(session: aiohttp.ClientSession, config: Dict[str, Any]) -> List[Tuple[str, str, datetime]]:
    """Scrapes headlines from Finviz News page."""
    url = config['url']
    base_url = config['base_url']
    html = await fetch_html(session, url)
    if not html: return []
    soup = BeautifulSoup(html, 'html.parser')
    headlines = []
    detection_time = datetime.now()
    news_table = soup.select_one('table.news-table')
    if news_table:
        rows = news_table.find_all('tr')
        for row in rows:
            headline_tag = row.select_one('td a.nn-tab-link, td a.tab-link-news')
            if headline_tag and headline_tag.text:
                headline = ' '.join(headline_tag.text.split())
                link = headline_tag.get('href')
                absolute_link = urljoin(base_url, link) if link else url
                if headline and absolute_link:
                    headlines.append((headline, absolute_link, detection_time))
                    if len(headlines) >= 15: break
    logger.info(f"Finviz: Found {len(headlines)} potential headlines.")
    return headlines


async def scrape_seeking_alpha(session: aiohttp.ClientSession, config: Dict[str, Any]) -> List[Tuple[str, str, datetime]]:
    """Scrapes headlines from Seeking Alpha Market News."""
    url = config['url']
    base_url = config['base_url']
    html = await fetch_html(session, url)
    if not html: return []
    soup = BeautifulSoup(html, 'html.parser')
    headlines = []
    detection_time = datetime.now()
    # Seeking Alpha often uses data attributes
    articles = soup.select('article[data-test-id="post-list-item"] a[data-test-id="post-list-item-title"], div[class*="media-body"] a[href]')
    for headline_tag in articles:
        headline = ' '.join(headline_tag.text.split())
        link = headline_tag.get('href')
        absolute_link = urljoin(base_url, link) if link else url
        if headline and absolute_link:
            headlines.append((headline, absolute_link, detection_time))
            if len(headlines) >= 15: break
    logger.info(f"Seeking Alpha: Found {len(headlines)} potential headlines.")
    return headlines


# --- Main Scraper Function ---
SCRAPER_FUNCTIONS = {
    "MarketWatch": scrape_marketwatch,
    "CNBC": scrape_cnbc,
    "Yahoo Finance": scrape_yahoo_finance,
    "Finviz": scrape_finviz,
    "Seeking Alpha": scrape_seeking_alpha,
}

async def scrape_all(sources_config: Dict[str, Dict[str, Any]]) -> Dict[str, List[Tuple[str, str, datetime]]]:
    """Scrapes all configured sources concurrently."""
    results = {}
    # Create connector with limited connections to avoid overwhelming sites or hitting local limits
    connector = aiohttp.TCPConnector(limit_per_host=2, limit=10, ssl=False) # Adjust limits as needed
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        source_names = []
        for name, config_data in sources_config.items():
            if name in SCRAPER_FUNCTIONS:
                tasks.append(asyncio.create_task(SCRAPER_FUNCTIONS[name](session, config_data)))
                source_names.append(name)
            else:
                logger.warning(f"No scraper function defined for source: {name}")

        scraped_data_list = await asyncio.gather(*tasks, return_exceptions=True)

        for i, data in enumerate(scraped_data_list):
            source_name = source_names[i]
            if isinstance(data, Exception):
                logger.error(f"Error scraping {source_name}: {data}", exc_info=False) # Keep log concise
                results[source_name] = []
            elif data is not None:
                results[source_name] = data
            else:
                results[source_name] = []

    return results

# Example Usage (for testing scraper.py directly)
if __name__ == "__main__":
    import config as app_config
    logging.basicConfig(level=logging.INFO)

    async def test_scrape():
        print("Testing scrapers...")
        all_headlines = await scrape_all(app_config.NEWS_SOURCES)
        print("\n--- Results ---")
        for source, headlines in all_headlines.items():
            print(f"\n{source} ({len(headlines)} found):")
            if headlines:
                for i, (title, link, time) in enumerate(headlines[:3]): # Print first 3
                    print(f"  - [{time.strftime('%H:%M:%S')}] {title} ({link})")
            else:
                print("  - No headlines found or error occurred.")

    try:
        asyncio.run(test_scrape())
    except RuntimeError as e:
        if "cannot run loop" in str(e):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(test_scrape())
        else:
            raise