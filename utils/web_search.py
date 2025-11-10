import logging
import requests
from typing import List, Tuple, Optional

logging.basicConfig(level=logging.DEBUG)  # DEBUG to check what's happening
logger = logging.getLogger(__name__)

# ---------------------------
# DuckDuckGo Instant Answer
# ---------------------------
def duckduckgo_search(query: str, max_results: int = 5) -> List[Tuple[str, str]]:
    """Use DuckDuckGo Instant Answer API (no key required)."""
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        results: List[Tuple[str, str]] = []

        abstract = data.get("AbstractText") or data.get("Answer")
        if abstract:
            results.append((abstract, data.get("AbstractURL", "")))

        for topic in data.get("RelatedTopics", []):
            if isinstance(topic, dict):
                text = topic.get("Text")
                link = topic.get("FirstURL", "")
                if text and link:
                    results.append((text, link))
            if len(results) >= max_results:
                break

        logger.debug(f"DuckDuckGo returned {len(results)} results")
        return results[:max_results]
    except Exception as e:
        logger.debug(f"DuckDuckGo search failed: {e}")
        return []

# ---------------------------
# Wikipedia summary (REST API)
# ---------------------------
def wikipedia_summary(query: str, sentences: int = 2) -> Optional[Tuple[str, str]]:
    """Try Wikipedia REST summary (no key)."""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.requote_uri(query)}"
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            extract = data.get("extract")
            web_url = data.get("content_urls", {}).get("desktop", {}).get("page")
            if extract:
                logger.debug("Wikipedia summary found")
                return (extract, web_url or f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}")
    except Exception as e:
        logger.debug(f"Wikipedia lookup failed: {e}")
    return None

# ---------------------------
# Public Searx instance fallback
# ---------------------------
def searx_search(query: str, max_results: int = 5) -> List[Tuple[str, str]]:
    """Try a public searx instance (no key)."""
    public_instances = [
        "https://searx.tiekoetter.com/search",
        "https://searx.be/search"
    ]
    results: List[Tuple[str, str]] = []
    for base in public_instances:
        try:
            resp = requests.get(base, params={"q": query, "format": "json", "language": "en"}, timeout=8)
            if resp.status_code != 200:
                continue
            data = resp.json()
            for r in data.get("results", [])[:max_results]:
                title = r.get("title") or r.get("content") or r.get("url")
                url = r.get("url", "")
                snippet = r.get("content") or ""
                results.append((f"{title} — {snippet}", url))
            if results:
                logger.debug(f"Searx returned {len(results)} results")
                return results[:max_results]
        except Exception as e:
            logger.debug(f"Searx instance failed: {e}")
            continue
    return []

# ---------------------------
# Stocks via yfinance (no API key)
# ---------------------------
def get_stock_price(ticker: str) -> Optional[str]:
    """Return a short stock summary using yfinance."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        currency = info.get("currency", "")
        if price is not None:
            return f"{ticker.upper()} current price: {price} {currency}"
    except Exception as e:
        logger.debug(f"yfinance lookup failed: {e}")
    return None

# ---------------------------
# Main helper
# ---------------------------
def get_web_results_text(query: str, max_results: int = 5) -> str:
    """Try multiple free sources and return formatted results string."""
    try:
        # 1) Wikipedia quick summary
        wiki = wikipedia_summary(query, sentences=3)
        if wiki:
            summary, url = wiki
            return f"📘 Wikipedia summary:\n\n{summary}\n\n🔗 {url}"

        # 2) DuckDuckGo
        ddg = duckduckgo_search(query, max_results=max_results)
        if ddg:
            formatted = "\n\n".join([f"{i+1}. {title}\n🔗 {link}" for i, (title, link) in enumerate(ddg)])
            return f"🌐 DuckDuckGo results for '{query}':\n\n{formatted}"

        # 3) Searx public instances fallback
        searx = searx_search(query, max_results=max_results)
        if searx:
            formatted = "\n\n".join([f"{i+1}. {title}\n🔗 {link}" for i, (title, link) in enumerate(searx)])
            return f"🌐 Searx results for '{query}':\n\n{formatted}"

        # 4) Stock lookup if query contains "stock" or "price"
        if any(token.lower() in query.lower() for token in ["stock", "price", "share"]):
            tokens = query.split()
            possible = [t.strip().upper() for t in tokens if t.strip()]
            for p in possible:
                stock_info = get_stock_price(p)
                if stock_info:
                    return f"💹 {stock_info}"

        return "No relevant search results found."
    except Exception as e:
        logger.exception(f"Error in get_web_results_text: {e}")
        return "Error retrieving web results."
