import os
import re
import urllib.parse
import asyncio
import aiohttp
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv()
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

def clean_fallback(text: str, query: str) -> str:
    text = re.sub(r"http[s]?://\\S+", "", text).strip()  # Remove URLs
    text = re.sub(r"\\[\\w{1,3}\\]", "", text)  # Remove [1], [a], etc.

    # If short or lacks punctuation, wrap with fallback context
    if len(text) < 40 or not any(p in text for p in '.!?\n'):
        return f"I couldn't find a detailed answer, but here's what I found about \"{query}\": {text}"

    return text

def is_news_query(query: str) -> bool:
    keywords = [
        "news", "latest", "breaking", "headline", "update", "happening",
        "event", "report", "incident", "alert", "current affairs"
    ]
    return any(k in query.lower() for k in keywords)


async def gnews_search(query: str) -> str:
    if not GNEWS_API_KEY:
        return "⚠️ GNews API key is not set."

    clean_query = re.sub(r"[^\w\s]", "", query.strip().lower())
    if len(clean_query.split()) < 2 or is_news_query(clean_query):
        clean_query += " latest news"

    encoded_query = urllib.parse.quote_plus(clean_query)
    url = f"https://gnews.io/api/v4/search?q={encoded_query}&lang=en&sortby=publishedAt&max=5&token={GNEWS_API_KEY}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    return f"⚠️ GNews error {response.status}"
                data = await response.json()
                articles = data.get("articles", [])

                if not articles:
                    return f"⚠️ No news found for '{query}'."

                result_lines = [f"📰 *Top News for:* _{query}_"]
                for article in articles:
                    title = article.get("title", "No title")
                    url = article.get("url", "#")
                    source = article.get("source", {}).get("name", "Unknown")
                    published = article.get("publishedAt", "")[:10]
                    result_lines.append(f"• [{title}]({url}) ({source}, {published})")

                return "\n\n".join(result_lines)

    except Exception as e:
        return f"⚠️ GNews failed: {e}"

from duckduckgo_search import DDGS

async def duckduckgo_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region="wt-wt", safesearch="Moderate", max_results=3)
            for r in results:
                snippet = r.get("body", r.get("snippet", ""))
                # Remove footnotes like [1], [a], etc.
                cleaned = re.sub(r"\[\w{1,3}\]", "", snippet)
                # Remove extra spaces and generic openings
                cleaned = re.sub(r"^(Find out|Learn more|Discover)[^:]*:\s*", "", cleaned, flags=re.IGNORECASE)
                # Get the first sentence only
                first_sentence = re.split(r'(?<=[.!?])\s+', cleaned.strip())[0]
                if len(first_sentence) > 5:
                    return clean_fallback(first_sentence, query)
            return f"⚠️ No clean results found for '{query}'."
    except Exception as e:
        return f"⚠️ DuckDuckGo error: {e}"

async def fallback_search(query: str) -> str:
    if is_news_query(query):
        return await gnews_search(query)
    return await duckduckgo_search(query)
