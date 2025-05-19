import os
import aiohttp
from dotenv import load_dotenv

load_dotenv()
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

async def get_gnews_results(query: str) -> str:
    """
    Fetch top 3 news articles from GNews API based on user query.
    Auto-expands for vague or news-related prompts.
    """
    if not GNEWS_API_KEY:
        return "⚠️ GNews API key is not set."

    trigger_keywords = [
        "news", "update", "breaking", "headline", "report",
        "happening", "incident", "current events", "live", "latest", "alert"
    ]

    # Normalize and extend query if too short or vague
    import urllib.parse
    clean_query = re.sub(r"[^\w\s]", "", query.strip().lower())  # Remove punctuation
    if len(clean_query.split()) < 2 or is_news_query(clean_query):
        clean_query += " latest news"

    encoded_query = urllib.parse.quote_plus(clean_query)
    url = f"https://gnews.io/api/v4/search?q={encoded_query}&lang=en&max=3&token={GNEWS_API_KEY}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    return f"⚠️ GNews error {response.status}"

                data = await response.json()
                articles = data.get("articles", [])
                if not articles:
                    return f"⚠️ No news found for '{query}'."

                result_lines = [f"📰 News for: *{query.title()}*"]
                for article in articles:
                    title = article.get("title", "No title")
                    url = article.get("url", "#")
                    source = article.get("source", {}).get("name", "Unknown")
                    result_lines.append(f"• [{title}]({url}) _({source})_")

                return "\n\n".join(result_lines)

    except Exception as e:
        return f"⚠️ News fetch failed: {e}"
