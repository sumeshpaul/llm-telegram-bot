# /app/utils/wikipedia_fallback.py
import aiohttp

async def get_wikipedia_weather_summary(city: str):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{city}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                extract = data.get("extract", "No summary found.")
                return {
                    "text": f"📚 [Wikipedia] {extract}",
                    "source": f"wiki-weather:{city}",
                    "fallback_reason": "OpenWeatherMap failed"
                }
            else:
                raise Exception(f"Wiki API error: {resp.status}")

async def get_wikipedia_summary(topic: str):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": topic,
        "explaintext": True,
        "exsentences": 10,
        "redirects": 1
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                pages = data["query"]["pages"]
                page = next(iter(pages.values()))
                extract = page.get("extract", "").strip()

                extract = extract.replace("()", "").replace("(lit.", "").strip()
                if not extract.endswith(('.', '!', '?')):
                    extract += '.'

                # Require at least 40 words or raise
                if len(extract.split()) < 40:
                    raise Exception("Wikipedia summary too short")

                return {
                    "text": extract,
                    "source": f"wiki:{topic}",
                    "fallback_reason": "LLM and DuckDuckGo failed"
                }
            else:
                raise Exception(f"Wikipedia API error: {resp.status}")
