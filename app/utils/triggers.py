import re
import string

# Weather keywords
WEATHER_TRIGGERS = [
    "weather", "temperature", "temp", "forecast", "climate", "humidity", "rain",
    "storm", "snow", "wind", "heat", "fog", "cold", "hot", "conditions", "sun",
    "air", "drizzle", "sandstorm", "dust", "chilly", "sunny", "thunderstorm", "breeze"
]

# Keywords that imply current or real-time data
FALLBACK_KEYWORDS = [
    "current", "latest", "today", "now", "recent", "new", "updated", "live", "real time",
    "breaking", "headline", "ongoing", "status", "alert", "news", "situation", "ongoing"
]

# Entities and topics that often require factual or dynamic answers
ENTITY_KEYWORDS = [
    "prime minister", "president", "ceo", "founder", "director", "capital", "population",
    "leader", "currency", "exchange rate", "stock", "market", "price", "ranking", "winner",
    "score", "flight", "airport", "visa", "holiday", "event", "tournament", "time in",
    "schedule", "map", "distance", "bank", "weather in", "real estate", "gold", "oil",
    "electricity", "internet outage", "temperature in", "who is", "what is", "is it"
]

# ========== Weather Detection ==========
def is_weather_related(prompt: str) -> bool:
    cleaned = prompt.lower().translate(str.maketrans('', '', string.punctuation))
    return any(word in cleaned for word in WEATHER_TRIGGERS)

# ========== Fallback Trigger Logic ==========
def is_fallback_needed(prompt: str, response: str) -> bool:
    weak_phrases = [
        "i don't have information",
        "i do not have information",
        "i'm not sure",
        "i am not sure",
        "as a language model",
        "i cannot provide",
        "i do not know",
        "i don't know",
        "no data available",
        "unable to retrieve",
        "i'm sorry, but",
        "i don't have access"
    ]

    if not response or response.strip() == "":
        return True

    for phrase in weak_phrases:
        if phrase in response.lower():
            return True

    # Also fallback based on prompt (for current/news queries)
    cleaned_prompt = prompt.lower().translate(str.maketrans('', '', string.punctuation))
    return any(k in cleaned_prompt for k in FALLBACK_KEYWORDS + ENTITY_KEYWORDS)

# ========== DuckDuckGo Results Formatter ==========
def format_duckduckgo_results(results: list, query: str) -> str:
    if not results:
        return f"⚠️ No results found for '{query}'."
    formatted = [f"🔎 *Top results for:* _{query}_\n"]
    for r in results[:5]:  # Return top 5
        title = r.get("title", "Untitled")
        href = r.get("href", "")
        formatted.append(f"• [{title}]({href})")
    return "\n".join(formatted)
