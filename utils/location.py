import re

def extract_city_name(prompt: str) -> str | None:
    prompt = prompt.lower()

    patterns = [
        r"weather\s+(?:in|at|for)?\s*([A-Za-z\s]+)",      # e.g., "weather in Kerala"
        r"(?:in|at|for)\s+([A-Za-z\s]+)\s+weather",       # e.g., "in Kerala weather"
        r"weather.*?in\s+([A-Za-z\s]+)",                  # e.g., "what is the weather in UAE"
        r"weather.*?([A-Za-z\s]+)$",                      # fallback
    ]

    for pattern in patterns:
        match = re.search(pattern, prompt)
        if match:
            city = match.group(1).strip().title()
            return city
    return None
