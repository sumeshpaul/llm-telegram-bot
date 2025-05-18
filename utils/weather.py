import aiohttp
import os

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

async def get_weather_for_city(city: str) -> str:
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"OpenWeather API error: {resp.status}")
            data = await resp.json()

            # 🔒 Sanity check: ensure valid city info
            if (
                "coord" not in data
                or not data.get("name")
                or data["name"].lower() == city.lower()  # suspicious if name is identical and generic
                or data["name"].lower() in ["atlantis", "city"]
            ):
                raise Exception("City not found in weather API.")

            weather = data["weather"][0]["description"].capitalize()
            temp = data["main"]["temp"]
            city_name = data["name"]
            return f"🌤️ Weather in {city_name} today: {weather}, {temp}°C."
