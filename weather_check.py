import requests

# Your OpenWeatherMap API key
api_key = "34e22345195b091617eac34bbfea742e"  # Replace this with your actual key

# City and parameters
city = "Dubai"
units = "metric"  # Use 'imperial' for Fahrenheit
url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units={units}&appid={api_key}"

# Send request
response = requests.get(url)
data = response.json()

# Extract data
if response.status_code == 200:
    weather = data['weather'][0]['description'].capitalize()
    temperature = data['main']['temp']
    print(f"The weather in {city} today is {weather} with a temperature of {temperature}°C.")
else:
    print(f"Failed to get weather data: {data.get('message', 'Unknown error')}")
