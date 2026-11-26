# %% Imports
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import json


# %% 1. Get daily weather for a few cities over the past week
def get_weather(latitude, longitude, start_date, end_date):
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min",
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "auto",
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()          # the full response, including data['daily']

today = datetime.now()
week_ago = today - timedelta(days=7)
start_date = week_ago.strftime("%Y-%m-%d")
end_date = today.strftime("%Y-%m-%d")

paris_data = get_weather(48.85, 2.35, start_date, end_date)
london_data = get_weather(51.50, -0.12, start_date, end_date)
tokyo_data = get_weather(35.68, 139.69, start_date, end_date)

print(f"Paris today (max): {paris_data['daily']['temperature_2m_max'][-1]}°C")
print(f"London today (max): {london_data['daily']['temperature_2m_max'][-1]}°C")
print(f"Tokyo today (max): {tokyo_data['daily']['temperature_2m_max'][-1]}°C")

data = paris_data


# %% 2. Process with pandas
df = pd.DataFrame({
    'date': pd.to_datetime(data['daily']['time']),
    'max_temp': data['daily']['temperature_2m_max'],
    'min_temp': data['daily']['temperature_2m_min']
})

# 3. Calculate average
df['avg_temp'] = (df['max_temp'] + df['min_temp']) / 2
print(df)

# %% 4. Create visualization
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['max_temp'], 'r-o', label='Max')
plt.plot(df['date'], df['min_temp'], 'b-o', label='Min')
plt.plot(df['date'], df['avg_temp'], 'g--', label='Average')

plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Paris Weather - Past Week')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# %% 5. Save everything
os.makedirs('data', exist_ok=True)

plt.savefig('data/weather_chart.png')
df.to_csv('data/paris_weather.csv', index=False)

with open('data/paris_data.json', 'w', encoding='utf-8') as json_file:
    json.dump(paris_data, json_file, indent=4, ensure_ascii=False)

print(f"Average temperature: {df['avg_temp'].mean():.1f}°C")
print("Files saved in 'data' folder")