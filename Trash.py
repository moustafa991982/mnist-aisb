# %% Imports
import datetime
import os
import json
import requests


# %% Current weather from OpenWeatherMap
# --- Current weather from OpenWeatherMap (New York) ---
# The API key is read from an environment variable so it never ends up in the code or on GitHub.
# It is loaded from the .env file next to this script (which is in .gitignore).
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd(), ".env")
if os.path.exists(env_path):
    with open(env_path) as env_file:
        for line in env_file:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

latitude = 40.7128
longitude = -74.0060
api_key = os.environ.get("OPENWEATHER_API_KEY")

if not api_key:
    print("OPENWEATHER_API_KEY is not set, skipping the weather request.")
else:
    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"lat": latitude, "lon": longitude, "appid": api_key, "units": "metric"},
        timeout=10,
    )
    if response.status_code == 200:
        weather_data = response.json()
        print(f"Weather in {weather_data['name']}: {weather_data['main']['temp']}°C, "
              f"{weather_data['weather'][0]['description']}")
    elif response.status_code == 401:
        print("401: invalid API key (new keys can take up to ~2 hours to activate).")
    else:
        print(f"Weather request failed: {response.status_code} {response.text}")

# %% Write data.json
data ={"name": "John", "age": 30, "city": "New York"}
with open("data.json", "w") as json_file:
    json.dump(data, json_file)
# %% Working directory and date
current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")
today = datetime.date.today()
print(f"Today's date: {today}")
# %% Tensors
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
print(f"Random Tensor: {tensor}")


# %% Strings and lists
message = "Hello, World!"
print(f"Does the message start with 'Hello'? {message.startswith('Hello')}")
mylist = [1, "Moustafa", 3, "Alice", 5]
for i in range(5):
    print(f"Iteration {i}: {message}")
    mylist.append(f"New Item {i}")
    print(f"Updated list: {mylist}")
mylist.insert(2, "Inserted Item")
print(f"List after insertion: {mylist}")
# %% Dictionary
PersonDictionary = {"name": "John", "age": 30, "city": "New York"}
print(f"Person Dictionary: {PersonDictionary}")
print(f"Person's name: {PersonDictionary['name']}")
# %% Functions
def greet(mylist):
    return f"greet, {mylist[1]}!", f"here is the list: {mylist}"


greet_message, list_message = greet(mylist)
print(f"Greeting message: {greet_message}")