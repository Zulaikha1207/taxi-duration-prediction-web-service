import requests

ride = [{"PULocationID": 1, "DOLocationID": 2,"trip_distance": 3.45 },    
    {"PULocationID": 3, "DOLocationID": 4, "trip_distance": 2.56},
    {"PULocationID": 12, "DOLocationID": 24, "trip_distance": 30},
    {"PULocationID": 19, "DOLocationID": 64, "trip_distance": 45}]


url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())