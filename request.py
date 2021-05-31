import requests
data = {
    "Customer_Name": 0
  }

url = "http://127.0.0.1:5000/predict"
response = requests.post(url, json=data)
print("Recommendations: "+ str(response.json()))
