import requests
data = {"name": "test", "author": "test", "price": 100}
response = requests.post("http://127.0.0.1:8000/book/add", json=data)
print(response)
