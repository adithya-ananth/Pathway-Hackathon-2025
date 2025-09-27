import requests

url = "http://localhost:8000/prompt"
payload = {"prompt": "RAG"}

response = requests.post(url, json=payload)

print("Status code:", response.status_code)
print("Response:")
print(response.json())
