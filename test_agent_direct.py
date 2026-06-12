import requests
import json
import time

url = "http://127.0.0.1:5001/api/chat"
payload = {"message": "hi"}
headers = {"Content-Type": "application/json"}

print("Sending request to agent...")
t0 = time.time()
try:
    response = requests.post(url, json=payload, headers=headers, timeout=120)
    print(f"Status code: {response.status_code}")
    print(f"Response body: {response.text}")
    print(f"Completed in {time.time() - t0:.2f} seconds")
except Exception as e:
    print(f"Request failed: {e}")
