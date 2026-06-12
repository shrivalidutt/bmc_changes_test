import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://st-ntx-3q0qb:8443/automation-api/session/login"
payload = {"username": "emuser", "password": "empass"}

print("Connecting to BMC API server at https://st-ntx-3q0qb:8443...")
try:
    response = requests.post(url, json=payload, verify=False, timeout=10)
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"BMC API request failed: {e}")
