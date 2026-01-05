import requests
import os
import sys

url = os.environ.get("SERVICE_URL")

if not url:
    print("ERROR: SERVICE_URL environment variable is not set.", file=sys.stderr)
    print(
        "Please set it to your public endpoint, e.g., http://<YOUR-IP>:3000/classify",
        file=sys.stderr,
    )
    sys.exit(1)

files = {"image": open("./imgs/dog.jpg", "rb")}

try:
    resp = requests.post(url, files=files)
    print("Status Code:", resp.status_code)
    print("Result:", resp.json())
except requests.exceptions.ConnectionError:
    print(f"Failed to connect to {url}. Is your VM running and port 3000 open?")
except Exception as e:
    print("Failed:", e)
