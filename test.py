import requests
import base64

with open("./odometer_images/5.jpg", "rb") as f:
    img_str = base64.b64encode(f.read()).decode()

payload = {
    "file": img_str,
}

response = requests.post("http://localhost:8000/inference/base64/", json=payload)
print(response.json())