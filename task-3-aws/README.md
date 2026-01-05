# Hosting a BentoML Model on AWS EC2

This guide walks through deploying an image classification model using **BentoML**, **Docker**, and **AWS EC2**. It’s written as a clean, repeatable checklist you can come back to later.

---

## Overview

**Stack**:

* Model: `google/mobilenet_v2_1.0_224`
* Frameworks: PyTorch, Transformers, BentoML
* Deployment: Docker on AWS EC2

**Result**:
A public HTTP endpoint that accepts an image and returns a predicted label.

---

## 1. Create a BentoML Service

Create a `service.py` file that defines your BentoML service.

```python
import bentoml
from PIL import Image
import torch
from transformers import AutoImageProcessor, MobileNetV2ForImageClassification

# Load processor and model
processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
model = MobileNetV2ForImageClassification.from_pretrained(
    "google/mobilenet_v2_1.0_224"
)


@bentoml.service(name="mobilenet_classifier")
class MobileNetService:
    @bentoml.api
    def classify(self, image: Image.Image) -> dict:
        # BentoML automatically converts input into a PIL Image
        inputs = processor(image, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_label = logits.argmax(-1).item()
        label = model.config.id2label[predicted_label]

        return {"label": label}
```

---

## 2. Create a Docker Image

### Dockerfile

```dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY service.py .

EXPOSE 3000

CMD ["bentoml", "serve", "--production"]
```

### Build the Image

```bash
docker build -t mobilenet-bentoml .
```

---

## 3. Create an EC2 Instance

Recommended settings:

* **Service**: EC2
* **OS**: Ubuntu
* **Instance type**: `c7i-flex.large` (sufficient for this model)
* **Storage**: Increase to ~30 GB
* **Network**: Allow HTTP and HTTPS traffic from anywhere

---

## 4. Configure Security Group Rules

Add an **Inbound Rule**:

* Type: Custom TCP
* Port: `3000`
* Source: `0.0.0.0/0`

> ⚠️ For production, restrict this to known IP ranges or use a load balancer.

---

## 5. Connect to the EC2 Instance

```bash
ssh -i <KEY_PATH> <USER>@<EC2_PUBLIC_IP>
```

---

## 6. Install Docker on EC2

```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
```

(Optional) Run Docker without `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

---

## 7. Transfer the Docker Image to EC2

From your local machine:

```bash
docker save mobilenet-bentoml | pv | ssh -i <KEY_PATH> <USER>@<EC2_PUBLIC_IP> "docker load"
```

---

## 8. Run the Container

On the EC2 instance:

```bash
docker run -p 3000:3000 mobilenet-bentoml
```

The service will be available at:

```
http://<EC2_PUBLIC_IP>:3000
```

---

## 9. Test the Deployment

### Test Script (`test_script.py`)

```python
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
```

### Run the Test

```bash
export SERVICE_URL=http://<EC2_PUBLIC_IP>:3000/classify
python3 test_script.py
```

---

## Notes & Improvements

* Add a reverse proxy (Nginx) for TLS and cleaner URLs
* Use an ECR registry instead of copying images manually
* Add autoscaling or a load balancer for production traffic
* Consider BentoML build + containerization instead of a custom Dockerfile

---

✅ You now have a BentoML-powered model running on AWS EC2.
