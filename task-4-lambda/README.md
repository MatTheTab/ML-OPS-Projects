# AWS Lambda Setup with CDK (TypeScript)

This repository contains an AWS CDK project for deploying an AWS Lambda function that performs image classification using a PyTorch model.

---

## 1. Prerequisites

Before you begin, ensure you have the following installed and configured:

* **Node.js & npm** (LTS version)
  [https://nodejs.org/](https://nodejs.org/)

* **AWS CLI** (configured with `aws configure`)
[https://aws.amazon.com/cli/](https://aws.amazon.com/cli/) (create user in AWS to configure with keys)

* **AWS CDK CLI** (installed globally)

```bash
npm install -g aws-cdk
```

---

## 2. Initialize a New CDK Project

Create a new directory and initialize a CDK application:

```bash
mkdir app
cd app
cdk init app --language python
```

This will generate the basic CDK project structure.

---

## 3. Train and Save the Model

Train your PyTorch model locally and save it to the following path:

```text
/app/model.pth
```

This file will be loaded by the Lambda function at runtime.

---

## 4. Create `main.py`

Create a file named `main.py` in your project root with the following contents:

```python
import json
import boto3
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load model

device = torch.device("cpu")
model = SimpleCNN()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# CIFAR-10 labels
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Image preprocessing pipeline
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

s3 = boto3.client("s3")


def handler(event, context):
    record = event["Records"][0]
    bucket_name = record["s3"]["bucket"]["name"]
    file_key = record["s3"]["object"]["key"]

    # Skip already-processed results
    if file_key.startswith("results/"):
        return

    # Download image into memory
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    image_bytes = response["Body"].read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run inference
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = CIFAR10_CLASSES[predicted.item()]

    # Upload prediction result
    result = {
        "filename": file_key,
        "prediction": label,
    }

    s3.put_object(
        Bucket=bucket_name,
        Key=f"results/{os.path.basename(file_key)}.json",
        Body=json.dumps(result),
    )

    return {
        "statusCode": 200,
        "prediction": label,
    }
```

---

## 5. Deploy the Stack

Bootstrap your AWS environment (required once per account/region):

```bash
cdk bootstrap
```

Deploy the CDK stack:

```bash
cdk deploy
```

---

## 6. Clean Up Resources

When you are finished, destroy the deployed resources to avoid ongoing charges:

```bash
cdk destroy
```

---