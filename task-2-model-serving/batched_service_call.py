import threading
import requests
import torch
import json


def send_request():
    dummy_input = torch.rand(1, 50).tolist()
    r = requests.post("http://localhost:3000/predict", json={"input": dummy_input})
    print(r.json())


threads = [threading.Thread(target=send_request) for _ in range(20)]
for t in threads:
    t.start()
for t in threads:
    t.join()
