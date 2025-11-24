import torch
from typing import List
import bentoml

dummy_inputs: torch.Tensor = torch.rand(1, 50)
dummy_inputs = dummy_inputs.tolist()

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    result = client.predict(input=dummy_inputs)
    print(f"Model Result: {result}")
