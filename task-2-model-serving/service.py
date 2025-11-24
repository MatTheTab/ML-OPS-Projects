from __future__ import annotations
import bentoml
from typing import List
import torch
from utils.model import LitAutoencoder


@bentoml.service
class TorchService:
    def __init__(self):
        self.model = LitAutoencoder.load_from_checkpoint("./data/model.ckpt")

    @bentoml.api(batchable=True)
    def predict(self, input: List[List[float]]) -> List[List[float]]:
        with torch.no_grad():
            input = torch.tensor(input).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            results = self.model(input).cpu().detach().numpy()
            results = results.tolist()
            return results
