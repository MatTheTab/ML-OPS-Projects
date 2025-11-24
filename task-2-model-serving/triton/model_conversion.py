import torch
from utils.model import LitAutoencoder

ckpt_path = "./data/model.ckpt"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = LitAutoencoder.load_from_checkpoint(ckpt_path).to(device)
model.eval()

dummy_input = torch.randn(1, 50).to(device)
traced_model = model.to_torchscript(
    file_path=None, example_inputs=dummy_input, method="trace"
)
traced_model.save("./models/lit-autoencoder/1/model.pt")
