import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


class LitAutoencoder(L.LightningModule):
    def __init__(
        self,
        input_dim=500,
        num_layers=3,
        neurons=128,
        latent_dim=32,
        lr=1e-3,
        gamma=0.995,
    ):
        super().__init__()
        self.save_hyperparameters()
        if num_layers < 2:
            raise ValueError(
                "num_layers must be >= 2 to include encoder, bottleneck, and decoder structure"
            )

        mid_sizes = (
            torch.linspace(neurons, latent_dim, steps=num_layers - 1)
            .round()
            .int()
            .tolist()
        )
        encoder_sizes = [input_dim, neurons] + mid_sizes

        encoder_layers = []
        for i in range(len(encoder_sizes) - 1):
            encoder_layers.append(nn.Linear(encoder_sizes[i], encoder_sizes[i + 1]))
            if i < len(encoder_sizes) - 1:
                encoder_layers.append(nn.ReLU(inplace=True))
        encoder_layers.append(nn.Linear(encoder_sizes[-1], latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_sizes = encoder_sizes[::-1]
        decoder_layers = []
        for i in range(len(decoder_sizes) - 1):
            decoder_layers.append(nn.Linear(decoder_sizes[i], decoder_sizes[i + 1]))
            if i < len(decoder_sizes) - 2:
                decoder_layers.append(nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(*decoder_layers)

        self.learning_rate = lr
        self.gamma = gamma

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def _shared_step(self, batch):
        x, y, z = batch
        recon = self(x)
        loss = F.mse_loss(recon, x)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("test_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
