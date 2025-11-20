from utils.models import LitAutoencoder
import numpy as np
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.tuner import Tuner
import logging


def optimize_Basic_Autoencoder(trial, dm, window_size):
    neurons = trial.suggest_int("neurons", 16, 64)
    lr = trial.suggest_float("lr", 2e-5, 1e-3)
    gamma = trial.suggest_float("gamma", 0.9, 0.9995)
    logging.info(f"Parameters: lr {lr}, gamma: {gamma}, neurons: {neurons}")

    num_layers = 3
    model = LitAutoencoder(
        input_dim=window_size,
        num_layers=num_layers,
        neurons=neurons,
        latent_dim=16,
        lr=lr,
        gamma=gamma,
    )

    mlflow_logger = MLFlowLogger(
        experiment_name="autoencoder", run_name=f"trial_{trial.number}"
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, mode="min", verbose=True
    )

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        dirpath=f"output/checkpoints/trial_{trial.number}/",
    )

    callbacks = [early_stop, checkpoint]

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=100,
        callbacks=callbacks,
        logger=mlflow_logger,
        default_root_dir="output/lightning_logs",
    )

    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, datamodule=dm, mode="power", max_trials=8)
    trainer.fit(model, datamodule=dm)

    val_loss = np.inf
    if checkpoint.best_model_path:
        best = model.__class__.load_from_checkpoint(checkpoint.best_model_path)
        val_result = trainer.validate(best, datamodule=dm)
        val_loss = val_result[0]["val_loss"]
    return val_loss
