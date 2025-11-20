import torch
import matplotlib.pyplot as plt


def plot_model_reconstructions(model, scaler, signals_list):
    """
    Plots original vs reconstructed signals for a list of 1D numpy arrays using the given model.

    Args:
        model: a trained LitAutoencoder model
        signals_list: list of numpy arrays, each representing a single signal (1D)
    """
    model.eval()
    num_signals = len(signals_list)

    fig, axes = plt.subplots(num_signals, 1, figsize=(10, 3 * num_signals))
    if num_signals == 1:
        axes = [axes]

    for i, signal in enumerate(signals_list):
        x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            recon = model(x).squeeze(0).cpu().numpy()
            recon = scaler.inverse_transform(recon.reshape(-1, 1)).flatten()

        # Plot
        axes[i].plot(
            scaler.inverse_transform(signal.reshape(-1, 1)).flatten(),
            label="Original",
            alpha=0.8,
        )
        axes[i].plot(recon, label="Reconstructed", linestyle="--", alpha=0.8)
        axes[i].set_title(f"Signal {i+1}")
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()


def plot_model_reconstructions_slide_signal(
    model, scaler, signals_list, window_size=None
):
    """
    Plots original vs reconstructed signals for a list of 1D numpy arrays using the given model.

    Args:
        model: a trained LitAutoencoder model
        signals_list: list of numpy arrays, each representing a single signal (1D)
    """
    model.eval()
    num_signals = len(signals_list)

    fig, axes = plt.subplots(num_signals, 1, figsize=(10, 3 * num_signals))
    if num_signals == 1:
        axes = [axes]

    for i, signal in enumerate(signals_list):
        entire_signal = []
        curr_step = 0
        while curr_step + window_size <= len(signal):
            windowed = signal[curr_step : curr_step + window_size]
            curr_step += window_size
            x = torch.tensor(windowed, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                recon = model(x).squeeze(0).cpu().numpy()
                recon = scaler.inverse_transform(recon.reshape(-1, 1)).flatten()
                entire_signal.append(torch.tensor(recon.copy()))
        result_tensor = torch.cat(entire_signal, dim=0)
        axes[i].plot(
            scaler.inverse_transform(signal.reshape(-1, 1)).flatten(),
            label="Original",
            alpha=0.8,
        )
        axes[i].plot(result_tensor, label="Reconstructed", linestyle="--", alpha=0.8)
        axes[i].set_title(f"Signal {i+1}")
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()
