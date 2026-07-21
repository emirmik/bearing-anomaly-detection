import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from model import LSTMAutoencoder

FULL_SEQ_PATH = Path("data/processed/full_sequences.npy")
MODEL_PATH = Path("data/processed/lstm_autoencoder.pt")
OUT_PLOT = Path("data/processed/reconstruction_error.png")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_data = np.load(FULL_SEQ_PATH)
    full_tensor = torch.tensor(full_data, dtype=torch.float32).to(device)
    n_features = full_tensor.shape[2]

    model = LSTMAutoencoder(n_features=n_features).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        reconstructed = model(full_tensor)
        # mean squared error per sequence (averaged over window and features)
        errors = torch.mean((reconstructed - full_tensor) ** 2, dim=(1, 2))
        errors = errors.cpu().numpy()

    print(f"Reconstruction error - min: {errors.min():.4f}, max: {errors.max():.4f}, mean: {errors.mean():.4f}")

    plt.figure(figsize=(12, 4))
    plt.plot(errors)
    plt.xlabel("Sequence index (time)")
    plt.ylabel("Reconstruction error (MSE)")
    plt.title("Anomaly score over time")
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    print(f"Saved plot to {OUT_PLOT}")