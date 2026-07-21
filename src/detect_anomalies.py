import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from model import LSTMAutoencoder

TRAIN_SEQ_PATH = Path("data/processed/train_sequences.npy")
FULL_SEQ_PATH = Path("data/processed/full_sequences.npy")
MODEL_PATH = Path("data/processed/lstm_autoencoder.pt")
OUT_PLOT = Path("data/processed/anomaly_detection.png")

WINDOW_SIZE = 10  # must match prepare_dataset.py


def compute_errors(model, tensor):
    with torch.no_grad():
        reconstructed = model(tensor)
        errors = torch.mean((reconstructed - tensor) ** 2, dim=(1, 2))
    return errors.cpu().numpy()

def flag_persistent_anomalies(errors: np.ndarray, threshold: float, min_consecutive: int = 5) -> np.ndarray:
    """Only flag a sequence as an anomaly if it's part of a run of at
    least `min_consecutive` consecutive sequences above the threshold.
    Filters out single-frame noise spikes."""
    raw_flags = errors > threshold
    persistent_flags = np.zeros_like(raw_flags)

    run_start = None
    for i, flag in enumerate(raw_flags):
        if flag:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= min_consecutive:
                persistent_flags[run_start:i] = True
            run_start = None
    # handle a run that reaches the end of the array
    if run_start is not None and (len(raw_flags) - run_start) >= min_consecutive:
        persistent_flags[run_start:] = True

    return persistent_flags


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = np.load(TRAIN_SEQ_PATH)
    full_data = np.load(FULL_SEQ_PATH)
    train_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
    full_tensor = torch.tensor(full_data, dtype=torch.float32).to(device)
    n_features = full_tensor.shape[2]

    model = LSTMAutoencoder(n_features=n_features).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    train_errors = compute_errors(model, train_tensor)
    full_errors = compute_errors(model, full_tensor)

    threshold = train_errors.mean() + 3 * train_errors.std()
    print(f"Healthy error - mean: {train_errors.mean():.4f}, std: {train_errors.std():.4f}")
    print(f"Threshold: {threshold:.4f}")

    is_anomaly = flag_persistent_anomalies(full_errors, threshold, min_consecutive=5)
    n_anomalies = is_anomaly.sum()
    first_anomaly_idx = np.argmax(is_anomaly) if n_anomalies > 0 else None

    print(f"Flagged {n_anomalies}/{len(full_errors)} sequences as anomalies")
    print(f"First anomaly at sequence index: {first_anomaly_idx} "
          f"(corresponds to original file index ~{first_anomaly_idx + WINDOW_SIZE})")

    plt.figure(figsize=(12, 4))
    plt.plot(full_errors, label="Reconstruction error", color="tab:blue")
    plt.axhline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.2f})")
    plt.scatter(
        np.where(is_anomaly)[0], full_errors[is_anomaly],
        color="red", s=10, zorder=5, label="Flagged anomaly"
    )
    plt.xlabel("Sequence index (time)")
    plt.ylabel("Reconstruction error (MSE)")
    plt.title("Bearing Anomaly Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    print(f"Saved plot to {OUT_PLOT}")