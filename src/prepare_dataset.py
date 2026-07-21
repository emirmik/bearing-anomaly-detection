import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

CSV_PATH = Path("data/processed/features_1st_test.csv")
HEALTHY_CUTOFF = 1700  # files before this index are considered "healthy"
WINDOW_SIZE = 10        # sequence length fed to the LSTM

def make_sequences(data: np.ndarray, window_size: int) -> np.ndarray:
    """Turn a (n_samples, n_features) array into overlapping sequences
    of shape (n_sequences, window_size, n_features)."""
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i:i + window_size])
    return np.array(sequences)


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    feature_cols = [c for c in df.columns if c != "timestamp"]
    values = df[feature_cols].values  # shape: (2156, 32)

    healthy = values[:HEALTHY_CUTOFF]
    print(f"Healthy portion: {healthy.shape}, full: {values.shape}")

    scaler = StandardScaler()
    scaler.fit(healthy)
    values_scaled = scaler.transform(values)

    train_seq = make_sequences(values_scaled[:HEALTHY_CUTOFF], WINDOW_SIZE)
    full_seq = make_sequences(values_scaled, WINDOW_SIZE)

    print(f"Train sequences: {train_seq.shape}")
    print(f"Full sequences (train+monitor): {full_seq.shape}")

    out_dir = Path("data/processed")
    np.save(out_dir / "train_sequences.npy", train_seq)
    np.save(out_dir / "full_sequences.npy", full_seq)
    print("Saved train_sequences.npy and full_sequences.npy")