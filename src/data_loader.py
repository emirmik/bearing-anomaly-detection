import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/1st_test/1st_test")
N_CHANNELS = 8


def extract_features(filepath: Path) -> dict:
    """Load one snapshot file (20480 samples x 8 channels) and compute
    stat features per channel."""
    data = np.loadtxt(filepath)  # shape: (20480, 8)
    features = {"timestamp": filepath.name}
    for ch in range(N_CHANNELS):
        signal = data[:, ch]
        rms = np.sqrt(np.mean(signal ** 2))
        kurtosis = pd.Series(signal).kurtosis()
        peak = np.max(np.abs(signal))
        crest_factor = peak / rms if rms > 0 else 0.0
        features[f"ch{ch+1}_rms"] = rms
        features[f"ch{ch+1}_kurtosis"] = kurtosis
        features[f"ch{ch+1}_peak"] = peak
        features[f"ch{ch+1}_crest"] = crest_factor
    return features


if __name__ == "__main__":
    files = sorted(RAW_DIR.iterdir())
    print(f"Found {len(files)} files")

    records = []
    for i, f in enumerate(files):
        records.append(extract_features(f))
        if (i + 1) % 200 == 0 or (i + 1) == len(files):
            print(f"Processed {i + 1}/{len(files)}")

    df = pd.DataFrame(records)
    out_path = Path("data/processed/features_1st_test.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")