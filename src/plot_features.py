import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/processed/features_1st_test.csv")

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
bearing_channels = {
    "Bearing 1": ["ch1_rms", "ch2_rms"],
    "Bearing 2": ["ch3_rms", "ch4_rms"],
    "Bearing 3": ["ch5_rms", "ch6_rms"],
    "Bearing 4": ["ch7_rms", "ch8_rms"],
}

for ax, (bearing, cols) in zip(axes, bearing_channels.items()):
    for col in cols:
        ax.plot(df[col], label=col)
    ax.set_ylabel("RMS")
    ax.set_title(bearing)
    ax.legend()

axes[-1].set_xlabel("File index (time)")
plt.tight_layout()

out_path = Path("data/processed/rms_trend.png")
plt.savefig(out_path, dpi=120)
print(f"Saved plot to {out_path}")