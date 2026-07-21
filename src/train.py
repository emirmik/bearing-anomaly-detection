import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from model import LSTMAutoencoder

DATA_PATH = Path("data/processed/train_sequences.npy")
MODEL_OUT = Path("data/processed/lstm_autoencoder.pt")

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data = np.load(DATA_PATH)  # (n_sequences, window_size, n_features)
    train_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
    n_features = train_tensor.shape[2]

    model = LSTMAutoencoder(n_features=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    n_samples = train_tensor.shape[0]

    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(n_samples)
        epoch_loss = 0.0

        for i in range(0, n_samples, BATCH_SIZE):
            indices = permutation[i:i + BATCH_SIZE]
            batch = train_tensor[indices]

            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= n_samples
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.6f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")