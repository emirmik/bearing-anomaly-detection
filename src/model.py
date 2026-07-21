import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, latent_size: int = 16):
        super().__init__()
        # Encoder: compress the sequence down to a single latent vector
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.encoder_fc = nn.Linear(hidden_size, latent_size)

        # Decoder: expand the latent vector back into a sequence
        self.decoder_fc = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output_fc = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        batch_size, seq_len, _ = x.shape

        # --- Encode ---
        _, (h_n, _) = self.encoder_lstm(x)   # h_n: (1, batch, hidden_size)
        latent = self.encoder_fc(h_n[-1])    # (batch, latent_size)

        # --- Decode ---
        # repeat the latent vector across the sequence length so the
        # decoder LSTM has something to unroll over
        decoder_input = self.decoder_fc(latent)               # (batch, hidden_size)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, seq_len, 1)
        decoded_seq, _ = self.decoder_lstm(decoder_input)     # (batch, seq_len, hidden_size)
        output = self.output_fc(decoded_seq)                  # (batch, seq_len, n_features)

        return output