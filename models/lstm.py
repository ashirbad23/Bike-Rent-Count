import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        # LSTM output
        _, (h_n, _) = self.lstm(x)

        # h_n shape: (num_layers, batch_size, hidden_dim)
        # Take last layer's hidden state
        last_hidden = h_n[-1]

        # Final prediction
        out = self.fc(last_hidden)

        return out
