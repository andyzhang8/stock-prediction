import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=19, hidden_size=256, num_layers=5, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate through LSTM layers
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)  # Apply dropout

        out = self.fc(out[:, -1, :])  # Pass through fully connected layer
        return out
