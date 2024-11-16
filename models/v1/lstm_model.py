import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=100, num_layers=4, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layers with the inferred parameters
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Additional dropout after LSTM layers
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer with hidden_size as input
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate through LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)  # Apply dropout

        # Pass the last time step's output through the fully connected layer
        out = self.fc(out[:, -1, :])
        return out
