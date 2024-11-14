import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_loader import StockDataLoader
from lstm_model import LSTMModel
import yfinance as yf

# Parameters
ticker = "AAPL"
sequence_length = 250
start_date = "2010-01-01"
end_date = "2020-01-01"
num_epochs = 100
batch_size = 64
learning_rate = 0.0001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fetch_yfinance_data(ticker, sequence_length=250):
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    # Calculate technical indicators
    data['MA100'] = data['Close'].rolling(window=100).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'], data['Signal Line'] = calculate_macd(data['Close'])
    data['Upper Band'], data['Lower Band'] = calculate_bollinger_bands(data['Close'])
    data.dropna(inplace=True)
    return data[['Close', 'MA100', 'MA200', 'RSI', 'MACD', 'Signal Line', 'Upper Band', 'Lower Band']]

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    exp1 = prices.ewm(span=fast_period, adjust=False).mean()
    exp2 = prices.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices, window=20, num_std=2):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

# Combined training on both datasets without resetting weights
def train_on_combined_data(model, datasets, num_epochs, batch_size):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    early_stop_patience = 100
    best_val_loss = float("inf")
    patience_counter = 0
    min_delta = 1e-4

    for epoch in range(num_epochs):
        model.train()
        total_epoch_loss = 0
        total_batches = 0

        # Train on both datasets without resetting weights
        for x, y, scaler in datasets:
            # Convert numpy arrays to PyTorch tensors and move to device
            x_train, x_test = x[:int(0.7 * len(x))], x[int(0.7 * len(x)):]
            y_train, y_test = y[:int(0.7 * len(y))], y[int(0.7 * len(y)):]

            x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
            y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
            x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
            y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

            permutation = torch.randperm(x_train.size(0))
            epoch_loss = 0

            for i in range(0, x_train.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = x_train[indices], y_train[indices]

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total_batches += 1

            total_epoch_loss += epoch_loss

        avg_epoch_loss = total_epoch_loss / total_batches

        # Validation on the last dataset's test set
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_test)
            val_loss = criterion(val_outputs.squeeze(), y_test).item()

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "temp_best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered")
                break

    # Load the best model from the combined training
    model.load_state_dict(torch.load("temp_best_model.pth"))

# Main script
input_size = 8  # Adjusted to include all indicators
model = LSTMModel(input_size=input_size, hidden_size=200, num_layers=5, dropout=0.2).to(device)

print("Preparing datasets for combined training...")

# Load yfinance data
data_loader_yfinance = StockDataLoader(ticker=ticker, sequence_length=sequence_length, data_dir="dataset")
x_yfinance, y_yfinance, scaler_yfinance = data_loader_yfinance.get_data()

# Load local data
data_loader_local = StockDataLoader(ticker=ticker, sequence_length=sequence_length, data_dir="dataset")
x_local, y_local, scaler_local = data_loader_local.get_data()

# Combine both datasets
datasets = [
    (x_yfinance, y_yfinance, scaler_yfinance),
    (x_local, y_local, scaler_local)
]

# Train on the combined datasets
train_on_combined_data(model, datasets, num_epochs, batch_size)

# Save the final model after combined training
torch.save(model.state_dict(), "final_lstm_stock_prediction_model.pth")
print("Final model saved as 'final_lstm_stock_prediction_model.pth'")
