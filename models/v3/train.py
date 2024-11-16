import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import StockDataLoader
from lstm_model import LSTMModel

# Parameters
ticker = "AAPL"
sequence_length = 250
start_date = "2010-01-01"
end_date = "2020-01-01"
num_epochs = 100
batch_size = 64
learning_rate = 1e-5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, x, y, num_epochs, batch_size):
    # Split data into training and testing sets
    train_size = int(0.7 * len(x))
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    early_stop_patience = 100
    best_val_loss = float("inf")
    patience_counter = 0
    min_delta = 1e-4

    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(x_train.size(0))
        epoch_loss = 0
        total_batches = 0

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

        avg_epoch_loss = epoch_loss / total_batches

        model.eval()
        with torch.no_grad():
            val_outputs = model(x_test)
            val_loss = criterion(val_outputs.squeeze(), y_test).item()

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_epoch_loss:.6f}, Validation Loss: {val_loss:.6f}')

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "temp_best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered")
                break

    model.load_state_dict(torch.load("temp_best_model.pth"))

# Main script
input_size = 8  # 'Close', 'MA100', 'MA200', 'RSI', 'MACD', 'Signal Line', 'Upper Band', 'Lower Band'
model = LSTMModel(input_size=input_size, hidden_size=256, num_layers=5, dropout=0.1).to(device)

print("Preparing dataset for training...")

# Use the data loader to fetch yfinance data
data_loader = StockDataLoader(ticker=ticker, sequence_length=sequence_length, start_date=start_date, end_date=end_date)
x_data, y_data, scaler = data_loader.get_data()

# Train the model
train_model(model, x_data, y_data, num_epochs, batch_size)

# Save the final model
torch.save(model.state_dict(), "final_lstm_stock_prediction_model.pth")
print("Final model saved as 'final_lstm_stock_prediction_model.pth'")
