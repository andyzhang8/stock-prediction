import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

def train_model(model, x_train, y_train, x_val, y_val, scaler, num_epochs, batch_size):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    early_stop_patience = 100
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
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

        avg_epoch_loss = epoch_loss / (x_train.size(0) // batch_size)

        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs.squeeze(), y_val).item()

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "temp_best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered")
                break

    model.load_state_dict(torch.load("temp_best_model.pth"))

def plot_predictions(model, x_val, y_val, scaler):
    # Generate predictions
    model.eval()
    with torch.no_grad():
        predictions = model(x_val).squeeze().cpu().numpy()

    # Inverse transform to get the actual prices
    predictions = scaler.inverse_transform(
        np.concatenate([predictions.reshape(-1, 1), np.zeros((predictions.shape[0], 7))], axis=1)
    )[:, 0]
    actuals = scaler.inverse_transform(
        np.concatenate([y_val.cpu().numpy().reshape(-1, 1), np.zeros((y_val.shape[0], 7))], axis=1)
    )[:, 0]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual Prices', color='blue')
    plt.plot(predictions, label='Predicted Prices', color='orange')
    plt.title(f'{ticker} - Predicted vs Actual Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Main script
data_loader = StockDataLoader(ticker=ticker, sequence_length=sequence_length, data_dir="dataset", start_date=start_date, end_date=end_date)

x, y, scaler = data_loader.get_data()

# Split into train and validation sets
split_index = int(0.7 * len(x))
x_train, x_val = x[:split_index], x[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# Convert numpy arrays to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

# Model setup and training
input_size = x_train.shape[2]
model = LSTMModel(input_size=input_size, hidden_size=200, num_layers=5, dropout=0.2).to(device)

train_model(model, x_train, y_train, x_val, y_val, scaler, num_epochs, batch_size)

# Plot the predicted vs actual prices
plot_predictions(model, x_val, y_val, scaler)

# Save the final model
torch.save(model.state_dict(), "final_lstm_stock_prediction_model.pth")
print("Final model saved as 'final_lstm_stock_prediction_model.pth'")
