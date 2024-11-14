# train.py

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
sequence_length = 200
start_date = "2010-01-01"
end_date = "2020-01-01"
num_epochs = 100  # Increased epochs to allow more learning time
batch_size = 64
learning_rate = 0.0005  # Reduced learning rate for gradual adjustments

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_evaluate(data_loader, model, num_epochs, checkpoint_path):
    x, y, scaler = data_loader.get_data()
    
    # Split data into training and testing sets
    train_size = int(len(x) * 0.7)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Convert numpy arrays to PyTorch tensors and move to device
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Define loss, optimizer, and learning rate scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15, verbose=True)  # Less aggressive factor and increased patience

    early_stop_patience = 30  # Increased patience to allow more time for improvement
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
            
            # Ensure batch_y has the same shape as outputs for loss calculation
            loss = criterion(outputs.squeeze(), batch_y.unsqueeze(-1).squeeze())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / (x_train.size(0) // batch_size)

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_test)
            val_loss = criterion(val_outputs.squeeze(), y_test.unsqueeze(-1).squeeze()).item()

        scheduler.step(val_loss)

        # Log training and validation loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Early stopping with increased patience
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered")
                break

    # Load the best model for evaluation
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    with torch.no_grad():
        predicted = model(x_test).squeeze().cpu().numpy()
        actual = y_test.cpu().numpy()

    # Inverse transform to get actual prices
    predicted_prices = scaler.inverse_transform(
        np.concatenate((predicted.reshape(-1, 1), np.zeros((len(predicted), 2))), axis=1)
    )[:, 0]
    actual_prices = scaler.inverse_transform(
        np.concatenate((actual.reshape(-1, 1), np.zeros((len(actual), 2))), axis=1)
    )[:, 0]

    # Calculate evaluation metrics
    mse = mean_squared_error(actual_prices, predicted_prices)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_prices, predicted_prices)

    print(f"\nEvaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual Price')
    plt.plot(predicted_prices, label='Predicted Price')
    plt.title(f"Actual vs Predicted Stock Price for {data_loader.ticker}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.savefig(f"prediction_plot_{data_loader.ticker}.png")
    plt.show()

# Initialize model
input_size = 3  # 'Close', 'MA100', 'MA200' columns
model = LSTMModel(input_size=input_size, hidden_size=100, num_layers=4, dropout=0.3).to(device)

print("Training on yfinance data...")
data_loader_yfinance = StockDataLoader(ticker=ticker, sequence_length=sequence_length, use_yfinance=True, start_date=start_date, end_date=end_date)
train_and_evaluate(data_loader_yfinance, model, num_epochs, checkpoint_path="best_model_yfinance.pth")

# Load the model with yfinance training and continue training on local dataset
print("Training on local dataset...")
data_loader_local = StockDataLoader(ticker=ticker, sequence_length=sequence_length, data_dir="dataset", use_yfinance=False)
train_and_evaluate(data_loader_local, model, num_epochs, checkpoint_path="best_model_local.pth")
