import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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


def evaluate_random_forest(x, y):
    """Train and evaluate a Random Forest model as a baseline."""
    print("\nTraining Random Forest Baseline...")
    x_flat = x.reshape(x.shape[0], -1)  # Flatten LSTM input for Random Forest
    x_train, x_test, y_train, y_test = train_test_split(x_flat, y, test_size=0.3, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)

    # Predictions
    y_pred = rf_model.predict(x_test)

    # Evaluation Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nRandom Forest Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")


def train_lstm_model(data_loader, model, num_epochs, batch_size, learning_rate, checkpoint_path):
    """Train and evaluate the LSTM model."""
    x, y, scaler = data_loader.get_data()
    
    x_train, x_test = x[:int(0.7 * len(x))], x[int(0.7 * len(x)):]
    y_train, y_test = y[:int(0.7 * len(y))], y[int(0.7 * len(y)):]

    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float("inf")

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

        model.eval()
        with torch.no_grad():
            val_outputs = model(x_test)
            val_loss = criterion(val_outputs.squeeze(), y_test).item()

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    print("\nLSTM model training complete. Best model saved.")
    return x, y


# Main Script
print("Loading dataset...")
data_loader = StockDataLoader(ticker=ticker, sequence_length=sequence_length, data_dir="dataset")
# Replace the input_size parameter to match the data features (19)
input_size = 19  # Adjusted to match the actual feature size
model = LSTMModel(input_size=input_size, hidden_size=200, num_layers=5, dropout=0.2).to(device)

# Train and evaluate LSTM
lstm_checkpoint_path = "final_lstm_stock_prediction_model.pth"
x, y = train_lstm_model(data_loader, model, num_epochs, batch_size, learning_rate, lstm_checkpoint_path)

# Benchmark with Random Forest
evaluate_random_forest(x, y)
