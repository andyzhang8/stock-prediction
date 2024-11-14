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
sequence_length = 100
start_date = "2010-01-01"
end_date = None
num_epochs = 100
batch_size = 32
learning_rate = 0.001

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and prepare data
data_loader = StockDataLoader(ticker, start_date, end_date, sequence_length)
x, y, scaler = data_loader.get_data()

train_size = int(len(x) * 0.7)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)

input_size = x_train.shape[2]
model = LSTMModel(input_size=input_size, hidden_size=50, num_layers=4, dropout=0.2).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass and loss calculation
    outputs = model(x_train)
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(x_test)
        val_loss = criterion(val_outputs.squeeze(), y_test)
    
    # Log training and validation loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

model.eval()
with torch.no_grad():
    predicted = model(x_test).squeeze().cpu().numpy()
    actual = y_test.cpu().numpy()

predicted_prices = scaler.inverse_transform(
    np.concatenate((predicted.reshape(-1, 1), np.zeros((len(predicted), 2))), axis=1)
)[:, 0]
actual_prices = scaler.inverse_transform(
    np.concatenate((actual.reshape(-1, 1), np.zeros((len(actual), 2))), axis=1)
)[:, 0]

mse = mean_squared_error(actual_prices, predicted_prices)
mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
r2 = r2_score(actual_prices, predicted_prices)

print(f"\nEvaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.title(f"Actual vs Predicted Stock Price for {ticker}")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.savefig("prediction_plot_pytorch.png")
plt.show()

torch.save(model.state_dict(), 'lstm_stock_prediction_model.pth')
