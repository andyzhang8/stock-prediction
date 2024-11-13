import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping

from data_loader import StockDataLoader
from lstm_model import LSTMModel

# Hyperparameters
ticker = "AAPL"
sequence_length = 100
start_date = "2010-01-01"
end_date = None
epochs = 100
batch_size = 32
learning_rate = 0.001

data_loader = StockDataLoader(ticker, start_date, end_date, sequence_length)
x, y, scaler = data_loader.get_data()

# Split data
train_size = int(len(x) * 0.7)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Initialize and compile model
input_shape = (x_train.shape[1], x_train.shape[2])
model_instance = LSTMModel(input_shape)
model = model_instance.get_model()
model_instance.compile_model(learning_rate=learning_rate)

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,
                    batch_size=batch_size, callbacks=[early_stop])

y_pred = model.predict(x_test)
y_test_rescaled = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), 2))], axis=1))[:, 0]
y_pred_rescaled = scaler.inverse_transform(np.concatenate([y_pred, np.zeros((len(y_pred), 2))], axis=1))[:, 0]

mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared: {r2}")

# Plot actual vs predicted stock prices
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label="Actual Price", color="blue")
plt.plot(y_pred_rescaled, label="Predicted Price", color="red")
plt.title(f"Actual vs Predicted Stock Price for {ticker}")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.savefig("prediction_plot.png")
plt.show()

# Moving Average Plot
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), 2))], axis=1))[:, 0], label="Close Price")
plt.plot(scaler.inverse_transform(np.concatenate([x_test[:, -1, 1].reshape(-1, 1), np.zeros((len(y_test), 2))], axis=1))[:, 0], label="100-Day MA", color="orange")
plt.plot(scaler.inverse_transform(np.concatenate([x_test[:, -1, 2].reshape(-1, 1), np.zeros((len(y_test), 2))], axis=1))[:, 0], label="200-Day MA", color="green")
plt.title(f"Stock Price and Moving Averages for {ticker}")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.savefig("moving_average_plot.png")
plt.show()

model.save("lstm_stock_prediction_model.h5")

