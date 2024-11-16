import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
from lstm_model import LSTMModel  # Import your updated LSTM model class

# Path to the saved model
MODEL_PATH = "best_model_yfinance.pth"

# Define constants
SEQUENCE_LENGTH = 200  # Same sequence length as in training
INPUT_SIZE = 3  # Number of input features used in the model ('Close', 'MA100', 'MA200')

def load_model(input_size=INPUT_SIZE, hidden_size=100, num_layers=4, dropout=0.2):
    """
    Load the LSTM model with the specified architecture.
    """
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

def fetch_data(ticker, sequence_length=SEQUENCE_LENGTH):
    """
    Fetch historical stock data from Yahoo Finance and preprocess it for the model.
    """
    periods = ["1y", "2y", "3y", "5y"]
    data = None
    
    for period in periods:
        # Download historical data
        data = yf.download(ticker, period=period, interval="1d")
        
        # Calculate additional features: Moving averages
        data['MA100'] = data['Close'].rolling(window=100).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        data.dropna(inplace=True)

        # Check if there's enough data after calculating moving averages
        if data.shape[0] >= sequence_length:
            print(f"Data fetched for {ticker} with period: {period}")
            break
        else:
            print(f"Not enough data with period {period}. Trying a longer period...")

    if data is None or data.shape[0] < sequence_length:
        raise ValueError(f"Not enough data available for {ticker} after applying moving averages.")

    # Prepare data for the model
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close', 'MA100', 'MA200']])

    x_input = scaled_data[-sequence_length:]  # Shape: (sequence_length, INPUT_SIZE)
    x_input = np.expand_dims(x_input, axis=0)  # Reshape for the model input: (1, sequence_length, INPUT_SIZE)
    x_input = torch.tensor(x_input, dtype=torch.float32)

    return x_input, scaler, float(data['Close'].values[-1])  # Ensure last close price is a scalar

def predict_trend(model, x_input, scaler, last_close_price):
    """
    Predict the stock trend and price using the LSTM model.
    """
    with torch.no_grad():
        predicted_price_scaled = model(x_input).item()

    # Inverse transform to get the actual predicted price
    predicted_price = scaler.inverse_transform(
        np.array([[predicted_price_scaled, 0, 0]])  # Shape for inverse transform
    )[0, 0]

    # Determine trend
    trend = "Bullish" if predicted_price > last_close_price else "Bearish"
    return predicted_price, trend

def main():
    """
    Main function to run the prediction script.
    """
    if len(sys.argv) < 2:
        print("Usage: python predict.py <STOCK_TICKER>")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    try:
        model = load_model(input_size=INPUT_SIZE)
        x_input, scaler, last_close_price = fetch_data(ticker)

        predicted_price, trend = predict_trend(model, x_input, scaler, last_close_price)

        print(f"Stock: {ticker}")
        print(f"Last Close Price: {last_close_price:.2f}")
        print(f"Predicted Price: {predicted_price:.2f}")
        print(f"Market Trend: {trend}")

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
