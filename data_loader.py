import pandas as pd
import numpy as np
import os
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

class StockDataLoader:
    def __init__(self, ticker, sequence_length=100, data_dir="dataset", use_yfinance=False, start_date="2010-01-01", end_date=None):
        self.ticker = ticker
        self.sequence_length = sequence_length
        self.data_dir = data_dir  # Directory containing the local dataset
        self.use_yfinance = use_yfinance  # Determine whether to use yfinance
        self.start_date = start_date
        self.end_date = end_date
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self):
        if self.use_yfinance:
            # Download data from yfinance
            data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        else:
            # Define the path to look for the stock data in the stocks or ETFs folder
            file_path_stocks = os.path.join(self.data_dir, 'stocks', f"{self.ticker}.csv")
            file_path_etfs = os.path.join(self.data_dir, 'etfs', f"{self.ticker}.csv")
            
            # Check if file exists in stocks or ETFs folder
            if os.path.exists(file_path_stocks):
                file_path = file_path_stocks
            elif os.path.exists(file_path_etfs):
                file_path = file_path_etfs
            else:
                raise FileNotFoundError(f"No data found for ticker {self.ticker}")

            # Load the CSV file and ensure columns are interpreted as floats
            data = pd.read_csv(file_path, dtype={
                "Open": float, "High": float, "Low": float, 
                "Close": float, "Adj Close": float, "Volume": float
            })

        # Verify that the required columns exist and are float types
        required_columns = ["Close"]
        for col in ["MA100", "MA200"]:
            data[col] = data['Close'].rolling(window=int(col[2:])).mean()  # Moving averages
        data.dropna(inplace=True)  # Drop rows with NaN values

        return data[['Close', 'MA100', 'MA200']]

    def preprocess_data(self, data):
        # Scale the data
        data_scaled = self.scaler.fit_transform(data)

        # Create sequences
        x, y = [], []
        for i in range(self.sequence_length, len(data_scaled)):
            x.append(data_scaled[i - self.sequence_length:i])
            y.append(data_scaled[i, 0])  # Predicting the close price
        x, y = np.array(x), np.array(y)
        return x, y

    def get_data(self):
        data = self.load_data()
        x, y = self.preprocess_data(data)
        return x, y, self.scaler
