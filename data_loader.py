import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

class StockDataLoader:
    def __init__(self, ticker, start_date="2010-01-01", end_date=None, sequence_length=100):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self):
        # Stock data from Yahoo Finance
        data = yf.download(self.ticker, self.start_date, self.end_date)
        # Moving averages
        data['MA100'] = data['Close'].rolling(window=100).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
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

