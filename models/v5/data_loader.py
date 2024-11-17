import pandas as pd
import numpy as np
import os
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

class StockDataLoader:
    def __init__(self, ticker, sequence_length=100, data_dir="dataset", start_date="2010-01-01", end_date="2020-01-01"):
        self.ticker = ticker
        self.sequence_length = sequence_length
        self.data_dir = data_dir
        self.start_date = start_date  # Store start_date
        self.end_date = end_date      # Store end_date
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.meta_data = self.load_meta_data()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.ohe.fit(self.meta_data[['Market Category', 'Listing Exchange', 'ETF']])

    def load_meta_data(self):
        meta_file_path = os.path.join(self.data_dir, "symbols_valid_meta.csv")
        meta_data = pd.read_csv(meta_file_path)
        meta_data = meta_data[['Symbol', 'Market Category', 'Listing Exchange', 'ETF']]
        meta_data.fillna("Unknown", inplace=True)
        return meta_data

    def get_meta_features(self):
        ticker_meta = self.meta_data[self.meta_data['Symbol'] == self.ticker]
        if ticker_meta.empty:
            raise ValueError(f"Ticker {self.ticker} not found in metadata.")
        return self.ohe.transform(ticker_meta[['Market Category', 'Listing Exchange', 'ETF']]).flatten()

    def load_data(self):
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date, interval="1d")  # Now uses class attributes

        # Calculate technical indicators
        data['MA100'] = data['Close'].rolling(window=100).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])
        data['MACD'], data['Signal Line'] = self.calculate_macd(data['Close'])
        data['Upper Band'], data['Lower Band'] = self.calculate_bollinger_bands(data['Close'])
        data.dropna(inplace=True)

        return data[['Close', 'MA100', 'MA200', 'RSI', 'MACD', 'Signal Line', 'Upper Band', 'Lower Band']]

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        exp1 = prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = prices.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal_line

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

    def preprocess_data(self, data):
        data_scaled = self.scaler.fit_transform(data)
        
        # Get metadata features for the ticker and append to each sequence
        meta_features = self.get_meta_features()
        meta_features = np.tile(meta_features, (self.sequence_length, 1))  # Repeat for each sequence step

        x, y = [], []
        for i in range(self.sequence_length, len(data_scaled)):
            sequence = np.concatenate((data_scaled[i - self.sequence_length:i], meta_features), axis=1)
            x.append(sequence)
            y.append(data_scaled[i, 0])  # Predicting the close price

        x, y = np.array(x), np.array(y)
        return x, y, self.scaler  # Return scaler for inverse transformations later

    def get_data(self):
        data = self.load_data()
        return self.preprocess_data(data)
