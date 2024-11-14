import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

class StockDataLoader:
    def __init__(self, ticker, sequence_length=100, data_dir="dataset"):
        self.ticker = ticker
        self.sequence_length = sequence_length
        self.data_dir = data_dir
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self):
        file_path_stocks = os.path.join(self.data_dir, 'stocks', f"{self.ticker}.csv")
        file_path_etfs = os.path.join(self.data_dir, 'etfs', f"{self.ticker}.csv")

        if os.path.exists(file_path_stocks):
            file_path = file_path_stocks
        elif os.path.exists(file_path_etfs):
            file_path = file_path_etfs
        else:
            raise FileNotFoundError(f"No data found for ticker {self.ticker}")

        data = pd.read_csv(file_path)
        required_columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        if not all(column in data.columns for column in required_columns):
            raise ValueError(f"CSV file for {self.ticker} is missing required columns")

        data = data.sort_values(by="Date")
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
