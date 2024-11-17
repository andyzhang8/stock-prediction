import yfinance as yf
from dotenv import load_dotenv
import os

env_path = os.path.join(os.path.dirname(__file__), '../env/.env')
load_dotenv(env_path)

def fetch_stock_data(ticker=None, period="1mo", interval="1d"):
    """
    Fetch historical stock data using Yahoo Finance API
    """
    try:
        ticker = ticker or os.getenv("DEFAULT_STOCK_TICKER")
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        return hist
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

if __name__ == "__main__":
    stock_data = fetch_stock_data()
    print("Live Stock Data:")
    print(stock_data.head())
