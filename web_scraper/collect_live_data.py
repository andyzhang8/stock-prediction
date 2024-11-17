from scrape_headlines import scrape_headlines
from fetch_tweets import fetch_tweets
from fetch_stock_data import fetch_stock_data
from dotenv import load_dotenv
import os

env_path = os.path.join(os.path.dirname(__file__), '../env/.env')
load_dotenv(env_path)

def collect_live_data():
    """
    Collect live data from financial headlines, Twitter, and stock prices.
    """
    stock_ticker = os.getenv("DEFAULT_STOCK_TICKER")
    query = stock_ticker

    headlines = scrape_headlines()

    tweets = fetch_tweets(query=query)

    stock_data = fetch_stock_data(ticker=stock_ticker)

    return {
        "headlines": headlines,
        "tweets": tweets,
        "stock_data": stock_data
    }

if __name__ == "__main__":
    live_data = collect_live_data()
    print("Live Data:")
    print("Headlines:", live_data["headlines"])
    print("Tweets:", live_data["tweets"])
    print("Stock Data:")
    print(live_data["stock_data"].head())
