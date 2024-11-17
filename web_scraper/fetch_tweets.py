import tweepy
from dotenv import load_dotenv
import os

env_path = os.path.join(os.path.dirname(__file__), '../env/.env')
load_dotenv(env_path)

def fetch_tweets(query="AAPL", count=100):
    """
    Fetch tweets using Twitter API.
    """
    try:
        api_key = os.getenv("TWITTER_API_KEY")
        api_secret = os.getenv("TWITTER_API_SECRET")
        access_token = os.getenv("TWITTER_ACCESS_TOKEN")
        access_secret = os.getenv("TWITTER_ACCESS_SECRET")

        auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_secret)
        api = tweepy.API(auth)

        tweets = api.search_tweets(q=query, lang="en", count=count)
        tweet_texts = [tweet.text for tweet in tweets]
        return tweet_texts
    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return []

if __name__ == "__main__":
    tweets = fetch_tweets(query="AAPL", count=50)
    print("Fetched Tweets:")
    print(tweets)
