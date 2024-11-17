import requests
from bs4 import BeautifulSoup

def scrape_headlines(url="https://finance.yahoo.com"):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        headlines = [headline.text.strip() for headline in soup.find_all('h3', class_='Mb(5px)')]
        return headlines
    except Exception as e:
        print(f"Error scraping headlines: {e}")
        return []

if __name__ == "__main__":
    headlines = scrape_headlines()
    print("Scraped Headlines:")
    print(headlines)
