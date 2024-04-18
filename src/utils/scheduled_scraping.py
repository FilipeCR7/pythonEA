import schedule
import time
from web_scraping_utils import get_article_urls, get_article_text
from sentiment_analysis_utils import get_sentiment


def job():
    print("Scraping and Analyzing Sentiment...")
    # Example usage with Reuters
    reuters_urls = get_article_urls('https://www.reuters.com/markets')  # Update with the correct URL

    for url in reuters_urls[:5]:  # Just as an example, limit to the first 5 articles
        try:
            article_text = get_article_text(url)
            sentiment_score = get_sentiment(article_text)
            print(f"URL: {url}, Sentiment Score: {sentiment_score}")
        except Exception as e:
            print(f"Error processing {url}: {e}")


# Schedule the job every 15 minutes
schedule.every(15).minutes.do(job)

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(1)
