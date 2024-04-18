from textblob import TextBlob


def get_sentiment(text):
    """
    Calculate the sentiment polarity of a given text.

    Parameters:
    text (str): The text to analyze.

    Returns:
    float: The sentiment polarity score.
    """
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Returns a value between -1 (negative) and 1 (positive)

for url in reuters_urls[:5]:
    article_text = get_article_text(url)
    sentiment_score = analyze_sentiment(article_text)
    print(f"URL: {url}, Sentiment Score: {sentiment_score}")