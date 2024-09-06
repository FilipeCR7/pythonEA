import requests
from bs4 import BeautifulSoup


def get_article_urls(url):
    """
    Scrapes the given URL for article links.

    :param url: URL of the page to scrape for articles.
    :return: List of article URLs.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Finds all article links based on their class, adjust as necessary for changes in site structure.
    article_links = soup.find_all('a', class_='media-story-card__heading__tFMEu')  # Adjust class as needed
    urls = ['https://www.reuters.com' + link['href'] for link in article_links if 'href' in link.attrs]

    return urls


def get_article_text(url):
    """
    Scrapes the full text of the article from the given URL.

    :param url: URL of the article to scrape.
    :return: The full text of the article.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Finds the article body based on its class, adjust if the structure of the site changes.
    paragraphs = soup.find_all('div', class_='article-body__content__17Yit')  # Adjust class as needed
    article_text = ' '.join([para.get_text() for para in paragraphs])

    return article_text


# Example usage: Get article URLs from Reuters Markets section and scrape the first 5 articles.
reuters_urls = get_article_urls('https://www.reuters.com/markets')  # Adjust URL as needed

for url in reuters_urls[:5]:
    article_text = get_article_text(url)
    print(f"Scraping article: {url}")
    print(f"Article text: {article_text[:500]}")  # Print the first 500 characters of the article
    print("..." * 10)
