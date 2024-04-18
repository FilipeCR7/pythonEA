import requests
from bs4 import BeautifulSoup

def get_article_urls(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article_links = soup.find_all('a', class_='media-story-card__heading__tFMEu')  # Adjust class as needed
    urls = ['https://www.reuters.com' + link['href'] for link in article_links if 'href' in link.attrs]
    return urls

def get_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('div', class_='article-body__content__17Yit')  # Adjust class as needed
    article_text = ' '.join([para.get_text() for para in paragraphs])
    return article_text

# Example usage
reuters_urls = get_article_urls('https://www.reuters.com/markets')  # Adjust URL as needed

for url in reuters_urls[:5]:
    article_text = get_article_text(url)
    print(f"Scraping article: {url}")
    print(f"Article text: {article_text[:500]}")
    print("..."*10)
