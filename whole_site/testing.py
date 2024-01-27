import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from langchain_community.document_loaders import BSHTMLLoader



from dotenv import load_dotenv
# Construct the path to the .env file
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')

# Load the .env file
load_dotenv(dotenv_path=env_path)

def fetch_and_parse(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup

base_url = 'https://www.dickinson.edu/'
main_page_soup = fetch_and_parse(base_url)

# Filter and complete the URLs
subpage_urls = [
    urljoin(base_url, a['href']) 
    for a in main_page_soup.find_all('a', href=True) 
    if a['href'] and not a['href'].startswith(('#', 'javascript:', 'mailto:', 'tel:'))
]
# Exclude fragment identifiers (URLs that contain '#')
subpage_urls = [url for url in subpage_urls if '#' not in url][:3]


subpage_contents = {}
for url in subpage_urls:
    try:
        # Fetch the HTML content
        html_content = fetch_and_parse(url)
        
        # Initialize BSHTMLLoader with the HTML content
        loader = BSHTMLLoader(str(html_content))
        
        # Load data
        data = loader.load()
        print(data)
    except Exception as e:
        print(f"Failed to process {url}: {e}")