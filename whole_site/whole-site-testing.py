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

base_url = 'https://www.georgetown.edu/'
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
        subpage_soup = fetch_and_parse(url)
        subpage_contents[url] = subpage_soup.get_text()
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {url}: {e}")

from langchain_openai import AzureOpenAI


# Initialize LangChain with a suitable language model
llm = AzureOpenAI(api_key=os.getenv("OPENAI_API_KEY"), azure_endpoint=os.getenv("AZURE_ENDPOINT"))


for url, content in subpage_contents.items():
    try:
        prompt = f"Summarize the following text: {content}"
        summary = llm.generate([prompt])  # Check if generate method accepts list
        # Handle the response
        # ...
    except Exception as e:
        print(f"Error processing {url}: {e}")
