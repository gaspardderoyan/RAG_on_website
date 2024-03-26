import requests
import xml.etree.ElementTree as ET

# Function to extract URLs from the sitemap
def extract_urls_from_sitemap(sitemap_url):
    """Extracts and returns URLs from the given sitemap URL."""
    response = requests.get(sitemap_url)
    root = ET.fromstring(response.content)
    urls = [elem.text for elem in root.iter('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')]
    return urls



if __name__ == "__main__":
    test_sitemap_url = 'https://manuel.fr/sitemap.xml'
    print(extract_urls_from_sitemap(test_sitemap_url))
    