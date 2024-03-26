from langchain_community.document_loaders import WebBaseLoader
import nest_asyncio  # To handle concurrent requests properly

def load_text_from_urls(urls):
    """
    Loads text content from a list of URLs using the WebBaseLoader from langchain_community.
    
    Args:
        urls (list of str): The URLs from which to load the text content.
    
    Returns:
        list: A list of document objects containing the text content from each URL.
    """
    
    # Apply necessary settings for asynchronous operations, especially useful in Jupyter environments
    nest_asyncio.apply()

    # Initialize the WebBaseLoader with the list of URLs
    loader = WebBaseLoader(urls)
    
    # You can adjust the rate of requests per second as needed
    loader.requests_per_second = 1

    # Load the documents
    docs = loader.load()

    return docs

if __name__ == "__main__":
    # Example usage, for testing purposes
    test_urls = ['https://example.com']
    documents = load_text_from_urls(test_urls)
    for doc in documents:
        print(doc)  # Adjust this line based on how you want to print or process the documents
