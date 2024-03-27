from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def store_in_vector_db(documents, use_ollama=False, ollama_model='nomic-embed-text:latest'):
    """
    Stores documents in a vector database using either OpenAI or Ollama embeddings.

    Args:
        documents (list): The documents to be stored in the vector database.
        use_ollama (bool): Flag to use Ollama embeddings instead of OpenAI. Defaults to False.
        ollama_model (str): The model name to use for Ollama embeddings. Defaults to 'nomic'.
    """
    
    # Choose the embeddings model based on user input
    if use_ollama:
        embeddings_model = OllamaEmbeddings(model= ollama_model)
    else:
        embeddings_model = OpenAIEmbeddings()

    persist_directory = 'db'
    # Create the vector store and populate it with documents
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings_model,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    return vectorstore
