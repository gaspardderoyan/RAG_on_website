from config_loader import load_environment_variable
from sitemap_parser import extract_urls_from_sitemap
from url_text_loader import load_text_from_urls
from text_splitter import recursive_text_splitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# IMPORT API KEYS INTO ENVIROMENT VARIABLES
load_environment_variable()

# PARSE THE SITEMAP INTO URLS 
sitemap_url = 'https://manuel.fr/sitemap.xml'
urls = extract_urls_from_sitemap(sitemap_url)

# LOAD URLS AS TEXT
docs = load_text_from_urls(urls)


# SPLIT THE TEXT
all_splits = recursive_text_splitter(docs, chunk_size=1500, chunk_overlap=300)

# all_splits[0].metadata
"""  
{'source': 'https://manuel.fr/blog',
 'title': 'Blog | Manuel.fr',
 'description': 'Blog',
 'language': 'fr',
 'start_index': 5}
"""

# all.splits[0].metadata['source']
# 'https://manuel.fr/blog'

# STORE IN VECTOR DB

# function to store split text embeddings in vector db
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
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

    persist_directory = 'db'
    # Create the vector store and populate it with documents
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings_model,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    return vectorstore

vectorstore = store_in_vector_db(all_splits, use_ollama= False, ollama_model='nomic-embed-text:latest')

# vectorstore.get()
# type(vectorstore.get())
# vectorstore.get(where={"source":"https://manuel.fr/docs/installation/En-savoir-plus/instruct"})
# type(vectorstore.get(where={"source":"https://manuel.fr/docs/installation/En-savoir-plus/instruct"}))

# vectorstore.get(where={"source":"https://manuel.fr/docs/installation/En-savoir-plus/instruct"})['metadatas']
# len(vectorstore.get(where={"source":"https://manuel.fr/docs/installation/En-savoir-plus/instruct"})['metadatas'])

# vectorstore.get(where={"source":"https://manuel.fr/docs/installation/En-savoir-plus/instruct"})['metadatas'][0]['source']

# RETRIEVER
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# CHOOSE LLM
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

from langchain_community.llms import Ollama

llm = Ollama(model="qwen:72b")

# llm.invoke("Tell me a joke")


# IMPORT PROMPT 
from langchain import hub

prompt = hub.pull("rlm/rag-prompt")

# IMPLEMENT RAG
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("Qui a cree ce site?"):
    print(chunk, end="", flush=True)



type(all_splits)
all_splits[0]
type(all_splits[0])