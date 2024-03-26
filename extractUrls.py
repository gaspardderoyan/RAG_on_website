# https://python.langchain.com/docs/use_cases/question_answering/quickstart

import requests
import xml.etree.ElementTree as ET

# Function to extract URLs from the sitemap
def extract_urls_from_sitemap(sitemap_url):
    response = requests.get(sitemap_url)
    root = ET.fromstring(response.content)
    urls = [elem.text for elem in root.iter('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')]
    return urls

# Example usage
sitemap_url = 'https://manuel.fr/sitemap.xml'
urls = extract_urls_from_sitemap(sitemap_url)
print(urls)


# from bs4 import BeautifulSoup
# import requests

# def extract_text_with_beautifulsoup(urls):
#     texts = []
#     for url in urls:
#         response = requests.get(url)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         text = soup.get_text()
#         texts.append(text)
#     return texts

# # Assuming 'urls' is your list of URLs
# texts = extract_text_with_beautifulsoup(urls)



# Import API keys into enviroment variables
import os
import dotenv

dotenv.load_dotenv()

langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key

###

# LOAD URLS AS TEXT

from langchain_community.document_loaders import WebBaseLoader
import nest_asyncio # to load urls concurently 

# fixes a bug in jupyter
nest_asyncio.apply()

loader = WebBaseLoader(urls)
loader.requests_per_second = 1
docs = loader.load()

###

# SPLIT THE TEXT

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

all_splits = text_splitter.split_documents(docs)

###

# STORE IN VECTOR DB

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# embeddings_model = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents= all_splits,
    embedding= OpenAIEmbeddings()
)


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