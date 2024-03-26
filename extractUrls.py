from config_loader import load_environment_variable
from sitemap_parser import extract_urls_from_sitemap
from url_text_loader import load_text_from_urls
from text_splitter import recursive_text_splitter
from embeddings_vector_store import store_in_vector_db

# IMPORT API KEYS INTO ENVIROMENT VARIABLES
load_environment_variable()

# PARSE THE SITEMAP INTO URLS 
sitemap_url = 'https://manuel.fr/sitemap.xml'
urls = extract_urls_from_sitemap(sitemap_url)

# LOAD URLS AS TEXT
docs = load_text_from_urls(urls)


# SPLIT THE TEXT
all_splits = recursive_text_splitter(docs)


# STORE IN VECTOR DB
vectorstore = store_in_vector_db(all_splits, use_ollama= True, ollama_model='nomic-embed-text:latest')


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