# PARSE THE SITEMAP INTO URLS 
from sitemap_parser import extract_urls_from_sitemap

sitemap_url = 'https://manuel.fr/sitemap.xml'
urls = extract_urls_from_sitemap(sitemap_url)

# IMPORT API KEYS INTO ENVIROMENT VARIABLES
from config_loader import load_environment_variable

load_environment_variable()


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