import os 
import dotenv

def load_environment_variable():
    """ Loads API keys and other configuration from a .env file and sets required environment variables"""

    # load environment variables from .env into os.environ
    dotenv.load_dotenv()
    
    # retrieve specific api keys
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Set additional environment variables that might be required by your application
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key if langchain_api_key else ""
    os.environ["OPENAI_API_KEY"] = openai_api_key if openai_api_key else ""





dotenv.load_dotenv()

langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key

if __name__ == "__main__":
   # for testing purposes. when ran standalones, loads the env vars and prints them
   load_environment_variable()

   # print the vars 
   print("LANGCHAIN_API_KEY: ", os.getenv("LANGCHAIN_API_KEY"))
   print("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))