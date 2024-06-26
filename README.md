Project to create a chatbot to ask questions on a website's data using RAG.

1. Parse the XML of a website to get all the URLs
2. Scrape all of the URLs as text
3. Embeddings
4. ... to Vector DB
5. RAG
6. Chatbot

- Interesting links

  - https://github.com/streamlit/example-app-langchain-rag
  - https://realpython.com/build-llm-rag-chatbot-with-langchain/
  - https://python.langchain.com/docs/use_cases/question_answering/quickstart#retrieval-and-generation-generate
  - https://medium.com/@vndee.huynh/build-your-own-rag-and-run-it-locally-langchain-ollama-streamlit-181d42805895
  - https://github.com/hwchase17/chroma-langchain/blob/master/qa.ipynb
  - https://github.com/hwchase17/chroma-langchain/blob/master/persistent-qa.ipynb
  - https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/
  - https://www.koyeb.com/tutorials/use-llamaindex-streamlit-and-openai-to-query-unstructured-data
  - https://github.com/carolinedlu/llamaindex-chat-with-streamlit-docs

- Stuff
  - Use Streamlit for frontend

### List of tasks

- [ ] Implement logic to check if document is already in vector DB
  - [ ] using log/txt file?
  - [ ] querying the db directly?
- [ ] Create a chatbot (has access to previous messages)
- [ ] using pinecone as a vectorstore?

- [ ] check the last link on koyeb.com, my main app, and the llamaindex_test notebook, mashup all 3
