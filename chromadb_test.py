import chromadb

client = chromadb.PersistentClient('db')

client.heartbeat()

collection = client.get_collection(name='langchain')
collection.peek()
collection.count()
collection.peek()['metadatas']
type(collection.peek()['metadatas'])

collection.peek()['metadatas'][0]['source']
