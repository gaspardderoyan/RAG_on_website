from langchain_text_splitters import RecursiveCharacterTextSplitter

def recursive_text_splitter(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    splits = text_splitter.split_documents(docs)

    return splits