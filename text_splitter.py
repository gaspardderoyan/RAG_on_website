from langchain_text_splitters import RecursiveCharacterTextSplitter

def recursive_text_splitter(docs, chunk_size=1000, chunk_overlap=200):
    """
    Splits documents into chunks with a specified size and overlap.

    :param docs: The documents to be split.
    :param chunk_size: The size of each chunk (default is 1000 characters).
    :param chunk_overlap: The number of characters to overlap between chunks (default is 200 characters).
    :return: A list of document splits.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )

    splits = text_splitter.split_documents(docs)

    return splits
