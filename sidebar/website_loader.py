import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from utils.text_utils import custom_text_splitter

def load_and_index_documents(urls_input, vector_store, client):
    urls = urls_input.split()
    for url_index, url in enumerate(urls):
        loader = WebBaseLoader(url)
        document = loader.load()
        document_chunks = custom_text_splitter(document[0].page_content, lines_per_chunk=10)
        id = 1
        for chunk in document_chunks:
            response = client.embeddings.create(model="text-embedding-ada-002", input=chunk)
            embedding = response.data[0].embedding
            embedding_with_metadata = {
                "id": str(id),
                "values": embedding,
                "metadata": {"text": chunk}
            }
            vector_store.upsert([embedding_with_metadata], namespace='Default')
            id += 1
