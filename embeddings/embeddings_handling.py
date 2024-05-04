import openai

def custom_text_splitter(text, lines_per_chunk):
    chunks = []
    lines = str(text).split('\n')
    non_empty_lines = [line for line in lines if line.strip()]  # Remove blank lines
    for i in range(0, len(non_empty_lines), lines_per_chunk):
        chunk_lines = non_empty_lines[i:i+lines_per_chunk]
        chunk = '\n'.join(chunk_lines)
        chunks.append(chunk)
    return chunks

def process_embeddings_and_store(document_texts, pc, index_name, client):
    vector_store = pc.Index(name=index_name)
    for full_text in document_texts:
        document_chunks = custom_text_splitter(full_text, lines_per_chunk=10)
        # Index documents
        for chunk_index, chunk in enumerate(document_chunks):
            # Create embedding for each chunk
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=chunk
            )
            # Extract the embedding from the response
            embedding = response.data[0].embedding

            # Store the embedding along with metadata
            embedding_with_metadata = {
                "id": str(chunk_index + 1),  # Incremented by 1 to start from 1
                "values": embedding,
                "metadata": {
                    "text": chunk  # Store the text of the chunk as metadata
                }
            }
            # Upsert the embedding into the vector store
            vector_store.upsert([embedding_with_metadata], namespace='Default')
