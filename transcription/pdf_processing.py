import pdfplumber

def process_pdf_files(uploaded_files, pc, index_name, client):
    vector_store = pc.Index(name=index_name)
    for uploaded_file in uploaded_files:
        with pdfplumber.open(uploaded_file) as pdf:
            full_text = ''
            for page in pdf.pages:
                full_text += page.extract_text() + '\n'
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
