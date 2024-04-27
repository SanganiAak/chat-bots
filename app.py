import streamlit as st
import openai
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
import shelve

# Load environment variables
load_dotenv()

# Pinecone initialization
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Set the index name
index_name = os.getenv('PINECONE_INDEX')
USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Check if the Pinecone index exists with the correct dimension
index_list = pc.list_indexes()
index_info = next((idx for idx in index_list if idx['name'] == index_name), None)
if not index_info or index_info.get('dimension', 0) != 1536:
    if index_info:
        pc.delete_index(name=index_name)
    pc.create_index(
        name=index_name,
        dimension=1536,  # Ensure this matches the embedding dimension
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-2')
    )

client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))

# Get a handle to the index
vector_store = pc.Index(name=index_name)

# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        messages_dict_list = db.get("messages", [])
    # Convert list of dictionaries back to messages
    messages = [message_dict.copy() for message_dict in messages_dict_list]
    return messages

# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

def custom_text_splitter(text, lines_per_chunk):
    chunks = []
    lines = str(text).split('\n')
    non_empty_lines = [line for line in lines if line.strip()]  # Remove blank lines
    for i in range(0, len(non_empty_lines), lines_per_chunk):
        chunk_lines = non_empty_lines[i:i+lines_per_chunk]
        chunk = '\n'.join(chunk_lines)
        chunks.append(chunk)
    return chunks

# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history() 

# Streamlit UI
st.title("Intelligent Chatbot")
with st.sidebar:
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)
    if st.button("Load and Index Documents"):
        for uploaded_file in uploaded_files:
            with pdfplumber.open(uploaded_file) as pdf:
                full_text = ''
                for page in pdf.pages:
                    full_text += page.extract_text() + '\n'
                document_chunks = custom_text_splitter(full_text, lines_per_chunk=10)
                # for chunk in custom_text_splitter(full_text, lines_per_chunk=10):
                #     st.write(chunk)
                
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

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# User query
if user_query := st.chat_input("Ask a question:"):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=user_query
    ).data[0].embedding
    
    top_docs = vector_store.query(vector=response, top_k=5, include_metadata=True, namespace='Default')

    context = " ".join([doc.metadata.get('text', '') if doc.metadata else '' for doc in top_docs.matches])

    query = "Answer the user's question based on the below context:\n\n "+ context +". The question is: "+ user_query

    message = [{"role": "user", "content": query}]

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=message,
        temperature=0.2
    )
    
    content = response.choices[0].message.content

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.write(content)
        st.session_state.messages.append({"role": "assistant", "content": content})

# Save chat history after each interaction
save_chat_history(st.session_state.messages)
