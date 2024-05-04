import streamlit as st
import openai
import os
from environment.environment_setup import init_environment
from chat.chat_functions import load_chat_history, save_chat_history, display_chat_messages, handle_user_query
from sidebar.website_loader import load_and_index_documents
from transcription.audio_transcription import transcribe_audio_files
from transcription.pdf_transcription import transcribe_pdf_files
from embeddings.embeddings_handling import process_embeddings_and_store

# Initialize environment and Pinecone
pc, index_name, USER_AVATAR, BOT_AVATAR = init_environment()
client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))
vector_store = pc.Index(name=index_name)

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Streamlit UI Setup
st.title("Intelligent Chatbot")

# Sidebar for document URLs
with st.sidebar:
    st.subheader("Upload and Process Documents")
    urls_input = st.text_area("Enter document URLs (one per line)", height=150)
    if st.button("Load and Index URLs"):
        load_and_index_documents(urls_input, vector_store, client)

    # Uploader for PDF files
    uploaded_pdf_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, key="pdf")
    if uploaded_pdf_files and st.button("Transcribe and Index PDFs"):
        document_texts = transcribe_pdf_files(uploaded_pdf_files)
        process_embeddings_and_store(document_texts, pc, index_name, client)

    # Uploader for Audio files
    uploaded_audio_files = st.file_uploader("Upload Audio files", type=['mp3', 'wav'], accept_multiple_files=True, key="audio")
    if uploaded_audio_files and st.button("Transcribe Audio"):
        transcribe_audio_files(uploaded_audio_files)

# Display chat messages
display_chat_messages(st.session_state.messages, USER_AVATAR, BOT_AVATAR)

# Handling user queries
handle_user_query(client, vector_store, USER_AVATAR, BOT_AVATAR)

# Save chat history after each interaction
save_chat_history(st.session_state.messages)
