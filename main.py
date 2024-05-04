import streamlit as st
import openai
import os
from environment.environment_setup import init_environment
from transcription.pdf_transcription import transcribe_pdf_files
from embeddings.embeddings_handling import process_embeddings_and_store
from chat.chat_functions import load_chat_history, save_chat_history, display_chat_messages, handle_user_query

pc, index_name, USER_AVATAR, BOT_AVATAR = init_environment()
client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))

# Streamlit UI
st.title("Intelligent Chatbot")
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

with st.sidebar:
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)
    if st.button("Load and Index Documents"):
        document_texts = transcribe_pdf_files(uploaded_files)
        process_embeddings_and_store(document_texts, pc, index_name, client)

display_chat_messages(st.session_state.messages, USER_AVATAR, BOT_AVATAR)
handle_user_query(client, pc.Index(name=index_name), USER_AVATAR, BOT_AVATAR)
save_chat_history(st.session_state.messages)
