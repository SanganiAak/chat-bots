import streamlit as st
import openai
import os
import shelve
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import speech_recognition as sr
from embeddings.embeddings_handling import process_embeddings_and_store

# Load environment variables
load_dotenv()

# Pinecone initialization
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv('PINECONE_INDEX')

# Check and create the Pinecone index
index_list = pc.list_indexes()
index_info = next((idx for idx in index_list if idx['name'] == index_name), None)
if not index_info or index_info.get('dimension', 0) != 1536:
    if index_info:
        pc.delete_index(name=index_name)
    pc.create_index(name=index_name, dimension=1536, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))

client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))
vector_store = pc.Index(name=index_name)

def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

# Streamlit UI
st.title("Intelligent Chatbot")
with st.sidebar:
    uploaded_audio_files = st.file_uploader("Upload Audio files", type=['mp3', 'wav'], accept_multiple_files=True)

    if st.button("Transcribe Audio"):
        for uploaded_audio in uploaded_audio_files:
            # Assuming the use of SpeechRecognition for transcription
            audio_path = uploaded_audio.name
            with open(audio_path, "wb") as f:
                f.write(uploaded_audio.getvalue())
            # Transcribe audio using SpeechRecognition
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                transcribed_text = recognizer.recognize_google(audio_data)
                print(transcribed_text)
                process_embeddings_and_store(transcribed_text, pc, index_name, client)
            # Store transcription in database or use it directly
            with shelve.open("transcriptions") as db:
                db[uploaded_audio.name] = transcribed_text


if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Display chat messages and transcriptions
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# User query handling
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
