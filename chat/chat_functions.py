import streamlit as st
import openai
import shelve

def load_chat_history():
    with shelve.open("chat_history") as db:
        messages_dict_list = db.get("messages", [])
    messages = [message_dict.copy() for message_dict in messages_dict_list]
    return messages

def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

def display_chat_messages(messages, USER_AVATAR, BOT_AVATAR):
    for message in messages:
        avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

def handle_user_query(client, vector_store, USER_AVATAR, BOT_AVATAR):
    if user_query := st.chat_input("Ask a question:"):
        # Generate embedding for the user query
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=user_query
        ).data[0].embedding

        # Query the vector store for the top documents with relevance scores
        top_docs = vector_store.query(vector=response, top_k=5, include_metadata=True, namespace='Default')

        # Compile the context and assess relevance
        context = ""
        relevance_threshold = 0.7  # Set a threshold for minimum relevance
        relevant_context_found = False
        for doc in top_docs.matches:
            if doc.score >= relevance_threshold:
                context += doc.metadata['text'] + " "
                relevant_context_found = True

        if relevant_context_found:
            # Prepare a message for the model to generate a response
            query = "Answer the user's question based on the below context:\n\n" + context + "The question is: " + user_query
            message = [{"role": "user", "content": query}]
            
            # Request a response from OpenAI based on the provided context
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=message,
                temperature=0.2
            )
            content = response.choices[0].message.content
        else:
            # Default response if no relevant context is found or the query is not related to the context
            content = "I'm not sure about that."

        # Display user query and bot response in the chat
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.write(content)
            st.session_state.messages.append({"role": "assistant", "content": content})
