# chat-bots
streamlit chatbot applications

# Intelligent Chatbot Application

This Streamlit-based chatbot application utilizes OpenAI's GPT models for generating responses and Pinecone's vector database for storing and retrieving document embeddings. It's designed to provide users with intelligent responses based on the content from indexed documents loaded into Pinecone.

## Features

- **Web-Based Document Loading**: Allows users to upload PDF documents directly through the interface for indexing.
- **Intelligent Response Generation**: Utilizes OpenAI's powerful models to generate context-aware responses to user queries.
- **Embedding Storage and Retrieval**: Leverages Pinecone's serverless vector database for efficient storage and retrieval of text embeddings, enhancing the response quality based on relevant document content.

## Prerequisites

Before setting up the application, ensure you have the following:
- Python 3.8 or higher
- Access to OpenAI's API
- Access to Pinecone's API
- An internet connection for accessing APIs and handling document uploads

## Installation and Setup


```bash
git clone [repository URL]
cd [repository name]

python -m venv venv
source venv/bin/activate  # For Unix or MacOS
venv\Scripts\activate  # For Windows

pip install -r requirements.txt

OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

streamlit run app.py
