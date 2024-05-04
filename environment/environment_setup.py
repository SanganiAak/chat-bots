import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

def init_environment():
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
    return pc, index_name, USER_AVATAR, BOT_AVATAR
