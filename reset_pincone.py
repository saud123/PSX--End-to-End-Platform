import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "psx-financial-intelligence"

# Delete the old index
if index_name in [idx.name for idx in pc.list_indexes()]:
    print(f"Deleting old index: {index_name}...")
    pc.delete_index(index_name)

# Recreate it as a standard Vector index (384 dimensions for MiniLM)
print("Creating new standard Vector index...")
pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
print("✅ Pinecone index reset successfully! You can now re-upload your PDF.")


