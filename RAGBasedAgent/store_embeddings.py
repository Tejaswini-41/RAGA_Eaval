import chromadb
from chromadb.utils import embedding_functions
import uuid
import os

def store_embeddings(changed_files, pull_requests, model_name="all-mpnet-base-v2"):
    """Store PR file embeddings in ChromaDB"""
    # Create directory if it doesn't exist
    CHROMA_DATA_PATH = "chroma_data/"
    os.makedirs(CHROMA_DATA_PATH, exist_ok=True)
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    
    # Create a unique collection name or use a fixed one
    COLLECTION_NAME = str(uuid.uuid4())  # Unique collection
    
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    
    try:
        # Create Collection
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func,
            metadata={"hnsw:space": "cosine"},
        )
        
        # Store PR file embeddings
        collection.add(
            documents=changed_files,
            ids=[f"id{i}" for i in range(len(changed_files))],
            metadatas=[{"pr_number": pr} for pr in pull_requests]
        )
        
        print(f"✅ ChromaDB initialized with collection '{COLLECTION_NAME}' and {len(changed_files)} embeddings stored!")
        return client, collection
        
    except Exception as e:
        print(f"Error storing embeddings: {e}")
        return None, None

if __name__ == "__main__":
    # If run directly, execute with sample data
    from fetch_prs import fetch_pull_requests
    
    repo_owner = 'Tejaswini-41'
    repo_name = 'RAGA_Eaval'
    
    changed_files, pull_requests = fetch_pull_requests(repo_owner, repo_name)
    
    if changed_files and pull_requests:
        store_embeddings(changed_files, pull_requests)