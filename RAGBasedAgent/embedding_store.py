import chromadb
import os
import uuid
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFEmbeddingFunction:
    def __init__(self, max_features=100):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.fitted = False
    
    def __call__(self, input):  
        if not self.fitted:
            # First fit the vectorizer on all texts
            self.vectorizer.fit(input)
            self.fitted = True
            # Save the vectorizer for query time
            with open('tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
        
        # Transform texts to vectors
        vectors = self.vectorizer.transform(input).toarray()
        return vectors.tolist()

def store_embeddings(changed_files, pull_requests):
    """Store PR file embeddings in ChromaDB using TF-IDF"""
    # Create directory if it doesn't exist
    CHROMA_DATA_PATH = "chroma_chunks/"
    os.makedirs(CHROMA_DATA_PATH, exist_ok=True)
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    
    # Create a unique collection name
    COLLECTION_NAME = str(uuid.uuid4())
    
    embedding_func = TFIDFEmbeddingFunction()
    
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
        
        print(f"✅ ChromaDB initialized with collection '{COLLECTION_NAME}' and {len(changed_files)} embeddings stored")
        return client, collection
        
    except Exception as e:
        print(f"Error storing embeddings: {e}")
        return None, None