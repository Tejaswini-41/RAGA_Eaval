import abc
import os
import uuid
import chromadb

class BaseEmbedder:
    """Base class for all embedding methods"""
    
    def __init__(self, embedder_type="base"):
        self.embedder_type = embedder_type
        self.collection = None
        self.client = None
    
    @abc.abstractmethod
    def __call__(self, input):
        """Generate embeddings for input text"""
        pass
    
    def initialize_store(self, chroma_path="chroma_chunks/"):
        """Initialize the ChromaDB store"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(chroma_path, exist_ok=True)
            
            # Initialize ChromaDB
            self.client = chromadb.PersistentClient(path=chroma_path)
            
            # Create a unique collection name
            collection_name = f"{self.embedder_type}_{str(uuid.uuid4())}"
            
            # Create Collection
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self,
                metadata={"hnsw:space": "cosine"},
            )
            
            print(f"✅ ChromaDB initialized with {self.embedder_type} collection '{collection_name}'")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing ChromaDB with {self.embedder_type}: {e}")
            return False
    
    def store_embeddings(self, documents, pr_numbers):
        """Store embeddings in ChromaDB"""
        if not self.collection:
            if not self.initialize_store():
                return False
                
        try:
            # Store embeddings
            self.collection.add(
                documents=documents,
                ids=[f"id{i}" for i in range(len(documents))],
                metadatas=[{"pr_number": pr} for pr in pr_numbers]
            )
            
            print(f"✅ {len(documents)} embeddings stored with {self.embedder_type}")
            return True
            
        except Exception as e:
            print(f"❌ Error storing embeddings with {self.embedder_type}: {e}")
            return False
    
    def query_similar(self, query_text, num_similar=3):
        """Query for similar documents"""
        if not self.collection:
            print(f"❌ Collection not initialized for {self.embedder_type}")
            return None
            
        try:
            # Prepare query
            query_embedding = self([query_text])
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding[0] if isinstance(query_embedding, list) else query_embedding],
                n_results=num_similar
            )
            
            return results
            
        except Exception as e:
            print(f"❌ Error querying with {self.embedder_type}: {e}")
            return None