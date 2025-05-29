from sentence_transformers import SentenceTransformer
import numpy as np
from .base_embedder import BaseEmbedder

class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence Transformer based embeddings"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        super().__init__(embedder_type="sentence_transformer")
        try:
            self.model = SentenceTransformer(model_name)
            print(f"✅ Loaded SentenceTransformer model: {model_name}")
        except Exception as e:
            print(f"❌ Error loading SentenceTransformer model: {e}")
            self.model = None
    
    def __call__(self, input):
        """Generate embeddings using SentenceTransformer"""
        if not self.model:
            raise ValueError("SentenceTransformer model not loaded")
            
        try:
            # Generate embeddings
            embeddings = self.model.encode(input)
            
            # Convert to list format for ChromaDB
            if len(input) == 1:
                return [embeddings.tolist()]
            return embeddings.tolist()
            
        except Exception as e:
            print(f"❌ Error generating SentenceTransformer embeddings: {e}")
            # Return zero embeddings as fallback
            dim = 384  # Default dimension for many sentence transformers
            return [[0.0] * dim] * len(input)