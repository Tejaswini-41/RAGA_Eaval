import numpy as np
from .base_embedder import BaseEmbedder
from .tfidf_embedder import TFIDFEmbedder
from .sentence_transformer_embedder import SentenceTransformerEmbedder

class HybridEmbedder(BaseEmbedder):
    """Hybrid embedding approach combining multiple methods"""
    
    def __init__(self, weights=None):
        super().__init__(embedder_type="hybrid")
        
        # Initialize individual embedders
        self.tfidf = TFIDFEmbedder()
        self.sentence_transformer = SentenceTransformerEmbedder()
        
        # Set weights for each method
        self.weights = weights or {"tfidf": 0.5, "sentence_transformer": 0.5}
        
    def __call__(self, input):
        """Generate hybrid embeddings by combining methods"""
        try:
            # Get embeddings from each method
            tfidf_embeddings = self.tfidf(input)
            st_embeddings = self.sentence_transformer(input)
            
            # Combine embeddings using simple concatenation
            # For actual implementation, normalize dimensions or use more sophisticated methods
            hybrid_embeddings = []
            
            for i in range(len(input)):
                # Since dimensions may vary, we normalize each embedding 
                # and then concatenate with weights
                tfidf_vec = np.array(tfidf_embeddings[i])
                st_vec = np.array(st_embeddings[i])
                
                # Normalize vectors if not zero
                tfidf_norm = tfidf_vec / np.linalg.norm(tfidf_vec) if np.linalg.norm(tfidf_vec) > 0 else tfidf_vec
                st_norm = st_vec / np.linalg.norm(st_vec) if np.linalg.norm(st_vec) > 0 else st_vec
                
                # Apply weights
                weighted_tfidf = tfidf_norm * self.weights["tfidf"]
                weighted_st = st_norm * self.weights["sentence_transformer"]
                
                # For now, we'll just concatenate (in practice you might want to reduce dimensions)
                combined = np.concatenate([weighted_tfidf, weighted_st])
                hybrid_embeddings.append(combined.tolist())
            
            return hybrid_embeddings
            
        except Exception as e:
            print(f"❌ Error generating hybrid embeddings: {e}")
            # Return empty embeddings as fallback
            return [[0.0] * 100] * len(input)