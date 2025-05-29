from .tfidf_embedder import TFIDFEmbedder
from .sentence_transformer_embedder import SentenceTransformerEmbedder
from .code_bert_embedder import CodeBertEmbedder
from .hybrid_embedder import HybridEmbedder
from .word2vec_embedder import Word2VecEmbedder  # Add this import

class EmbeddingFactory:
    """Factory for creating different embedding methods"""
    
    @staticmethod
    def get_embedder(embedder_type="tfidf", **kwargs):
        """Get an appropriate embedder based on type"""
        embedder_map = {
            "tfidf": TFIDFEmbedder,
            "sentence_transformer": SentenceTransformerEmbedder,
            "codebert": CodeBertEmbedder,
            "hybrid": HybridEmbedder,
            "word2vec": Word2VecEmbedder  # Add this line
        }
        
        if embedder_type not in embedder_map:
            print(f"❌ Unknown embedder type: {embedder_type}. Defaulting to TF-IDF.")
            embedder_type = "tfidf"
        
        # Create and return the embedder
        return embedder_map[embedder_type](**kwargs)
    
    @staticmethod
    def get_available_embedders():
        """Return a list of available embedding methods"""
        return {
            "tfidf": "TF-IDF (Term Frequency-Inverse Document Frequency)",
            "sentence_transformer": "Sentence Transformer (all-MiniLM-L6-v2)",
            "codebert": "CodeBERT (microsoft/codebert-base)",
            "hybrid": "Hybrid (TF-IDF + Sentence Transformer)",
            "word2vec": "Word2Vec (Trained on-the-fly)"  # Add this line
        }