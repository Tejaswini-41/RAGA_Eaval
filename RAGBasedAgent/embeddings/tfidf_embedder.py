import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from .base_embedder import BaseEmbedder

class TFIDFEmbedder(BaseEmbedder):
    """TF-IDF based embedding method"""
    
    def __init__(self, max_features=100):
        super().__init__(embedder_type="tfidf")
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.fitted = False
        self.pickle_path = 'tfidf_vectorizer.pkl'
    
    def __call__(self, input):
        """Generate TF-IDF embeddings for input text"""
        if not self.fitted:
            # First fit the vectorizer on all texts
            self.vectorizer.fit(input)
            self.fitted = True
            # Save the vectorizer for query time
            with open(self.pickle_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
        
        # Transform texts to vectors
        vectors = self.vectorizer.transform(input).toarray()
        return vectors.tolist()
    
    def load_vectorizer(self):
        """Load vectorizer from pickle if it exists"""
        if os.path.exists(self.pickle_path):
            with open(self.pickle_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
                self.fitted = True
                return True
        return False