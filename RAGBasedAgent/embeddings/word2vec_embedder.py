from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import numpy as np
import os
import nltk
from nltk.tokenize import word_tokenize
from .base_embedder import BaseEmbedder

class Word2VecEmbedder(BaseEmbedder):
    """Word2Vec based embedding method"""
    
    def __init__(self, pretrained_path=None, vector_size=100, window=5, min_count=1):
        super().__init__(embedder_type="word2vec")
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
        
        # Try to load pretrained model if path is provided
        if pretrained_path and os.path.exists(pretrained_path):
            try:
                self.model = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
                print(f"✅ Loaded pretrained Word2Vec model from {pretrained_path}")
            except Exception as e:
                print(f"❌ Error loading pretrained Word2Vec model: {e}")
        
        # Download NLTK resources
        try:
            # Try to download punkt tokenizer
            print("⚙️ Checking NLTK punkt tokenizer...")
            try:
                nltk.data.find('tokenizers/punkt')
                print("✅ NLTK punkt tokenizer found")
            except LookupError:
                print("⚙️ Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
                print("✅ Downloaded NLTK punkt tokenizer")
        except Exception as e:
            print(f"⚠️ NLTK download error: {e}")
            print("Will fall back to simple tokenization if needed")
    
    def __call__(self, input):
        """Generate Word2Vec embeddings for input text"""
        try:
            # If we don't have a pretrained model, train one on the input
            if self.model is None:
                # Tokenize input texts
                tokenized_texts = self._tokenize_texts(input)
                
                # Train Word2Vec model
                self.model = Word2Vec(
                    sentences=tokenized_texts,
                    vector_size=self.vector_size,
                    window=self.window,
                    min_count=self.min_count,
                    workers=4
                )
                print(f"✅ Trained Word2Vec model on {len(input)} texts")
            
            # Generate embeddings for each text
            embeddings = []
            
            for text in input:
                # Tokenize
                tokens = self._tokenize_text(text)
                
                # Filter words that are in the vocabulary
                if hasattr(self.model, 'wv'):
                    # For trained Word2Vec model
                    tokens = [token for token in tokens if token in self.model.wv.key_to_index]
                    
                    if tokens:
                        # Get embeddings for all tokens and average them
                        token_embeddings = [self.model.wv[token] for token in tokens]
                        text_embedding = np.mean(token_embeddings, axis=0)
                    else:
                        # If no tokens match, return zero vector
                        text_embedding = np.zeros(self.vector_size)
                else:
                    # For KeyedVectors model
                    tokens = [token for token in tokens if token in self.model]
                    
                    if tokens:
                        # Get embeddings for all tokens and average them
                        token_embeddings = [self.model[token] for token in tokens]
                        text_embedding = np.mean(token_embeddings, axis=0)
                    else:
                        # If no tokens match, return zero vector
                        text_embedding = np.zeros(self.vector_size)
                
                embeddings.append(text_embedding.tolist())
            
            return embeddings
            
        except Exception as e:
            print(f"❌ Error generating Word2Vec embeddings: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * self.vector_size] * len(input)
    
    def _tokenize_texts(self, texts):
        """Tokenize a list of texts with fallback if NLTK fails"""
        tokenized_texts = []
        for text in texts:
            tokens = self._tokenize_text(text)
            tokenized_texts.append(tokens)
        return tokenized_texts
    
    def _tokenize_text(self, text):
        """Tokenize a single text with fallback if NLTK fails"""
        try:
            # Try using NLTK's word_tokenize
            return word_tokenize(text.lower())
        except Exception as e:
            # Fall back to simple tokenization
            return text.lower().split()
    
    def load_or_train_model(self, texts=None, save_path='word2vec.model'):
        """Load existing model or train new one and save it"""
        if os.path.exists(save_path):
            try:
                self.model = Word2Vec.load(save_path)
                print(f"✅ Loaded existing Word2Vec model from {save_path}")
                return True
            except Exception as e:
                print(f"❌ Error loading Word2Vec model: {e}")
        
        if texts:
            try:
                # Tokenize input texts
                tokenized_texts = self._tokenize_texts(texts)
                
                # Train Word2Vec model
                self.model = Word2Vec(
                    sentences=tokenized_texts,
                    vector_size=self.vector_size,
                    window=self.window,
                    min_count=self.min_count,
                    workers=4
                )
                
                # Save model
                self.model.save(save_path)
                print(f"✅ Trained and saved Word2Vec model to {save_path}")
                return True
            except Exception as e:
                print(f"❌ Error training Word2Vec model: {e}")
        
        return False