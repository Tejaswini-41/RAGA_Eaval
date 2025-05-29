from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from .base_embedder import BaseEmbedder

class CodeBertEmbedder(BaseEmbedder):
    """CodeBERT-based embeddings for code"""
    
    def __init__(self, model_name='microsoft/codebert-base'):
        super().__init__(embedder_type="codebert")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            print(f"✅ Loaded CodeBERT model: {model_name}")
        except Exception as e:
            print(f"❌ Error loading CodeBERT model: {e}")
            self.tokenizer = None
            self.model = None
    
    def __call__(self, input):
        """Generate embeddings using CodeBERT"""
        if not self.model or not self.tokenizer:
            raise ValueError("CodeBERT model not loaded")
            
        try:
            embeddings = []
            for text in input:
                # Truncate to avoid token length issues
                if len(text) > 8000:
                    text = text[:8000]
                
                # Tokenize and get model outputs
                inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=512)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Use CLS token as the embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                embeddings.append(embedding.tolist())
            
            return embeddings
            
        except Exception as e:
            print(f"❌ Error generating CodeBERT embeddings: {e}")
            # Return zero embeddings as fallback
            dim = 768  # Default dimension for CodeBERT
            return [[0.0] * dim] * len(input)