import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    try:
        vec1 = np.array(vec1).reshape(1, -1)
        vec2 = np.array(vec2).reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0]
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return 0.0

def normalize_vector(vector):
    """Normalize a vector to unit length"""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def save_embeddings_to_file(embeddings, filename):
    """Save embeddings to pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings_from_file(filename):
    """Load embeddings from pickle file"""
    if not os.path.exists(filename):
        return None
    with open(filename, 'rb') as f:
        return pickle.load(f)

def visualize_embeddings(embeddings, labels=None):
    """Visualize embeddings in 2D using dimensionality reduction"""
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        # Reduce dimensions to 2D
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
        
        if labels is not None:
            for i, label in enumerate(labels):
                plt.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
        
        plt.title("Embedding Visualization")
        plt.show()
        
    except Exception as e:
        print(f"Error visualizing embeddings: {e}")