from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import numpy as np
from GithubAuth import get_pull_request

def text_embedding(text):
    """Generate TF-IDF based embedding for text"""
    # Load the vectorizer that was used during embedding
    if os.path.exists('tfidf_vectorizer.pkl'):
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    else:
        # If no saved vectorizer exists, create a new one
        # This is not ideal but handles the case where this is called directly
        print("Warning: No saved vectorizer found. Creating new one.")
        vectorizer = TfidfVectorizer(max_features=100)
        vectorizer.fit(["dummy text for fitting"])
    
    # Transform text to vector
    vector = vectorizer.transform([text]).toarray()[0]
    return vector.tolist()

def query_similar_prs(pr_number, repo_owner, repo_name, collection, num_similar=3):
    """Find PRs similar to the specified PR"""
    try:
        # Get PR files
        pull_request = get_pull_request(pr_number, repo_owner, repo_name)
        
        # If PR doesn't exist, use sample data
        if not pull_request:
            print(f"⚠️ PR #{pr_number} not found. Using sample data.")
            
            # Create a sample file list for testing
            class SampleFile:
                def __init__(self, filename, patch):
                    self.filename = filename
                    self.patch = patch
            
            sample_files = [
                SampleFile("README.md", "@@ -1,5 +1,7 @@\n-# Project\n+# Awesome Project\n+\n This is a sample project.\n+Adding more description."),
                SampleFile("src/main.py", "@@ -10,6 +10,8 @@\n def main():\n-    print('Hello')\n+    print('Hello, World!')\n+    return True")
            ]
            
            filenames = ", ".join([file.filename for file in sample_files])
            print(f"Sample files: {filenames}")
            
            # Create embedding using TF-IDF
            vector = text_embedding(filenames)
            
            # Query collection - get top 3 similar PRs
            results = collection.query(
                query_embeddings=[vector],
                n_results=num_similar + 1  # +1 because current PR might be included
            )
            
            return results, sample_files
            
        # Get files from the PR
        files = pull_request.get_files()
        
        # Extract filenames
        filenames = ", ".join([file.filename for file in files])
        
        # Create embedding using TF-IDF
        vector = text_embedding(filenames)
        
        print(f"Querying for PRs similar to PR #{pr_number} with files: {filenames[:100]}...")
        
        # Query collection - get top 3 similar PRs
        results = collection.query(
            query_embeddings=[vector],
            n_results=num_similar + 1  # +1 because current PR might be included
        )
        
        print(f"Found {len(results['ids'][0])} similar PRs")
        
        return results, list(files)
        
    except Exception as e:
        print(f"Error querying similar PRs: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None

def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    import numpy as np
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2) if norm1 * norm2 != 0 else 0