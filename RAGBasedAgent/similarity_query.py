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

def query_similar_prs(pr_number, repo_owner, repo_name, collection):
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
            
            # Query collection
            query_results = collection.query(
                query_embeddings=[vector],
                n_results=1
            )
            
            print("🔍 Found similar pull request (sample data):")
            if query_results and len(query_results["metadatas"]) > 0:
                similar_pr = query_results["metadatas"][0][0]["pr_number"]
                print(f"PR #{similar_pr}")
            
            return query_results, sample_files
            
        # Normal flow if PR exists
        files = pull_request.get_files()
        
        # Extract file names
        filenames = ", ".join([file.filename for file in files])
        print(f"Files in PR #{pull_request.number}: {filenames}")
        
        # Create embedding
        vector = text_embedding(filenames)
        
        # Get all results first
        all_results = collection.get()
        
        # Manually filter out the current PR
        filtered_indices = []
        filtered_distances = []
        
        # Calculate similarity to each PR
        for i, meta in enumerate(all_results["metadatas"]):
            # Skip if this is the current PR
            if meta["pr_number"] == pr_number:
                continue
                
            # Otherwise calculate similarity and add to list
            doc_vector = text_embedding(all_results["documents"][i])
            similarity = calculate_cosine_similarity(vector, doc_vector)
            filtered_indices.append(i)
            filtered_distances.append(similarity)
        
        if not filtered_indices:
            print("⚠️ No other PRs found to compare. Using a random PR instead.")
            # Pick any PR that isn't the current one
            for meta in all_results["metadatas"]:
                if meta["pr_number"] != pr_number:
                    similar_pr = meta["pr_number"]
                    break
        else:
            # Find the most similar PR (highest similarity)
            best_index = filtered_indices[filtered_distances.index(max(filtered_distances))]
            similar_pr = all_results["metadatas"][best_index]["pr_number"]
        
        print(f"🔍 Found similar pull request: PR #{similar_pr}")
        
        # Format results to match expected structure
        query_results = {"metadatas": [[{"pr_number": similar_pr}]]}
        
        return query_results, files
        
    except Exception as e:
        print(f"Error querying similar PRs: {e}")
        return None, None

def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    import numpy as np
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2) if norm1 * norm2 != 0 else 0