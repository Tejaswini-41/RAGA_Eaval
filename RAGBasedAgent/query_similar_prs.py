from sentence_transformers import SentenceTransformer
from github import Github
import os
from dotenv import load_dotenv

def text_embedding(text, model_name='BAAI/bge-large-en'):
    """Generate embeddings for text using given model"""
    model = SentenceTransformer(model_name)
    return model.encode(text).tolist()

def query_similar_prs(pr_number, repo_owner, repo_name, collection):
    """Find PRs similar to the specified PR"""
    # Load environment variables
    load_dotenv()
    
    # Get GitHub access token
    github_access_token = os.getenv("GITHUB_ACCESS_TOKEN")
    if not github_access_token:
        print("GitHub token not found! Make sure it's set in .env file.")
        return None, None
    
    # Initialize GitHub client
    g = Github(github_access_token)
    
    try:
        # Connect to repository
        repo = g.get_repo(f"{repo_owner}/{repo_name}")
        
        # Get PR files
        pull_request = repo.get_pull(pr_number)
        files = pull_request.get_files()
        
        # Extract file names
        filenames = ", ".join([file.filename for file in files])
        print(f"Files in PR #{pull_request.number}: {filenames}")
        
        # Create embedding
        vector = text_embedding(filenames)
        
        # Query collection
        query_results = collection.query(
            query_embeddings=[vector],
            n_results=1
        )
        
        print("🔍 Query Results:")
        print(query_results)
        
        return query_results, files
        
    except Exception as e:
        print(f"Error querying similar PRs: {e}")
        return None, None

if __name__ == "__main__":
    # If run directly, execute with sample data
    from store_embeddings import store_embeddings
    from fetch_prs import fetch_pull_requests
    
    # Setup
    repo_owner = 'Tejaswini-41'
    repo_name = 'RAGA_Eaval'
    pr_number = 1  # Example PR number - change as needed
    
    # Get PR data
    changed_files, pull_requests = fetch_pull_requests(repo_owner, repo_name)
    
    if changed_files and pull_requests:
        # Store embeddings
        _, collection = store_embeddings(changed_files, pull_requests)
        
        if collection:
            # Query similar PRs
            query_similar_prs(pr_number, repo_owner, repo_name, collection)