from github import Github
import os
from dotenv import load_dotenv

def fetch_pull_requests(repo_owner, repo_name, limit=10):
    """Fetch closed pull requests and their changed files"""
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
        print(f"Connected to repository: {repo.full_name}")
        
        # Fetch closed pull requests
        Changed_Files = []
        Pull_Requests = []
        count = 0
        
        for pull_request in repo.get_pulls(state='closed'):
            print(f"Checking Pull Request #{pull_request.number} - {pull_request.title}")
            
            # Get list of changed files
            files = pull_request.get_files()
            sFiles = ""
            
            for file in files:
                print(f"File: {file.filename}")
                sFiles += file.filename + ", "
            
            Changed_Files.append(sFiles.strip(", "))
            Pull_Requests.append(pull_request.number)
            
            count += 1
            if count >= limit:
                break  # Limit to specified number of PRs
        
        print("\n=== List of Changed Files ===")
        print(Changed_Files)
        print("\n=== List of Pull Requests ===")
        print(Pull_Requests)
        
        return Changed_Files, Pull_Requests
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    # If run directly, execute the function
    repo_owner = 'Tejaswini-41'
    repo_name = 'RAGA_Eaval'
    fetch_pull_requests(repo_owner, repo_name)
