from github import Github
import os
from dotenv import load_dotenv

def authenticate_github():
    """Authenticate with GitHub using token from env"""
    # Load environment variables
    load_dotenv()
    
    # Get GitHub access token
    github_access_token = os.getenv("GITHUB_ACCESS_TOKEN")
    if not github_access_token:
        print("GitHub token not found! Make sure it's set in .env file.")
        return None
    
    # Initialize GitHub client
    try:
        g = Github(github_access_token)
        return g
    except Exception as e:
        print(f"GitHub authentication error: {e}")
        return None

def get_pull_request(pr_number, repo_owner, repo_name):
    """Get a specific pull request by number"""
    g = authenticate_github()
    if not g:
        return None
    
    try:
        repo = g.get_repo(f"{repo_owner}/{repo_name}")
        pull_request = repo.get_pull(pr_number)
        return pull_request
    except Exception as e:
        print(f"Error getting pull request #{pr_number}: {e}")
        return None

def fetch_pull_requests(repo_owner, repo_name, limit=20):
    """Fetch closed pull requests and their changed files Close without merging."""
    g = authenticate_github()
    if not g:
        return None, None
    
    try:
        # Connect to repository
        repo = g.get_repo(f"{repo_owner}/{repo_name}")
        print(f"Connected to repository: {repo.full_name}")
        
        # Fetch closed pull requests
        Changed_Files = []
        Pull_Requests = []
        count = 0
        
        # Try to fetch both closed and open PRs
        for state in ['closed', 'open']:
            print(f"Fetching {state} pull requests...")
            for pull_request in repo.get_pulls(state=state):
                print(f"Checking Pull Request #{pull_request.number} - {pull_request.title}")
                
                # Get list of changed files
                files = pull_request.get_files()
                sFiles = ""
                
                for file in files:
                    print(f"File: {file.filename}")
                    sFiles += file.filename + ", "
                
                if sFiles:  # Only add if there are actually files
                    Changed_Files.append(sFiles.strip(", "))
                    Pull_Requests.append(pull_request.number)
                    
                    count += 1
                    if count >= limit:
                        break  # Limit to specified number of PRs
            
            # If we found enough PRs, no need to check other state
            if count >= limit:
                break
                
        # If no PRs found, use sample data for testing
        if not Changed_Files:
            print("\n⚠️ No pull requests found in repository. Using sample data for testing.")
            Changed_Files = [
                "README.md, docs/setup.md", 
                "src/main.py, src/utils.py",
                "tests/test_main.py, src/config.py",
                "src/auth.py, src/models.py",
                "docs/usage.md, src/api/endpoints.py"
            ]
            Pull_Requests = [999, 998, 997, 996, 995]  # Dummy PR numbers
        
        # print("\n=== List of Changed Files ===")
        print(Changed_Files)
        print("\n=== List of Pull Requests ===")
        print(Pull_Requests)
        
        return Changed_Files, Pull_Requests
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# def get_file_changes(files):
#     """Extract before/after changes from files in a PR"""
#     final_changes = ""
#     for file in files:
#         if file.patch:
#             added_lines = [line[1:] for line in file.patch.split('\n') 
#                           if line.startswith('+') and not line.startswith('+++')]
#             removed_lines = [line[1:] for line in file.patch.split('\n') 
#                             if line.startswith('-') and not line.startswith('---')]

#             before_change = "\n".join(removed_lines)
#             after_change = "\n".join(added_lines)

#             final_changes += f"\nFile: {file.filename}\nBefore:\n{before_change}\nAfter:\n{after_change}\n"

#     return final_changes

# def compare_pr_changes(pr_files, similar_pr_number, repo_owner, repo_name):
#     """Compare changes between PR files and a similar PR"""
#     try:
#         # Get similar PR
#         similar_pr = get_pull_request(similar_pr_number, repo_owner, repo_name)
#         if not similar_pr:
#             return None, None
            
#         similar_files = similar_pr.get_files()
        
#         # Get changes for current PR
#         current_pr_changes = get_file_changes(pr_files)
        
#         # Get changes for similar PR
#         similar_pr_changes = get_file_changes(similar_files)
        
#         print("\n🔍 Changes in Current PR:")
#         print(current_pr_changes[:300] + "..." if len(current_pr_changes) > 300 else current_pr_changes)
#         print("\n🔍 Changes in Similar PR:")
#         print(similar_pr_changes[:300] + "..." if len(similar_pr_changes) > 300 else similar_pr_changes)
        
#         return current_pr_changes, similar_pr_changes
        
#     except Exception as e:
#         print(f"Error comparing PR changes: {e}")
#         return None, None

if __name__ == "__main__":
    # If run directly, execute the function
    repo_owner = 'explodinggradients'
    repo_name = 'ragas'
    fetch_pull_requests(repo_owner, repo_name)
