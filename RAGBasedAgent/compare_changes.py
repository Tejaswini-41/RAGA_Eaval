from github import Github
import os
from dotenv import load_dotenv

def get_file_changes(files):
    """Extract before/after changes from files in a PR"""
    final_changes = ""
    for file in files:
        if file.patch:
            added_lines = [line[1:] for line in file.patch.split('\n') 
                          if line.startswith('+') and not line.startswith('+++')]
            removed_lines = [line[1:] for line in file.patch.split('\n') 
                            if line.startswith('-') and not line.startswith('---')]

            before_change = "\n".join(removed_lines)
            after_change = "\n".join(added_lines)

            final_changes += f"\nFile: {file.filename}\nBefore:\n{before_change}\nAfter:\n{after_change}\n"

    return final_changes

def compare_pr_changes(pr_files, similar_pr_number, repo_owner, repo_name):
    """Compare changes between PR files and a similar PR"""
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
        
        # Get changes for current PR
        current_pr_changes = get_file_changes(pr_files)
        
        # Get similar PR
        similar_pr = repo.get_pull(similar_pr_number)
        similar_files = similar_pr.get_files()
        
        # Get changes for similar PR
        similar_pr_changes = get_file_changes(similar_files)
        
        print("\n🔍 Changes in Current PR:")
        print(current_pr_changes)
        print("\n🔍 Changes in Similar PR:")
        print(similar_pr_changes)
        
        return current_pr_changes, similar_pr_changes
        
    except Exception as e:
        print(f"Error comparing PR changes: {e}")
        return None, None

if __name__ == "__main__":
    # If run directly, execute with sample data
    from query_similar_prs import query_similar_prs
    from store_embeddings import store_embeddings
    from fetch_prs import fetch_pull_requests
    
    # Setup
    repo_owner = 'Tejaswini-41'
    repo_name = 'RAGA_Eaval'
    pr_number = 1  # Example PR number - change as needed
    
    # Get PR data and store embeddings
    changed_files, pull_requests = fetch_pull_requests(repo_owner, repo_name)
    
    if changed_files and pull_requests:
        _, collection = store_embeddings(changed_files, pull_requests)
        
        if collection:
            # Query similar PRs
            query_results, pr_files = query_similar_prs(pr_number, repo_owner, repo_name, collection)
            
            if query_results and pr_files:
                # Get similar PR number from results
                similar_pr_number = query_results["metadatas"][0][0]["pr_number"]
                
                # Compare changes
                compare_pr_changes(pr_files, similar_pr_number, repo_owner, repo_name)