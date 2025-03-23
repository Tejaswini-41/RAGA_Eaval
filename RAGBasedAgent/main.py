import os
from dotenv import load_dotenv
from GithubAuth import fetch_pull_requests
from embedding_store import store_embeddings
from similarity_query import query_similar_prs
from change_analyzer import compare_pr_changes
from review_generator import generate_review

def setup_environment():
    """Setup environment and check required variables"""
    load_dotenv()
    github_token = os.getenv("GITHUB_ACCESS_TOKEN")
    
    if not github_token:
        print("⚠️ GitHub token not found! Please set GITHUB_ACCESS_TOKEN in .env file")
        return False
    
    return True

def run_rag_review(repo_owner, repo_name, pr_number):
    """Run the complete RAG-based PR review process"""
    print(f"🚀 Starting RAG-based review for PR #{pr_number} in {repo_owner}/{repo_name}")
    
    # Step 1 & 2: Fetch repository PRs and store changed files
    print("\n📦 Step 1-2: Fetching pull requests data...")
    changed_files, pull_requests = fetch_pull_requests(repo_owner, repo_name)
    
    if not changed_files or not pull_requests:
        print("❌ Failed to fetch pull requests")
        return
    
    # Step 3: Store embeddings
    print("\n📊 Step 3: Creating embeddings and storing in ChromaDB...")
    _, collection = store_embeddings(changed_files, pull_requests)
    
    if not collection:
        print("❌ Failed to store embeddings")
        return
    
    # Step 4: Query similar PRs
    print(f"\n🔍 Step 4: Finding similar PRs to PR #{pr_number}...")
    query_results, pr_files = query_similar_prs(pr_number, repo_owner, repo_name, collection)
    
    if not query_results or not pr_files:
        print("❌ Failed to find similar PRs")
        return
    
    # Get similar PR number from results
    similar_pr_number = query_results["metadatas"][0][0]["pr_number"]
    
    # Step 5: Compare changes
    print(f"\n📈 Step 5: Comparing changes between PR #{pr_number} and similar PR #{similar_pr_number}...")
    current_pr_changes, similar_pr_changes = compare_pr_changes(pr_files, similar_pr_number, repo_owner, repo_name)
    
    if not current_pr_changes or not similar_pr_changes:
        print("❌ Failed to compare PR changes")
        return
    
    # Step 6: Generate AI review
    print("\n🤖 Step 6: Generating AI-based review...")
    review = generate_review(current_pr_changes, similar_pr_changes)
    
    if not review:
        print("❌ Failed to generate review")
        return
    
    print("\n✅ RAG-based review process completed successfully!")
    return review

if __name__ == "__main__":
    # Check environment setup
    if not setup_environment():
        exit(1)
    
    # Repository and PR settings
    repo_owner = 'microsoft'
    repo_name = 'vscode'
    pr_number = 244354 
    
    # Run RAG-based review process
    run_rag_review(repo_owner, repo_name, pr_number)