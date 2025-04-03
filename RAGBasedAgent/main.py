import os
from dotenv import load_dotenv
from GithubAuth import fetch_pull_requests
from embedding_store import store_embeddings
from similarity_query import query_similar_prs
from change_analyzer import compare_pr_changes
from review_generator import generate_review
from review_evaluator import ReviewEvaluator

def setup_environment():
    """Setup environment and check required variables"""
    load_dotenv()
    github_token = os.getenv("GITHUB_ACCESS_TOKEN")
    
    if not github_token:
        print("⚠️ GitHub token not found! Please set GITHUB_ACCESS_TOKEN in .env file")
        return False
    
    return True

async def run_rag_review(repo_owner, repo_name, pr_number):
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
    
    if similar_pr_number == pr_number:
        print(f"⚠️ Warning: Most similar PR is the same PR (#{pr_number}). Using next best match.")
        # Try to get next best match if available
        if len(query_results["metadatas"][0]) > 1:
            similar_pr_number = query_results["metadatas"][0][1]["pr_number"]
        else:
            # Fallback to first PR in the list that isn't the current PR
            for pr in pull_requests:
                if pr != pr_number:
                    similar_pr_number = pr
                    break
    
    # Step 5: Compare changes
    print(f"\n📈 Step 5: Comparing changes between PR #{pr_number} and similar PR #{similar_pr_number}...")
    current_pr_changes, similar_pr_changes = compare_pr_changes(pr_files, similar_pr_number, repo_owner, repo_name)
    
    if not current_pr_changes or not similar_pr_changes:
        print("❌ Failed to compare PR changes")
        return
    
    # Get list of files modified in current PR
    file_paths = []
    for file in pr_files:
        file_paths.append(file.filename)
    
    current_pr_file = ", ".join(file_paths) if file_paths else None
    
    # Step 6: Evaluate models to select best one (NEW)
    print("\n🧪 Step 6: Evaluating AI models for review quality...")
    evaluator = ReviewEvaluator()
    best_model, model_metrics = await evaluator.evaluate_models(
        current_pr_changes, 
        similar_pr_changes
    )
    
    # Step 7: Generate AI review using best model
    print(f"\n🤖 Step 7: Generating AI-based review using {best_model}...")
    review = generate_review(
        current_pr_changes, 
        similar_pr_changes,
        pr_number=pr_number,
        similar_pr_number=similar_pr_number,
        current_pr_file=current_pr_file,
        model_name=best_model
    )
    
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
    pr_number = 244354  # Use a PR number that exists
    
    # Run RAG-based review process using asyncio
    import asyncio
    review = asyncio.run(run_rag_review(repo_owner, repo_name, pr_number))