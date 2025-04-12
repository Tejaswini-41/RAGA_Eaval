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
    query_results, pr_files = query_similar_prs(pr_number, repo_owner, repo_name, collection, num_similar=3)
    
    if not query_results or not pr_files:
        print("❌ Failed to find similar PRs")
        return
    
    # Get top 3 similar PR numbers from results
    similar_pr_numbers = []
    for i in range(min(3, len(query_results["metadatas"][0]))):
        similar_pr = query_results["metadatas"][0][i]["pr_number"]
        
        # Don't include the same PR
        if similar_pr != pr_number:
            similar_pr_numbers.append(similar_pr)
            
    if not similar_pr_numbers:
        # Fallback to first PR in the list that isn't the current PR
        for pr in pull_requests:
            if pr != pr_number:
                similar_pr_numbers.append(pr)
                if len(similar_pr_numbers) >= 3:
                    break

    # Display the list of similar PRs clearly
    print("\n📋 Top Similar PRs:")
    for i, pr in enumerate(similar_pr_numbers):
        print(f"  {i+1}. PR #{pr}")
    
    # For display purposes, show the most similar PR
    most_similar_pr = similar_pr_numbers[0] if similar_pr_numbers else None
    
    # Step 5: Compare changes
    print(f"\n📈 Step 5: Comparing changes between PR #{pr_number} and {len(similar_pr_numbers)} similar PRs...")
    print(f"Most similar PR: #{most_similar_pr}")
    current_pr_changes, similar_prs_changes = compare_pr_changes(pr_files, similar_pr_numbers, repo_owner, repo_name)
    
    # Add content truncation here to avoid token limits
    max_tokens = 4000
    if current_pr_changes and len(current_pr_changes) > max_tokens:
        from evaluation.metrics import MetricsCalculator
        metrics_calculator = MetricsCalculator()
        original_length = len(current_pr_changes)
        current_pr_changes = metrics_calculator.extract_relevant_pr_content(current_pr_changes, max_tokens)
        print(f"⚠️ Intelligently extracted PR content from {original_length} chars to {len(current_pr_changes)}")

    if similar_prs_changes:
        for i, pr_data in enumerate(similar_prs_changes):
            if len(pr_data['changes']) > (max_tokens//2):
                similar_prs_changes[i]['changes'] = pr_data['changes'][:(max_tokens//2)] + "\n...[content truncated]"

    if not current_pr_changes or not similar_prs_changes:
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
        similar_prs_changes
    )
    
    # Step 7: Generate AI review using best model
    print(f"\n🤖 Step 7: Generating AI-based review using {best_model}...")
    review = generate_review(
        current_pr_changes, 
        similar_prs_changes,
        pr_number=pr_number,
        similar_pr_number=most_similar_pr,  
        current_pr_file=current_pr_file,
        model_name=best_model
    )
    
    if not review:
        print("❌ Failed to generate review")
        return
    
    print("\n✅ RAG-based review process completed successfully!")
    return review

def display_menu():
    """Display the options menu"""
    print("\n" + "="*50)
    print("🚀 RAG-BASED PR REVIEW SYSTEM")
    print("="*50)
    print("\nEnhancement Options:")
    print("0. 🔄 Run standard review (default)")
    print("1. 🔍 Add confidence scores to review suggestions")
    print("2. 📝 Use enhanced prompts for better specificity")
    print("3. 📊 Implement improved DB chunking")
    print("4. 💡 Add interactive feedback system for RAGAS improvement")
    print("5. ❌ Exit")
    print("-"*50)
    choice = input("\nSelect an option (0-5): ")
    return choice

if __name__ == "__main__":
    # Check environment setup
    if not setup_environment():
        exit(1)
    
    while True:
        choice = display_menu()
        
        if choice == "0":
            # Repository and PR settings
            repo_owner = 'microsoft'
            repo_name = 'vscode'
            pr_number = 246149  # Use a PR number that exists
            
            # Run RAG-based review process using asyncio
            import asyncio
            review = asyncio.run(run_rag_review(repo_owner, repo_name, pr_number))
        
        elif choice == "5":
            print("\n👋 Exiting the program. Goodbye!")
            exit(0)
        
        elif choice in ["1", "2", "3", "4"]:
            print("\n🚧 COMING SOON! This feature is under development.")
            input("\nPress Enter to continue...")
        
        else:
            print("\n⚠️ Invalid option. Please try again.")
            input("\nPress Enter to continue...")