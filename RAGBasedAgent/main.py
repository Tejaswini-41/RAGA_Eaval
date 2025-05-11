import os
import datetime
import json
from dotenv import load_dotenv
from GithubAuth import fetch_pull_requests
from embedding_store import store_embeddings
from similarity_query import query_similar_prs
from change_analyzer import compare_pr_changes
from review_generator import generate_review
from review_evaluator import ReviewEvaluator
from prompts.review_prompts import ReviewPrompts
from datetime import datetime  # Update the import statement
from Confidence_Scorer import enhance_review_with_confidence_scores
from chunking_advice import ChunkingAdvisor  # Assuming this is the correct import path
import uuid
import time

# Global constants for repository settings
REPO_OWNER = 'microsoft'
REPO_NAME = 'vscode'
PR_NUMBER = 246149

# REPO_OWNER = 'explodinggradients'
# REPO_NAME = 'ragas'
# PR_NUMBER = 2030

def generate_session_id():
    """Generate a unique session ID"""
    return f"session_{int(time.time())}_{str(uuid.uuid4())[:8]}"

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
        similar_prs_changes if 'similar_prs_changes' in locals() else None,
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
    print("3. 📊 DB chunking Advice")
    print("4. 💡 Add interactive feedback system for RAGAS improvement")
    print("5. 🧪 Test Chunking Strategy")  # New option
    print("6. ❌ Exit")

    print("-"*50)
    choice = input("\nSelect an option (0-6): ")
    return choice

# Update the save_results function to handle markdown files
def save_results(data, prefix, session_id, evaluation_type=None):
    """
    Save results with session tracking and evaluation type
    
    Args:
        data: Data to save
        prefix: Prefix for filename
        session_id: Current session ID
        evaluation_type: Type of evaluation (baseline/enhanced)
    """
    results_dir = "RAG_based_Analysis_2"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Add evaluation type to filename if provided
    eval_suffix = f"_{evaluation_type}" if evaluation_type else ""
    
    # Handle markdown files for enhanced reviews
    if prefix == "confidence_enhanced_review":
        filename = f"{prefix}_{session_id}_{timestamp}{eval_suffix}.md"
        filepath = os.path.join(results_dir, filename)
        
        # Save as formatted markdown
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(data)
    else:
        # Save other data as JSON
        filename = f"{prefix}_{session_id}_{timestamp}{eval_suffix}.json"
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    return filepath

def get_latest_session_file(session_id=None, evaluation_type=None):
    """Get the most recent file for the current session and evaluation type"""
    results_dir = "RAG_based_Analysis_2"
    if not os.path.exists(results_dir):
        return None
        
    files = []
    eval_suffix = f"_{evaluation_type}" if evaluation_type else ""
    
    for f in os.listdir(results_dir):
        if f.endswith(".json") and (not session_id or session_id in f):
            if evaluation_type and eval_suffix in f:
                files.append(os.path.join(results_dir, f))
            elif not evaluation_type:
                files.append(os.path.join(results_dir, f))
    
    return max(files, key=os.path.getctime) if files else None

async def get_chunking_advice(pr_data):
    """Get chunking advice for the current PR"""
    if not pr_data.get("current_pr_changes"):
        print("❌ No PR data available. Please run a review first.")
        return None

    print("\n🔍 Analyzing PR content for chunking recommendations...")
    advisor = ChunkingAdvisor()
    advice = await advisor.get_chunking_advice(pr_data)

    if advice:
        # Save advice to file
        os.makedirs("recommendations", exist_ok=True)
        advice_file = "recommendations/chunking_advice.md"
        with open(advice_file, "w", encoding="utf-8") as f:
            f.write(advice)
        print(f"\n✅ Chunking recommendations saved to {advice_file}")
        return advice
    return None

def load_stored_prompts(session_id=None):
    """Load stored results with session validation"""
    latest_file = get_latest_session_file(session_id)
    if not latest_file:
        return None
        
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
            
        # Validate session and required fields
        if not data.get("session_id"):
            print("❌ No session ID found in stored results")
            return None
            
        if session_id and data["session_id"] != session_id:
            print("❌ Results from different session found. Please run option 0 first")
            return None
            
        required_fields = ["baseline_review", "baseline_metrics"]
        if not all(field in data for field in required_fields):
            print("❌ Stored results file is missing required data")
            return None
            
        return data
        
    except Exception as e:
        print(f"Error loading stored prompts: {e}")
        return None

async def initial_review():
    try:
        # Remove duplicate declarations and use constants
        print(f"\n📦 Fetching pull request data...")
        changed_files, pull_requests = fetch_pull_requests(REPO_OWNER, REPO_NAME)
        
        if not changed_files or not pull_requests:
            raise Exception("Failed to fetch pull request data")
        
        # Create embeddings and get similar PRs
        print("\n🔄 Creating embeddings and finding similar PRs...")
        _, collection = store_embeddings(changed_files, pull_requests)
        query_results, pr_files = query_similar_prs(PR_NUMBER, REPO_OWNER, REPO_NAME, collection, num_similar=3)
        
        # Get similar PR numbers and compare changes (reuse code from run_rag_review)
        similar_pr_numbers = []
        if query_results and "metadatas" in query_results:
            for i, similar_pr in enumerate(query_results["metadatas"][0]):
                pr_num = similar_pr["pr_number"]
                if pr_num != PR_NUMBER:
                    similar_pr_numbers.append(pr_num)
                    if len(similar_pr_numbers) >= 3:
                        break

        # Compare changes
        print("\n📊 Analyzing PR changes...")
        current_pr_changes, similar_prs_changes = compare_pr_changes(pr_files, similar_pr_numbers, REPO_OWNER, REPO_NAME)
        
        if not current_pr_changes or not similar_prs_changes:
            raise Exception("Failed to analyze PR changes")

        # Initialize ReviewEvaluator and evaluate models
        evaluator = ReviewEvaluator()
        best_model, model_metrics = await evaluator.evaluate_models(
            current_pr_changes, 
            similar_prs_changes
        )

        # Generate review using best model directly
        print(f"\n🤖 Generating review with {best_model}...")
        current_pr_file = ", ".join(f.filename for f in pr_files) if pr_files else None
        baseline_review = generate_review(
            current_pr_changes,
            similar_prs_changes,
            pr_number=PR_NUMBER,
            similar_pr_number=similar_pr_numbers[0] if similar_pr_numbers else None,
            current_pr_file=current_pr_file,
            model_name=best_model
        )

        # Get current prompts and generate enhanced prompt
        current_prompt, current_system = ReviewPrompts.get_current_prompt()
        print("\n🔄 Generating enhanced system prompt...")
        new_system_prompt = await evaluator.generate_enhanced_prompt(
            current_metrics=model_metrics[best_model],
            current_prompt=current_system
        )
        
        # Store results - ADD THESE NEW FIELDS
        results = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "original_system_prompt": current_system,
            "enhanced_system_prompt": new_system_prompt,
            "baseline_metrics": model_metrics,
            "best_model": best_model,
            "baseline_review": baseline_review,
            "session_id": session_id,
            # Add these new fields
            "current_pr_changes": current_pr_changes,
            "similar_prs_changes": similar_prs_changes,
            "pr_files": [f.filename for f in pr_files] if pr_files else []
        }
        
        # Store baseline results
        file_path = save_results(results, "prompt_enhancement", session_id, "baseline")
        print(f"\n💾 Baseline results saved to {file_path}")
        return results
        
    except Exception as e:
        print(f"❌ Error during initial review: {e}")
        return None

# async def test_stored_prompt():
    try:
        stored_results = load_stored_prompts(session_id)
        if not stored_results:
            raise Exception("Missing stored results")

        # Get required data
        current_pr_changes = stored_results.get("current_pr_changes")
        similar_prs_changes = stored_results.get("similar_prs_changes")
        enhanced_prompt = stored_results.get("enhanced_system_prompt")
        pr_files = stored_results.get("pr_files", [])
        
        if not all([current_pr_changes, similar_prs_changes, enhanced_prompt]):
            raise Exception("Missing required PR data")

        print("\n📝 Using Enhanced System Prompt...")
        ReviewPrompts.update_system_prompt(enhanced_prompt)
        
        # Initialize evaluator
        evaluator = ReviewEvaluator()
        
        # Generate reviews with all models using enhanced prompt
        print("\n🤖 Generating Reviews with Enhanced Prompt:")
        
        # Use the same models as in initial review
        models = ["gemini", "llama", "alibaba", "deepseek"]
        enhanced_reviews = {}
        
        print("\n🔄 Generating reviews with all models...")
        for model in models:
            print(f"\n📋 Model: {model.upper()}")
            
            # Generate review using enhanced prompt
            review = generate_review(
                current_pr_changes,
                similar_prs_changes,
                pr_number=PR_NUMBER,
                current_pr_file=", ".join(pr_files),
                model_name=model
            )
            
            if review:
                print(f"✅ Review generated successfully")
                enhanced_reviews[model] = review
                if model == "gemini":  # Use gemini as reference
                    reference_review = review

        # Calculate metrics for all models against reference
        print("\n📊 Calculating RAGAS Metrics...")
        best_score = 0
        best_model = None
        enhanced_metrics = {}
        
        for model in enhanced_reviews:
            if model != "gemini":
                metrics = await evaluator._calculate_metrics(
                    reference_review,
                    enhanced_reviews[model]
                )
                enhanced_metrics[model] = metrics
                
                if metrics["Overall"] > best_score:
                    best_score = metrics["Overall"]
                    best_model = model

        # Show detailed metrics comparison
        print("\n📈 RAGAS Metrics Comparison:")
        print("=" * 90)
        print(f"{'Model':<10} | {'Metric':<15} | {'Baseline':>8} | {'Enhanced':>8} | {'Change':>8} | {'% Change':>8} |")
        print("=" * 90)

        baseline_metrics = stored_results["baseline_metrics"]
        
        # Rest of the comparison and saving logic remains the same
        # ...existing code...

    except Exception as e:
        print(f"❌ Error during prompt testing: {e}")

async def test_stored_prompt():
    try:
        stored_results = load_stored_prompts(session_id)
        if not stored_results:
            raise Exception("Missing stored results")

        # Get required data
        current_pr_changes = stored_results.get("current_pr_changes")
        similar_prs_changes = stored_results.get("similar_prs_changes")
        enhanced_prompt = stored_results.get("enhanced_system_prompt")
        pr_files = stored_results.get("pr_files", [])

        if not all([current_pr_changes, similar_prs_changes, enhanced_prompt]):
            raise Exception("Missing required PR data")

        print("\n📝 Using Enhanced System Prompt...")
        ReviewPrompts.update_system_prompt(enhanced_prompt)
        print("=" * 80)
        print(enhanced_prompt)
        print("=" * 80)

        # Initialize evaluator
        evaluator = ReviewEvaluator()

        # Generate reviews with all models using enhanced prompt
        print("\n🤖 Generating Reviews with Enhanced Prompt:")
        models = ["gemini", "llama", "alibaba", "deepseek"]
        enhanced_reviews = {}

        print("\n🔄 Generating reviews with all models...")
        for model in models:
            print(f"\n📋 Model: {model.upper()}")
            review = generate_review(
                current_pr_changes,
                similar_prs_changes,
                pr_number=PR_NUMBER,
                current_pr_file=", ".join(pr_files),
                model_name=model
            )
            if review:
                print(f"✅ Review generated successfully")
                enhanced_reviews[model] = review

        # Calculate metrics for all models
        print("\n📊 Calculating RAGAS Metrics...")
        best_score = 0
        best_model = None
        enhanced_metrics = {}

        for model in enhanced_reviews:
            metrics = await evaluator._calculate_metrics(
                enhanced_reviews["gemini"],  # Use "gemini" as reference
                enhanced_reviews[model]
            )
            enhanced_metrics[model] = metrics
            if metrics["Overall"] > best_score:
                best_score = metrics["Overall"]
                best_model = model

        # Save comprehensive results
        enhanced_results = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "session_id": stored_results["session_id"],
            "prompts": {
                "original": stored_results.get("original_system_prompt"),
                "enhanced": enhanced_prompt
            },
            "reviews": enhanced_reviews,
            "metrics_comparison": {
                "baseline": stored_results["baseline_metrics"],
                "enhanced": enhanced_metrics
            },
            "best_model": {
                "baseline": stored_results["best_model"],
                "enhanced": best_model
            }
        }

        enhanced_file = save_results(
            enhanced_results,
            "prompt_enhancement",
            stored_results["session_id"],
            "enhanced"
        )
        print(f"\n💾 Enhanced results saved to: {enhanced_file}")

        # Display comparison of baseline and enhanced metrics
        print("\n📈 RAGAS Metrics Comparison:")
        print("=" * 90)
        print(f"{'Model':<10} | {'Metric':<15} | {'Baseline':>8} | {'Enhanced':>8} | {'Change':>8} | {'% Change':>8} |")
        print("=" * 90)

        baseline_metrics = stored_results["baseline_metrics"]

        for model in enhanced_metrics:
            if model in baseline_metrics:
                for metric in enhanced_metrics[model]:
                    baseline = baseline_metrics[model].get(metric, 0)
                    enhanced = enhanced_metrics[model].get(metric, 0)
                    change = enhanced - baseline
                    pct_change = (change / baseline * 100) if baseline else 0

                    print(f"{model:<10} | {metric:<15} | {baseline:>8.3f} | {enhanced:>8.3f} | {change:>+8.3f} | {pct_change:>7.1f}% |")

    except Exception as e:
        print(f"❌ Error during enhanced prompt evaluation: {e}")

if __name__ == "__main__":
    # Check environment setup
    if not setup_environment():
        exit(1)
    
    session_id = None
    
    while True:
        choice = display_menu()
        
        if choice == "0":
            # Generate new session ID when starting fresh
            session_id = generate_session_id()
            print(f"\n🔑 Starting new session: {session_id}")
            
            print("\n🔍 Running initial review and prompt generation...")
            
            import asyncio
            
            # Run initial review
            results = asyncio.run(initial_review())
            input("\nPress Enter to continue...")
        
        elif choice == "1":
            if not session_id:
                print("❌ No active session found. Please run option 0 first")
                input("\nPress Enter to continue...")
                continue
                
            stored_results = load_stored_prompts(session_id)
            if not stored_results:
                print("❌ Please run option 0 first to generate baseline review")
                input("\nPress Enter to continue...")
                continue
                
            print("\n🎯 Adding confidence scores to review...")
            
            # Load stored results from option 0
            stored_results = load_stored_prompts(session_id)
            if not stored_results:
                print("❌ No previously generated review found. Please run option 0 first.")
                input("\nPress Enter to continue...")
                continue
            
            import asyncio
            
            async def add_confidence_scores():
                try:
                     # Use the stored baseline review instead of generating a new one
                    base_review = stored_results["baseline_review"]
                    current_pr_changes = stored_results["current_pr_changes"]
                    similar_prs_changes = stored_results["similar_prs_changes"]
                    pr_files = stored_results.get("pr_files", [])
                    query_results = stored_results.get("query_results", None)

                    # Add confidence scores
                    print("\n🎯 Adding confidence scores to review...")
                    enhanced_review = enhance_review_with_confidence_scores(
                        current_pr_changes,
                        similar_prs_changes,
                        base_review,
                        pr_files,
                        query_results
                    )

                    # Inside the `add_confidence_scores` function in option 1
                    pr_files = stored_results.get("pr_files", [])

                    # Ensure `pr_files` is handled as a list of strings
                    if isinstance(pr_files, list) and all(isinstance(f, str) for f in pr_files):
                        print("\n📋 Loaded PR files as strings.")
                    else:
                        print("\n⚠️ PR files are not in the expected format.")
                    
                    # Format the review as markdown
                    markdown_review = f"""# Pull Request Review with Confidence Scores

{enhanced_review}
"""
                    
                    # Save enhanced review as markdown
                    file_path = save_results(markdown_review, "confidence_enhanced_review", session_id)
                    
                    print(f"\n✅ Enhanced review with confidence scores saved to {file_path}")
                    
                    # Display summary of changes
                    print("\n📋 Summary of Confidence Assessment:")
                    print("="*50)
                    if "Confidence Assessment" in enhanced_review:
                        confidence_section = enhanced_review.split("## 🎯 Confidence Assessment")[1].split("##")[0]
                        print(confidence_section)
                    
                except Exception as e:
                    print(f"❌ Error adding confidence scores: {e}")
            
            # Run confidence scoring
            asyncio.run(add_confidence_scores())
            input("\nPress Enter to continue...")
        
        elif choice == "2":
            if not session_id:
                print("❌ No active session found. Please run option 0 first")
                input("\nPress Enter to continue...")
                continue
                
            stored_results = load_stored_prompts(session_id)
            if not stored_results:
                print("❌ Please run option 0 first to generate baseline review")
                input("\nPress Enter to continue...")
                continue

            print("\n🔍 Testing enhanced prompt from previous analysis...")
            
            async def test_stored_prompt():
                try:
                    stored_results = load_stored_prompts(session_id)
                    if not stored_results:
                        raise Exception("Missing stored results")

                    # Get required data
                    current_pr_changes = stored_results.get("current_pr_changes")
                    similar_prs_changes = stored_results.get("similar_prs_changes")
                    enhanced_prompt = stored_results.get("enhanced_system_prompt")
                    pr_files = stored_results.get("pr_files", [])
                    
                    if not all([current_pr_changes, similar_prs_changes, enhanced_prompt]):
                        raise Exception("Missing required PR data")

                    print("\n📝 Using Enhanced System Prompt...")
                    ReviewPrompts.update_system_prompt(enhanced_prompt)
                    print("=" * 80)
                    print(enhanced_prompt)
                    print("=" * 80)

                    # Verify prompt update
                    ReviewPrompts.update_system_prompt(enhanced_prompt)
                    current_prompt, current_system = ReviewPrompts.get_current_prompt()
                    if current_system != enhanced_prompt:
                        print("⚠ Warning: System prompt may not have updated correctly")
                    
                    # Initialize evaluator
                    evaluator = ReviewEvaluator()
                    
                    # Generate reviews with all models using enhanced prompt
                    print("\n🤖 Generating Reviews with Enhanced Prompt:")
                    
                    # Use the same models as in initial review
                    models = ["gemini", "llama", "alibaba", "deepseek"]
                    enhanced_reviews = {}
                    
                    print("\n🔄 Generating reviews with all models...")
                    for model in models:
                        print(f"\n📋 Model: {model.upper()}")
                        
                        # Generate review using enhanced prompt
                        review = generate_review(
                            current_pr_changes,
                            similar_prs_changes,
                            pr_number=PR_NUMBER,
                            current_pr_file=", ".join(pr_files),
                            model_name=model
                        )
                        
                        if review:
                            print(f"✅ Review generated successfully")
                            enhanced_reviews[model] = review
                            if model == "gemini":  # Use gemini as reference
                                reference_review = review

                    # Calculate metrics for all models against reference
                    print("\n📊 Calculating RAGAS Metrics...")
                    best_score = 0
                    best_model = None
                    enhanced_metrics = {}
                    
                    for model in enhanced_reviews:
                        if model != "gemini":
                            metrics = await evaluator._calculate_metrics(
                                reference_review,
                                enhanced_reviews[model]
                            )
                            enhanced_metrics[model] = metrics
                            
                            if metrics["Overall"] > best_score:
                                best_score = metrics["Overall"]
                                best_model = model

                    # Show detailed metrics comparison
                    print("\n📈 RAGAS Metrics Comparison:")
                    print("=" * 90)
                    print(f"{'Model':<10} | {'Metric':<15} | {'Baseline':>8} | {'Enhanced':>8} | {'Change':>8} | {'% Change':>8} |")
                    print("=" * 90)

                    baseline_metrics = stored_results["baseline_metrics"]
                    
                    for model in enhanced_metrics:
                        if model in baseline_metrics:
                            print(f"\n📊 {model.upper()}")
                            print("-" * 90)
                
                            for metric in enhanced_metrics[model]:
                                if metric != "Overall":
                                    baseline = baseline_metrics[model][metric]
                                    enhanced = enhanced_metrics[model][metric]
                                    change = enhanced - baseline
                                    pct_change = (change / baseline * 100) if baseline else 0
                        
                                    print(f"{'':<10} | {metric:<15} | {baseline:>8.3f} | {enhanced:>8.3f} | {change:>+8.3f} | {pct_change:>7.1f}% |")

                    # Save comprehensive results
                    enhanced_results = {
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "session_id": stored_results["session_id"],
                        "prompts": {
                            "original": stored_results["original_system_prompt"],
                            "enhanced": enhanced_prompt
                        },
                        "reviews": enhanced_reviews,
                        "metrics_comparison": {
                            "baseline": baseline_metrics,
                             "enhanced": enhanced_metrics
                        },
                        "best_model": {
                            "baseline": stored_results["best_model"],
                            "enhanced": best_model
                        }
                    }

                    enhanced_file = save_results(
                        enhanced_results, 
                        "prompt_enhancement", 
                        stored_results["session_id"],
                        "enhanced"
                    )
        
                    print(f"\n💾 Enhanced results saved to: {enhanced_file}")
        
                except Exception as e:
                    print(f"❌ Error during enhanced prompt evaluation: {e}")
                    
                # except Exception as e:
                #     print(f"❌ Error during prompt testing: {e}")
            
            # Run the prompt testing
            asyncio.run(test_stored_prompt())
            input("\nPress Enter to continue...")

        elif choice == "3":
            if not session_id:
                print("❌ No active session found. Please run option 0 first")
                input("\nPress Enter to continue...")
                continue
                
            # Load stored results from previous run
            stored_results = load_stored_prompts(session_id)
            if not stored_results:
                print("❌ Please run option 0 first to generate baseline review")
                input("\nPress Enter to continue...")
                continue
            
            print("\n📊 Analyzing PR for chunking advice...")
            
            # Prepare PR data for chunking analysis
            pr_data = {
                "current_pr_changes": stored_results.get("baseline_review", ""),
                "pr_files": stored_results.get("pr_files", []),  # These are now string file paths
                "metrics": stored_results.get("baseline_metrics", {})
            }
            
            # Get chunking advice
            import asyncio
            advice = asyncio.run(get_chunking_advice(pr_data))
            
            if advice:
                print("\n" + advice)
            else:
                print("\n⚠ Could not generate chunking advice")
            
            input("\nPress Enter to continue...")
        
        elif choice == "4":
            if not session_id:
                print("❌ No active session found. Please run option 0 first")
                input("\nPress Enter to continue...")
                continue
                
            # Load stored results from previous run
            stored_results = load_stored_prompts(session_id)
            if not stored_results:
                print("❌ Please run option 0 first to generate baseline review")
                input("\nPress Enter to continue...")
                continue
            
            print("\n🧮 Evaluating Chunking Strategy with RAGAS Metrics...")
            
            # Import needed modules
            from chunk_tester import ChunkingStrategyTester
            from review_evaluator import ReviewEvaluator
            import asyncio
            
            async def test_chunking_with_ragas():
                try:
                    # Prepare PR data for chunking tests - reuse stored data
                    pr_data = {
                        "current_pr_changes": stored_results.get("current_pr_changes", ""),
                        "similar_prs_changes": stored_results.get("similar_prs_changes", []),
                        "pr_files": stored_results.get("pr_files", [])
                    }
                    
                    # Get baseline review
                    baseline_review = stored_results.get("baseline_review", "")
                    if not baseline_review:
                        print("❌ No baseline review found in stored results")
                        return
                    
                    # Initialize chunking tester and evaluator
                    chunking_tester = ChunkingStrategyTester()
                    evaluator = ReviewEvaluator()
                    
                    # Generate chunked review using existing code from chunk_tester
                    print("\n📄 Generating chunked review...")
                    chunked_review = await chunking_tester._generate_chunked_review(pr_data)
                    
                    if not chunked_review:
                        print("❌ Failed to generate chunked review")
                        return
                    
                    # Use reference for metrics calculation
                    reference_review = stored_results.get("reference_review", baseline_review)
                    
                    print("\n📊 Calculating RAGAS metrics...")
                    
                    # Calculate metrics manually using only reliable metrics from metrics.py
                    baseline_metrics = {}
                    chunked_metrics = {}
                    
                    # Use metrics calculator directly to avoid issues with RAGAS
                    metrics_calculator = evaluator.metrics_calculator
                    
                    # Calculate only reliable metrics directly from metrics.py
                    baseline_metrics["Relevance"] = metrics_calculator.compute_relevance(reference_review, baseline_review)
                    baseline_metrics["Accuracy"] = metrics_calculator.compute_accuracy(baseline_review)
                    baseline_metrics["Groundedness"] = metrics_calculator.compute_groundedness(reference_review, baseline_review)
                    baseline_metrics["Completeness"] = metrics_calculator.compute_completeness(reference_review, baseline_review)
                    baseline_metrics["ContextualPrecision"] = metrics_calculator.compute_contextual_precision(reference_review, baseline_review)
                    baseline_metrics["AnswerRelevance"] = metrics_calculator.compute_answer_relevance(reference_review, baseline_review)
                    
                    # Same for chunked review
                    chunked_metrics["Relevance"] = metrics_calculator.compute_relevance(reference_review, chunked_review)
                    chunked_metrics["Accuracy"] = metrics_calculator.compute_accuracy(chunked_review)
                    chunked_metrics["Groundedness"] = metrics_calculator.compute_groundedness(reference_review, chunked_review)
                    chunked_metrics["Completeness"] = metrics_calculator.compute_completeness(reference_review, chunked_review)
                    chunked_metrics["ContextualPrecision"] = metrics_calculator.compute_contextual_precision(reference_review, chunked_review)
                    chunked_metrics["AnswerRelevance"] = metrics_calculator.compute_answer_relevance(reference_review, chunked_review)
                    
                    # Calculate overall scores
                    weights = evaluator.weights
                    
                    # Only use available metrics for weighted score
                    available_weights = {k: v for k, v in weights.items() if k in baseline_metrics}
                    weight_sum = sum(available_weights.values())
                    
                    # Calculate overall scores
                    baseline_overall = sum(baseline_metrics[m] * (weights[m]/weight_sum) 
                                          for m in baseline_metrics)
                    chunked_overall = sum(chunked_metrics[m] * (weights[m]/weight_sum)
                                         for m in chunked_metrics)
                    
                    baseline_metrics["Overall"] = round(baseline_overall, 3)
                    chunked_metrics["Overall"] = round(chunked_overall, 3)
                    
                    # Save results for future reference
                    metrics_comparison = {
                        "session_id": session_id,
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "baseline_metrics": baseline_metrics,
                        "chunked_metrics": chunked_metrics
                    }
                    
                    os.makedirs("ragas_results", exist_ok=True)
                    metrics_file = f"ragas_results/chunking_metrics_{session_id}.json"
                    with open(metrics_file, "w", encoding="utf-8") as f:
                        json.dump(metrics_comparison, f, indent=2)
                    
                    # Display metrics comparison in a formatted table
                    print("\n📈 RAGAS Metrics Comparison (Default vs. Chunked):")
                    print("=" * 70)
                    print(f"{'Metric':<18} | {'Default':>8} | {'Chunked':>8} | {'Change':>8} | {'%':>7} |")
                    print("=" * 70)
                    
                    # Track changes for analysis
                    improvements = []
                    declines = []
                    
                    for metric in baseline_metrics:
                        if metric in chunked_metrics:
                            baseline_val = baseline_metrics[metric]
                            chunked_val = chunked_metrics[metric]
                            change = chunked_val - baseline_val
                            
                            # Avoid division by zero
                            if baseline_val > 0:
                                pct_change = (change / baseline_val) * 100
                            else:
                                pct_change = 0 if change == 0 else 100
                                
                            print(f"{metric:<18} | {baseline_val:>8.3f} | {chunked_val:>8.3f} | {change:>+8.3f} | {pct_change:>+7.1f}% |")
                            
                            if metric != "Overall":  # Don't include Overall in improvements/declines
                                if change > 0:
                                    improvements.append((metric, pct_change))
                                elif change < 0:
                                    declines.append((metric, pct_change))
                    
                    # Calculate overall change
                    overall_change = chunked_metrics["Overall"] - baseline_metrics["Overall"] 
                    overall_pct = (overall_change / max(0.001, baseline_metrics["Overall"])) * 100
                    
                    print("=" * 70)
                    print(f"{'Overall':<18} | {baseline_metrics['Overall']:>8.3f} | {chunked_metrics['Overall']:>8.3f} | {overall_change:>+8.3f} | {overall_pct:>+7.1f}% |")
                    
                    # Show analysis based on results
                    if overall_pct >= 0:
                        print(f"\n✅ Chunking strategy IMPROVED RAGAS scores by {overall_pct:.1f}%")
                    else:
                        print(f"\n⚠ The chunking strategy did NOT improve RAGAS scores ({overall_pct:.1f}%).")
                    
                    if improvements:
                        print("\nImprovements in:")
                        for metric, pct in sorted(improvements, key=lambda x: x[1], reverse=True):
                            print(f"  • {metric}: {pct:+.1f}%")
                            
                    if declines:
                        print("\nDeclines in:")
                        for metric, pct in sorted(declines, key=lambda x: x[1]):
                            print(f"  • {metric}: {pct:+.1f}%")
                    
                    print(f"\n📁 Detailed metrics saved to {metrics_file}")
                    
                    # Ask if user wants to use chunked strategy for future operations
                    use_chunked = input("\nDo you want to use the chunked strategy for future operations? (y/n): ").strip().lower()
                    if use_chunked == 'y':
                        stored_results["baseline_review"] = chunked_review
                        stored_results["chunked_metrics"] = chunked_metrics
                        stored_results["chunking_applied"] = True
                        
                        # Save updated stored results
                        save_results(stored_results, "stored_prompts", session_id)
                        print("\n✅ Chunking strategy set as the new baseline!")
                    
                except Exception as e:
                    print(f"\n❌ Error during chunking strategy evaluation: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Run the test asynchronously
            asyncio.run(test_chunking_with_ragas())
            input("\nPress Enter to continue...")
        
        elif choice == "5":
            print("\n📈 Analyzing system performance and generating improvement suggestions...")
            
            if not session_id:
                print("❌ No active session found. Please run option 0 first")
                input("\nPress Enter to continue...")
                continue
            
            import asyncio
            report = asyncio.run(analyze_improvements(session_id))
            
            if report:
                print("\n✅ Analysis complete!")
                print("\nFirst 500 characters of report:")
                print("-" * 50)
                print(report[:500] + "...\n")
                print("-" * 50)
                print("\nFull report saved to recommendations directory.")
            else:
                print("❌ Failed to generate improvement analysis")
            
            input("\nPress Enter to continue...")

        elif choice == "6": 
            print("\n👋 Exiting the program. Goodbye!")
            exit(0)
        
        else:
            print("\n⚠ Invalid option. Please try again.")
            input("\nPress Enter to continue...")