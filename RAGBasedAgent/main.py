import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime 
from dotenv import load_dotenv
import glob
from GithubAuth import fetch_pull_requests
from embedding_store import store_embeddings
from similarity_query import query_similar_prs
from change_analyzer import compare_pr_changes
from review_generator import generate_review
from review_evaluator import ReviewEvaluator
from prompts.review_prompts import ReviewPrompts
from improvement_analyzer import analyze_improvements
from Confidence_Scorer import enhance_review_with_confidence_scores
from models.model_factory import ModelFactory
from chunking_advice import ChunkingAdvisor  
from chunking import HybridSemanticChunker
from chunked_review_generator import ChunkedReviewGenerator

# Make sure to add these paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Global constants for repository settings
# REPO_OWNER = 'microsoft'
# REPO_NAME = 'vscode'
# PR_NUMBER = 246149

REPO_OWNER = 'explodinggradients'
REPO_NAME = 'ragas'
PR_NUMBER = 2030

def generate_session_id():
    """Generate a unique session ID"""
    timestamp = int(time.time())
    random_id = uuid.uuid4().hex[:8]  # Use hex representation which is alphanumeric
    return f"session_{timestamp}_{random_id}"

def setup_environment():
    """Setup environment and check required variables"""
    load_dotenv()
    github_token = os.getenv("GITHUB_ACCESS_TOKEN")
    
    if not github_token:
        print("⚠️ GitHub token not found! Please set GITHUB_ACCESS_TOKEN in .env file")
        return False
    
    return True

async def custom_pr_review_generator():
    """
    Generate custom PR review with user-selectable components:
    - Chunking strategy
    - Embedding method
    - Prompt type
    - Model
    """
    # Import necessary modules
    import asyncio
    import uuid
    import os
    import json
    import nltk
    from datetime import datetime
    from models.model_factory import ModelFactory
    from chunked_review_generator import ChunkedReviewGenerator
    from review_generator import generate_review
    from prompts.review_prompts import ReviewPrompts
    from review_evaluator import ReviewEvaluator
    from evaluation.metrics import MetricsCalculator

    # Check for session data
    stored_results = load_stored_prompts(session_id)
    if not stored_results:
        print("❌ Please run option 0 first to generate baseline review")
        input("\nPress Enter to continue...")
        return False
    
    print(f"\n🛠️ Custom PR Review Generator")
    
    # Print session information
    print(f"\n📊 Session {stored_results.get('session_id', 'Unknown')} data loaded:")
    print(f"  - Timestamp: {stored_results.get('timestamp', 'Unknown')}")
    print(f"  - PR Files: {len(stored_results.get('pr_files', []))} files")
    print(f"  - Similar PRs: {len(stored_results.get('similar_prs_changes', []))} PRs")
    
    # Get PR data from stored results
    current_pr_changes = stored_results.get("current_pr_changes")
    similar_prs_changes = stored_results.get("similar_prs_changes", [])
    pr_files = stored_results.get("pr_files", [])
    pr_number = PR_NUMBER  # Using global constant

    if not current_pr_changes or not similar_prs_changes:
        print("❌ Missing PR data in session. Please run option 0 first.")
        input("\nPress Enter to continue...")
        return False
    
    # Step 1: Choose Chunking Strategy
    print("\n🧩 Step 1: Choose Chunking Strategy")
    print("-" * 50)
    print("1. Hybrid Semantic (balances structure with size)")
    print("2. Pure Semantic (preserves natural code boundaries)")
    print("3. Fixed Size (consistent chunk sizes)")
    print("4. Hierarchical (maintains parent-child relationships)")
    print("5. No chunking (process entire PR at once)")
    
    chunking_choice = input("\nSelect chunking strategy (1-5): ")
    chunking_strategy = None
    use_chunking = True
    
    if chunking_choice == "1":
        chunking_strategy = "hybrid"
    elif chunking_choice == "2":
        chunking_strategy = "semantic"
    elif chunking_choice == "3":
        chunking_strategy = "fixed"
    elif chunking_choice == "4":
        chunking_strategy = "hierarchical"
    elif chunking_choice == "5":
        use_chunking = False
    else:
        print("⚠️ Invalid choice, defaulting to hybrid chunking")
        chunking_strategy = "hybrid"
    
    # Step 2: Choose Embedding Method
    print("\n📊 Step 2: Choose Embedding Method")
    print("-" * 50)
    print("1. tfidf: TF-IDF (Term Frequency-Inverse Document Frequency)")
    print("2. sentence_transformer: Sentence Transformer (all-MiniLM-L6-v2)")
    print("3. codebert: CodeBERT (microsoft/codebert-base)")
    print("4. hybrid: Hybrid (TF-IDF + Sentence Transformer)")
    print("5. word2vec: Word2Vec (Trained on-the-fly)")
    
    embedding_choice = input("\nSelect embedding method (1-5): ")
    embedding_method = None
    
    if embedding_choice == "1":
        embedding_method = "tfidf"
    elif embedding_choice == "2":
        embedding_method = "sentence_transformer"
    elif embedding_choice == "3":
        embedding_method = "codebert"
    elif embedding_choice == "4":
        embedding_method = "hybrid"
    elif embedding_choice == "5":
        embedding_method = "word2vec"
    else:
        print("⚠️ Invalid choice, defaulting to tfidf embedding")
        embedding_method = "tfidf"
    
    # Step 3: Choose Prompt Type
    print("\n📝 Step 3: Choose Prompt Type")
    print("-" * 50)
    print("1. Default system prompt")
    print("2. Enhanced system prompt (from option 4)")
    print("3. Custom system prompt (enter your own)")
    
    prompt_choice = input("\nSelect prompt type (1-3): ")
    system_prompt = None
    
    if prompt_choice == "1":
        current_prompt, system_prompt = ReviewPrompts.get_current_prompt()
        print("✅ Using default system prompt")
    elif prompt_choice == "2":
        enhanced_prompt = stored_results.get("enhanced_system_prompt")
        if enhanced_prompt:
            system_prompt = enhanced_prompt
            print("✅ Using enhanced system prompt")
        else:
            current_prompt, system_prompt = ReviewPrompts.get_current_prompt()
            print("⚠️ No enhanced prompt found, using default")
    elif prompt_choice == "3":
        print("\nEnter your custom system prompt (finish with Enter + Ctrl+Z on Windows):")
        custom_lines = []
        try:
            while True:
                line = input()
                if not line.strip():  # Empty line to terminate input
                    break
                custom_lines.append(line)
        except EOFError:
            pass
        
        system_prompt = "\n".join(custom_lines)
        if system_prompt:
            print("✅ Custom prompt received")
        else:
            current_prompt, system_prompt = ReviewPrompts.get_current_prompt()
            print("⚠️ No prompt entered, using default")
    else:
        current_prompt, system_prompt = ReviewPrompts.get_current_prompt()
        print("⚠️ Invalid choice, using default system prompt")
    
    # Step 4: Choose Model
    print("\n🤖 Step 4: Choose Model")
    print("-" * 50)
    
    # Get available models from model factory
    model_factory = ModelFactory()
    available_models = model_factory.get_model_names()
    
    # Display available models
    for i, model_name in enumerate(available_models, 1):
        print(f"{i}. {model_name}")
    
    model_choice = input(f"\nSelect model (1-{len(available_models)}): ")
    
    try:
        model_index = int(model_choice) - 1
        if 0 <= model_index < len(available_models):
            model_name = available_models[model_index]
        else:
            model_name = "gemini"  # Default to gemini if invalid choice
            print("⚠️ Invalid choice, defaulting to gemini")
    except ValueError:
        model_name = "gemini"  # Default to gemini if invalid choice
        print("⚠️ Invalid choice, defaulting to gemini")
    
    # Display configuration summary
    print(f"\n🚀 Generating custom PR review with:")
    print(f"- Chunking: {chunking_strategy if use_chunking else 'none'}")
    print(f"- Embedding: {embedding_method}")
    print(f"- Model: {model_name}")
    print(f"- Prompt: {'Enhanced' if prompt_choice == '2' else 'Custom' if prompt_choice == '3' else 'Default'}")
    print("-" * 50)
    
    # Import embedding factory and configure embedding method
    try:
        # Try to import EmbeddingFactory
        try:
            from embeddings.embedding_factory import EmbeddingFactory
            
            # Setup the embedding factory with selected method
            print(f"⚙️ Checking NLTK punkt tokenizer...")
            try:
                nltk.data.find('tokenizers/punkt')
                print(f"✅ NLTK punkt tokenizer found")
            except LookupError:
                print(f"⚠️ Downloading NLTK punkt tokenizer...")
                nltk.download('punkt')
            
            print(f"✅ Using {embedding_method} embedding method")
            embedding_factory = EmbeddingFactory()
            embedder = embedding_factory.get_embedder(embedding_method)
        except ImportError:
            # If EmbeddingFactory not available, continue with default embedding
            print("⚠️ EmbeddingFactory not available, using default embedding")
    except Exception as e:
        print(f"❌ Error configuring embedding: {e}")
    
    # Apply selected configuration to generate review
    if use_chunking:
        # Use chunking strategy with selected configuration
        try:
            # Create chunking collection name with unique ID
            collection_name = f"{chunking_strategy}_chunks_{uuid.uuid4().hex[:8]}"
            print(f"\n🧩 Applying {chunking_strategy} chunking strategy...")
            
            # Create ChunkedReviewGenerator with selected chunking strategy
            generator = ChunkedReviewGenerator(
                chunk_collection_name=collection_name,
                chunking_strategy=chunking_strategy
            )
            
            # Process PR with chunking and selected model
            print(f"\n🔄 Processing PR with {chunking_strategy} chunking...")
            result = await generator.process_pr_with_chunking(
                current_pr_changes,
                similar_prs_changes,
                pr_number,
                model_name=model_name,
                chunking_strategy=chunking_strategy
            )
            
            if not result or not result.get("success"):
                print(f"❌ Failed to generate chunked review: {result.get('error', 'Unknown error')}")
                input("\nPress Enter to continue...")
                return False
            
            chunked_review = result.get("chunked_review")
            
            # Calculate RAGAS metrics for the generated review
            print(f"\n📊 Calculating RAGAS metrics for review quality...")
            evaluator = ReviewEvaluator()
            
            # Get a reference review (using best model from baseline)
            reference_model = stored_results.get("best_model", "gemini")
            # Add this code to handle dictionary case:
            if isinstance(reference_model, dict):
                # Extract the best model name from dict, preferring 'enhanced' version
                reference_model = reference_model.get('enhanced', reference_model.get('baseline', 'gemini'))
                print(f"Using {reference_model} from best model dictionary")

            reference_review = generate_review(
                current_pr_changes,
                similar_prs_changes,
                pr_number=pr_number,
                model_name=reference_model
            )
            
            # Calculate metrics against reference
            metrics = await evaluator._calculate_metrics(reference_review, chunked_review)
            
            # Save the results
            custom_results = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "session_id": session_id,
                "pr_number": pr_number,
                "configuration": {
                    "chunking_strategy": chunking_strategy,
                    "embedding_method": embedding_method,
                    "model_name": model_name,
                    "prompt_type": prompt_choice
                },
                "chunked_review": chunked_review,
                "metrics": metrics,
                "chunking_stats": result.get("chunking_stats", {})
            }
            
            # Save results
            json_path = save_results(
                custom_results,
                "custom_pr_review",
                session_id
            )
            
            # Also save the review as markdown for easier reading
            review_path = json_path.replace(".json", ".md")
            with open(review_path, "w", encoding="utf-8") as f:
                f.write(chunked_review)
            
            # Display review summary
            print("\n📝 Custom PR Review:")
            print("-" * 50)
            preview_length = min(1000, len(chunked_review))
            print(chunked_review[:preview_length] + ("..." if len(chunked_review) > preview_length else ""))
            print("-" * 50)
            print(f"\n💾 Results saved to:")
            print(f"  - JSON: {json_path}")
            print(f"  - Markdown: {review_path}")
            
            # Display metrics summary
            print("\n📊 RAGAS Metrics:")
            print("-" * 50)
            print(f"Overall Score: {metrics.get('Overall', 0):.3f}")
            for metric, score in metrics.items():
                if metric != "Overall":
                    print(f"{metric}: {score:.3f}")
            
            input("\nPress Enter to continue...")
            return True
            
        except Exception as e:
            print(f"❌ Error during chunked review generation: {e}")
            import traceback
            traceback.print_exc()
            input("\nPress Enter to continue...")
            return False
    else:
        # Use direct review generation without chunking
        try:
            print("\n🤖 Generating review without chunking...")
            
            # Update system prompt if needed
            if system_prompt:
                ReviewPrompts.update_system_prompt(system_prompt)
            
            # Generate review with selected model
            review = generate_review(
                current_pr_changes,
                similar_prs_changes,
                pr_number=pr_number,
                current_pr_file=", ".join(pr_files) if pr_files else None,
                model_name=model_name
            )
            
            if not review:
                print("❌ Failed to generate review")
                input("\nPress Enter to continue...")
                return False
            
            # Calculate RAGAS metrics
            print(f"\n📊 Calculating RAGAS metrics for review quality...")
            evaluator = ReviewEvaluator()
            reference_model = stored_results.get("best_model", "gemini")
            # Add the same dictionary handling code:
            if isinstance(reference_model, dict):
                # Extract the best model name from dict, preferring 'enhanced' version
                reference_model = reference_model.get('enhanced', reference_model.get('baseline', 'gemini'))
                print(f"Using {reference_model} from best model dictionary")
            
            # Generate reference review if needed
            if reference_model != model_name:
                reference_review = generate_review(
                    current_pr_changes,
                    similar_prs_changes,
                    pr_number=pr_number,
                    current_pr_file=", ".join(pr_files) if pr_files else None,
                    model_name=reference_model
                )
            else:
                # If using the same model as reference, use a different model for comparison
                reference_model = "gemini" if model_name != "gemini" else "deepseek"
                reference_review = generate_review(
                    current_pr_changes,
                    similar_prs_changes,
                    pr_number=pr_number,
                    current_pr_file=", ".join(pr_files) if pr_files else None,
                    model_name=reference_model
                )
            
            # Calculate metrics against reference
            metrics = await evaluator._calculate_metrics(reference_review, review)
            
            # Save the results
            custom_results = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "session_id": session_id,
                "pr_number": pr_number,
                "configuration": {
                    "chunking_strategy": "none",
                    "embedding_method": embedding_method,
                    "model_name": model_name,
                    "prompt_type": prompt_choice
                },
                "review": review,
                "metrics": metrics
            }
            
            # Save results
            json_path = save_results(
                custom_results,
                "custom_pr_review",
                session_id
            )
            
            # Also save the review as markdown for easier reading
            review_path = json_path.replace(".json", ".md")
            with open(review_path, "w", encoding="utf-8") as f:
                f.write(review)
            
            # Display review summary
            print("\n📝 Custom PR Review:")
            print("-" * 50)
            preview_length = min(1000, len(review))
            print(review[:preview_length] + ("..." if len(review) > preview_length else ""))
            print("-" * 50)
            print(f"\n💾 Results saved to:")
            print(f"  - JSON: {json_path}")
            print(f"  - Markdown: {review_path}")
            
            # Display metrics summary
            print("\n📊 RAGAS Metrics:")
            print("-" * 50)
            print(f"Overall Score: {metrics.get('Overall', 0):.3f}")
            for metric, score in metrics.items():
                if metric != "Overall":
                    print(f"{metric}: {score:.3f}")
            
            input("\nPress Enter to continue...")
            return True
            
        except Exception as e:
            print(f"❌ Error during review generation: {e}")
            import traceback
            traceback.print_exc()
            input("\nPress Enter to continue...")
            return False

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

async def run_chunked_rag_review(repo_owner, repo_name, pr_number):
    """Run the RAG-based PR review process with hybrid semantic chunking"""
    print(f"🚀 Starting chunked RAG-based review for PR #{pr_number} in {repo_owner}/{repo_name}")
    
    # Try to get existing data from previous session
    print("\n📦 Step 1: Getting PR data...")
    current_pr_changes = None
    similar_prs_changes = None
    
    # Check if we have existing PR data from a previous session
    stored_results = load_stored_prompts(session_id)
    
    if stored_results:
        try:
            print(f"\n📂 Using existing PR data from session {session_id}")
            
            # Get PR data from existing session
            current_pr_changes = stored_results.get("current_pr_changes")
            similar_prs_changes = stored_results.get("similar_prs_changes", [])
            
            if current_pr_changes and similar_prs_changes:
                print(f"✅ Loaded existing PR data ({len(current_pr_changes)} chars)")
                print(f"✅ Loaded {len(similar_prs_changes)} similar PRs")
            else:
                print("⚠️ Incomplete data in stored session")
        except Exception as e:
            print(f"⚠️ Error reading session data: {e}")
    
    # If no existing data, fetch it fresh
    if not current_pr_changes or not similar_prs_changes:
        print("\n⚙️ Fetching fresh PR data...")
        
        # Use the fetch_pr_data helper function to get data
        current_pr_changes, similar_prs_changes = await fetch_pr_data(repo_owner, repo_name, pr_number)
        
        if not current_pr_changes or not similar_prs_changes:
            print("❌ Failed to get PR data")
            return
    
    # Step 2: Apply hybrid semantic chunking for review generation
    print("\n🧩 Step 2: Applying hybrid semantic chunking and generating review...")
    
    # Create a valid ChromaDB collection name (alphanumeric only)
    collection_name = f"chunks{pr_number}_{uuid.uuid4().hex[:8]}"
    
    # Create generator and process PR with chunking
    generator = ChunkedReviewGenerator(chunk_collection_name=collection_name)
    
    result = await generator.process_pr_with_chunking(
        current_pr_changes, 
        similar_prs_changes,
        pr_number
    )
    
    if not result.get("success"):
        print(f"❌ Failed to generate chunked review: {result.get('error')}")
        return
    
    chunked_review = result.get("chunked_review")
    best_model = result.get("best_model")
    model_metrics = result.get("model_metrics")
    
    # No need to display model metrics again - they were already shown during evaluation
    print("\n✅ Chunked RAG-based review process completed successfully!")
    
    # Save results
    print("\n💾 Saving results...")
    
    # Save results and get the correct filepath
    json_path = save_results({
        "pr_number": pr_number,
        "chunked_review": chunked_review,
        "best_model": best_model,
        "model_metrics": model_metrics,
        "chunking_stats": result.get("chunking_stats"),
        "current_pr_changes": current_pr_changes,
        "similar_prs_changes": similar_prs_changes
    }, "chunked_rag_review", session_id)

    # Display the review
    print("\n📝 Summary of chunked review:")
    print("-" * 50)
    # Show a larger preview of the review
    preview_length = min(1000, len(chunked_review))
    print(chunked_review[:preview_length] + ("..." if len(chunked_review) > preview_length else ""))
    print("-" * 50)
    print(f"\nFull review saved to {json_path}")

    return chunked_review

# Helper function to fetch PR data
async def fetch_pr_data(repo_owner, repo_name, pr_number):
    """Fetch PR data including current changes and similar PR changes"""
    # Step 1: Fetch repository PRs and store changed files
    print("\n📦 Step 1: Fetching pull requests data...")
    changed_files, pull_requests = fetch_pull_requests(repo_owner, repo_name)
    
    if not changed_files or not pull_requests:
        print("❌ Failed to fetch pull requests")
        return None, None
    
    # Step 2: Create regular embeddings
    print("\n📊 Step 2: Creating embeddings for PR similarity detection...")
    _, collection = store_embeddings(changed_files, pull_requests)
    
    if not collection:
        print("❌ Failed to store embeddings")
        return None, None
    
    # Step 3: Query similar PRs
    print(f"\n🔍 Step 3: Finding similar PRs to PR #{pr_number}...")
    query_results, pr_files = query_similar_prs(pr_number, repo_owner, repo_name, collection)
    
    if not query_results or not pr_files:
        print("❌ Failed to find similar PRs")
        return None, None
    
    # Get similar PR numbers from results
    similar_pr_numbers = []
    for i in range(min(3, len(query_results["metadatas"][0]))):
        similar_pr = query_results["metadatas"][0][i]["pr_number"]
        
        # Don't include the same PR
        if similar_pr != pr_number:
            similar_pr_numbers.append(similar_pr)
    
    # Step 4: Compare PR changes
    print(f"\n📈 Step 4: Comparing changes between PR #{pr_number} and {len(similar_pr_numbers)} similar PRs...")
    current_pr_changes, similar_prs_changes = compare_pr_changes(pr_files, similar_pr_numbers, repo_owner, repo_name)
    
    return current_pr_changes, similar_prs_changes
    

def display_menu():
    """Display the options menu"""
    print("\n" + "="*50)
    print("🚀 RAG-BASED PR REVIEW SYSTEM")
    print("="*50)
    # print("\nEnhancement Options:")
    # print("0. 🔄 Run standard review (default)")
    # print("1. 🔍 Add confidence scores to review suggestions")
    # print("2. 📝 Use enhanced prompts for better specificity")
    # print("3. 📊 DB chunking Advice")
    # print("4. 🧪 Test Chunking Strategy")
    # print("5. 💡 Add interactive feedback system for RAGAS improvement")
    # print("6. 🧩 Compare different embedding methods")  
    # print("7. ❌ Exit")

    print("\nEnhancement Options:")
    print("0. 🔄 Run standard review (default)")
    print("1. 🔍 Add confidence scores to review suggestions")
    print("2. 📊 DB chunking Advice")
    print("3. 🧪 Test Chunking Strategy")
    print("4. 📝 Use enhanced prompts")
    print("5. 🧩 Test embedding methods") 
    print("6. 🏆 Top Scoring Techniques")
    print("7. 🛠️  Custom PR Review Generator")
    print("8. ❌ Exit")

    print("-"*50)
    choice = input("\nSelect an option (0-8): ")
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
    
    Returns:
        Absolute path to the saved file
    """
    results_dir = "RAG_based_Analysis_2"
    # Ensure directory exists
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
    
    # Return the absolute path that actually exists
    abs_path = os.path.abspath(filepath)
    
    # Verify file exists before returning path
    if os.path.exists(abs_path):
        return abs_path
    else:
        # If path doesn't exist (shouldn't happen), return the expected path with warning
        print(f"⚠️ Warning: File may not have been saved correctly at {abs_path}")
        return abs_path

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
    # Find the most recent session file
    results_dir = "RAG_based_Analysis_2"
    if not os.path.exists(results_dir):
        return None
    
    # Look for any file with this session ID
    files = []
    for f in os.listdir(results_dir):
        if f.endswith(".json") and session_id in f:
            files.append(os.path.join(results_dir, f))
    
    if not files:
        return None
    
    # Try to find both baseline and enhanced files
    baseline_file = None
    enhanced_file = None
    
    for file in files:
        if "baseline" in file:
            baseline_file = file
        elif "enhanced" in file:
            enhanced_file = file
    
    # Always start with baseline data since it has the PR data
    if not baseline_file:
        print("❌ No baseline file found for session")
        return None
    
    try:
        with open(baseline_file, 'r') as f:
            merged_data = json.load(f)
            
        # If we have an enhanced file, merge metrics, reviews and prompts from it
        if enhanced_file:
            with open(enhanced_file, 'r') as f:
                enhanced_data = json.load(f)
                
            # Keep PR data from baseline file but use enhanced metrics and reviews
            if "metrics_comparison" in enhanced_data:
                merged_data["metrics_comparison"] = enhanced_data["metrics_comparison"]
            if "reviews" in enhanced_data:
                merged_data["reviews"] = enhanced_data["reviews"]
            if "prompts" in enhanced_data:
                merged_data["prompts"] = enhanced_data["prompts"]
            if "best_model" in enhanced_data:
                merged_data["best_model"] = enhanced_data["best_model"]
        
        # Print diagnostic info about loaded data
        print(f"\n📊 Session {merged_data['session_id']} data loaded:")
        print(f"  - Timestamp: {merged_data.get('timestamp', 'Unknown')}")
        print(f"  - PR Files: {len(merged_data.get('pr_files', []))} files")
        print(f"  - Similar PRs: {len(merged_data.get('similar_prs_changes', []))} PRs")
        
        return merged_data
        
    except Exception as e:
        print(f"❌ Error loading stored prompts: {e}")
        return None


class SessionManager:
    """Manages session data across options"""
    
    def __init__(self):
        self.current_session_id = None
        self.session_data = None
    
    def start_new_session(self):
        """Start a new session and return session ID"""
        self.current_session_id = generate_session_id()
        self.session_data = None
        # print(f"\n🔑 Starting new session: {self.current_session_id}")
        return self.current_session_id
    
    def load_session_data(self):
        """Load data for current session"""
        if not self.current_session_id:
            print("❌ No active session found")
            return None
            
        self.session_data = load_stored_prompts(self.current_session_id)
        return self.session_data
    
    def get_session_id(self):
        """Get current session ID"""
        return self.current_session_id

# Create a global instance
session_manager = SessionManager()

# def load_stored_prompts(session_id=None):
#     """Load stored results with session validation"""
#     latest_file = get_latest_session_file(session_id)
#     if not latest_file:
#         return None
        
#     try:
#         with open(latest_file, 'r') as f:
#             data = json.load(f)
            
#         # Validate session ID
#         if not data.get("session_id"):
#             print("❌ No session ID found in stored results")
#             return None
            
#         if session_id and data["session_id"] != session_id:
#             print("❌ Results from different session found. Please run option 0 first")
#             return None
            
#         # Check for either original structure or enhanced structure
#         has_original_data = all(key in data for key in ["baseline_review", "baseline_metrics"])
#         has_enhanced_data = all(key in data for key in ["reviews", "metrics_comparison"])
        
#         if not (has_original_data or has_enhanced_data):
#             print("❌ Stored results file is missing required data structure")
#             return None
            
#         # If we have enhanced data structure, reconstruct original format
#         if has_enhanced_data and not has_original_data:
#             # Convert enhanced structure back to original format
#             data["baseline_review"] = data["reviews"].get("gemini", "")
#             data["baseline_metrics"] = data["metrics_comparison"].get("baseline", {})
            
#         return data
        
#     except Exception as e:
#         print(f"Error loading stored prompts: {e}")
#         return None

async def initial_review():
    try:
        # Remove duplicate declarations and use constants
        # print(f"\n📦 Fetching pull request data...")
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

        baseline_metrics = stored_results["baseline_metrics"];

        for model in enhanced_metrics:
            if model in baseline_metrics:
                for metric in enhanced_metrics[model]:
                    baseline = baseline_metrics[model].get(metric, 0)
                    enhanced = enhanced_metrics[model].get(metric, 0)
                    change = enhanced - baseline
                    pct_change = (change / baseline * 100) if baseline else 0;

                    print(f"{model:<10} | {metric:<15} | {baseline:>8.3f} | {enhanced:>8.3f} | {change:>+8.3f} | {pct_change:>7.1f}% |");

    except Exception as e:
        print(f"❌ Error during enhanced prompt evaluation: {e}")
    
def load_complete_session_data(session_id):
    """Load all data for a session by finding and merging baseline and enhanced files"""
    data_dir = "RAG_based_Analysis_2"
    
    # Look for baseline file first
    baseline_pattern = os.path.join(data_dir, f"*{session_id}*baseline.json")
    baseline_files = glob.glob(baseline_pattern)
    
    if not baseline_files:
        print(f"❌ No baseline file found for session {session_id}")
        return None
    
    try:
        with open(baseline_files[0], 'r') as f:
            merged_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading baseline file: {e}")
        return None
    
    # Look for enhanced file if it exists
    enhanced_pattern = os.path.join(data_dir, f"*{session_id}*enhanced.json")
    enhanced_files = glob.glob(enhanced_pattern)
    
    if enhanced_files:
        try:
            with open(enhanced_files[0], 'r') as f:
                enhanced_data = json.load(f)
                
            # Merge only specific fields
            for field in ["metrics_comparison", "best_model", "prompts", "reviews"]:
                if field in enhanced_data:
                    merged_data[field] = enhanced_data[field]
        except Exception as e:
            print(f"❌ Error loading enhanced file: {e}")
    
    return merged_data

if __name__ == "__main__":
    # Check environment setup
    if not setup_environment():
        exit(1)
    
    # Create session manager instance
    session_manager = SessionManager()
    
    while True:
        choice = display_menu()
        
        if choice == "0":
            # Generate new session ID when starting fresh
            session_id = session_manager.start_new_session()
            print(f"\n🔑 Starting new session: {session_id}")
            
            print("\n🔍 Running initial review and prompt generation...")
            
            import asyncio
            
            # Run initial review and store the session ID in the manager
            results = asyncio.run(initial_review())
            input("\nPress Enter to continue...")
        
        elif choice == "1":
            # Use manager to get session ID
            session_id = session_manager.get_session_id()
            if not session_id:
                print("❌ No active session found. Please run option 0 first")
                input("\nPress Enter to continue...")
                continue
                
            # Use manager to load data
            stored_results = session_manager.load_session_data()
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
                
            # Load stored results from previous run
            stored_results = load_stored_prompts(session_id)
            if not stored_results:
                print("❌ Please run option 0 first to generate baseline review")
                input("\nPress Enter to continue...")
                continue
                
            
            print("\n📊 Analyzing PR for chunking advice...")
            stored_results = load_stored_prompts(session_id)
            
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
                print("\n⚠️ Could not generate chunking advice")
            
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
            
            print("\n🧮 Evaluating Chunking Strategies with RAGAS Metrics...")
            
            from chunk_tester import ChunkingStrategyTester
            from chunked_review_generator import ChunkedReviewGenerator
            import webbrowser
            import os
            
            async def test_chunking_with_ragas():
                try:
                    # Prepare PR data for chunking tests
                    print("\n📊 Getting PR data for chunking comparison...")
                    
                    # Get PR content from stored results - use consistent field names
                    current_pr_changes = stored_results.get("current_pr_changes", "")
                    if not current_pr_changes:
                        # Try alternate fields as fallback
                        current_pr_changes = (stored_results.get("baseline_review") or 
                                            stored_results.get("reviews", {}).get("gemini", ""))
                    
                    # Get similar PRs data
                    similar_prs_data = stored_results.get("similar_prs_changes", [])
                    similar_prs_changes = []
                    
                    # Standard format for similar PRs across all options
                    if similar_prs_data:
                        for pr_data in similar_prs_data:
                            # Standardize the format
                            if isinstance(pr_data, dict):
                                pr_number = pr_data.get('pr_number', 0)
                                changes = pr_data.get('changes', '')
                                if not changes and 'content' in pr_data:
                                    changes = pr_data['content']
                                
                                similar_prs_changes.append({
                                    'pr_number': pr_number,
                                    'changes': changes
                                })
                        
                    # If no similar PRs were found, manually add them to the session data
                    if not similar_prs_changes and current_pr_changes:
                        print("⚠️ No similar PRs found, but current PR data is available")
                        
                        # Manually add current PR changes as the only "similar" PR for testing
                        similar_prs_changes = [{
                            'pr_number': PR_NUMBER,
                            'changes': current_pr_changes
                        }]
                        print("✅ Added current PR changes as a similar PR for testing")
                    
                    pr_number = stored_results.get("pr_number", PR_NUMBER)
                    
                    if not current_pr_changes:
                        print("❌ Could not find PR content in stored results")
                        return False
                        
                    print(f"✅ Loaded PR data ({len(current_pr_changes)} chars)")
                    
                    if not similar_prs_changes:
                        print("⚠️ No similar PRs found, continuing with current PR only")
                    else:
                        print(f"✅ Found {len(similar_prs_changes)} similar PRs")
                    
                    # Initialize chunking tester
                    chunking_tester = ChunkingStrategyTester()
                    
                    # Run chunking comparison
                    print("\n🔄 Running chunking strategy comparison (this may take several minutes)...")
                    result = await chunking_tester.run_chunking_test(
                        current_pr_changes,
                        similar_prs_changes,
                        pr_number
                    )
                    
                    if not result:
                        print("❌ Chunking comparison failed")
                        return False
                        
                    print("\n✅ Chunking comparison completed successfully!")
                    
                    # Extract best strategy and score
                    comparison_results = result.get("results", {})
                    best_strategy = comparison_results.get("best_strategy")
                    best_score = comparison_results.get("best_score", 0)
                    
                    # Display concise recommendation
                    print("\n📋 Chunking Strategy Recommendation:")
                    print("═" * 60)
                    
                    if best_strategy:
                        best_name = next((s["name"] for s in comparison_results.get("comparison_table", []) 
                                    if s["strategy"] == best_strategy), "Unknown")
                        print(f"✅ Best strategy: {best_name} (Score: {best_score:.3f})")
                        print(f"🔍 Key strengths: {get_strategy_strengths(best_strategy)}")
                    else:
                        print("⚠️ Could not determine best chunking strategy")
                    
                    # Generate review using best chunking strategy
                    if best_strategy:
                        print(f"\n🤖 Generating PR review using {best_strategy} chunking strategy...")
                        generator = ChunkedReviewGenerator(chunking_strategy=best_strategy)
                        review_result = await generator.process_pr_with_chunking(
                            current_pr_changes,
                            similar_prs_changes,
                            pr_number
                        )
                        
                        if review_result.get("success"):
                            chunked_review = review_result.get("chunked_review")
                            best_model = review_result.get("best_model")
                            
                            # Save the results in JSON
                            output = {
                                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                                "session_id": session_id,
                                "pr_number": pr_number,
                                "best_chunking_strategy": best_strategy,
                                "best_model": best_model,
                                "chunked_review": chunked_review,
                                "chunking_score": best_score,
                                "chunking_stats": review_result.get("chunking_stats", {})
                            }


                            if output:
                                # Add additional data to the result before saving
                                output["baseline_data"] = {
                                    "baseline_metrics": stored_results.get("baseline_metrics", {}),
                                    "enhanced_metrics": stored_results.get("metrics_comparison", {}).get("enhanced", {}),
                                    "current_pr_changes": current_pr_changes,
                                    "similar_prs_changes": similar_prs_changes,
                                    "pr_files": stored_results.get("pr_files", [])
                                }

                                # # Save the enriched results
                                # json_path = save_results(
                                #     output,
                                #     "chunking_comparison",
                                #     session_id
                                # )
                                # print(f"\n💾 Chunking comparison results saved to {json_path}")

                            
                            # Save results to JSON file
                            json_path = save_results(
                                output,
                                "best_chunking_review",
                                session_id
                            )
                            
                            # Display preview of the generated review
                            print("\n📝 PR Review (using best chunking strategy):")
                            print("-" * 60)
                            preview_length = min(500, len(chunked_review))
                            print(chunked_review[:preview_length] + ("..." if len(chunked_review) > preview_length else ""))
                            print("-" * 60)
                            print(f"\n💾 Chunking comparison results saved to {json_path}")
                        else:
                            print(f"❌ Failed to generate review: {review_result.get('error', 'Unknown error')}")
                    
                    # Handle visualizations and report
                    if result.get("report_path"):
                        html_report = result.get("report_path").replace(".md", "_report.html")
                        if os.path.exists(html_report):
                            print(f"\n🌐 Opening HTML report in browser...")
                            webbrowser.open(f"file://{os.path.abspath(html_report)}")

                    
                    # ---------------------

                    # Get baseline metrics from stored results
                    # Update the metrics comparison section in test_chunking_with_ragas():
                    # ---------------------

                    # Update the metrics comparison section in test_chunking_with_ragas() function:

                    # Get baseline metrics from stored results
                    baseline_metrics = stored_results.get("baseline_metrics", {}).get(stored_results.get("best_model", "gemini"), {})

                    if baseline_metrics and comparison_results.get("strategies"):
                        # Create metrics comparison table
                        chunking_metrics_comparison = {
                            "metrics_table": {
                                "strategies": {
                                    "Baseline": baseline_metrics,
                                    "Hybrid_Semantic": comparison_results["strategies"]["hybrid"]["metrics"],
                                    "Pure_Semantic": comparison_results["strategies"]["semantic"]["metrics"],
                                    "Fixed_Size": comparison_results["strategies"]["fixed"]["metrics"],
                                    "Hierarchical": comparison_results["strategies"]["hierarchical"]["metrics"]
                                },
                                "metrics_list": [
                                    "Relevance", "Accuracy", "Groundedness", "Completeness", 
                                    "Faithfulness", "ContextualPrecision", "ContextRecall", 
                                    "AnswerRelevance", "BLEU", "ROUGE", "Overall"
                                ]
                            }
                        }

                        # Add metrics comparison to output
                        output["chunking_metrics_comparison"] = chunking_metrics_comparison

                        # Save the results with metrics comparison
                        json_path = save_results(
                            output,
                            "best_chunking_review",
                            session_id
                        )
                    
                    # ---------------------

                    # Display comparison of baseline and enhanced metrics
                    # print("\n📈 RAGAS Metrics Comparison:")
                    # print("=" * 90)
                    # print(f"{'Model':<10} | {'Metric':<15} | {'Baseline':>8} | {'Enhanced':>8} | {'Change':>8} | {'% Change':>8} |")
                    # print("=" * 90)

                    # baseline_metrics = stored_results["baseline_metrics"];

                    # for model in enhanced_metrics:
                    #     if model in baseline_metrics:
                    #         for metric in enhanced_metrics[model]:
                    #             baseline = baseline_metrics[model].get(metric, 0)
                    #             enhanced = enhanced_metrics[model].get(metric, 0)
                    #             change = enhanced - baseline
                    #             pct_change = (change / baseline * 100) if baseline else 0;

                    #             print(f"{model:<10} | {metric:<15} | {baseline:>8.3f} | {enhanced:>8.3f} | {change:>+8.3f} | {pct_change:>7.1f}% |");

                except Exception as e:
                    print(f"❌ Error during chunking test: {str(e)}")
                    return False
            
            # Define the get_strategy_strengths helper function here (outside of test_chunking_with_ragas)
            def get_strategy_strengths(strategy):
                strengths = {
                    "semantic": "Preserves natural code boundaries, ideal for well-structured code",
                    "hybrid": "Balances structure awareness with consistent sizing, versatile for mixed content",
                    "hierarchical": "Maintains parent-child relationships, good for complex nested code",
                    "fixed": "Consistent chunk sizing with predictable behavior"
                }
                return strengths.get(strategy, "Unknown strategy")
            
            # Run the test and generate review using best strategy
            success = asyncio.run(test_chunking_with_ragas())
            # if not success:
            #     print("\n⚠️ Could not complete chunking strategy comparison")
            
            input("\nPress Enter to continue...")
        
        elif choice == "4":
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

                    baseline_metrics = stored_results["baseline_metrics"];
                    
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
                            
                                    print(f"{'':<10} | {metric:<15} | {baseline:>8.3f} | {enhanced:>8.3f} | {change:>+8.3f} | {pct_change:>7.1f}%")

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
                        },
                        # Add these fields to match baseline format
                        "current_pr_changes": stored_results.get("current_pr_changes"),
                        "similar_prs_changes": stored_results.get("similar_prs_changes"),
                        "pr_files": stored_results.get("pr_files", [])
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

                    baseline_metrics = stored_results["baseline_metrics"];

                    for model in enhanced_metrics:
                        if model in baseline_metrics:
                            for metric in enhanced_metrics[model]:
                                baseline = baseline_metrics[model].get(metric, 0)
                                enhanced = enhanced_metrics[model].get(metric, 0)
                                change = enhanced - baseline
                                pct_change = (change / baseline * 100) if baseline else 0;

                                print(f"{model:<10} | {metric:<15} | {baseline:>8.3f} | {enhanced:>8.3f} | {change:>+8.3f} | {pct_change:>7.1f}% |");

                except Exception as e:
                    print(f"❌ Error during enhanced prompt evaluation: {e}")
            
            # Run the prompt testing
            asyncio.run(test_stored_prompt())
            input("\nPress Enter to continue...")
       
        elif choice == "5":
            print("\n🔍 Running embedding method comparison with RAGAS metrics...")
            
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
            
            # Get PR content from stored results
            current_pr_changes = stored_results.get("current_pr_changes", "")
            similar_prs_changes = stored_results.get("similar_prs_changes", [])
            
            if not current_pr_changes or not similar_prs_changes:
                print("❌ No PR data found in session. Please run option 0 first.")
                input("\nPress Enter to continue...")
                continue
            
            # Import embedding evaluator and factory
            try:
                from embeddings.embedding_factory import EmbeddingFactory
                from embeddings.embedding_evaluator import EmbeddingEvaluator
                from embeddings.tfidf_embedder import TFIDFEmbedder
            except ImportError as e:
                print(f"❌ Error importing embedding modules: {e}")
                print("Make sure the embeddings package is correctly installed")
                input("\nPress Enter to continue...")
                continue
            
            # Get available embedding methods
            available_embedders = EmbeddingFactory.get_available_embedders()
            print(f"\n📊 Available embedding methods: {', '.join(available_embedders.keys())}")
            
            # Create current PR data structure
            current_pr_data = {"changes": current_pr_changes}
            
            # Get first similar PR for comparison
            similar_pr_data = {"changes": similar_prs_changes[0]['changes']} if similar_prs_changes else {"changes": ""}
            
            print("\n🔄 Comparing embedding methods using RAGAS metrics...")
            
            # Update the compare_embeddings() function in option 6:

            async def compare_embeddings():
                try:
                    # Initialize embedding evaluator
                    evaluator = EmbeddingEvaluator()
                    
                    # Evaluate all embedding methods
                    best_embedder, evaluation_results = await evaluator.evaluate_embedders(
                        current_pr_data,
                        similar_pr_data
                    )
                    
                    print(f"\n✅ Evaluation complete! Best embedding method: {best_embedder}")
                    
                    # Use the best embedder to generate a PR review
                    if best_embedder:
                        print(f"\n🤖 Testing models with {best_embedder} embeddings...")
                        
                        # Create embedder instance - this doesn't actually affect the PR review generation
                        best_embedding = EmbeddingFactory.get_embedder(best_embedder)
                        
                        # Initialize ReviewEvaluator for model comparison
                        review_evaluator = ReviewEvaluator()
                        
                        # Evaluate all models and get the best one
                        print("\n📊 Evaluating models for review generation...")
                        best_model, model_metrics = await review_evaluator.evaluate_models(
                            current_pr_changes,
                            similar_prs_changes
                        )
                        
                        print(f"\n✅ Best model determined: {best_model}")
                        
                        # Generate review using best model
                        print(f"\n🤖 Generating PR review using {best_embedder} embeddings and {best_model} model...")
            
                        # FIX: Add print_full_review=False parameter to avoid duplicate display
                        review = generate_review(
                            current_pr_changes,
                            similar_prs_changes,
                            pr_number=PR_NUMBER,
                            model_name=best_model
                        )
                        
                        if review:
                            print("\n📝 PR Review (using best embedding method and model):")
                            print("-" * 60)
                            preview_length = min(500, len(review))
                            print(review[:preview_length] + ("..." if len(review) > preview_length else ""))
                            print("-" * 60)
                            
                            # Save comprehensive results
                            results = {
                                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                                "session_id": session_id,
                                "embedding_evaluation": evaluation_results,
                                "best_embedder": best_embedder,
                                "model_evaluation": {
                                    "best_model": best_model,
                                    "model_metrics": model_metrics
                                },
                                "pr_review": review
                            }
                            
                            json_path = save_results(
                                results,
                                "embedding_evaluation",
                                session_id
                            )
                            
                            # Display preview of the generated review
                            print("\n📝 PR Review (using best embedding method):")
                            print("-" * 60)
                            preview_length = min(500, len(review))
                            print(review[:preview_length] + ("..." if len(review) > preview_length else ""))
                            print("-" * 60)
                            print(f"\n💾 Results saved to {json_path}")
                            

                            # # Display metrics comparison
                            # print("\n📈 Model Performance Metrics:")
                            # print("=" * 80)
                            # print(f"{'Model':<15} | {'Metric':<20} | {'Score':>10}")
                            # print("-" * 80)
                            
                            # for model, metrics in model_metrics.items():
                            #     for metric, score in metrics.items():
                            #         print(f"{model:<15} | {metric:<20} | {score:>10.3f}")
                        else:
                            print("❌ Failed to generate PR review")
                    
                    return best_embedder, evaluation_results
                except Exception as e:
                    print(f"❌ Error during embedding comparison: {e}")
                    import traceback
                    traceback.print_exc()
                    return "tfidf", {}  # Default fallback
            
            # Run embedding comparison
            best_embedder, evaluation_results = asyncio.run(compare_embeddings())
            
            input("\nPress Enter to continue...")

        elif choice == "6":
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
            
        elif choice == "7":
            # Call the custom PR review generator function
            asyncio.run(custom_pr_review_generator())
                
        elif choice == "8": 
            print("\n👋 Exiting the program. Goodbye!")
            exit(0)
        
        else:
            print("\n⚠ Invalid option. Please try again.")
            input("\nPress Enter to continue...")