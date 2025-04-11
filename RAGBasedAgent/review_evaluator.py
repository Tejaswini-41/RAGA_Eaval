import os
import sys
import asyncio
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from existing evaluation system
from evaluation.metrics import MetricsCalculator 
from models.model_factory import ModelFactory

class ReviewEvaluator:
    """Evaluates models for PR review quality using existing RAGA metrics"""
    
    def __init__(self):
        self.metrics_calculator = MetricsCalculator()
        self.model_factory = ModelFactory()
        
        # Updated weights with enhanced RAGA metrics
        self.weights = {
            # Original metrics with adjusted weights
            "Relevance": 0.15,
            "Accuracy": 0.15,
            "Groundedness": 0.15,
            "Completeness": 0.10,
            
            # New RAGA metrics
            "Faithfulness": 0.20,     # How well the response avoids hallucinations
            "ContextualPrecision": 0.15,  # How precisely the review references specific PR elements
            "AnswerRelevance": 0.05,   # How well the review addresses PR-specific issues
            
            # Existing metrics with reduced weights
            "BLEU": 0.025,
            "ROUGE": 0.025
        }
    
    async def evaluate_models(self, current_pr_changes, similar_pr_changes, 
                             models=None, test_prompt_size=0.3):
        """
        Evaluate multiple models on PR review task to select best performer
        
        Args:
            current_pr_changes: The changes in current PR
            similar_pr_changes: The changes in similar PR
            models: List of model names to evaluate
            test_prompt_size: Fraction of PR content to use for quick evaluation
            
        Returns:
            best_model: Name of the best performing model
            all_metrics: Dict of metrics for all models
        """
        # Truncate PR content to avoid token limit errors
        max_tokens = 4000  # Set a reasonable token limit
        
        # Truncate PR content
        if len(current_pr_changes) > max_tokens:
            current_pr_changes = current_pr_changes[:max_tokens] + "...[truncated]"
        
        # Truncate similar PR content
        if isinstance(similar_pr_changes, list):
            truncated_similar = []
            for pr in similar_pr_changes:
                if len(pr['changes']) > max_tokens//2:
                    pr['changes'] = pr['changes'][:max_tokens//2] + "...[truncated]"
                truncated_similar.append(pr)
            similar_pr_changes = truncated_similar
        else:
            if len(similar_pr_changes) > max_tokens:
                similar_pr_changes = similar_pr_changes[:max_tokens] + "...[truncated]"
        
        # Get available models if not specified
        if not models:
            models = self.model_factory.get_model_names()
            
        # Build comprehensive prompt template for evaluation - using FULL content
        prompt = f"""Compare these pull requests:
    
Similar PR:
{similar_pr_changes}

Current PR:
{current_pr_changes}

Please provide a detailed code review including:
1. Summary of the changes
2. File Change Suggestions - Identify files that might get affected based on changes
3. Conflict Prediction - Flag files changed in multiple PRs that could cause conflicts
4. Breakage Risk Warning - Note which changes might break existing functionality
5. Test Coverage Advice - Recommend test files that should be updated
6. Code Quality Suggestions - Point out potential code smells or duplication

Be specific with file names, function names, and line numbers when possible.
"""
        
        print("\n🔍 Evaluating models to select best one for PR review...")
        
        # Generate initial review with reference model (using Gemini as reference)
        reference_model = "gemini"
        print(f"• Generating reference review with {reference_model}...")
        reference_review = self.model_factory.generate_response_with_prompt(
            reference_model, 
            prompt, 
            "You are an expert code reviewer. Focus on practical, specific suggestions. Mention specific files, functions, and line numbers when relevant."
        )
        
        # Generate and evaluate reviews from other models
        model_metrics = {}
        all_reviews = {reference_model: reference_review}
        
        for model_name in models:
            if model_name == reference_model:
                continue
                
            print(f"• Testing {model_name} model...")
            
            try:
                # Generate sample review
                test_review = self.model_factory.generate_response_with_prompt(
                    model_name, 
                    prompt, 
                    "You are an expert code reviewer. Focus on practical, specific suggestions. Mention specific files, functions, and line numbers when relevant."
                )
                all_reviews[model_name] = test_review
                
                # Calculate metrics (using similar approach to interactive_eval.py)
                metrics = await self._calculate_metrics(reference_review, test_review)
                model_metrics[model_name] = metrics
                print(f"  - Overall score: {metrics['Overall']:.3f}")
                
            except Exception as e:
                print(f"  ⚠️ Error evaluating {model_name}: {e}")
        
        # Find best model
        if model_metrics:
            best_model = max(model_metrics.items(), key=lambda x: x[1]["Overall"])
            print(f"\n✅ Best model for PR review: {best_model[0]} (Score: {best_model[1]['Overall']:.3f})")
            
            # Print comparison table (like in interactive_eval.py)
            self._display_comparison_table(reference_model, model_metrics)
            
            return best_model[0], model_metrics
        else:
            print(f"\n⚠️ No models successfully evaluated. Using {reference_model} as default.")
            return reference_model, {}
    
    async def generate_detailed_pr_review(self, current_pr_data, similar_prs_data, model_name=None):
        """
        Generate detailed PR review with specific suggestions based on PR comparison
        
        Args:
            current_pr_data: Dict with current PR info (changes, files, PR number)
            similar_prs_data: List of dicts with similar PR info
            model_name: Name of model to use (uses best model if None)
            
        Returns:
            Dict with detailed review sections
        """
        if not model_name:
            # Use best model from evaluation or default to gemini
            model_name = "gemini"
        
        # Extract key information
        current_files = current_pr_data.get('changed_files', [])
        current_changes = current_pr_data.get('changes', '')
        
        # Build context from similar PRs
        similar_prs_context = self._build_similar_prs_context(similar_prs_data)
        
        prompt = f"""As an expert code reviewer, analyze this PR and provide specific suggestions:

CURRENT PR DETAILS:
PR #{current_pr_data.get('number', 'Unknown')}
Changed Files: {', '.join(current_files)}

CODE CHANGES:
{current_changes}

SIMILAR PRS HISTORY:
{similar_prs_context}

PROVIDE THE FOLLOWING SECTIONS:
1. Summary of Changes - Brief overview of what this PR does
2. File Change Suggestions - Identify additional files that might need changes based on modified files
3. Conflict Prediction - Flag files with high change frequency that could cause conflicts
4. Breakage Risk Warning - Note which changes might break existing functionality
5. Test Coverage Advice - Recommend test files that should be updated
6. Code Quality Suggestions - Point out potential code smells or duplication

Be specific with file names, function names, and line numbers when possible.
"""
        
        print("\n🔍 Generating detailed PR review...")
        
        detailed_review = self.model_factory.generate_response_with_prompt(
            model_name, 
            prompt, 
            "You are an expert code reviewer. Focus on practical, specific suggestions. Mention specific files, functions, and line numbers when relevant."
        )
        
        return detailed_review
    
    async def _calculate_metrics(self, reference, response):
        """Calculate all metrics for a model response compared to reference"""
        metrics = {}
        
        # Calculate original metrics
        metrics["Relevance"] = self.metrics_calculator.compute_relevance(reference, response)
        metrics["Accuracy"] = self.metrics_calculator.compute_accuracy(response)
        metrics["Groundedness"] = self.metrics_calculator.compute_groundedness(reference, response)
        metrics["Completeness"] = self.metrics_calculator.compute_completeness(reference, response)
        
        # Use RAGAS faithfulness with free LLM if available (with fallback)
        try:
            # Use RAGAS with free LLM
            metrics["Faithfulness"] = await self.metrics_calculator.get_ragas_faithfulness(
                reference, response, use_custom=False
            )
        except Exception as e:
            print(f"Error using RAGAS faithfulness: {e}")
            # Fallback to custom implementation
            metrics["Faithfulness"] = self.metrics_calculator.compute_faithfulness(reference, response)
        
        # Calculate other custom metrics
        metrics["ContextualPrecision"] = self.metrics_calculator.compute_contextual_precision(reference, response)
        metrics["AnswerRelevance"] = self.metrics_calculator.compute_answer_relevance(reference, response)
        
        # Calculate BLEU and ROUGE
        metrics["BLEU"] = await self.metrics_calculator.compute_bleu_ragas(reference, response)
        metrics["ROUGE"] = await self.metrics_calculator.compute_rouge_ragas(reference, response)
        
        # Calculate overall score with weights
        weighted_score = sum(metrics[metric] * weight for metric, weight in self.weights.items())
        metrics["Overall"] = round(weighted_score, 3)
        
        return metrics
    
    def _display_comparison_table(self, reference_model, model_metrics):
        """Display comparison table of model metrics with improved terminal readability"""
        print("\n" + "═" * 80)
        print(f"📊 MODEL COMPARISON (Reference: {reference_model.upper()})")
        print("═" * 80)
        
        # Use more compact header names with better spacing
        header = f"{'Model':<12} | {'Overall':<7} |"
        
        # Short forms for metrics to improve terminal display
        metric_labels = {
            "Relevance": "Relev", 
            "Accuracy": "Accur", 
            "Groundedness": "Grnd",  # Fixed typo: was Groundness
            "Completeness": "Comp",
            "Faithfulness": "Faith",
            "ContextualPrecision": "Ctx",
            "AnswerRelevance": "Answ", 
            "BLEU": "BLEU",
            "ROUGE": "ROUG"
        }
        
        # Use correct metric keys from the weights dictionary
        metrics_to_display = [
            "Relevance", "Accuracy", "Groundedness", "Completeness", 
            "Faithfulness", "ContextualPrecision", "AnswerRelevance",
            "BLEU", "ROUGE"
        ]
        
        # Add metrics with clear column separation
        for metric in metrics_to_display:
            header += f" {metric_labels[metric]:<5} |"
        
        print(header)
        print("─" * 80)
        
        # Print each model's metrics with clear column separation
        for model_name, scores in sorted(model_metrics.items(), key=lambda x: x[1]["Overall"], reverse=True):
            row = f"{model_name:<12} | {scores['Overall']:<7.3f} |"
            
            for metric in metrics_to_display:
                # Handle metrics that might not be available
                value = scores.get(metric, 0)
                row += f" {value:<5.2f} |"  # Use .2f for more compact display
            
            print(row)
        
        print("─" * 80)