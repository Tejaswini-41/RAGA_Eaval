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
        self.weights = {
            "Relevance": 0.20,
            "Accuracy": 0.25,
            "Groundedness": 0.20,
            "Completeness": 0.15,
            "BLEU": 0.10,
            "ROUGE": 0.10
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
        # Get available models if not specified
        if not models:
            models = self.model_factory.get_model_names()
            
        # Create a shortened version of PR changes for quick evaluation
        shortened_current = self._shorten_content(current_pr_changes, test_prompt_size)
        shortened_similar = self._shorten_content(similar_pr_changes, test_prompt_size)
        
        # Build prompt template for evaluation
        prompt = f"""Compare these pull requests and provide a detailed code review:
        
Similar PR:
{shortened_similar}

Current PR:
{shortened_current}

Provide a focused code review that includes:
1. Summary of changes
2. Potential issues or bugs 
3. Improvement suggestions
4. Patterns from similar PR that could apply
"""
        
        print("\n🔍 Evaluating models to select best one for PR review...")
        
        # Generate initial review with reference model (using Gemini as reference)
        reference_model = "gemini"
        print(f"• Generating reference review with {reference_model}...")
        reference_review = self.model_factory.generate_response_with_prompt(
            reference_model, 
            prompt, 
            "You are an expert code reviewer. Be concise, technical and specific."
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
                    "You are an expert code reviewer. Be concise, technical and specific."
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
    
    async def _calculate_metrics(self, reference, response):
        """Calculate all metrics for a model response compared to reference"""
        metrics = {}
        
        # Calculate individual metrics (similar to interactive_eval.py)
        metrics["Relevance"] = self.metrics_calculator.compute_relevance(reference, response)
        metrics["Accuracy"] = self.metrics_calculator.compute_accuracy(response)
        metrics["Groundedness"] = self.metrics_calculator.compute_groundedness(reference, response)
        metrics["Completeness"] = self.metrics_calculator.compute_completeness(reference, response)
        
        # Calculate BLEU and ROUGE using existing metrics calculator
        metrics["BLEU"] = await self.metrics_calculator.compute_bleu_ragas(reference, response)
        metrics["ROUGE"] = await self.metrics_calculator.compute_rouge_ragas(reference, response)
        
        # Calculate overall score with weights
        weighted_score = sum(metrics[metric] * weight for metric, weight in self.weights.items())
        metrics["Overall"] = round(weighted_score, 3)
        
        return metrics
    
    def _display_comparison_table(self, reference_model, model_metrics):
        """Display comparison table of model metrics"""
        print("\n" + "═" * 80)
        print(f"📊 MODEL COMPARISON (Reference: {reference_model.upper()})")
        print("═" * 80)
        
        # Header
        header = f"{'Model':<15} {'Overall':<10}"
        metrics = ["Relevance", "Accuracy", "Groundedness", "Completeness", "BLEU", "ROUGE"]
        for metric in metrics:
            header += f"{metric:<12}"
        print(header)
        print("─" * 80)
        
        # Print each model's metrics
        for model_name, scores in sorted(model_metrics.items(), key=lambda x: x[1]["Overall"], reverse=True):
            row = f"{model_name:<15} {scores['Overall']:<10.3f}"
            for metric in metrics:
                row += f"{scores[metric]:<12.3f}"
            print(row)
    
    def _shorten_content(self, content, ratio=0.3):
        """Shorten content for quick evaluation while preserving structure"""
        lines = content.split('\n')
        sample_size = max(5, int(len(lines) * ratio))
        
        # Take some lines from beginning, middle and end
        beginning = lines[:sample_size // 3]
        middle_start = len(lines) // 2 - sample_size // 6
        middle = lines[middle_start:middle_start + sample_size // 3]
        end = lines[-(sample_size // 3):]
        
        return '\n'.join(beginning + ['...'] + middle + ['...'] + end)