import os
import sys
import time
import inspect  # Add this import
import pandas as pd
import numpy as np
from tabulate import tabulate
import asyncio

# Import MetricsCalculator from evaluation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from evaluation.metrics import MetricsCalculator

from .embedding_factory import EmbeddingFactory
from .base_embedder import BaseEmbedder
from .tfidf_embedder import TFIDFEmbedder

class EmbeddingEvaluator:
    """Evaluate different embedding methods for PR review"""
    
    def __init__(self):
        self.metrics_calculator = MetricsCalculator()
        self.evaluation_results = {}
        self.best_embedder = None
    
    async def evaluate_embedders(self, current_pr_data, similar_prs_data, 
                               embedder_types=None, generate_review=True):
        """
        Evaluate multiple embedding methods on PR similarity task
        
        Args:
            current_pr_data: Dict with current PR info
            similar_prs_data: Dict with similar PR info
            embedder_types: List of embedder types to evaluate
            generate_review: Whether to generate review after evaluation
            
        Returns:
            best_embedder: Best performing embedder
            evaluation_results: Dict of metrics for all embedders
        """
        print("\n📊 Evaluating different embedding methods...")
        
        # Get embedder types to evaluate
        if not embedder_types:
            embedder_types = list(EmbeddingFactory.get_available_embedders().keys())
        
        # Extract text content
        current_pr_text = current_pr_data.get("changes", "")
        similar_pr_text = similar_prs_data.get("changes", "")
        
        # Generate reference embedding with TFIDF (baseline)
        reference_embedder = TFIDFEmbedder()
        
        # Prepare documents for embeddings
        documents = [current_pr_text, similar_pr_text]
        
        # Create and evaluate each embedder
        for embedder_type in embedder_types:
            print(f"\n• Testing {embedder_type} embedder...")
            
            # Create embedder
            embedder = EmbeddingFactory.get_embedder(embedder_type)
            
            # Measure embedding time
            start_time = time.time()
            embeddings = embedder(documents)
            embedding_time = time.time() - start_time
            
            # Calculate metrics
            metrics = await self._calculate_metrics(current_pr_text, similar_pr_text, embedder_type)
            
            # Add time metrics
            metrics["embedding_time"] = embedding_time
            
            # Store results
            self.evaluation_results[embedder_type] = metrics
        
        # Find best embedder based on weighted metrics
        self.best_embedder = self._select_best_embedder()
        
        # Display results
        self._display_comparison_table()
        
        return self.best_embedder, self.evaluation_results
    
    # Fix the _calculate_metrics function to properly handle both async and non-async functions

    async def _calculate_metrics(self, current_pr_text, similar_pr_text, embedder_type):
        """Calculate RAGAS metrics for embedder evaluation using existing MetricsCalculator"""
        try:
            # Compute metrics using the MetricsCalculator - handle non-async methods correctly
            relevance = self.metrics_calculator.compute_relevance(current_pr_text, similar_pr_text)
            
            # For groundedness and completeness, check if they're actually coroutines or coroutine functions
            try:
                # Check if the groundedness function is async
                import inspect
                if inspect.iscoroutinefunction(self.metrics_calculator.compute_groundedness):
                    groundedness = await self.metrics_calculator.compute_groundedness(current_pr_text, similar_pr_text)
                else:
                    # If not async, call directly
                    groundedness = self.metrics_calculator.compute_groundedness(current_pr_text, similar_pr_text)
            except Exception as e:
                print(f"Error computing groundedness: {e}")
                groundedness = 0.5  # Default value
        
            try:
                # Check if the completeness function is async
                if inspect.iscoroutinefunction(self.metrics_calculator.compute_completeness):
                    completeness = await self.metrics_calculator.compute_completeness(current_pr_text, similar_pr_text)
                else:
                    # If not async, call directly
                    completeness = self.metrics_calculator.compute_completeness(current_pr_text, similar_pr_text)
            except Exception as e:
                print(f"Error computing completeness: {e}")
                completeness = 0.5  # Default value
        
            # Regular metrics
            try:
                faithfulness = self.metrics_calculator.compute_faithfulness(current_pr_text, similar_pr_text)
            except Exception as e:
                print(f"Error computing faithfulness: {e}")
                faithfulness = 0.5  # Default value
                
            try:
                answer_relevance = self.metrics_calculator.compute_answer_relevance(current_pr_text, similar_pr_text)
            except Exception as e:
                print(f"Error computing answer relevance: {e}")
                answer_relevance = 0.5  # Default value
        
            # BLEU and ROUGE
            try:
                if inspect.iscoroutinefunction(self.metrics_calculator.compute_bleu_ragas):
                    bleu = await self.metrics_calculator.compute_bleu_ragas(current_pr_text, similar_pr_text)
                else:
                    bleu = self.metrics_calculator.compute_bleu_ragas(current_pr_text, similar_pr_text)
            except Exception as e:
                print(f"Error computing BLEU: {e}")
                bleu = 0.5  # Default value
            
            try:
                if inspect.iscoroutinefunction(self.metrics_calculator.compute_rouge_ragas):
                    rouge = await self.metrics_calculator.compute_rouge_ragas(current_pr_text, similar_pr_text)
                else:
                    rouge = self.metrics.calculator.compute_rouge_ragas(current_pr_text, similar_pr_text)
            except Exception as e:
                print(f"Error computing ROUGE: {e}")
                rouge = 0.5  # Default value
        
            # Calculate embedding time for each method
            import time
            start_time = time.time()
            # Use the imported EmbeddingFactory
            embedder = EmbeddingFactory.get_embedder(embedder_type)
            documents = [current_pr_text, similar_pr_text]
            _ = embedder(documents)
            embedding_time = time.time() - start_time
        
            metrics = {
                "Relevance": relevance,
                "Groundedness": groundedness,
                "Completeness": completeness, 
                "Faithfulness": faithfulness,
                "AnswerRelevance": answer_relevance,
                "BLEU": bleu,
                "ROUGE": rouge,
                "EmbeddingTime": embedding_time,
                "Average": (relevance + groundedness + completeness + bleu + rouge + faithfulness + answer_relevance) / 7
            }
            
            return metrics
        
        except Exception as e:
            print(f"❌ Error calculating metrics for {embedder_type}: {e}")
            import traceback
            traceback.print_exc()
            # Return default metrics
            return {"Relevance": 0.5, "Groundedness": 0.5, "Completeness": 0.5, 
                   "Faithfulness": 0.5, "AnswerRelevance": 0.5, "BLEU": 0.5, "ROUGE": 0.5,
                   "EmbeddingTime": 0.0, "Average": 0.5}
    
    def _select_best_embedder(self):
        """Select the best embedder based on metrics"""
        if not self.evaluation_results:
            print("❌ No evaluation results available")
            return "tfidf"  # Default to TF-IDF
        
        # Calculate weighted scores
        weighted_scores = {}
        for embedder_type, metrics in self.evaluation_results.items():
            # Use average score for simple ranking
            weighted_scores[embedder_type] = metrics.get("Average", 0)
        
        # Find embedder with highest score
        best_embedder = max(weighted_scores.items(), key=lambda x: x[1])[0]
        
        print(f"\n✅ Best embedder: {best_embedder} with score {weighted_scores[best_embedder]:.4f}")
        return best_embedder

    def _display_comparison_table(self):
        """Display a comparison table of embedder metrics"""
        if not self.evaluation_results:
            print("❌ No evaluation results to display")
            return

        print("\n📊 Embedding Methods Comparison:")
        print("=" * 100)

        # Define metrics columns with shortened names
        metrics = [
            "Overall", "Relev", "Ground", "Compl", "Faith", 
            "AnsRel", "BLEU", "ROUGE", "Time"
        ]
        
        metrics_mapping = {
            "Overall": "Average",
            "Relev": "Relevance", 
            "Ground": "Groundedness", 
            "Compl": "Completeness", 
            "Faith": "Faithfulness",
            "AnsRel": "AnswerRelevance", 
            "BLEU": "BLEU", 
            "ROUGE": "ROUGE", 
            "Time": "EmbeddingTime"
        }

        # Print header
        header = "| {:<15} |".format("Embedder")
        for metric in metrics:
            header += " {:>6} |".format(metric)
        print(header)
        print("-" * len(header))

        # Print each embedder's metrics
        for embedder_type, embedder_metrics in self.evaluation_results.items():
            # Prepare embedder name with BEST label if applicable
            embedder_name = f"{embedder_type} (BEST)" if embedder_type == self.best_embedder else embedder_type
            row = "| {:<15} |".format(embedder_name[:15])

            # Add each metric value
            for metric_short in metrics:
                metric_full = metrics_mapping.get(metric_short)
                value = embedder_metrics.get(metric_full, 0.0)
                row += " {:>6.3f} |".format(value)
            print(row)

        print("=" * len(header))

        # Print summary
        print(f"\n🏆 Best Overall Embedder: {self.best_embedder}")

        # Show key strengths of best embedder
        if self.best_embedder and self.best_embedder in self.evaluation_results:
            print("\nKey Strengths:")
            best_metrics = self.evaluation_results[self.best_embedder]
            baseline_metrics = self.evaluation_results.get('tfidf', {})
            
            found_strengths = False
            for short_name, full_name in metrics_mapping.items():
                if full_name in best_metrics and full_name in baseline_metrics:
                    value = best_metrics[full_name]
                    baseline = baseline_metrics[full_name]
                    if value > baseline and baseline > 0:
                        found_strengths = True
                        improvement = ((value - baseline) / baseline * 100)
                        print(f"• {short_name}: +{improvement:.1f}% vs baseline")
            
            if not found_strengths:
                print("• No significant improvements over baseline")

        print("=" * 120)