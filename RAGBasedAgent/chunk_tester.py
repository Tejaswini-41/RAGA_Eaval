import os
import json
import time
import asyncio
import uuid
from typing import Dict, List, Optional
from datetime import datetime

# Import all chunking strategies
from hybrid_chunker import HybridSemanticChunker
from semantic_chunker import SemanticChunker
from fixed_size_chunker import FixedSizeChunker
from hierarchical_chunker import HierarchicalChunker
from chunked_review_generator import ChunkedReviewGenerator
from review_evaluator import ReviewEvaluator
from evaluation.metrics import MetricsCalculator

class ChunkingStrategyTester:
    """
    Tests and compares different chunking strategies using RAGAS metrics
    """
    
    def __init__(self):
        """Initialize the chunking strategy tester"""
        self.evaluator = ReviewEvaluator()
        self.metrics_calculator = MetricsCalculator()
        self.strategies = {
            "hybrid": {
                "name": "Hybrid Semantic Chunking",
                "description": "Combines fixed-size and semantic boundaries while respecting code structure",
                "chunker_class": HybridSemanticChunker,
                "results": {}
            },
            "semantic": {
                "name": "Pure Semantic Chunking",
                "description": "Chunks based on semantic boundaries like functions, classes, and code blocks",
                "chunker_class": SemanticChunker,
                "results": {}
            },
            "fixed": {
                "name": "Fixed Size Chunking",
                "description": "Divides text into chunks of equal size with overlap",
                "chunker_class": FixedSizeChunker, 
                "results": {}
            },
            "hierarchical": {
                "name": "Hierarchical Chunking",
                "description": "Creates a tree of chunks with parent-child relationships",
                "chunker_class": HierarchicalChunker,
                "results": {}
            }
        }
        
    async def test_all_strategies(self, 
                                current_pr_changes: str,
                                similar_prs_changes: List[Dict],
                                pr_number: int) -> Dict:
        """
        Test all chunking strategies and evaluate with RAGAS
        
        Args:
            current_pr_changes: Current PR content
            similar_prs_changes: List of similar PR changes
            pr_number: PR number being analyzed
            
        Returns:
            Dict with results from all strategies
        """
        print("\n🧪 Testing all chunking strategies with RAGAS metrics...")
        
        # Generate a baseline (non-chunked) review first to use as reference
        baseline_review = await self._generate_baseline_review(
            current_pr_changes,
            similar_prs_changes,
            pr_number
        )
        
        if not baseline_review:
            print("❌ Failed to generate baseline review")
            return {"success": False, "error": "Failed to generate baseline review"}
        
        # Track overall results including chunking statistics
        overall_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pr_number": pr_number,
            "baseline_review_length": len(baseline_review),
            "baseline_review": baseline_review[:1000] + "..." if len(baseline_review) > 1000 else baseline_review,
            "strategies": {},
            "best_strategy": None,
            "best_score": 0,
            "comparison_table": []
        }
        
        # Test each strategy
        for strategy_id, strategy_info in self.strategies.items():
            print(f"\n📋 Testing {strategy_info['name']}...")
            
            # Create a unique collection name for each strategy
            collection_name = f"{strategy_id}_pr{pr_number}_{uuid.uuid4().hex[:8]}"
            
            try:
                start_time = time.time()
                
                # Initialize chunker with unique collection name
                chunker = strategy_info["chunker_class"](collection_name=collection_name)
                
                # Create review generator with this chunker
                generator = ChunkedReviewGenerator(chunk_collection_name=collection_name, 
                                                chunking_strategy=strategy_id)
                
                # Process PR with this chunking strategy
                result = await generator.process_pr_with_chunking(
                    current_pr_changes, 
                    similar_prs_changes,
                    pr_number
                )
                
                processing_time = time.time() - start_time
                
                if not result.get("success"):
                    print(f"⚠️ {strategy_info['name']} failed: {result.get('error')}")
                    continue
                
                # Get the chunked review
                chunked_review = result.get("chunked_review")
                
                # Calculate RAGAS metrics compared to baseline
                print(f"📊 Calculating metrics for {strategy_info['name']}...")
                metrics = await self.evaluator._calculate_metrics(
                    baseline_review,
                    chunked_review
                )
                
                # Store results
                strategy_results = {
                    "review": chunked_review[:1000] + "..." if len(chunked_review) > 1000 else chunked_review,
                    "review_length": len(chunked_review),
                    "metrics": metrics,
                    "chunking_stats": result.get("chunking_stats"),
                    "processing_time": processing_time
                }
                
                # Check if this is the best strategy
                overall_score = metrics.get("Overall", 0)
                if overall_score > overall_results["best_score"]:
                    overall_results["best_score"] = overall_score
                    overall_results["best_strategy"] = strategy_id
                
                # Add to overall results
                overall_results["strategies"][strategy_id] = strategy_results
                
                # Add to comparison table
                overall_results["comparison_table"].append({
                    "strategy": strategy_id,
                    "name": strategy_info["name"],
                    "overall_score": overall_score,
                    "faithfulness": metrics.get("Faithfulness", 0),
                    "answer_relevancy": metrics.get("Answer Relevancy", 0),
                    "context_precision": metrics.get("Context Precision", 0),
                    "context_recall": metrics.get("Context Recall", 0),
                    "chunk_count": result.get("chunking_stats", {}).get("total_chunks", 0),
                    "processing_time": processing_time
                })
                
                print(f"✅ {strategy_info['name']} evaluation complete")
                
            except Exception as e:
                print(f"❌ Error testing {strategy_info['name']}: {str(e)}")
        
        # Sort comparison table by overall score
        overall_results["comparison_table"] = sorted(
            overall_results["comparison_table"], 
            key=lambda x: x["overall_score"], 
            reverse=True
        )
        
        # Generate recommendations based on results
        overall_results["recommendations"] = await self._generate_recommendations(
            overall_results["comparison_table"],
            pr_number
        )
        
        # Save results to file
        self._save_results(overall_results, pr_number)
        
        return {"success": True, "results": overall_results}
    
    async def _generate_baseline_review(self, 
                                      current_pr_changes: str,
                                      similar_prs_changes: List[Dict],
                                      pr_number: int) -> str:
        """Generate a baseline review without chunking to use as reference"""
        from review_generator import generate_review
        
        print("🔄 Generating baseline review (without chunking)...")
        
        # Use the best model according to evaluator
        best_model, _ = await self.evaluator.evaluate_models(
            current_pr_changes, 
            similar_prs_changes
        )
        
        # Get most similar PR number
        most_similar_pr = similar_prs_changes[0]['pr_number'] if similar_prs_changes else None
        
        # Generate baseline review
        baseline_review = generate_review(
            current_pr_changes,
            similar_prs_changes,
            pr_number=pr_number,
            similar_pr_number=most_similar_pr,
            model_name=best_model
        )
        
        return baseline_review
    
    async def _generate_recommendations(self, comparison_table: List[Dict], pr_number: int) -> str:
        """
        Generate recommendations based on chunking strategy comparison
        
        Args:
            comparison_table: Comparison data of all chunking strategies
            pr_number: PR number being analyzed
            
        Returns:
            Recommendation text with analysis
        """
        if not comparison_table:
            return "No chunking strategy comparisons available."
        
        # Best strategy is first one after sorting
        best_strategy = comparison_table[0]["strategy"]
        best_name = comparison_table[0]["name"]
        best_score = comparison_table[0]["overall_score"]
        
        # Generate recommendations using the evaluator's language model
        prompt = f"""
        Analyze the PR review chunking strategy comparison data and provide recommendations. 
        The data shows metrics for different chunking strategies for PR #{pr_number}.
        
        Comparison data:
        {json.dumps(comparison_table, indent=2)}
        
        Based on RAGAS metrics, the {best_name} strategy performed best with an overall score of {best_score:.3f}.
        
        Please provide:
        1. A brief analysis of why this strategy likely performed best for this PR
        2. When each chunking strategy might be most appropriate based on the data
        3. A concise recommendation on which chunking strategy to use for similar PRs in the future
        """
        
        # Use the gemini model for recommendations
        try:
            from models.model_factory import ModelFactory
            
            factory = ModelFactory()
            recommendations = factory.generate_response_with_prompt(
                "gemini",
                prompt,
                system_prompt="You are an AI assistant that specializes in analyzing code review systems and text chunking strategies. Provide clear, concise analysis based on the data provided."
            )
            
            return recommendations
        except Exception as e:
            print(f"⚠️ Error generating recommendations: {str(e)}")
            # Fallback to simple recommendation
            return f"Based on RAGAS metrics, the {best_name} performed best for PR #{pr_number} with an overall score of {best_score:.3f}."
    
    def _save_results(self, results: Dict, pr_number: int) -> str:
        """Save chunking comparison results to file"""
        results_dir = "chunking_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chunking_comparison_pr{pr_number}_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Chunking comparison results saved to {filepath}")
        return filepath
    
    def format_comparison_table(self, results: Dict) -> str:
        """
        Format comparison table as markdown for display
        
        Args:
            results: Chunking comparison results
            
        Returns:
            Markdown formatted comparison table
        """
        if not results.get("comparison_table"):
            return "No comparison data available."
        
        # Build markdown table
        table = "# Chunking Strategy Comparison\n\n"
        table += "| Strategy | Overall Score | Faithfulness | Answer Relevancy | Context Precision | Context Recall | # Chunks | Time (s) |\n"
        table += "|----------|--------------|------------|-----------------|------------------|---------------|----------|----------|\n"
        
        for row in results["comparison_table"]:
            table += f"| {row['name']} | **{row['overall_score']:.3f}** | {row['faithfulness']:.3f} | {row['answer_relevancy']:.3f} | "
            table += f"{row['context_precision']:.3f} | {row['context_recall']:.3f} | {row['chunk_count']} | {row['processing_time']:.2f} |\n"
        
        # Add recommendations
        table += "\n## Recommendations\n\n"
        table += results.get("recommendations", "No recommendations available.")
        
        return table

    async def run_chunking_test(self, current_pr_changes, similar_prs_changes, pr_number):
        """
        Execute a full chunking test and provide results
        
        Args:
            current_pr_changes: PR content to analyze
            similar_prs_changes: List of similar PR contents
            pr_number: PR number
            
        Returns:
            Dict with results
        """
        # Run the comparison
        results = await self.test_all_strategies(
            current_pr_changes, 
            similar_prs_changes, 
            pr_number
        )
        
        if not results.get("success"):
            print(f"❌ Chunking comparison failed: {results.get('error')}")
            return None

        # Generate formatted markdown report        
        formatted_report = self.format_comparison_table(results.get("results"))
        
        # Save report to file
        report_dir = "chunking_results"
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"chunking_report_pr{pr_number}_{timestamp}.md"
        report_path = os.path.join(report_dir, report_file)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(formatted_report)
        
        print(f"✅ Chunking comparison report saved to {report_path}")
        
        # Print the formatted table to console
        print("\n")
        print(formatted_report)
        
        return {
            "success": True,
            "results": results.get("results"),
            "report_path": report_path,
            "formatted_report": formatted_report
        }