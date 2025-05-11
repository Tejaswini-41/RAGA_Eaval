import os
import json
import pandas as pd
from datetime import datetime
# import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from models.groq_models import GroqModelFactory
from review_evaluator import ReviewEvaluator
import time

class ImprovementAnalyzer:
    """
    Analyzes RAG system performance metrics and suggests improvements
    based on model performance, prompt strategies, and chunking results.
    """
    
    def __init__(self, results_dir="RAG_based_Analysis_2"):
        self.results_dir = results_dir
        self.evaluator = ReviewEvaluator()
        self.best_model = None
        self.model = None
        self.session_data = {}
        
    async def analyze_and_suggest_improvements(self, session_id: str = None) -> str:
        """
        Analyze system performance and provide actionable suggestions
        
        Args:
            session_id: Current session ID for loading specific results
            
        Returns:
            str: Markdown-formatted improvement suggestions
        """
        print("\n🔍 Analyzing RAG system performance...")
        
        if not session_id:
            print("⚠ No session ID provided. Analysis may be incomplete.")
        else:
            print(f"📂 Loading session data for: {session_id}")
            
        # Check for cached analysis results
        cache_file = f"recommendations/cache_{session_id}.json"
        if os.path.exists(cache_file) and os.path.getmtime(cache_file) > time.time() - 3600:  # Cache valid for 1 hour
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    print("✅ Using cached analysis results")
                    return f.read()
            except:
                pass  # Continue with regular analysis if cache read fails
        
        # Load all results for the session
        results = self._load_session_results(session_id)
        if not results:
            return "No results found to analyze. Please run reviews first."
            
        # Load session-specific metadata
        self._load_session_metadata(session_id, results)
        
        # Find the best model based on metrics
        self.best_model = self._determine_best_model(results)
        
        # Initialize the best model for suggestion generation
        if self.best_model:
            self.model = GroqModelFactory.create_model(self.best_model)
        else:
            self.model = GroqModelFactory.create_model("llama")  # Fallback to llama
        
        # Extract key metrics from the results
        metrics_summary = self._extract_metrics_summary(results)
        
        # Analyze prompt effectiveness
        prompt_analysis = self._analyze_prompt_effectiveness(results)
        
        # Analyze chunking effectiveness
        chunking_analysis = self._analyze_chunking_effectiveness(results)
        
        # Generate improvement suggestions using the best model
        improvement_suggestions = await self._generate_improvement_suggestions(
            metrics_summary,
            prompt_analysis,
            chunking_analysis
        )
        
        # Format the final report
        report = self._format_improvement_report(
            metrics_summary, 
            prompt_analysis,
            chunking_analysis,
            improvement_suggestions
        )
        
        # Save the report
        self._save_improvement_report(report, session_id)
        
        return report
    
    def _load_session_metadata(self, session_id: str, results: List[Dict]) -> None:
        """
        Load session-specific metadata from results
        
        This extracts key information about the session, such as:
        - Session start time
        - PR information
        - Model configurations
        - Applied strategies
        """
        if not session_id or not results:
            return
            
        # Extract session metadata from all results
        self.session_data = {
            "session_id": session_id,
            "timestamp": None,
            "pr_files": [],
            "chunking_applied": False,
            "confidence_scores_applied": False,
            "enhanced_prompt_applied": False,
            "strategies_tested": [],
        }
        
        # Process all results to extract session info
        for result in results:
            # Get timestamp from first result (earliest)
            if not self.session_data["timestamp"] and "timestamp" in result:
                self.session_data["timestamp"] = result["timestamp"]
                
            # Get PR files
            if "pr_files" in result and not self.session_data["pr_files"]:
                self.session_data["pr_files"] = result["pr_files"]
                
            # Track applied strategies
            if "chunking_applied" in result:
                self.session_data["chunking_applied"] = True
                
            if "enhanced_system_prompt" in result:
                self.session_data["enhanced_prompt_applied"] = True
                
            # Check for confidence scores by looking for confidence review files
            if result.get('filename', '').startswith('confidence_enhanced_review'):
                self.session_data["confidence_scores_applied"] = True
                
            # Track tested strategies
            strategy_name = result.get("chunking_strategy", "")
            if strategy_name and strategy_name not in self.session_data["strategies_tested"]:
                self.session_data["strategies_tested"].append(strategy_name)
        
        print(f"📊 Session {session_id} metadata loaded:")
        print(f"  - Timestamp: {self.session_data['timestamp']}")
        print(f"  - PR Files: {len(self.session_data['pr_files'])} files")
        print(f"  - Strategies applied: " + 
              f"{'Chunking ' if self.session_data['chunking_applied'] else ''}" +
              f"{'Enhanced Prompt ' if self.session_data['enhanced_prompt_applied'] else ''}" +
              f"{'Confidence Scores' if self.session_data['confidence_scores_applied'] else ''}")
    
    def _load_session_results(self, session_id: str = None) -> List[Dict]:
        """
        Load all results files for a specific session
        
        Ensures that only files belonging to the specified session are loaded,
        maintaining data isolation between different analysis runs.
        """
        all_results = []
        
        try:
            if not os.path.exists(self.results_dir):
                print(f"⚠ Results directory not found: {self.results_dir}")
                return all_results
                
            for filename in os.listdir(self.results_dir):
                if not filename.endswith('.json'):
                    continue
                    
                # Only process files for this session when session_id is provided
                if session_id and session_id not in filename:
                    continue
                
                filepath = os.path.join(self.results_dir, filename)
                
                with open(filepath, 'r', encoding='utf-8') as file:
                    try:
                        data = json.load(file)
                        
                        # Add filename for reference and verify session ID match
                        data['filename'] = filename
                        
                        # Skip files that don't match our session ID
                        if session_id and data.get('session_id') and data['session_id'] != session_id:
                            continue
                            
                        all_results.append(data)
                    except json.JSONDecodeError:
                        print(f"⚠ Could not parse JSON from {filename}")
        
        except Exception as e:
            print(f"Error loading results: {e}")
        
        print(f"📊 Loaded {len(all_results)} result files for session {session_id}")
        return all_results
    
    def _determine_best_model(self, results: List[Dict]) -> Optional[str]:
        """Determine the best performing model based on metrics"""
        best_model = None
        best_score = 0
        
        for result in results:
            if "baseline_metrics" in result:
                metrics = result["baseline_metrics"]
                for model, model_metrics in metrics.items():
                    overall_score = model_metrics.get("Overall", 0)
                    if overall_score > best_score:
                        best_score = overall_score
                        best_model = model
            
            # Also check enhanced metrics if available
            if "metrics_comparison" in result and "enhanced" in result["metrics_comparison"]:
                enhanced = result["metrics_comparison"]["enhanced"]
                for model, model_metrics in enhanced.items():
                    overall_score = model_metrics.get("Overall", 0)
                    if overall_score > best_score:
                        best_score = overall_score
                        best_model = model
        
        if best_model:
            print(f"🏆 Best performing model: {best_model.upper()} (Score: {best_score:.3f})")
        
        return best_model
    
    def _extract_metrics_summary(self, results: List[Dict]) -> Dict:
        """Extract key metrics from all results"""
        summary = {
            "models": {},
            "metrics": {
                "baseline": {},
                "enhanced": {}
            },
            "overall_best": {
                "model": None,
                "score": 0
            }
        }
        
        # First pass - collect all metrics
        for result in results:
            # Process baseline metrics
            if "baseline_metrics" in result:
                for model, metrics in result["baseline_metrics"].items():
                    if model not in summary["metrics"]["baseline"]:
                        summary["metrics"]["baseline"][model] = {}
                    
                    for metric_name, metric_value in metrics.items():
                        if metric_name not in summary["metrics"]["baseline"][model]:
                            summary["metrics"]["baseline"][model][metric_name] = []
                        summary["metrics"]["baseline"][model][metric_name].append(metric_value)
            
            # Process enhanced metrics if available
            if "metrics_comparison" in result and "enhanced" in result["metrics_comparison"]:
                enhanced = result["metrics_comparison"]["enhanced"]
                for model, metrics in enhanced.items():
                    if model not in summary["metrics"]["enhanced"]:
                        summary["metrics"]["enhanced"][model] = {}
                    
                    for metric_name, metric_value in metrics.items():
                        if metric_name not in summary["metrics"]["enhanced"][model]:
                            summary["metrics"]["enhanced"][model][metric_name] = []
                        summary["metrics"]["enhanced"][model][metric_name].append(metric_value)
        
        # Second pass - calculate averages
        for phase in ["baseline", "enhanced"]:
            for model in summary["metrics"][phase]:
                for metric in summary["metrics"][phase][model]:
                    values = summary["metrics"][phase][model][metric]
                    if values:
                        avg_value = sum(values) / len(values)
                        # Replace list with average for cleaner output
                        summary["metrics"][phase][model][metric] = round(avg_value, 3)
        
        # Find overall best model and score
        for phase in ["baseline", "enhanced"]:
            for model, metrics in summary["metrics"][phase].items():
                overall = metrics.get("Overall", 0)
                if overall > summary["overall_best"]["score"]:
                    summary["overall_best"]["score"] = overall
                    summary["overall_best"]["model"] = model
        
        return summary
    
    def _analyze_prompt_effectiveness(self, results: List[Dict]) -> Dict:
        """Analyze the effectiveness of different prompt strategies"""
        analysis = {
            "baseline_vs_enhanced": {},
            "prompt_improvements": {},
            "prompt_patterns": []
        }
        
        # Extract and compare baseline vs enhanced metrics
        for result in results:
            if "metrics_comparison" in result and "enhanced" in result["metrics_comparison"]:
                baseline = result["metrics_comparison"].get("baseline", {})
                enhanced = result["metrics_comparison"].get("enhanced", {})
                
                for model in enhanced:
                    if model in baseline:
                        if model not in analysis["baseline_vs_enhanced"]:
                            analysis["baseline_vs_enhanced"][model] = {}
                            
                        for metric in enhanced[model]:
                            if metric in baseline[model]:
                                baseline_value = baseline[model][metric]
                                enhanced_value = enhanced[model][metric]
                                
                                if metric not in analysis["baseline_vs_enhanced"][model]:
                                    analysis["baseline_vs_enhanced"][model][metric] = {
                                        "baseline": baseline_value,
                                        "enhanced": enhanced_value,
                                        "change": enhanced_value - baseline_value,
                                        "percent": ((enhanced_value - baseline_value) / max(0.001, baseline_value)) * 100
                                    }
        
        # Extract key patterns from successful prompt improvements
        success_patterns = []
        
        for result in results:
            if "original_system_prompt" in result and "enhanced_system_prompt" in result:
                original = result.get("original_system_prompt", "")
                enhanced = result.get("enhanced_system_prompt", "")
                
                # Simple pattern extraction - identify added keywords/phrases
                # This is a basic implementation - could be improved with NLP techniques
                original_words = set(original.lower().split())
                enhanced_words = set(enhanced.lower().split())
                
                new_words = enhanced_words - original_words
                for word in new_words:
                    if len(word) > 5:  # Filter out short words
                        success_patterns.append(word)
                        
        # Extract most common patterns
        from collections import Counter
        pattern_counts = Counter(success_patterns)
        analysis["prompt_patterns"] = [
            {"pattern": pattern, "count": count} 
            for pattern, count in pattern_counts.most_common(10)
        ]
                
        return analysis
    
    def _analyze_chunking_effectiveness(self, results: List[Dict]) -> Dict:
        """Analyze the effectiveness of different chunking strategies"""
        analysis = {
            "strategies_compared": [],
            "best_strategy": None,
            "metrics_impact": {},
            "recommendations": []
        }
        
        # Look for chunking-specific metrics in the results
        for result in results:
            if "chunked_metrics" in result and "baseline_metrics" in result:
                strategy = {
                    "name": result.get("chunking_strategy", "Default Chunking"),
                    "baseline": result["baseline_metrics"].get("Overall", 0),
                    "chunked": result["chunked_metrics"].get("Overall", 0),
                    "improvement": result["chunked_metrics"].get("Overall", 0) - 
                                 result["baseline_metrics"].get("Overall", 0)
                }
                
                analysis["strategies_compared"].append(strategy)
                
                # Track impact on specific metrics
                for metric in result["chunked_metrics"]:
                    if metric != "Overall" and metric in result["baseline_metrics"]:
                        if metric not in analysis["metrics_impact"]:
                            analysis["metrics_impact"][metric] = []
                            
                        impact = result["chunked_metrics"][metric] - result["baseline_metrics"][metric]
                        analysis["metrics_impact"][metric].append(impact)
        
        # Determine best strategy
        if analysis["strategies_compared"]:
            analysis["best_strategy"] = max(
                analysis["strategies_compared"], 
                key=lambda x: x["improvement"]
            )
        
        # Average impact on each metric
        for metric, impacts in analysis["metrics_impact"].items():
            if impacts:
                analysis["metrics_impact"][metric] = sum(impacts) / len(impacts)
            else:
                analysis["metrics_impact"][metric] = 0
        
        return analysis
    
    async def _generate_improvement_suggestions(
        self, 
        metrics_summary: Dict, 
        prompt_analysis: Dict,
        chunking_analysis: Dict
    ) -> Dict:
        """Generate AI-based improvement suggestions using the best model"""
        
        if not self.model:
            return {"error": "Model not initialized"}
        
        # Prepare input for the model
        input_data = {
            "metrics_summary": metrics_summary,
            "prompt_analysis": prompt_analysis,
            "chunking_analysis": chunking_analysis,
            "session_data": self.session_data
        }
        
        # Convert to formatted string for the model
        input_str = json.dumps(input_data, indent=2)
        
        # Generate the prompt
        prompt = f"""You are an expert in Retrieval-Augmented Generation (RAG) systems and RAGAS evaluation metrics.
Based on the following performance data from a RAG-based PR review system for session {self.session_data.get('session_id', 'unknown')}, 
provide specific, actionable suggestions for improvements in these areas:

1. Model Selection
2. Prompt Engineering
3. Chunking Strategies
4. Overall RAG Pipeline Optimization

Here is the performance data:
{input_str}

Focus on practical, implementation-ready suggestions that will maximize:
- Relevance of retrieved information
- Accuracy of generated reviews
- Groundedness in the source code
- Completeness of analysis
- Contextual precision
- Answer relevance

Format your response as JSON with these sections:
{{"model_selection": [...], "prompt_engineering": [...], "chunking_strategies": [...], "pipeline_optimization": [...]}}

Each suggestion should be specific, actionable, and clearly explain the expected benefit.
"""
        
        # Generate suggestions
        try:
            response = self.model.generate_response(prompt)
            try:
                suggestions = json.loads(response)
            except json.JSONDecodeError:
                print("⚠ Model response was not valid JSON, using fallback suggestions")
                # Use fallback suggestions
            return suggestions
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            # Fallback to manual suggestions
            return {
                "model_selection": ["Consider evaluating more models to find the optimal balance of performance and cost"],
                "prompt_engineering": ["Increase specificity in prompts by including file and function references"],
                "chunking_strategies": ["Adjust chunk size based on the nature of the content: larger for code, smaller for documentation"],
                "pipeline_optimization": ["Implement caching for similar PRs to reduce processing time"]
            }
    
    def _format_improvement_report(
        self, 
        metrics_summary: Dict, 
        prompt_analysis: Dict,
        chunking_analysis: Dict,
        suggestions: Dict
    ) -> str:
        """Format the final improvement report as markdown"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add session-specific information to the report header
        session_info = ""
        if self.session_data:
            session_id = self.session_data.get("session_id", "Unknown")
            session_time = self.session_data.get("timestamp", "Unknown")
            strategies = []
            if self.session_data.get("chunking_applied"):
                strategies.append("Chunking")
            if self.session_data.get("enhanced_prompt_applied"):
                strategies.append("Enhanced Prompts")
            if self.session_data.get("confidence_scores_applied"):
                strategies.append("Confidence Scoring")
            
            strategies_text = ", ".join(strategies) if strategies else "None"
            
            session_info = f"""
## 📋 Session Details

- *Session ID*: {session_id}
- *Started*: {session_time}
- *Strategies Applied*: {strategies_text}
- *PR Files*: {len(self.session_data.get("pr_files", []))} files
"""
        
        report = f"""# 🚀 RAG System Improvement Analysis

Generated on: {timestamp}
{session_info}

## 📊 Performance Summary

### Model Performance
| Model | Overall Score | Relevance | Accuracy | Completeness | Groundedness |
|-------|--------------|-----------|----------|-------------|--------------|
"""
        
        # Add model performance data
        for model, metrics in metrics_summary.get("metrics", {}).get("baseline", {}).items():
            overall = metrics.get("Overall", 0)
            relevance = metrics.get("Relevance", 0)
            accuracy = metrics.get("Accuracy", 0)
            completeness = metrics.get("Completeness", 0)
            groundedness = metrics.get("Groundedness", 0)
            
            report += f"| {model.upper()} | {overall:.3f} | {relevance:.3f} | {accuracy:.3f} | {completeness:.3f} | {groundedness:.3f} |\n"
        
        # Best model
        best_model = metrics_summary.get("overall_best", {}).get("model", "")
        best_score = metrics_summary.get("overall_best", {}).get("score", 0)
        
        report += f"\n*Best Performing Model:* {best_model.upper() if best_model else 'Unknown'} (Score: {best_score:.3f})\n"
        
        # Prompt effectiveness
        report += f"""
## 📝 Prompt Engineering Analysis

### Baseline vs Enhanced Performance
"""
        
        for model, metrics in prompt_analysis.get("baseline_vs_enhanced", {}).items():
            report += f"\n#### {model.upper()}\n"
            report += "| Metric | Baseline | Enhanced | Change | % Change |\n"
            report += "|--------|----------|----------|--------|----------|\n"
            
            for metric, values in metrics.items():
                baseline = values.get("baseline", 0)
                enhanced = values.get("enhanced", 0)
                change = values.get("change", 0)
                percent = values.get("percent", 0)
                
                report += f"| {metric} | {baseline:.3f} | {enhanced:.3f} | {change:+.3f} | {percent:+.1f}% |\n"
        
        # Chunking effectiveness
        report += f"""
## 🧩 Chunking Strategy Analysis

### Impact on Metrics
"""
        
        # Only add this section if we have chunking data
        if chunking_analysis.get("metrics_impact"):
            report += "| Metric | Average Impact |\n"
            report += "|--------|---------------|\n"
            
            for metric, impact in chunking_analysis.get("metrics_impact", {}).items():
                report += f"| {metric} | {impact:+.3f} |\n"
        
            # Best chunking strategy
            best_strategy = chunking_analysis.get("best_strategy", {})
            if best_strategy:
                name = best_strategy.get("name", "Unknown")
                improvement = best_strategy.get("improvement", 0)
                
                report += f"\n*Best Chunking Strategy:* {name} (Improvement: {improvement:+.3f})\n"
        else:
            report += "No chunking strategy comparison data available\n"
        
        # Improvement suggestions
        report += f"""
## 💡 Improvement Suggestions

### Model Selection
"""
        
        for suggestion in suggestions.get("model_selection", []):
            report += f"- {suggestion}\n"
            
        report += f"""
### Prompt Engineering
"""

        for suggestion in suggestions.get("prompt_engineering", []):
            report += f"- {suggestion}\n"
            
        report += f"""
### Chunking Strategies
"""

        for suggestion in suggestions.get("chunking_strategies", []):
            report += f"- {suggestion}\n"
            
        report += f"""
### Pipeline Optimization
"""

        for suggestion in suggestions.get("pipeline_optimization", []):
            report += f"- {suggestion}\n"
        
        return report
    
    def _save_improvement_report(self, report: str, session_id: str = None) -> str:
        """Save the improvement report to file"""
        os.makedirs("recommendations", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_part = f"_{session_id}" if session_id else ""
        filename = f"recommendations/improvement_analysis{session_part}_{timestamp}.md"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"✅ Improvement analysis saved to {filename}")
        return filename

    def visualize_metrics(self, session_id: str = None, output_file: str = None):
        """Generate visualizations of metrics for better analysis"""
        # This could be implemented to generate charts for the session's metrics
        pass

async def analyze_improvements(session_id: str = None) -> str:
    """Run the improvement analysis process"""
    analyzer = ImprovementAnalyzer()
    report = await analyzer.analyze_and_suggest_improvements(session_id)
    return report

if __name__ == "__main__":
    # If run directly, execute the analysis
    import asyncio
    import sys
    
    # Allow optional session ID argument when run directly
    session_id = sys.argv[1] if len(sys.argv) > 1 else None
    report = asyncio.run(analyze_improvements(session_id))
    print(report)