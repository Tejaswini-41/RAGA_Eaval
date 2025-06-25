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

        # Extract prompts and store as attributes
        self.original_prompt, self.enhanced_prompt = self._extract_prompts(results)
        
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
        
        # Analyze embedding effectiveness
        embedding_analysis = self._analyze_embedding_effectiveness(results)
        
        # Format the final report with all analyses
        report = self._format_improvement_report(
            metrics_summary, 
            prompt_analysis,
            chunking_analysis,
            embedding_analysis
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
    
    def _extract_prompts(self, results: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract the original and enhanced system prompts from the session results.
        Returns a tuple: (original_prompt, enhanced_prompt)
        """
        original_prompt = None
        enhanced_prompt = None
        for result in results:
            if not original_prompt and "original_system_prompt" in result:
                original_prompt = result["original_system_prompt"]
            if not enhanced_prompt and "enhanced_system_prompt" in result:
                enhanced_prompt = result["enhanced_system_prompt"]
            if original_prompt and enhanced_prompt:
                break
        return original_prompt, enhanced_prompt


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

        # First try to get metrics from stored results
        for result in results:
            if "chunking_metrics_comparison" in result:
                stored_metrics = result["chunking_metrics_comparison"]
                strategies = stored_metrics["metrics_table"]["strategies"]

                # Process each strategy's metrics
                for strategy_name, metrics in strategies.items():
                    if strategy_name != "Baseline":
                        strategy_data = {
                            "name": strategy_name,
                            "baseline": strategies["Baseline"].get("Overall", 0),
                            "chunked": metrics.get("Overall", 0),
                            "improvement": metrics.get("Overall", 0) - strategies["Baseline"].get("Overall", 0)
                        }
                        analysis["strategies_compared"].append(strategy_data)

                        # Track metric-specific improvements
                        for metric in metrics:
                            if metric != "Overall":
                                if metric not in analysis["metrics_impact"]:
                                    analysis["metrics_impact"][metric] = []
                                impact = metrics[metric] - strategies["Baseline"].get(metric, 0)
                                analysis["metrics_impact"][metric].append(impact)

        # Continue with existing analysis if no stored metrics found
        if not analysis["strategies_compared"]:
            # ... existing analysis code ...
            pass

        return analysis
    
    def _analyze_embedding_effectiveness(self, results: List[Dict]) -> Dict:
        """Analyze the effectiveness of different embedding strategies"""
        analysis = {
            "strategies_compared": [],
            "best_strategy": None,
            "metrics_impact": {},
            "performance_summary": {}
        }
        
        # Look for embedding-specific metrics in the results
        for result in results:
            if "embedding_evaluation" in result:
                evaluation = result["embedding_evaluation"]
                
                # Track strategies and their metrics
                for strategy, metrics in evaluation.items():
                    strategy_data = {
                        "name": strategy,
                        "relevance": metrics.get("Relevance", 0),
                        "groundedness": metrics.get("Groundedness", 0),
                        "completeness": metrics.get("Completeness", 0),
                        "answer_relevance": metrics.get("AnswerRelevance", 0),
                        "overall_score": metrics.get("Average", 0),
                        "embedding_time": metrics.get("EmbeddingTime", 0)
                    }
                    
                    analysis["strategies_compared"].append(strategy_data)
                    
                    # Track metric-specific performance
                    for metric, value in metrics.items():
                        if metric not in analysis["metrics_impact"]:
                            analysis["metrics_impact"][metric] = []
                        analysis["metrics_impact"][metric].append(value)
        
        # Determine best strategy
        if analysis["strategies_compared"]:
            analysis["best_strategy"] = max(
                analysis["strategies_compared"],
                key=lambda x: x["overall_score"]
            )
            
            # Calculate average performance for each metric
            for metric, values in analysis["metrics_impact"].items():
                if values:
                    analysis["performance_summary"][metric] = {
                        "average": sum(values) / len(values),
                        "best": max(values),
                        "worst": min(values)
                    }
        
        return analysis


    def _suggest_optimal_combinations(self, chunking_analysis: Dict, embedding_analysis: Dict) -> List[Dict]:
        """Get optimal combinations of chunking and embedding strategies using Groq's Llama model"""
        import httpx
        from dotenv import load_dotenv
        import os
        import json

        print("\n🔄 Generating optimal strategy combinations...")

        # Load Groq API key
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ No Groq API key found in .env file")
            return []

        print(f"📊 Found {len(chunking_analysis.get('strategies_compared', []))} chunking strategies")
        print(f"📊 Found {len(embedding_analysis.get('strategies_compared', []))} embedding strategies")

        if not chunking_analysis.get('strategies_compared') or not embedding_analysis.get('strategies_compared'):
            print("⚠ No strategies found to analyze")
            return []

        # Create a more structured prompt
        system_prompt = """As an expert in RAG systems analysis, analyze the following performance metrics and suggest optimal strategy combinations.

    Performance data summary:

    CHUNKING STRATEGIES:
    """
        # Add chunking data
        chunking_strategies = chunking_analysis.get("strategies_compared", [])
        if chunking_strategies:
            for strategy in chunking_strategies:
                system_prompt += f"""
    - {strategy['name']}:
    * Overall Score: {strategy.get('overall_score', 0):.3f}
    * Faithfulness: {strategy.get('faithfulness', 0):.3f}
    * Context Precision: {strategy.get('context_precision', 0):.3f}
    * Answer Relevance: {strategy.get('answer_relevance', 0):.3f}
    * Processing Time: {strategy.get('processing_time', 0):.2f}s
    """

        system_prompt += "\nEMBEDDING STRATEGIES:"
        
        # Add embedding data
        embedding_strategies = embedding_analysis.get("strategies_compared", [])
        if embedding_strategies:
            for strategy in embedding_strategies:
                system_prompt += f"""
    - {strategy['name']}:
    * Overall Score: {strategy.get('overall_score', 0):.3f}
    * Relevance: {strategy.get('relevance', 0):.3f}
    * Groundedness: {strategy.get('groundedness', 0):.3f}
    * Embedding Time: {strategy.get('embedding_time', 0):.2f}s
    """

        system_prompt += """
    Based on these metrics, provide exactly 2 optimal combinations of chunking method, embedding method, and prompt type (enhanced or original) that together will give the highest RAGAS score.
    For each combination, specify:
    - Chunking: [strategy name]
    - Embedding: [method name]
    - Prompt: [enhanced/original]
    - Reasoning: [explain why this combination works well and why this prompt type is best]

    Use this exact format for your response:

    COMBINATION 1:
    - Chunking: [strategy name]
    - Embedding: [method name]
    - Prompt: [enhanced/original]
    - Reasoning: [detailed explanation]

    COMBINATION 2:
    - Chunking: [strategy name]
    - Embedding: [method name]
    - Prompt: [enhanced/original]
    - Reasoning: [detailed explanation]

    Prioritize combinations that prompt effectiveness, computational costs and give good ragas score.
    """

        # Setup Groq API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Updated payload with different model and clearer instructions
        payload = {
            "model": "llama3-70b-8192",  # Using the base model instead of guard
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert RAG system analyst. Provide clear, structured recommendations."
                },
                {
                    "role": "user",
                    "content": system_prompt
                }
            ],
            "temperature": 0.2,  # Lower temperature for more consistent output
            "max_tokens": 1000,
            "top_p": 0.95,
            "stream": False
        }

        try:
            response = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code != 200:
                print(f"❌ API request failed: {response.status_code}")
                print(f"Error: {response.text}")
                return []

            result = response.json()
            suggestions = result["choices"][0]["message"]["content"]
            
            print(f"\n✅ Received response from Llama: {len(suggestions)} characters")

            # Improved parsing logic
            combinations = []
            current_combo = {}
            current_section = None
            
            for line in suggestions.split('\n'):
                line = line.strip()
                if not line:
                    continue

                if line.startswith('COMBINATION'):
                    if current_combo:
                        combinations.append(current_combo)
                    current_combo = {'id': line}
                    continue

                if line.startswith('- '):
                    parts = line[2:].split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].lower().strip()
                        value = parts[1].strip()
                        if key in ['chunking', 'embedding', 'prompt', 'reasoning']:
                            current_combo[key.replace(' ', '_')] = value
                            current_section = key.replace(' ', '_')
                elif current_section and current_combo.get(current_section):
                    current_combo[current_section] += f"\n{line}"

            if current_combo:
                combinations.append(current_combo)

            # Validate combinations
            valid_combinations = []
            required_keys = ['id', 'chunking', 'embedding','prompt','reasoning']
            
            for combo in combinations:
                if all(key in combo for key in required_keys):
                    valid_combinations.append(combo)
                    print(f"✅ Valid combination found: {combo['id']}")
                else:
                    missing = [key for key in required_keys if key not in combo]
                    print(f"⚠ Invalid combination - missing: {missing}")

            return valid_combinations

        except Exception as e:
            print(f"❌ Error getting strategy suggestions: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []




    def _format_improvement_report(
        self, 
        metrics_summary: Dict, 
        prompt_analysis: Dict,
        chunking_analysis: Dict,
        embedding_analysis: Dict
        # suggestions: Dict,
        # optimal_combinations: List[Dict]
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

- Session ID: {session_id}
- Started: {session_time}
- Strategies Applied: {strategies_text}
- PR Files: {len(self.session_data.get("pr_files", []))} files
"""
            
        # Add prompts section
        prompts_section = ""
        if getattr(self, "original_prompt", None) or getattr(self, "enhanced_prompt", None):
            prompts_section = "## 📝 System Prompts Used\n"
            if getattr(self, "original_prompt", None):
                prompts_section += f"**Original Prompt:**\n\n```text\n{self.original_prompt}\n```\n\n"
            if getattr(self, "enhanced_prompt", None):
                prompts_section += f"**Enhanced Prompt:**\n\n```text\n{self.enhanced_prompt}\n```\n\n"

        
        report = f"""# 🚀 RAG System Improvement Analysis

Generated on: {timestamp}
{session_info}

{prompts_section}

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
        
        
        # Add detailed comparison tables
        report += "\n## 📊 Strategy Performance Analysis\n"
        
        # Add chunking comparison table
        report += self._format_chunking_comparison_table(chunking_analysis)
        
        # Add embedding comparison table
        report += self._format_embedding_comparison_table(embedding_analysis)

        # Get and add optimal combinations with error handling
        combinations = self._suggest_optimal_combinations(chunking_analysis, embedding_analysis)
        
        if combinations:
            report += "\n## 🔄 Optimal Strategy Combinations\n\n"
            
            # for combo in combinations:
            #     try:
            #         report += f"### {combo.get('id', 'Combination')}\n\n"
            #         report += f"*Chunking Strategy:* {combo.get('chunking', 'Not specified')}\n"
            #         report += f"*Embedding Method:* {combo.get('embedding', 'Not specified')}\n\n"
            #         report += f"*Reasoning:*\n{combo.get('reasoning', 'No reasoning provided')}\n\n"
            #         # report += f"*Expected Benefits:*\n{combo.get('benefits', 'No benefits specified')}\n\n"
            #         report += "---\n\n"
            #     except Exception as e:
            #         print(f"⚠ Error formatting combination: {e}")
            #         continue
            for combo in combinations:
                try:
                    report += f"### {combo.get('id', 'Combination')}\n\n"
                    report += f"*Chunking Strategy:* {combo.get('chunking', 'Not specified')}\n"
                    report += f"*Embedding Method:* {combo.get('embedding', 'Not specified')}\n"
                    report += f"*Prompt Type:* {combo.get('prompt', 'Not specified').capitalize()}\n\n"
                    report += f"*Reasoning:*\n{combo.get('reasoning', 'No reasoning provided')}\n\n"
                    report += "---\n\n"
                except Exception as e:
                    print(f"⚠ Error formatting combination: {e}")
                    continue
        else:
            report += "\n## 🔄 Optimal Strategy Combinations\n\n"
            report += "No valid strategy combinations could be generated.\n\n"

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

    def _format_chunking_comparison_table(self, chunking_analysis: Dict) -> str:
        """Format chunking strategies comparison table"""
        # First try to get metrics from stored results
        stored_metrics = None
        for result in self._load_session_results(self.session_data.get("session_id")):
            if "chunking_metrics_comparison" in result:
                stored_metrics = result["chunking_metrics_comparison"]
                break

        if stored_metrics:
            # Create table from stored metrics
            table = """
### Chunking Strategies Comparison
| Strategy | Relevance | Accuracy | Groundedness | Completeness | Faithfulness | Context Precision | Context Recall | Answer Relevance | BLEU | ROUGE | Overall |
|----------|---------|-----------|--------------|--------------|--------------|------------------|----------------|-----------------| ----------------|-----------------|-------------|
"""
            strategies = stored_metrics["metrics_table"]["strategies"]
            metrics_list = stored_metrics["metrics_table"]["metrics_list"]

            for strategy_name, metrics in strategies.items():
                row = f"| {strategy_name} |"
                for metric in metrics_list:
                    value = metrics.get(metric, 0)
                    row += f" {value:.3f} |"
                table += row + "\n"

            return table

        else:
            # Fallback to original implementation if no stored metrics found
            table = """
### Chunking Strategies Comparison
| Strategy | Overall Score | Faithfulness | Context Precision | Context Recall | Answer Relevance | Processing Time |
|----------|--------------|--------------|------------------|----------------|------------------|-----------------|
"""
            strategies = chunking_analysis.get("strategies_compared", [])
            for strategy in sorted(strategies, key=lambda x: x.get("overall_score", 0), reverse=True):
                table += (f"| {strategy['name']} | {strategy.get('overall_score', 0):.3f} | "
                         f"{strategy.get('faithfulness', 0):.3f} | {strategy.get('context_precision', 0):.3f} | "
                         f"{strategy.get('context_recall', 0):.3f} | {strategy.get('answer_relevance', 0):.3f} | "
                         f"{strategy.get('processing_time', 0):.2f}s |\n")
            return table

    def _format_embedding_comparison_table(self, embedding_analysis: Dict) -> str:
        """Format embedding strategies comparison table"""
        table = """
### Embedding Strategies Comparison
| Strategy | Overall Score | Relevance | Groundedness | Completeness | Answer Relevance | Embedding Time |
|----------|--------------|-----------|--------------|--------------|------------------|----------------|
"""
        strategies = embedding_analysis.get("strategies_compared", [])
        for strategy in sorted(strategies, key=lambda x: x.get("overall_score", 0), reverse=True):
            table += (f"| {strategy['name']} | {strategy.get('overall_score', 0):.3f} | "
                     f"{strategy.get('relevance', 0):.3f} | {strategy.get('groundedness', 0):.3f} | "
                     f"{strategy.get('completeness', 0):.3f} | {strategy.get('answer_relevance', 0):.3f} | "
                     f"{strategy.get('embedding_time', 0):.2f}s |\n")
        return table
    

async def analyze_improvements(session_id: str = None) -> str:
    """
    Analyze the RAG system performance and generate improvement suggestions
    
    Args:
        session_id: Session ID to analyze
        
    Returns:
        str: Markdown formatted analysis report
    """
    # Create analyzer instance
    analyzer = ImprovementAnalyzer()
    
    # Run analysis and get report
    report = await analyzer.analyze_and_suggest_improvements(session_id)
    
    return report