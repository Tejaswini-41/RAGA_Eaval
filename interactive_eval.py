import asyncio
import json
import os
import datetime
import csv
from models.model_factory import ModelFactory
from evaluation.evaluator import ResponseEvaluator

class InteractiveEvaluator:
    def __init__(self):
        self.model_factory = ModelFactory()
        self.evaluator = ResponseEvaluator()
        self.reference_model = "gemini"  # Default reference model
        self.ground_truth_responses = {}  # Store Gemini responses
        
        
        # Create responses directory if it doesn't exist
        if not os.path.exists("responses"):
            os.makedirs("responses")


    def _load_original_gemini_response(self, question):
        """Load original Gemini response from stored responses"""
        try:
            # Try to find the response file for this question
            for filename in os.listdir("responses"):
                if filename.endswith(".json"):
                    filepath = os.path.join("responses", filename)
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        if data["question"] == question and "gemini" in data["responses"]:
                            return data["responses"]["gemini"]
            return None
        except Exception as e:
            print(f"Error loading Gemini response: {e}")
            return None
    
    def load_sample_questions(self):
        """Load sample questions from questions.json"""
        try:
            with open("questions.json", "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading sample questions: {e}")
            return ["What are the best strategies for overcoming impostor syndrome?"]
    
    async def evaluate_question(self, question):
        """Generate responses and evaluate for a single question"""
        print(f"\n📝 Evaluating question: {question}")
        
        # Get current system prompt
        system_prompt = self.model_factory.get_system_prompt()
        
        # Generate responses from all models
        responses = {}
        available_models = self.model_factory.get_model_names()
        
        for model_name in available_models:
            try:
                response = self.model_factory.generate_response(model_name, question)
                responses[model_name] = response
            except Exception as e:
                print(f"⚠️ Skipping unavailable model {model_name}: {e}")
        
        if not responses:
            print("❌ No models available to generate responses")
            return None
        
        # Run evaluation with question and system prompt
        results = await self.evaluator.evaluate_responses(
        self.reference_model, responses, question, system_prompt
        )
    
        # Extract metrics from results
        metrics = {
            model: scores 
            for model, scores in results["rankings"]
        }
    
        # Save responses and metrics to file
        self._save_responses(question, responses, metrics)
    
        # Display results
        self._display_results(question, results)
    
        return results
    
    def _save_responses(self, question, responses, metrics=None):
        """Save responses and metrics to JSON file"""
        filename = f"responses/question_{hash(question) % 10000}.json"
        data = {
            "question": question,
            "responses": responses,
            "metrics": metrics  # Add metrics to saved data
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Responses saved to {filename}")
    
    # Save improved prompts for each question in a unique file
    def _save_improved_prompts_collection(self, model_name, improved_prompt, question):
        """Save improved prompts for each question in a unique file"""
        # Generate unique ID for the question
        question_id = self._generate_question_id(question)
        collection_file = f"prompts/improved_prompts_{question_id}.json"
    
        # Create prompts directory if it doesn't exist
        if not os.path.exists("prompts"):
            os.makedirs("prompts")
    
        # Load existing collection or create new one for this question
        try:
            with open(collection_file, "r") as f:
                prompts_collection = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            prompts_collection = {
                "question": question,
                "question_id": question_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "prompts": []
            }
    
        # Add new prompt entry
        prompt_entry = {
            "model": model_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "improved_prompt": improved_prompt
        }
    
        prompts_collection["prompts"].append(prompt_entry)
    
        # Save updated collection
        with open(collection_file, "w") as f:
            json.dump(prompts_collection, f, indent=2)
        print(f"\nImproved prompt saved to: {collection_file}")

    def _get_latest_improved_prompts(self, question):
        """Get latest improved prompts for a specific question"""
        try:
            # Find the prompts file for this question
            for filename in os.listdir("prompts"):
                if not filename.startswith("improved_prompts_"):
                    continue
                
                filepath = os.path.join("prompts", filename)
                with open(filepath, "r") as f:
                    collection = json.load(f)
                
                    if collection["question"] == question:
                        # Create a dictionary to store the latest prompt for each model
                        latest_prompts = {}
                
                        # Group prompts by model and get the latest one for each
                        for entry in collection["prompts"]:
                            model_name = entry["model"]
                            timestamp = datetime.datetime.fromisoformat(entry["timestamp"])
                    
                            # Update if this is the first or a newer prompt for this model
                            if (model_name not in latest_prompts or 
                                timestamp > datetime.datetime.fromisoformat(latest_prompts[model_name]["timestamp"])):
                                latest_prompts[model_name] = entry
                    
                        return latest_prompts
        
            print("No improved prompts found for this question.")
            return {}
                    
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading improved prompts: {e}")
            return {}

    def _generate_question_id(self, question):
        """Generate a unique ID for a question combining timestamp and hash"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        question_hash = abs(hash(question)) % 10000  # Use abs() to ensure positive hash
        return f"{timestamp}_{question_hash:04d}"


    def _load_original_results(self, question):
        """Load original evaluation results for a question"""
        try:
            for filename in os.listdir("responses"):
                if filename.endswith(".json"):
                    filepath = os.path.join("responses", filename)
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        if data["question"] == question:
                            return {
                                "responses": data["responses"],
                                "metrics": data.get("metrics", {})
                            }
            return None
        except Exception as e:
            print(f"Error loading original results: {e}")
            return None

    async def evaluate_with_improved_prompts(self, question):
        """Generate and evaluate responses using iteratively improved prompts"""
        print(f"\n📝 Evaluating question with iterative prompt improvements: {question}")

        # Load original Gemini response and results
        gemini_response = self._load_original_gemini_response(question)
        if not gemini_response:
            print("\n⚠️ Original Gemini response not found for this question.")
            print("Please evaluate this question first with Gemini to create ground truth.")
            return

        original_results = self._load_original_results(question)
        if not original_results:
            print("\n⚠️ Original evaluation results not found.")
            return

        # Track metrics across iterations
        iteration_metrics = {
            "original": original_results["metrics"]
        }
    
        current_responses = original_results["responses"]
        current_prompts = self._get_latest_improved_prompts(question)  # Initial improved prompts
    
        # Perform 3 iterations with evolving prompts
        for iteration in range(1, 4):
            print(f"\n{'═' * 50}")
            print(f"ITERATION {iteration}")
            print(f"{'═' * 50}")
            
            # Add progressive delay before iterations 2 and 3
            if iteration > 1:
                # Progressive delay: 10 seconds for iteration 2, 20 seconds for iteration 3
                delay_seconds = (iteration - 1) * 30  
                print(f"\n⏱️ Adding {delay_seconds} second delay before iteration {iteration} to avoid API rate limits...")
                await asyncio.sleep(delay_seconds)
                print("Continuing with evaluation...")

            improved_responses = {}
        
            # Generate responses using current prompts
            for model_name, prompt_data in current_prompts.items():
                print(f"\n⚡ Generating improved response for {model_name.upper()}")
                improved_prompt = prompt_data["improved_prompt"]
                improved_response = self.model_factory.generate_response_with_prompt(
                    model_name, question, improved_prompt
                )
                improved_responses[model_name] = improved_response

            improved_responses["gemini"] = gemini_response

            # Evaluate responses
            improved_results = await self.evaluator.evaluate_responses(
                "gemini",
                improved_responses,
                question,
                current_prompts[next(iter(current_prompts))]["improved_prompt"]
            )

            # Extract metrics
            improved_metrics = {
                model: scores 
                for model, scores in improved_results["rankings"]
            }

            # Store metrics for this iteration
            iteration_metrics[f"iteration_{iteration}"] = improved_metrics

            # Save comparison data
            question_id = self._generate_question_id(question)
            filename = f"improved_responses/comparison_{question_id}_iteration_{iteration}.json"
        
            comparison_data = {
                "question": question,
                "timestamp": datetime.datetime.now().isoformat(),
                "iteration": iteration,
                "ground_truth_model": "gemini",
                "ground_truth_response": gemini_response,
                "original_responses": original_results["responses"],
                "previous_responses": current_responses,
                "improved_responses": improved_responses,
                "original_metrics": original_results["metrics"],
                "improved_metrics": improved_metrics,
                "prompts_used": current_prompts
            }

            with open(filename, "w") as f:
                json.dump(comparison_data, f, indent=2)
            print(f"\nIteration {iteration} data saved to {filename}")

            # Generate new improved prompts based on current results
            if iteration < 3:  # Don't need new prompts after last iteration
                current_prompts = self._generate_new_improved_prompts(
                    question, 
                    improved_results, 
                    current_prompts
                )
            
                # Save the new prompts for next iteration
                # for model_name, prompt_data in current_prompts.items():
                #     self._save_improved_prompts_collection(
                #         model_name, 
                #         prompt_data["improved_prompt"],
                #         question
                #     )

            current_responses = improved_responses

        # Display final comparison
        self._display_multi_iteration_comparison(iteration_metrics)
        self._save_comparison_to_csv(question, iteration_metrics, current_prompts)


    def _generate_new_improved_prompts(self, question, current_results, current_prompts):
        """Generate new improved prompts based on cumulative analysis of all previous iterations"""
        new_prompts = {}
        
        # Collect historical data from all previous iterations
        iteration_history = self._collect_iteration_history(question)
        
        for model_name, prompt_data in current_prompts.items():
            # Get current performance metrics
            current_metrics = next(
                (scores for model, scores in current_results["rankings"] 
                if model == model_name),
                None
            )
            
            if current_metrics:
                # Analyze cumulative performance trends
                cumulative_analysis = {
                    "metric_progression": self._analyze_metric_progression(model_name, iteration_history),
                    "successful_patterns": self._identify_successful_patterns(model_name, iteration_history),
                    "failed_approaches": self._identify_failed_approaches(model_name, iteration_history),
                    "current_weaknesses": self._identify_current_weaknesses(current_metrics)
                }
                
                # Generate improved prompt using cumulative insights
                new_prompt = self._generate_cumulative_improved_prompt(
                    prompt_data["improved_prompt"],
                    cumulative_analysis,
                    iteration_history
                )
                
                new_prompts[model_name] = {
                    "model": model_name,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "improved_prompt": new_prompt,
                    "analysis_basis": cumulative_analysis,
                    "iteration_history": iteration_history
                }
        
        return new_prompts

    def _collect_iteration_history(self, question):
        """Collect and organize data from all previous iterations"""
        history = {
            "iterations": {},
            "progression_summary": {}
        }
        
        try:
            for filename in os.listdir("improved_responses"):
                if filename.startswith(f"comparison_{self._generate_question_id(question)}"):
                    with open(os.path.join("improved_responses", filename), "r") as f:
                        data = json.load(f)
                        iteration = data["iteration"]
                        history["iterations"][iteration] = {
                            "prompts": data["prompts_used"],
                            "metrics": data["improved_metrics"],
                            "responses": data["improved_responses"],
                            "original_metrics": data["original_metrics"]
                        }
        except Exception as e:
            print(f"Warning: Could not load complete iteration history: {e}")
        
        return history

    def _analyze_metric_progression(self, model_name, history):
        """Analyze how metrics have progressed across all iterations"""
        progression = {}
        metrics = ["Relevance", "Accuracy", "Groundedness", "Completeness", "BLEU", "ROUGE"]
        
        for metric in metrics:
            values = []
            changes = []
            
            # Collect values across iterations
            for iter_num in sorted(history["iterations"].keys()):
                iter_data = history["iterations"][iter_num]
                value = iter_data["metrics"].get(model_name, {}).get(metric, 0.0)
                values.append(value)
                
                if len(values) > 1:
                    change = ((value - values[-2]) / values[-2] * 100) if values[-2] != 0 else 0
                    changes.append(change)
            
            progression[metric] = {
                "values": values,
                "trend": self._calculate_trend(values),
                "changes": changes,
                "consistently_improving": all(c > 0 for c in changes),
                "best_value": max(values) if values else 0.0,
                "worst_value": min(values) if values else 0.0
            }
        
        return progression

    def _generate_cumulative_improved_prompt(self, current_prompt, analysis, history):
        """Generate new prompt incorporating learnings from all iterations"""
        improved_prompt = current_prompt
        
        # Extract successful patterns that consistently improved metrics
        successful_patterns = analysis["successful_patterns"]
        for pattern in successful_patterns:
            if pattern not in improved_prompt:
                improved_prompt = self._incorporate_pattern(improved_prompt, pattern)
        
        # Remove or modify approaches that consistently failed
        failed_approaches = analysis["failed_approaches"]
        for approach in failed_approaches:
            improved_prompt = self._remove_failed_approach(improved_prompt, approach)
        
        # Add specific improvements for current weaknesses
        for weakness, details in analysis["current_weaknesses"].items():
            improvement = self._generate_targeted_improvement(weakness, details, history)
            improved_prompt = self._incorporate_improvement(improved_prompt, improvement)
        
        # Add learnings from metric progression
        for metric, progression in analysis["metric_progression"].items():
            if not progression["consistently_improving"]:
                enhancement = self._generate_metric_enhancement(metric, progression)
                improved_prompt = self._incorporate_enhancement(improved_prompt, enhancement)
        
        return improved_prompt

    def _generate_new_improved_prompts(self, question, current_results, current_prompts):
        """Generate new improved prompts based on analysis of all previous iterations"""
        new_prompts = {}

        # Get metrics from all previous iterations
        iteration_metrics = {}
        try:
            # Find all iteration files for this question
            for filename in os.listdir("improved_responses"):
                if filename.startswith(f"comparison_{self._generate_question_id(question)}"):
                    with open(os.path.join("improved_responses", filename), "r") as f:
                        data = json.load(f)
                        iteration = data["iteration"]
                        iteration_metrics[iteration] = {
                            "metrics": data["improved_metrics"],
                            "prompts": data["prompts_used"],
                            "responses": data["improved_responses"]
                        }
        except Exception as e:
            print(f"Warning: Could not load previous iteration data: {e}")

        for model_name, prompt_data in current_prompts.items():
            # Get current metrics
            current_metrics = next(
                (scores for model, scores in current_results["rankings"] 
                if model == model_name),
                None
            )
            
            if current_metrics:
                # Analyze trends across iterations
                metric_trends = self._analyze_metric_trends(
                    model_name, 
                    current_metrics,
                    iteration_metrics
                )

                # Generate improved prompt based on historical analysis
                new_prompt = self._generate_prompt_from_trends(
                    prompt_data["improved_prompt"],
                    metric_trends,
                    iteration_metrics
                )

                new_prompts[model_name] = {
                    "model": model_name,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "improved_prompt": new_prompt
                }

        return new_prompts

    def _analyze_metric_trends(self, model_name, current_metrics, iteration_metrics):
        """Analyze trends in metrics across iterations"""
        trends = {}
        
        # Metrics to track
        key_metrics = ["Relevance", "Accuracy", "Groundedness", "Completeness", "BLEU", "ROUGE"]
        
        for metric in key_metrics:
            scores = []
            prompts = []
            
            # Collect scores and prompts from each iteration
            for iter_num, iter_data in sorted(iteration_metrics.items()):
                if model_name in iter_data["metrics"]:
                    score = iter_data["metrics"][model_name].get(metric, 0.0)
                    prompt = iter_data["prompts"].get(model_name, {}).get("improved_prompt", "")
                    scores.append(score)
                    prompts.append(prompt)
            
            # Add current score
            scores.append(current_metrics.get(metric, 0.0))
            
            # Analyze trend
            trends[metric] = {
                "scores": scores,
                "best_score": max(scores),
                "best_prompt": prompts[scores.index(max(scores[:-1]))] if scores[:-1] else None,
                "is_improving": scores[-1] > scores[-2] if len(scores) > 1 else False,
                "improvement_needed": scores[-1] < 0.7  # Threshold for improvement
            }
        
        return trends

    def _generate_prompt_from_trends(self, current_prompt, metric_trends, iteration_metrics):
        """Generate improved prompt based on historical trends"""
        improved_prompt = current_prompt
        
        # Identify areas needing most improvement
        weak_metrics = [
            metric for metric, data in metric_trends.items() 
            if data["improvement_needed"]
        ]
        
        # Use best performing prompts as reference
        for metric in weak_metrics:
            if metric_trends[metric]["best_prompt"]:
                # Extract successful patterns from best performing prompt
                best_prompt = metric_trends[metric]["best_prompt"]
                
                # Add relevant instructions from successful prompts
                key_phrases = self._extract_key_instructions(best_prompt)
                for phrase in key_phrases:
                    if phrase not in improved_prompt:
                        improved_prompt += f"\n\n{phrase}"
        
        # Add targeted improvements based on trends
        improvement_suggestions = self._get_trend_based_improvements(metric_trends)
        for suggestion in improvement_suggestions:
            if suggestion not in improved_prompt:
                improved_prompt += f"\n\n{suggestion}"
        
        return improved_prompt

    def _extract_key_instructions(self, prompt):
        """Extract key instruction phrases from a prompt"""
        # Split into sentences and identify instruction-like phrases
        sentences = prompt.split('.')
        key_phrases = []
        
        instruction_indicators = ['must', 'should', 'need to', 'ensure', 'focus on', 'avoid']
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in instruction_indicators):
                key_phrases.append(sentence.strip() + '.')
        
        return key_phrases

    def _get_trend_based_improvements(self, metric_trends):
        """Generate improvement suggestions based on metric trends"""
        suggestions = []
        
        improvement_templates = {
            "Relevance": [
                "Ensure responses directly address the core question",
                "Focus on maintaining topical consistency throughout"
            ],
            "Accuracy": [
                "Verify all factual claims before including",
                "Provide specific, concrete examples"
            ],
            "Groundedness": [
                "Base responses on established knowledge",
                "Connect statements to reliable sources"
            ],
            "Completeness": [
                "Cover all essential aspects of the topic",
                "Address potential counterarguments"
            ],
            "BLEU": [
                "Use standard terminology consistently",
                "Maintain consistent phrasing patterns"
            ],
            "ROUGE": [
                "Structure responses with clear sections",
                "Use key terms consistently throughout"
            ]
        }
        
        for metric, data in metric_trends.items():
            if data["improvement_needed"]:
                suggestions.extend(improvement_templates.get(metric, []))
        
        return suggestions


    def _display_multi_iteration_comparison(self, iteration_metrics):
        """Display comparison table of original and iteration metrics with per-iteration gains"""
        print("\n" + "═" * 160)
        print("📊 MULTI-ITERATION METRICS COMPARISON TABLE")
        print("═" * 160)

        # Header
        print("\n" + "─" * 120)
        # Calculate and display both per-iteration and total gains
        header = f"{'Model':<15} {'Metric':<15} {'Original':<12}"  # Changed from 12 to 15 for Model/Metric
        for i in range(1, 4):
            header += f"{'Iter '+str(i):<10} {'Gain '+str(i)+'%':<10}"
        header += f"{'Total Gain':<12}"
        print(header)
        print("─" * 160)

        # Get all models and metrics
        all_models = set()
        all_metrics = set()
        for metrics in iteration_metrics.values():
            for model, scores in metrics.items():
                if model != "gemini":
                    all_models.add(model)
                    all_metrics.update(scores.keys())

        for model_name in sorted(all_models):
            first_line = True
            for metric in sorted(all_metrics):
                # Get original value
                orig_val = iteration_metrics["original"].get(model_name, {}).get(metric, 0.0)
                
                # Get iteration values and calculate gains
                iter_vals = []
                iter_gains = []
                for i in range(1, 4):
                    iter_val = iteration_metrics.get(f"iteration_{i}", {}).get(model_name, {}).get(metric, 0.0)
                    iter_vals.append(iter_val)
                    
                    # Calculate gain percentage compared to original
                    gain_pct = ((iter_val - orig_val) / orig_val * 100) if orig_val != 0 else 0.0
                    iter_gains.append(gain_pct)
                
                # Calculate total gain
                final_val = iter_vals[-1]
                total_change_pct = ((final_val - orig_val) / orig_val * 100) if orig_val != 0 else 0.0
                
                # Format the line
                model_col = f"{model_name:<15}" if first_line else " " * 15
                line = f"{model_col} {metric:<15} {orig_val:<12.3f}"
            
                # Add iteration values and their respective gains
                for i, val in enumerate(iter_vals):
                    # Add the iteration value
                    line += f"{val:<10.3f}"
                    
                    # Calculate gain for this iteration compared to original
                    iter_gain_pct = ((val - orig_val) / orig_val * 100) if orig_val != 0 else 0.0
                    
                    # Format gain with indicator
                    iter_gain_str = f"{iter_gain_pct:+.1f}%"
                    if iter_gain_pct > 0:
                        iter_gain_str = f"✅{iter_gain_str}"
                    elif iter_gain_pct < 0:
                        iter_gain_str = f"🚩{iter_gain_str}"
                    
                    # Add the gain percentage
                    line += f"{iter_gain_str:<10}"

                # Add total change with indicator (keep existing code)
                change_str = f"{total_change_pct:+.1f}%"
                if total_change_pct > 0:
                    change_str = f"✅ {change_str}"
                elif total_change_pct < 0:
                    change_str = f"🔻 {change_str}"
                line += f"{change_str:<15}"
                
                print(line)
                first_line = False
            
            print("─" * 160)

        print("\n" + "═" * 120)
        
        # NEW CODE: Add conclusion about best iteration
        
        # Calculate average performance gains for each iteration
        iteration_avg_gains = {}
        iteration_success_rate = {}
        
        # Track metrics for all iterations
        for i in range(1, 4):
            total_gain = 0
            total_metrics = 0
            improvements = 0
            
            # Calculate across all models and metrics
            for model_name in all_models:
                for metric in all_metrics:
                    orig_val = iteration_metrics["original"].get(model_name, {}).get(metric, 0.0)
                    if orig_val == 0:
                        continue
                        
                    iter_val = iteration_metrics.get(f"iteration_{i}", {}).get(model_name, {}).get(metric, 0.0)
                    gain_pct = ((iter_val - orig_val) / orig_val * 100)
                    
                    total_gain += gain_pct
                    total_metrics += 1
                    
                    if gain_pct > 0:
                        improvements += 1
            
            if total_metrics > 0:
                iteration_avg_gains[i] = total_gain / total_metrics
                iteration_success_rate[i] = (improvements / total_metrics) * 100
        
        # Find best iteration
        best_iteration = max(iteration_avg_gains, key=iteration_avg_gains.get, default=None)
        
        if best_iteration:
            print(f"🏆 BEST ITERATION: Iteration {best_iteration} with average gain of {iteration_avg_gains[best_iteration]:.1f}%")
            print(f"    Success rate: {iteration_success_rate[best_iteration]:.1f}% of metrics improved")
            print(f"    Recommended prompt set: prompts/improved_prompts_*_iteration_{best_iteration}.json")
        else:
            print("⚠️ No clear best iteration identified. All iterations showed similar or negative performance.")
        
        print("═" * 120)

    def _save_comparison_to_csv(self, question, iteration_metrics, current_prompts):
        """Save the multi-iteration metrics comparison to a CSV file with enhanced formatting"""
        question_id = self._generate_question_id(question)
        filename = f"results/metrics_comparison_{question_id}.csv"
        
        # Create results directory if it doesn't exist
        if not os.path.exists("results"):
            os.makedirs("results")
        
        # Get all models and metrics
        all_models = set()
        all_metrics = set()
        for metrics in iteration_metrics.values():
            for model, scores in metrics.items():
                if model != "gemini":
                    all_models.add(model)
                    all_metrics.update(scores.keys())
        
        all_metrics = sorted(all_metrics)
        all_models = sorted(all_models)
        
        # Create professional CSV structure
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write title and metadata
            writer.writerow(["Model Performance Analysis"])
            writer.writerow(["Question:", question])
            writer.writerow(["Generated:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow([])  # Empty row for spacing
            
            # For each model, create a complete section
            for model_name in all_models:
                # Model header
                writer.writerow([f"MODEL: {model_name.upper()}"])
                
                # Headers for this model's table
                model_headers = ["Metric", "Original"]
                for i in range(1, 4):
                    model_headers.extend([f"Iteration {i}", f"Gain {i}"])
                model_headers.append("Total Gain")
                writer.writerow(model_headers)
                
                # Add metric rows for this model
                for metric in all_metrics:
                    orig_val = iteration_metrics["original"].get(model_name, {}).get(metric, 0.0)
                    
                    # Get values and gains for each iteration
                    row = [metric, f"{orig_val:.3f}"]
                    
                    iter_vals = []
                    for i in range(1, 4):
                        iter_val = iteration_metrics.get(f"iteration_{i}", {}).get(model_name, {}).get(metric, 0.0)
                        iter_vals.append(iter_val)
                        
                        # Calculate gain
                        iter_gain = ((iter_val - orig_val) / orig_val * 100) if orig_val != 0 else 0.0
                        
                        # Add value and gain
                        row.append(f"{iter_val:.3f}")
                        row.append(f"{iter_gain:+.1f}%")
                    
                    # Calculate total gain
                    final_val = iter_vals[-1]
                    total_gain = ((final_val - orig_val) / orig_val * 100) if orig_val != 0 else 0.0
                    row.append(f"{total_gain:+.1f}%")
                    
                    writer.writerow(row)
                
                # Add empty row for spacing between models
                writer.writerow([])
            
            # Add summary section
            writer.writerow(["PERFORMANCE SUMMARY"])
            summary_headers = ["Model", "Initial Overall", "Final Overall", "Change", "Best Score", "Best Iteration"]
            writer.writerow(summary_headers)
            
            # Calculate which iteration was best for each model
            for model_name in all_models:
                orig_overall = iteration_metrics["original"].get(model_name, {}).get("Overall", 0.0)
                final_overall = iteration_metrics.get("iteration_3", {}).get(model_name, {}).get("Overall", 0.0)
                change_pct = ((final_overall - orig_overall) / orig_overall * 100) if orig_overall != 0 else 0.0
                
                # Find best iteration
                best_iter = 0
                best_score = orig_overall
                for i in range(1, 4):
                    iter_score = iteration_metrics.get(f"iteration_{i}", {}).get(model_name, {}).get("Overall", 0.0)
                    if iter_score > best_score:
                        best_score = iter_score
                        best_iter = i
                
                best_iter_text = f"Iteration {best_iter}" if best_iter > 0 else "Original"
                
                writer.writerow([
                    model_name, 
                    f"{orig_overall:.3f}", 
                    f"{final_overall:.3f}", 
                    f"{change_pct:+.1f}%",
                    f"{best_score:.3f}",
                    best_iter_text
                ])
        
        print(f"\n📊 Enhanced metrics report saved to CSV: {filename}")

    def _display_results(self, question, results):
        """Display evaluation results with focus on prompt comparison"""
        import textwrap
        
        # Header section
        print("\n" + "═" * 100)
        print(f"📊  EVALUATION RESULTS: \"{question}\"")
        print("═" * 100)
        
        # Rankings section (brief)
        print(f"\n🏆  RANKINGS (Reference: {results['reference_model'].upper()})")
        print("─" * 50)
        
        for i, (model_name, scores) in enumerate(results['rankings'], 1):
            print(f"\n  {i}. {model_name.upper()} - Overall Score: {scores['Overall']}")
            
            # Format metrics in two columns for better space utilization
            metrics = [m for m in scores.keys() if m != "Overall"]
            mid = len(metrics) // 2
            col1 = metrics[:mid+1]
            col2 = metrics[mid+1:]
            
            for i in range(max(len(col1), len(col2))):
                line = "     "
                if i < len(col1):
                    line += f"{col1[i]}: {scores[col1[i]]:.3f}".ljust(25)
                if i < len(col2):
                    line += f"{col2[i]}: {scores[col2[i]]:.3f}"
                print(line)
        
        # Prompt improvements section with focus on prompt comparison
        if "prompt_improvements" in results:
            # Get current system prompt
            current_prompt = self.model_factory.get_system_prompt()
            
            print("\n" + "═" * 100)
            print("💡  PROMPT IMPROVEMENT RECOMMENDATIONS")
            print("═" * 100)
            
            for model_name, improvements in results["prompt_improvements"].items():
                # Model header
                print(f"\n📈  {model_name.upper()}")
                print("─" * 50)
                
                if "error" in improvements:
                    print(f"\n  ⚠️  Error: {improvements['error']}")
                    continue
                
                # Display key issues in simplified format
                if "prompt_improvements" in improvements:
                    print("\n  🔍 KEY ISSUES SUMMARY:")
                    
                    # Combine all target metrics and gains for display
                    all_targets = set()
                    all_gains = {}
                    
                    for imp in improvements["prompt_improvements"]:
                        # Brief issue statement
                        print(f"  • {imp.get('issue', 'Issue not specified')}")
                        
                        # Collect targets and gains
                        if "target_metrics" in imp:
                            for metric in imp["target_metrics"]:
                                all_targets.add(metric)
                        
                        if "estimated_gains" in imp:
                            for metric, gain in imp["estimated_gains"].items():
                                if metric in all_gains:
                                    # Take the higher gain value if the metric appears multiple times
                                    current = all_gains[metric].replace('+', '').replace('%', '')
                                    new = gain.replace('+', '').replace('%', '')
                                    if float(new.split('-')[0]) > float(current.split('-')[0]):
                                        all_gains[metric] = gain
                                else:
                                    all_gains[metric] = gain
                
                    # Show combined targets and gains
                    if all_targets:
                        print(f"\n  📌 TARGET METRICS: {', '.join(sorted(all_targets))}")
                    
                    if all_gains:
                        gains_str = ", ".join([f"{m}: {g}" for m, g in sorted(all_gains.items())])
                        print(f"  📈 POTENTIAL GAINS: {gains_str}")
                
                # PROMPT COMPARISON (highlighted section)
                print("\n" + "┄" * 50)
                print("  🔄 PROMPT COMPARISON")
                print("┄" * 50)
                
                # Display current prompt
                print("\n  📝 CURRENT PROMPT:")
                box_width = 90
                print(f"  ┌{'─' * box_width}┐")
                current_lines = textwrap.wrap(current_prompt, width=box_width-2)
                for line in current_lines:
                    print(f"  │ {line:{box_width-2}} │")
                print(f"  └{'─' * box_width}┘")
                
                # Display improved prompt with highlight
                if "improved_system_prompt" in improvements:
                    print("\n  ✨ IMPROVED PROMPT:")
                    print(f"  ┌{'─' * box_width}┐")
                    
                    # Format the prompt with proper wrapping and handle paragraphs
                    improved_prompt = improvements["improved_system_prompt"]
                    paragraphs = improved_prompt.split('\n\n')
                    
                    for i, paragraph in enumerate(paragraphs):
                        lines = textwrap.wrap(paragraph.strip(), width=box_width-2)
                        for line in lines:
                            print(f"  │ {line:{box_width-2}} │")
                        
                        # Add a blank line between paragraphs (except after the last one)
                        if i < len(paragraphs) - 1:
                            print(f"  │ {'':{box_width-2}} │")
                    
                    print(f"  └{'─' * box_width}┘")
                    # Save the improved prompt
                    self._save_improved_prompts_collection(model_name, improved_prompt, question)
                
                # Overall improvement estimate (simplified)
                if "overall_estimated_improvement" in improvements:
                    estimate = improvements["overall_estimated_improvement"]
                    # Extract just the percentage if possible
                    import re
                    match = re.search(r'(\+\d+(?:-\d+)?%)', estimate)
                    if match:
                        estimate = f"Estimated overall improvement: {match.group(1)}"
                        
                    print(f"\n  🚀 {estimate}")
        
        print("\n" + "═" * 100)
    
    def _get_user_choice(self, options, prompt):
        """Get user choice from menu"""
        while True:
            print(f"\n{prompt}")
            for i, option in enumerate(options, 1):
                print(f"{i}. {option}")
            
            choice = input("\nEnter your choice (number): ")
            try:
                index = int(choice) - 1
                if 0 <= index < len(options):
                    return options[index]
                print("Invalid option number.")
            except ValueError:
                print("Please enter a number.")


    async def interactive_menu(self):
        """Main interactive menu"""
        print("\n🤖 Welcome to the Interactive Model Evaluator")
        last_question = None  # Track the last evaluated question
    
        while True:
            try:
                print("\n" + "="*50)
                print("MAIN MENU")
                print("="*50)
        
                options = [
                    "Enter your own question",
                    "Choose from sample questions",
                    "Set reference model",
                    "Exit"
                ]
        
                # Add "Evaluate with improved prompts" only if there's a previous question
                # and improved prompts exist
                if last_question:
                    # Check if improved prompts exist for this question
                    has_improved_prompts = False
                    for filename in os.listdir("prompts"):
                        if filename.startswith("improved_prompts_"):
                            filepath = os.path.join("prompts", filename)
                            with open(filepath, "r") as f:
                                collection = json.load(f)
                                if collection.get("question") == last_question:
                                    has_improved_prompts = True
                                    break
                
                    if has_improved_prompts:
                        options.insert(-1, f"Evaluate previous question with improved prompts: '{last_question}'")
        
                choice = self._get_user_choice(options, "What would you like to do?")
        
                if choice == "Enter your own question":
                    question = input("\nEnter your question: ")
                    if question.strip():
                        last_question = question  # Store the question
                        await self.evaluate_question(question)
                        input("\nPress Enter to continue...")
        
                elif choice == "Choose from sample questions":
                    sample_questions = self.load_sample_questions()
                    question = self._get_user_choice(sample_questions, "Choose a question:")
                    last_question = question  # Store the question
                    await self.evaluate_question(question)
                    input("\nPress Enter to continue...")
        
                elif choice.startswith("Evaluate previous question"):
                    print(f"\n📝 Re-evaluating: {last_question}")
                    await self.evaluate_with_improved_prompts(last_question)
                    input("\nPress Enter to continue...")
        
                elif choice == "Set reference model":
                    model_names = self.model_factory.get_model_names()
                    self.reference_model = self._get_user_choice(
                        model_names, 
                        "Choose reference model (ground truth):"
                    )
                    print(f"Reference model set to: {self.reference_model}")
        
                elif choice == "Exit":
                    print("\nThank you for using the Interactive Model Evaluator. Goodbye!")
                    break
            
            except Exception as e:
                print(f"\nAn error occurred: {e}")
                print("Returning to main menu...")
                input("\nPress Enter to continue...")


async def main():
    evaluator = InteractiveEvaluator()
    await evaluator.interactive_menu()

if __name__ == "__main__":
    asyncio.run(main())