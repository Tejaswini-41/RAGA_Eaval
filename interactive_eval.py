import asyncio
import json
import os
import datetime
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
        responses = self.model_factory.generate_all_responses(question)
    
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
    
    # def _save_responses(self, question, responses):
    #     """Save responses to JSON file"""
    #     filename = f"responses/question_{hash(question) % 10000}.json"
    #     with open(filename, "w") as f:
    #         json.dump({
    #             "question": question,
    #             "responses": responses
    #         }, f, indent=2)
    #     print(f"Responses saved to {filename}")
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
    
    # Add this new method to the InteractiveEvaluator class
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



    # async def evaluate_with_improved_prompts(self, question):
    #     """Generate and evaluate responses using latest improved prompts with 3 iterations"""
    #     print(f"\n📝 Evaluating question with improved prompts (3 iterations): {question}")

    #     # Load original Gemini response and original results
    #     gemini_response = self._load_original_gemini_response(question)
    #     if not gemini_response:
    #         print("\n⚠️ Original Gemini response not found for this question.")
    #         print("Please evaluate this question first with Gemini to create ground truth.")
    #         return

    #     # Load original evaluation results
    #     original_results = self._load_original_results(question)
    #     if not original_results:
    #         print("\n⚠️ Original evaluation results not found.")
    #         return

    #     # Track metrics across iterations
    #     iteration_metrics = {
    #         "original": original_results["metrics"]
    #     }
    
    #     current_responses = original_results["responses"]
    
    #     # Perform 3 iterations
    #     for iteration in range(1, 4):
    #         print(f"\n{'═' * 50}")
    #         print(f"ITERATION {iteration}")
    #         print(f"{'═' * 50}")

    #         # Get latest improved prompts
    #         latest_prompts = self._get_latest_improved_prompts(question)
    #         improved_responses = {}
        
    #         for model_name, prompt_data in latest_prompts.items():
    #             print(f"\n⚡ Generating improved response for {model_name.upper()}")
    #             improved_prompt = prompt_data["improved_prompt"]
    #             improved_response = self.model_factory.generate_response_with_prompt(
    #                 model_name, question, improved_prompt
    #             )
    #             improved_responses[model_name] = improved_response

    #         # Add Gemini response as ground truth
    #         improved_responses["gemini"] = gemini_response

    #         # Evaluate improved responses against Gemini ground truth
    #         improved_results = await self.evaluator.evaluate_responses(
    #             "gemini",
    #             improved_responses,
    #             question,
    #             latest_prompts[next(iter(latest_prompts))]["improved_prompt"]
    #         )

    #         # Extract metrics from improved results
    #         improved_metrics = {
    #             model: scores 
    #             for model, scores in improved_results["rankings"]
    #         }

    #         # Store metrics for this iteration
    #         iteration_metrics[f"iteration_{iteration}"] = improved_metrics

    #         # Save comparison data for this iteration
    #         question_id = self._generate_question_id(question)
    #         filename = f"improved_responses/comparison_{question_id}_iteration_{iteration}.json"
        
    #         comparison_data = {
    #             "question": question,
    #             "timestamp": datetime.datetime.now().isoformat(),
    #             "iteration": iteration,
    #             "ground_truth_model": "gemini",
    #             "ground_truth_response": gemini_response,
    #             "original_responses": original_results["responses"],
    #             "previous_responses": current_responses,
    #             "improved_responses": improved_responses,
    #             "original_metrics": original_results["metrics"],
    #             "improved_metrics": improved_metrics,
    #             "prompts_used": latest_prompts
    #         }

    #         with open(filename, "w") as f:
    #             json.dump(comparison_data, f, indent=2)
    #         print(f"\nIteration {iteration} data saved to {filename}")

    #         # Update current responses for next iteration
    #         current_responses = improved_responses

    #     # Display final comparison using original and all iteration metrics
    #     self._display_multi_iteration_comparison(iteration_metrics)



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


    def _generate_new_improved_prompts(self, question, evaluation_results, current_prompts):
        """Generate new improved prompts based on evaluation results"""
        new_prompts = {}
    
        for model_name, prompt_data in current_prompts.items():
            # Get current prompt and its performance metrics
            current_prompt = prompt_data["improved_prompt"]
            model_metrics = next(
                (scores for model, scores in evaluation_results["rankings"] 
                if model == model_name), 
                None
         )
        
            if model_metrics:
                # Analyze metrics and generate improved prompt
                # This is where you would implement your prompt improvement logic
                # based on the evaluation results
                new_prompt = self._improve_prompt_based_on_metrics(
                    current_prompt, 
                    model_metrics
                )
            
                new_prompts[model_name] = {
                    "model": model_name,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "improved_prompt": new_prompt
                }
    
        return new_prompts

    def _improve_prompt_based_on_metrics(self, current_prompt, model_metrics):
        """Generate improved prompt based on model performance metrics"""
    
        # Initialize base score weights
        metric_weights = {
            'Relevance': 0.3,
            'Coherence': 0.2,
            'Accuracy': 0.3,
            'Completeness': 0.2
        }
    
        # Analyze weak points in metrics
        weak_points = []
        for metric, score in model_metrics.items():
            if metric != 'Overall':
                if score < 0.7:  # Consider scores below 0.7 as areas needing improvement
                    weak_points.append((metric, score))
    
        # Sort weak points by score (ascending)
        weak_points.sort(key=lambda x: x[1])
    
        # Generate improvements based on weak points
        improvements = []
    
        # Basic prompt enhancement templates
        enhancement_templates = {
            'Relevance': "Focus on providing information that directly addresses the question. Stay on topic and avoid tangential information.",
            'Coherence': "Structure your response logically with clear transitions between ideas. Maintain a consistent flow throughout.",
            'Accuracy': "Ensure all statements are factual and well-supported. Verify information before including it.",
            'Completeness': "Cover all essential aspects of the topic. Include relevant examples and explanations where necessary."
        }
    
        # Build improved prompt
        improved_prompt = current_prompt
    
        # Add specific improvements based on weak points
        for metric, score in weak_points:
            if metric in enhancement_templates:
                improvement = enhancement_templates[metric]
                if improvement not in improved_prompt:
                    improved_prompt += f"\n\n{improvement}"
    
        # Add general quality reminder if needed
        if len(weak_points) > 0:
            improved_prompt += "\n\nEnsure your response maintains high standards of clarity, precision, and relevance."
    
        return improved_prompt










    def _display_multi_iteration_comparison(self, iteration_metrics):
        """Display comparison table of original and all iteration metrics"""
        print("\n" + "═" * 120)
        print("📊 MULTI-ITERATION METRICS COMPARISON TABLE")
        print("═" * 120)

        # Header
        print("\n" + "─" * 120)
        # Calculate and display both per-iteration and total gains
        header = f"{'Model':<12} {'Metric':<12} {'Original':<10}"
        for i in range(1, 4):
            header += f"{'Iter '+str(i):<10} {'Gain '+str(i)+'%':<10}"
        header += f"{'Total Gain':<12}"
        print(header)
        print("─" * 120)

        # Get all models from all results (excluding reference model)
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
                # Get values across iterations
                orig_val = iteration_metrics["original"].get(model_name, {}).get(metric, 0.0)
                iter_vals = []
                for i in range(1, 4):
                    iter_val = iteration_metrics.get(f"iteration_{i}", {}).get(model_name, {}).get(metric, 0.0)
                    iter_vals.append(iter_val)
            
                # Calculate total improvement
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
                    change_str = f"🚩 {change_str}"
                line += f"{change_str:<12}"
            
                print(line)
                first_line = False
        
            print("─" * 120)

        print("\n" + "═" * 120)























    
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