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
        
        # Save responses to file
        self._save_responses(question, responses)
        
        # Run evaluation with question and system prompt
        results = await self.evaluator.evaluate_responses(
            self.reference_model, responses, question, system_prompt
        )
        
        # Display results
        self._display_results(question, results)
        
        return results
    
    def _save_responses(self, question, responses):
        """Save responses to JSON file"""
        filename = f"responses/question_{hash(question) % 10000}.json"
        with open(filename, "w") as f:
            json.dump({
                "question": question,
                "responses": responses
            }, f, indent=2)
        print(f"Responses saved to {filename}")
    
    # Add this new method to the InteractiveEvaluator class
    def _save_improved_prompts_collection(self, model_name, improved_prompt):
        """Save all improved prompts to a single JSON file"""
        collection_file = "prompts/improved_prompts_collection.json"
    
        # Create prompts directory if it doesn't exist
        if not os.path.exists("prompts"):
            os.makedirs("prompts")
    
        # Load existing collection or create new one
        try:
            with open(collection_file, "r") as f:
                prompts_collection = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            prompts_collection = {"prompts": []}
    
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
        print(f"\nImproved prompt saved to collection: {collection_file}")

    def _get_latest_improved_prompts(self):
        """Dynamically load the latest improved prompts for each model"""
        collection_file = "prompts/improved_prompts_collection.json"
        try:
            with open(collection_file, "r") as f:
                collection = json.load(f)
            
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
        except (FileNotFoundError, json.JSONDecodeError):
            print("No improved prompts found. Using original prompts.")
            return {}

    def _generate_question_id(self, question):
        """Generate a unique ID for a question combining timestamp and hash"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        question_hash = abs(hash(question)) % 10000  # Use abs() to ensure positive hash
        return f"{timestamp}_{question_hash:04d}"

    async def evaluate_with_improved_prompts(self, question):
        """Generate and evaluate responses using latest improved prompts"""
        print(f"\n📝 Evaluating question with latest improved prompts: {question}")

            # Load original Gemini response as ground truth
        gemini_response = self._load_original_gemini_response(question)
        if not gemini_response:
            print("\n⚠️ Original Gemini response not found for this question.")
            print("Please evaluate this question first with Gemini to create ground truth.")
            return
    
        # Create improved_responses directory if it doesn't exist
        if not os.path.exists("improved_responses"):
            os.makedirs("improved_responses")
    
        # Get latest improved prompts
        latest_prompts = self._get_latest_improved_prompts()
    
        # Generate and evaluate original responses
        # original_responses = self.model_factory.generate_all_responses(question)
        # original_results = await self.evaluator.evaluate_responses(
        #     self.reference_model,
        #     original_responses,
        #     question,
        #     self.model_factory.get_system_prompt()
        # )
    
        # Generate and evaluate improved responses
        improved_responses = {}
        for model_name, prompt_data in latest_prompts.items():
            print(f"\n⚡ Generating improved response for {model_name.upper()}")
            improved_prompt = prompt_data["improved_prompt"]
            improved_response = self.model_factory.generate_response_with_prompt(
                model_name, question, improved_prompt
            )
            improved_responses[model_name] = improved_response

        # Add Gemini response as ground truth
        improved_responses["gemini"] = gemini_response
    
        improved_results = await self.evaluator.evaluate_responses(
            "gemini",
            improved_responses,
            question,
            latest_prompts[next(iter(latest_prompts))]["improved_prompt"]  # Use first improved prompt as reference
        )
    
        # Save comparison data with Gemini ground truth
        question_id = self._generate_question_id(question)
        filename = f"improved_responses/comparison_{question_id}.json"
        comparison_data = {
            "question": question,
            "timestamp": datetime.datetime.now().isoformat(),
            "ground_truth_model": "gemini",
            "ground_truth_response": gemini_response,
            "improved_responses": improved_responses,
            "evaluation_results": improved_results,
            "prompts_used": latest_prompts
        }
    
        with open(filename, "w") as f:
            json.dump(comparison_data, f, indent=2)
        print(f"\nComparison data saved to {filename}")
    
        # Display comparison excluding Gemini from results
        filtered_results = {
            "rankings": [(model, scores) for model, scores in improved_results["rankings"] 
                    if model != "gemini"]
        }
        self._display_metrics_comparison(filtered_results, improved_results)

    def _display_metrics_comparison(self, original_results, improved_results):
        """Display comparison table of original and improved metrics"""
        print("\n" + "═" * 100)
        print("📊 METRICS COMPARISON TABLE")
        print("═" * 100)
    
        # Header
        print("\n" + "─" * 100)
        header = f"{'Model':<15} {'Metric':<15} {'Original':<12} {'Improved':<12} {'Change':<12}"
        print(header)
        print("─" * 100)
    
        # Get all models that appear in either result set
        all_models = set(dict(original_results['rankings']).keys()) | set(dict(improved_results['rankings']).keys())
    
        for model_name in sorted(all_models):
            # Get scores for this model
            orig_scores = dict(original_results['rankings']).get(model_name, {})
            impr_scores = dict(improved_results['rankings']).get(model_name, {})
        
            # Get all metrics for this model
            all_metrics = set(orig_scores.keys()) | set(impr_scores.keys())
        
            first_line = True
            for metric in sorted(all_metrics):
                orig_val = orig_scores.get(metric, 0.0)
                impr_val = impr_scores.get(metric, 0.0)
            
                # Calculate change
                if orig_val != 0:
                    change_pct = ((impr_val - orig_val) / orig_val) * 100
                    change_str = f"{change_pct:+.1f}%"
                    # Add arrows for visual indication
                    if change_pct > 0:
                        change_str = f"🔺 {change_str}"
                    elif change_pct < 0:
                        change_str = f"🔻 {change_str}"
                else:
                    change_str = "N/A"
            
                # Format the line
                model_col = f"{model_name:<15}" if first_line else " " * 15
                line = f"{model_col} {metric:<15} {orig_val:<12.3f} {impr_val:<12.3f} {change_str:<12}"
                print(line)
            
                first_line = False
        
            print("─" * 100)
    
        print("\n" + "═" * 100)
    
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
                    self._save_improved_prompts_collection(model_name, improved_prompt)
                
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
    
    # async def interactive_menu(self):
    #     """Main interactive menu"""
    #     print("\n🤖 Welcome to the Interactive Model Evaluator")
        
    #     while True:
    #         try:
    #             print("\n" + "="*50)
    #             print("MAIN MENU")
    #             print("="*50)
                
    #             # In the interactive_menu method, update the choices:
    #             choice = self._get_user_choice([
    #                 "Enter your own question",
    #                 "Choose from sample questions",
    #                 "Set reference model",
    #                 "Evaluate with improved prompts",  # New option
    #                 "Exit"
    #             ], "What would you like to do?")
                
    #             if choice == "Enter your own question":
    #                 question = input("\nEnter your question: ")
    #                 if question.strip():
    #                     await self.evaluate_question(question)
    #                     # Add a pause here to make sure user sees results
    #                     input("\nPress Enter to continue...")
                
    #             elif choice == "Choose from sample questions":
    #                 sample_questions = self.load_sample_questions()
    #                 question = self._get_user_choice(sample_questions, "Choose a question:")
    #                 await self.evaluate_question(question)
    #                 # Add a pause here to make sure user sees results
    #                 input("\nPress Enter to continue...")
                
    #             elif choice == "Set reference model":
    #                 model_names = self.model_factory.get_model_names()
    #                 self.reference_model = self._get_user_choice(
    #                     model_names, 
    #                     "Choose reference model (ground truth):"
    #                 )
    #                 print(f"Reference model set to: {self.reference_model}")
                
    #             elif choice == "Exit":
    #                 print("\nThank you for using the Interactive Model Evaluator. Goodbye!")
    #                 break
                    
    #         except Exception as e:
    #             print(f"\nAn error occurred: {e}")
    #             print("Returning to main menu...")
    #             input("\nPress Enter to continue...")

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
                if last_question:
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
                    if not os.path.exists("prompts/improved_prompts_collection.json"):
                        print("\n⚠️ No improved prompts found. Please evaluate some questions first.")
                    else:
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