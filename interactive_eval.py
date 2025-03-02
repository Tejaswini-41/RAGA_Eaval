import asyncio
import json
import os
from models.model_factory import ModelFactory
from evaluation.evaluator import ResponseEvaluator

class InteractiveEvaluator:
    def __init__(self):
        self.model_factory = ModelFactory()
        self.evaluator = ResponseEvaluator()
        self.reference_model = "gemini"  # Default reference model
        
        # Create responses directory if it doesn't exist
        if not os.path.exists("responses"):
            os.makedirs("responses")
    
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
        
        while True:
            try:
                print("\n" + "="*50)
                print("MAIN MENU")
                print("="*50)
                
                choice = self._get_user_choice([
                    "Enter your own question",
                    "Choose from sample questions",
                    "Set reference model",
                    "Exit"
                ], "What would you like to do?")
                
                if choice == "Enter your own question":
                    question = input("\nEnter your question: ")
                    if question.strip():
                        await self.evaluate_question(question)
                        # Add a pause here to make sure user sees results
                        input("\nPress Enter to continue...")
                
                elif choice == "Choose from sample questions":
                    sample_questions = self.load_sample_questions()
                    question = self._get_user_choice(sample_questions, "Choose a question:")
                    await self.evaluate_question(question)
                    # Add a pause here to make sure user sees results
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