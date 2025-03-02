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
        
        # Generate responses from all models
        responses = self.model_factory.generate_all_responses(question)
        
        # Save responses to file
        self._save_responses(question, responses)
        
        # Run evaluation
        results = await self.evaluator.evaluate_responses(self.reference_model, responses)
        
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
        """Display evaluation results"""
        print("\n" + "="*80)
        print(f"📊 Evaluation Results for Question: {question}")
        print("="*80)
        
        print(f"\n🏆 Rankings (Reference: {results['reference_model']}):")
        for i, (model_name, scores) in enumerate(results['rankings'], 1):
            print(f"\n{i}. {model_name.upper()} - Overall Score: {scores['Overall']}")
            for metric, score in scores.items():
                if metric != "Overall":
                    print(f"   - {metric}: {score}")
        
        print("\n" + "="*80)
    
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
        print("\n🤖 Welcome to the Interactive Model Evaluator 🤖")
        
        while True:
            print("\n" + "=-"*50)
            print("MAIN MENU")
            print("=-"*50)
            
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
            
            elif choice == "Choose from sample questions":
                sample_questions = self.load_sample_questions()
                question = self._get_user_choice(sample_questions, "Choose a question:")
                await self.evaluate_question(question)
            
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

async def main():
    evaluator = InteractiveEvaluator()
    await evaluator.interactive_menu()

if __name__ == "__main__":
    asyncio.run(main())