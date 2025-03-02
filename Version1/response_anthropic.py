import anthropic
from dotenv import load_dotenv
import os
from typing import Dict, Any
import json
import warnings

# Configuration
INPUT_FILE = "questions.json"
SYSTEM_PROMPT = """You are a helpful AI assistant. Please follow these guidelines:
- Provide concise and accurate answers
- Focus on factual information
- Use clear and simple language
- Avoid speculation or uncertain statements
- Keep responses brief but complete
- If unsure, acknowledge limitations
- Maintain professional tone"""

class ClaudeGenerator:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        self.client = anthropic.Anthropic(api_key=api_key)
        # Suppress deprecation warning
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def get_response(self, question: str) -> str:
        try:
            message = self.client.messages.create(
                model="claude-3-opus-20240229",  # Updated to latest model
                max_tokens=500,
                temperature=0.7,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": question}
                ]
            )
            return message.content
        except Exception as e:
            return f"Error: {str(e)}"

    def process_questions(self) -> None:
        try:
            with open(INPUT_FILE, "r", encoding="utf-8") as f:
                questions = json.load(f)
            
            responses = {}
            for i, question in enumerate(questions, 1):
                print(f"Processing [{i}/{len(questions)}]: {question}")
                responses[question] = self.get_response(question)
            
            with open("responses_claude.json", "w", encoding="utf-8") as f:
                json.dump(responses, f, indent=4, ensure_ascii=False)
            print("✓ Responses saved to responses_claude.json")
                
        except Exception as e:
            print(f"Error processing questions: {e}")

def main():
    try:
        generator = ClaudeGenerator()
        generator.process_questions()
    except Exception as e:
        print(f"Application error: {e}")

if __name__ == "__main__":
    main()