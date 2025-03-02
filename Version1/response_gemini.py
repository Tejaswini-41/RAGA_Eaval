import os
import json
from typing import Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

# Configuration
INPUT_FILE = "questions.json"
OUTPUT_FOLDER = "responses"
MODEL_NAME = "gemini-1.5-flash"
SYSTEM_PROMPT = """You are a helpful AI assistant. Please follow these guidelines:
- Provide concise and accurate answers
- Focus on factual information
- Use clear and simple language
- Avoid speculation or uncertain statements
- Keep responses brief but complete
- If unsure, acknowledge limitations
- Maintain professional tone"""

def setup_model() -> genai.GenerativeModel:
    """Setup and return Gemini model with API key."""
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)

def get_response(question: str, model: genai.GenerativeModel) -> str:
    """Get response from Gemini model with system prompt."""
    try:
        prompt = f"{SYSTEM_PROMPT}\nQuestion: {question}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error getting response: {e}")
        return f"Error: {str(e)}"

def process_questions(model: genai.GenerativeModel) -> None:
    """Process questions and save responses."""
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as file:
            questions = json.load(file)

        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        responses = {}
        for i, question in enumerate(questions, 1):
            print(f"Processing [{i}/{len(questions)}]: {question}")
            responses[question] = get_response(question, model)
        
        output_file = os.path.join(OUTPUT_FOLDER, f"{MODEL_NAME.lower()}.json")
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(responses, file, indent=4, ensure_ascii=False)
        print(f"✓ Responses saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing questions: {e}")

def main():
    """Main execution flow."""
    try:
        model = setup_model()
        process_questions(model)
    except Exception as e:
        print(f"Application error: {e}")

if __name__ == "__main__":
    main()