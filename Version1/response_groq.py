import os
import json
from dotenv import load_dotenv
from typing import Dict, Any
import groq

# Configuration
INPUT_FILE = "questions.json"
OUTPUT_FOLDER = "responses"
SYSTEM_PROMPT = """You are a helpful AI assistant. Please follow these guidelines:
- Provide concise and accurate answers
- Focus on factual information
- Use clear and simple language
- Avoid speculation or uncertain statements
- Keep responses brief but complete
- If unsure, acknowledge limitations
- Maintain professional tone"""

# Define available models
MODELS = {
    "DeepSeek": "deepseek-r1-distill-qwen-32b",
    "Llama": "llama3-70b-8192",
    "Ali-baba": "qwen-2.5-32b",
    "Mixtral": "mixtral-8x7b-32768"
}

def setup_client() -> groq.Client:
    """Setup and return Groq client with API key."""
    load_dotenv()
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    return groq.Client(api_key=api_key)

def get_response(question: str, model_name: str, client: groq.Client) -> str:
    """Get response from Groq model with system prompt."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting response: {e}")
        return f"Error: {str(e)}"

def process_questions(client: groq.Client) -> None:
    """Process questions for each model."""
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as file:
            questions = json.load(file)

        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        for model_name, model_id in MODELS.items():
            responses = {}
            print(f"\nProcessing with {model_name} model...")
            
            for i, question in enumerate(questions, 1):
                print(f"Processing [{i}/{len(questions)}]: {question}")
                responses[question] = get_response(question, model_id, client)
            
            output_file = os.path.join(OUTPUT_FOLDER, f"{model_name.lower()}.json")
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(responses, file, indent=4, ensure_ascii=False)
            print(f"✓ Responses saved to {output_file}")
            
    except Exception as e:
        print(f"Error processing questions: {e}")

def main():
    """Main execution flow."""
    try:
        client = setup_client()
        process_questions(client)
    except Exception as e:
        print(f"Application error: {e}")

if __name__ == "__main__":
    main()
