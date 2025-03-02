import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables and setup OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_answer(question):
    """Generate an answer for a given question using OpenAI API."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a helpful AI assistant. Please follow these guidelines:
                - Provide concise and accurate answers
                - Focus on factual information
                - Use clear and simple language
                - Avoid speculation or uncertain statements
                - Keep responses brief but complete
                - If unsure, acknowledge limitations
                - Maintain professional tone"""},
                {"role": "user", "content": question}
            ],
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def process_questions(input_file='questions.txt', output_file='ground_truth.txt'):
    """Process questions from input file and save answers to output file."""
    # Read questions
    with open(input_file, 'r') as f:
        questions = [q.strip() for q in f.readlines() if q.strip()]

    # Generate answers
    with open(output_file, 'w') as f:
        for i, question in enumerate(questions, 1):
            print(f"Processing question {i}: {question}")
            
            answer = generate_answer(question)
            f.write(f"Q{i}: {question}\n")
            f.write(f"A: {answer or '[Failed to generate response]'}\n\n")
            
            time.sleep(1)  # Rate limiting

if __name__ == "__main__":
    process_questions()
    print("Ground truth generation completed!")