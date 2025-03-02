import os
import json
import asyncio
import aiohttp
from typing import Dict, List
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Configuration
INPUT_FILE = "questions.json"
BATCH_SIZE = 5  # Process questions in batches
MAX_RETRIES = 3
TIMEOUT = 30  # seconds

MODELS = {
    "Mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "Falcon": "tiiuae/falcon-7b-instruct",
    "BLOOM": "bigscience/bloomz-7b1"
}

class HFGenerator:
    def __init__(self):
        load_dotenv()
        self.token = os.getenv('HF_TOKEN')
        if not self.token:
            raise ValueError("HF_TOKEN not found in environment variables")
        self.client = InferenceClient(
            token=self.token,
            timeout=TIMEOUT,
            headers={"Authorization": f"Bearer {self.token}"}
        )

    async def generate_response(self, question: str, model_name: str) -> str:
        """Generate response with retry logic."""
        for attempt in range(MAX_RETRIES):
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    ThreadPoolExecutor(),
                    lambda: self.client.text_generation(
                        prompt=f"Question: {question}\nAnswer:",
                        model=model_name,
                        max_new_tokens=150,  # Reduced for faster response
                        temperature=0.7,
                        repetition_penalty=1.1
                    )
                )
                return response.strip()
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return f"Error after {MAX_RETRIES} attempts: {str(e)}"
                await asyncio.sleep(1)  # Wait before retry

    async def process_batch(self, questions: List[str], model_name: str) -> Dict[str, str]:
        """Process a batch of questions concurrently."""
        tasks = [self.generate_response(q, model_name) for q in questions]
        responses = await asyncio.gather(*tasks)
        return dict(zip(questions, responses))

    async def process_all_questions(self, questions: List[str]) -> None:
        """Process all questions with progress bar."""
        for model_name, model_id in MODELS.items():
            responses = {}
            print(f"\nProcessing with {model_name} model...")
            
            # Process questions in batches with progress bar
            with tqdm(total=len(questions), desc="Processing") as pbar:
                for i in range(0, len(questions), BATCH_SIZE):
                    batch = questions[i:i + BATCH_SIZE]
                    batch_responses = await self.process_batch(batch, model_id)
                    responses.update(batch_responses)
                    pbar.update(len(batch))

            # Save responses
            output_file = f"responses_{model_name.lower()}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(responses, f, indent=4, ensure_ascii=False)
            print(f"✓ Saved to {output_file}")

async def main():
    """Main execution flow with async support."""
    try:
        # Load questions
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            questions = json.load(f)

        generator = HFGenerator()
        await generator.process_all_questions(questions)

    except Exception as e:
        print(f"Application error: {e}")

if __name__ == "__main__":
    asyncio.run(main())