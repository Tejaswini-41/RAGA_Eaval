from ragas.metrics import Faithfulness
from ragas.dataset_schema import SingleTurnSample
from openai import OpenAI
import asyncio
from typing import List, Any

class OpenAIWrapper:
    def __init__(self, client: OpenAI):
        self.client = client

    async def generate(self, prompts: List[str], **kwargs) -> List[str]:
        responses = []
        for prompt in prompts:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="gpt-4o",
                temperature=0,
                max_tokens=1000
            )
            responses.append(response.choices[0].message.content)
        return responses

# Initialize OpenAI client
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key="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",  # Replace with your actual API key
)

# Create evaluator LLM with wrapper
evaluator_llm = OpenAIWrapper(client)

# Sample data for evaluation
samples = [
    SingleTurnSample(
        user_input="When was the first super bowl?",
        response="The first Super Bowl was held on January 15, 1967 at the Los Angeles Memorial Coliseum.",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ]
    )
]

async def evaluate_faithfulness():
    # Initialize Faithfulness metric
    faithfulness = Faithfulness(llm=evaluator_llm)
    
    # Evaluate each sample
    for sample in samples:
        score = await faithfulness.single_turn_ascore(sample)
        print(f"\nQuestion: {sample.user_input}")
        print(f"Response: {sample.response}")
        print(f"Context: {sample.retrieved_contexts[0]}")
        print(f"Faithfulness Score: {score:.2f}")

if __name__ == "__main__":
    # Run the evaluation
    asyncio.run(evaluate_faithfulness())