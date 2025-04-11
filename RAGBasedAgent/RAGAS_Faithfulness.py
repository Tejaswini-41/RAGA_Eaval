from ragas.metrics import Faithfulness
from ragas.dataset_schema import SingleTurnSample
import asyncio
from typing import List, Any
from FreeLLM_Wrapper import FreeLLMWrapper

async def evaluate_faithfulness_with_free_api(reference, response, model_type="claude"):
    """Evaluate faithfulness using free LLM API"""
    # Initialize free LLM wrapper
    free_llm = FreeLLMWrapper(model_type=model_type)
    
    # Initialize Faithfulness metric
    faithfulness = Faithfulness(llm=free_llm)
    
    # Create sample format expected by RAGAS
    sample = SingleTurnSample(
        user_input="Review this PR",
        response=response,
        retrieved_contexts=[reference]
    )
    
    # Get faithfulness score
    score = await faithfulness.single_turn_ascore(sample)
    return score

# Example usage
async def test_free_llm_faithfulness():
    reference = """The First AFL–NFL World Championship Game was an American football game 
                  played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."""
    
    response = """The first Super Bowl was held on January 15, 1967 at the Los Angeles Memorial Coliseum."""
    
    # Try with different models
    claude_score = await evaluate_faithfulness_with_free_api(reference, response, "claude")
    llama_score = await evaluate_faithfulness_with_free_api(reference, response, "llama")
    
    print(f"Claude Faithfulness Score: {claude_score:.2f}")
    print(f"Llama Faithfulness Score: {llama_score:.2f}")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_free_llm_faithfulness())