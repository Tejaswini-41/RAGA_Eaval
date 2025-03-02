"""
Use SBERT for Relevance
Use Gemini  for Groundedness
Use Gemini for now with 10 que sets for Completeness
Avoid Gemini API for Accuracy
"""

import os
import json
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import asyncio
from ragas.metrics import BleuScore, RougeScore
from ragas.dataset_schema import SingleTurnSample
import re

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Load sentence transformer for semantic similarity
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to read response from a JSON file
def read_response(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        return data  # Return the entire JSON object

# Compute relevance using SBERT embeddings (like RAGAS)
def compute_relevance(reference, response):
    if not reference or not response:
        print("Empty reference or response for relevance computation.")
        return 0.0
    ref_embedding = embedder.encode(reference, convert_to_tensor=True)
    resp_embedding = embedder.encode(response, convert_to_tensor=True)
    return util.pytorch_cos_sim(ref_embedding, resp_embedding).item()

# Compute factual accuracy using Gemini
def compute_accuracy(response):
    prompt = f"Evaluate factual correctness of:\n\n'{response}'\n\nReturn a score between 0 (false) and 1 (fully accurate). Provide only the score."
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return float(response.text.strip())  # Extract numerical score
    except Exception as e:
        print(f"Error computing accuracy: {e}")
        return 0.5  # Default score if Gemini fails

# Compute groundedness using Gemini (replacing TF-IDF)
def compute_groundedness(reference, response):
    if not reference or not response:
        print("Empty reference or response for groundedness computation.")
        return 0.0
    
    prompt = f"""Evaluate how well the response is grounded in the reference text:

    Reference: '{reference}'
    Response: '{response}'

    Score groundedness on a scale from 0 to 1, where:
    - 0 means the response contains information completely absent from the reference
    - 1 means all information in the response can be directly traced to the reference

    Return only a numerical score.
    """
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content(prompt)
        import re
        match = re.search(r'(\d+\.\d+|\d+)', result.text.strip())
        if match:
            return float(match.group(0))
        return float(result.text.strip())
    except Exception as e:
        print(f"Error computing groundedness: {e}")
        return 0.5  # Default score if Gemini fails

# Compute answer completeness using Gemini (more sophisticated than word overlap)
def compute_completeness(reference, response):
    if not reference or not response:
        print("Empty reference or response for completeness computation.")
        return 0.0
    
    prompt = f"""Evaluate how complete the response is compared to the reference:

    Reference: '{reference}'
    Response: '{response}'

    Score completeness on a scale from 0 to 1, where:
    - 0 means the response misses all key information from the reference
    - 1 means the response covers all key information from the reference

    Return only a numerical score.
    """
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content(prompt)
        import re
        match = re.search(r'(\d+\.\d+|\d+)', result.text.strip())
        if match:
            return float(match.group(0))
        return float(result.text.strip())
    except Exception as e:
        print(f"Error computing completeness: {e}")
        return 0.5  # Default score if Gemini fails

async def compute_bleu_ragas(reference, response):
    """Compute BLEU score using RAGAS implementation."""
    if not reference or not response:
        print("Empty reference or response for BLEU computation.")
        return 0.0
    
    try:
        sample = SingleTurnSample(
            response=response,
            reference=reference
        )
        
        scorer = BleuScore()
        score = await scorer.single_turn_ascore(sample)
        return score
    except Exception as e:
        print(f"Error computing BLEU with RAGAS: {e}")
        return 0.0

async def compute_rouge_ragas(reference, response):
    """Compute ROUGE score using RAGAS implementation."""
    if not reference or not response:
        print("Empty reference or response for ROUGE computation.")
        return 0.0
    
    try:
        sample = SingleTurnSample(
            response=response,
            reference=reference
        )
        
        scorer = RougeScore()
        score = await scorer.single_turn_ascore(sample)
        return score
    except Exception as e:
        print(f"Error computing ROUGE with RAGAS: {e}")
        return 0.0

# Modified evaluate_raga function to include BLEU and ROUGE metrics
async def evaluate_raga_with_bleu_rouge(reference, responses):
    scores = {}
    for model_name, response in responses.items():
        print(f"Evaluating {model_name}...")
        
        # Original RAGAS metrics
        base_scores = {
            "Relevance": round(compute_relevance(reference, response), 3),
            "Accuracy": round(compute_accuracy(response), 3),
            "Groundedness": round(compute_groundedness(reference, response), 3),
            "Completeness": round(compute_completeness(reference, response), 3)
        }
        
        # RAGAS BLEU and ROUGE scores
        bleu_score = round(await compute_bleu_ragas(reference, response), 3)
        rouge_score = round(await compute_rouge_ragas(reference, response), 3)
        
        # Combine all scores
        scores[model_name] = {
            **base_scores,
            "BLEU": bleu_score,
            "ROUGE": rouge_score
        }
    
    return scores

# Updated calculate_overall_scores function to include new metrics
def calculate_overall_scores_with_bleu_rouge(raga_scores):
    """
    Calculate overall score including BLEU and ROUGE metrics
    """
    # Define weights for each metric (adjusted to include new metrics)
    weights = {
        "Relevance": 0.20,
        "Accuracy": 0.25,
        "Groundedness": 0.20,
        "Completeness": 0.15,
        "BLEU Score": 0.10,
        "ROUGE Score": 0.10
    }
    
    # Calculate overall weighted score for each model
    overall_scores = {}
    for model, metrics in raga_scores.items():
        weighted_sum = sum(metrics.get(metric, 0) * weight 
                          for metric, weight in weights.items() 
                          if metric in metrics)
        overall_scores[model] = round(weighted_sum, 3)
    
    # Sort models by overall score (descending)
    sorted_models = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_models, overall_scores

# File paths (update as needed)
response_files = {
    "Gemini": "responses/gemini-1.5-flash.json",  # Ground truth
    "DeepSeek": "responses/deepseek.json",
    "Llama": "responses/llama.json",
    "Alibaba": "responses/ali-baba.json",
    "Mixtral": "responses/mixtral.json"
}

# Replace the main execution code with this async version
async def main():
    # Read responses
    gemini_response = read_response(response_files["Gemini"])  # Ground truth
    model_responses = {k: read_response(v) for k, v in response_files.items() if k != "Gemini"}

    # Ensure responses are strings
    gemini_response = str(gemini_response)
    model_responses = {k: str(v) for k, v in model_responses.items()}

    # Compute RAGA scores with BLEU and ROUGE
    raga_scores = await evaluate_raga_with_bleu_rouge(gemini_response, model_responses)

    # Calculate overall scores with new metrics
    sorted_models, overall_scores = calculate_overall_scores_with_bleu_rouge(raga_scores)

    # Print overall results in sorted order
    print("\n🔍 Extended RAGAS Scores (Sorted by Performance):")
    for rank, (model, score) in enumerate(sorted_models, 1):
        print(f"{rank}. {model}: {score}")
        metrics = raga_scores[model]
        print(f"   - Relevance: {metrics['Relevance']}")
        print(f"   - Accuracy: {metrics['Accuracy']}")
        print(f"   - Groundedness: {metrics['Groundedness']}")
        print(f"   - Completeness: {metrics['Completeness']}")
        print(f"   - BLEU: {metrics['BLEU']}")
        print(f"   - ROUGE: {metrics['ROUGE']}")
        print()

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())