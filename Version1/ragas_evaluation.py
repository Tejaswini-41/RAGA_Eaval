""" 
Use SBERT for Relevance
Use TF-IDF similarity for Groundedness
Use Python sets for Completeness
Avoid Gemini API for Accuracy
"""


import os
import json
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

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

# Compute relevance using SBERT embeddings
def compute_relevance(reference, response):
    if not reference or not response:
        print("Empty reference or response for relevance computation.")
        return 0.0
    ref_embedding = embedder.encode(reference, convert_to_tensor=True)
    resp_embedding = embedder.encode(response, convert_to_tensor=True)
    return util.pytorch_cos_sim(ref_embedding, resp_embedding).item()

# Compute factual accuracy using Gemini
def compute_accuracy(response):
    prompt = f"""Evaluate the factual correctness of the following text:

'{response}'

Score the factual accuracy on a scale of 0.0 to 1.0 where:
- 0.0 means completely incorrect or misleading
- 0.5 means partially accurate with some errors
- 1.0 means completely accurate with no factual errors

Carefully analyze the statements for factual accuracy, logical consistency, and truthfulness.
Provide only a numerical score between 0.0 and 1.0, with no additional text."""
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        generated_response = model.generate_content(prompt)
        # Extract just the number from the response
        import re
        match = re.search(r'(\d+\.\d+|\d+)', generated_response.text.strip())
        if match:
            return float(match.group(0))
        return float(generated_response.text.strip())  # Fallback to original approach
    except Exception as e:
        print(f"Error computing accuracy: {e}")
        return 0.5  # Default score if Gemini fails

# Compute groundedness using TF-IDF cosine similarity
def compute_groundedness(reference, response):
    if not reference or not response:
        print("Empty reference or response for groundedness computation.")
        return 0.0
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([reference, response])
    return (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]

# Compute answer completeness (word overlap)
def compute_completeness(reference, response):
    if not reference or not response:
        print("Empty reference or response for completeness computation.")
        return 0.0
    ref_words = set(reference.lower().split())
    resp_words = set(response.lower().split())
    return len(resp_words.intersection(ref_words)) / len(ref_words) if ref_words else 0

# Evaluate RAGA metrics for all models
def evaluate_raga(reference, responses):
    scores = {}
    for model_name, response in responses.items():
        print(f"Evaluating {model_name}...")
        scores[model_name] = {
            "Relevance": round(compute_relevance(reference, response), 3),
            "Accuracy": round(compute_accuracy(response), 3),
            "Groundedness": round(compute_groundedness(reference, response), 3),
            "Completeness": round(compute_completeness(reference, response), 3)
        }
    return scores

# File paths (update as needed)
response_files = {
    "Gemini": "responses/gemini-1.5-flash.json",  # Ground truth
    "DeepSeek": "responses/deepseek.json",
    "Llama": "responses/llama.json",
    "Alibaba": "responses/ali-baba.json",
    "Mixtral": "responses/mixtral.json"
}

# Read responses
gemini_response = read_response(response_files["Gemini"])  # Ground truth
model_responses = {k: read_response(v) for k, v in response_files.items() if k != "Gemini"}

# Ensure responses are strings
gemini_response = str(gemini_response)
model_responses = {k: str(v) for k, v in model_responses.items()}

# Compute RAGA scores
raga_scores = evaluate_raga(gemini_response, model_responses)

# Print results
for model, scores in raga_scores.items():
    print(f"\n🔹 {model} RAGA Scores:")
    for metric, value in scores.items():
        print(f"  - {metric}: {value}")
