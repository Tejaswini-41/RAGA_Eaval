import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AspectCritic
from ragas import evaluate
from langchain_google_genai import ChatGoogleGenerativeAI
from datasets import Dataset
import pandas as pd

# Load API keys
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini LLM for RAGAS using LangChain Wrapper
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", api_key=GOOGLE_API_KEY)
evaluator_llm = LangchainLLMWrapper(llm)

# Define AspectCritic with name and definition
aspect_critic = AspectCritic(
    name="response_quality",
    definition="""Evaluate the response based on the following aspects:
    1. Faithfulness: Does the response accurately reflect the ground truth?
    2. Relevance: Is the response relevant to the query?
    3. Coherence: Is the response well-structured and logically coherent?
    4. Informativeness: Does the response provide comprehensive information?"""
)

# Function to read response from a JSON file
def read_response(file_path):
    """Read and parse JSON responses."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        # Convert dictionary to string representation if needed
        if isinstance(data, dict):
            return "\n".join([f"Q: {q}\nA: {a}" for q, a in data.items()])
        return str(data)

# Prepare dataset for evaluation
def prepare_dataset(reference, responses):
    """Prepare dataset in RAGAS format."""
    data = {
        "query": [],
        "ground_truth": [],
        "retrieved_docs": [],
        "generated_response": []
    }
    
    for model_name, response in responses.items():
        # Ensure all inputs are strings
        query = "Evaluate the response quality"
        ground_truth = str(reference)
        retrieved_doc = str(reference)  # Using reference as context
        generated_resp = str(response)
        
        data["query"].append(query)
        data["ground_truth"].append(ground_truth)
        data["retrieved_docs"].append([retrieved_doc])
        data["generated_response"].append(generated_resp)
    
    return Dataset.from_dict(data)

# Evaluate RAGAS metrics using AspectCritic
def evaluate_aspect_critic(reference, responses):
    """Evaluate responses using AspectCritic."""
    try:
        dataset = prepare_dataset(reference, responses)
        print("Dataset prepared successfully")
        print(f"Number of samples: {len(dataset)}")
        
        results = evaluate(
            dataset=dataset,
            metrics=[aspect_critic],
            llm=evaluator_llm  # Use Gemini as LLM for evaluation
        )
        return results.to_pandas()  # Convert results to pandas DataFrame
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None

# File paths (update as needed)
response_files = {
    "Gemini": "responses/gemini-pro.json",  # Ground truth
    "DeepSeek": "responses/deepseek.json",
    "Llama": "responses/llama.json",
    "Alibaba": "responses/ali-baba.json",
    "Mixtral": "responses/mixtral.json"
}

# Read and process responses
try:
    gemini_response = read_response(response_files["Gemini"])  # Ground truth
    model_responses = {k: read_response(v) for k, v in response_files.items() if k != "Gemini"}
    
    print("Ground truth response loaded:", type(gemini_response))
    print("Model responses loaded:", len(model_responses))
    
    # Compute AspectCritic scores
    aspect_scores_df = evaluate_aspect_critic(gemini_response, model_responses)
    
    if aspect_scores_df is not None:
        print("\n🔹 AspectCritic Scores:")
        print(aspect_scores_df)
        
        # Save results to CSV
        output_file = "responses/aspect_critic_scores.csv"
        aspect_scores_df.to_csv(output_file)
        print(f"\nScores saved to: {output_file}")
    else:
        print("Evaluation failed to produce scores.")
        
except Exception as e:
    print(f"Error in main execution: {str(e)}")