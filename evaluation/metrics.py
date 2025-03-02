import re
import asyncio
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from ragas.metrics import BleuScore, RougeScore
from ragas.dataset_schema import SingleTurnSample
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Load sentence transformer for semantic similarity
embedder = SentenceTransformer('all-MiniLM-L6-v2')

class MetricsCalculator:
    def __init__(self):
        self.embedder = embedder
        self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        
    def compute_relevance(self, reference, response):
        """Compute relevance using SBERT embeddings"""
        if not reference or not response:
            print("Empty reference or response for relevance computation.")
            return 0.0
        ref_embedding = self.embedder.encode(reference, convert_to_tensor=True)
        resp_embedding = self.embedder.encode(response, convert_to_tensor=True)
        return util.pytorch_cos_sim(ref_embedding, resp_embedding).item()
    
    def compute_accuracy(self, response):
        """Compute factual accuracy using Gemini"""
        prompt = f"Evaluate factual correctness of:\n\n'{response}'\n\nReturn a score between 0 (false) and 1 (fully accurate). Provide only the score."
        
        try:
            result = self.gemini_model.generate_content(prompt)
            return float(result.text.strip())
        except Exception as e:
            print(f"Error computing accuracy: {e}")
            return 0.5
    
    def compute_groundedness(self, reference, response):
        """Compute groundedness using Gemini"""
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
            result = self.gemini_model.generate_content(prompt)
            match = re.search(r'(\d+\.\d+|\d+)', result.text.strip())
            if match:
                return float(match.group(0))
            return float(result.text.strip())
        except Exception as e:
            print(f"Error computing groundedness: {e}")
            return 0.5
    
    def compute_completeness(self, reference, response):
        """Compute answer completeness using Gemini"""
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
            result = self.gemini_model.generate_content(prompt)
            match = re.search(r'(\d+\.\d+|\d+)', result.text.strip())
            if match:
                return float(match.group(0))
            return float(result.text.strip())
        except Exception as e:
            print(f"Error computing completeness: {e}")
            return 0.5
    
    async def compute_bleu_ragas(self, reference, response):
        """Compute BLEU score using RAGAS implementation"""
        if not reference or not response:
            print("Empty reference or response for BLEU computation.")
            return 0.0
        
        try:
            sample = SingleTurnSample(response=response, reference=reference)
            scorer = BleuScore()
            score = await scorer.single_turn_ascore(sample)
            return score
        except Exception as e:
            print(f"Error computing BLEU with RAGAS: {e}")
            return 0.0
    
    async def compute_rouge_ragas(self, reference, response):
        """Compute ROUGE score using RAGAS implementation"""
        if not reference or not response:
            print("Empty reference or response for ROUGE computation.")
            return 0.0
        
        try:
            sample = SingleTurnSample(response=response, reference=reference)
            scorer = RougeScore()
            score = await scorer.single_turn_ascore(sample)
            return score
        except Exception as e:
            print(f"Error computing ROUGE with RAGAS: {e}")
            return 0.0