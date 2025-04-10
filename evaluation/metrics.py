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

    def compute_faithfulness(self, reference, response):
        """
        Measures how factually consistent the response is with the source PR context.
        Detects hallucinated file names, line numbers, or issues.
        """
        try:
            # Extract file mentions from both reference and response
            import re
            file_pattern = r'(?:src|test|lib)\/[\w\/\-\.]+\.\w+'
            ref_files = set(re.findall(file_pattern, reference))
            response_files = set(re.findall(file_pattern, response))
            
            # Files mentioned in response but not in reference are potential hallucinations
            hallucinated_files = response_files - ref_files
            
            # Calculate basic faithfulness score based on file references
            if len(response_files) == 0:
                return 0.5  # No file references at all
            
            file_faithfulness = 1.0 - (len(hallucinated_files) / len(response_files))
            
            # Additional checks for hallucinated function names could be added here
            # For now, return the file-based faithfulness score
            return max(0.0, min(1.0, file_faithfulness))
        
        except Exception as e:
            print(f"Error computing faithfulness: {e}")
            return 0.5  # Default value

    def compute_contextual_precision(self, reference, response):
        """
        Measures how precisely the response references specific elements from the PR context
        (file names, function names, line numbers, etc.)
        """
        try:
            # Count specific references to code elements
            file_refs = len(re.findall(r'`[\w\/\-\.]+\.\w+`', response))
            func_refs = len(re.findall(r'`[\w]+\(`', response))
            line_refs = len(re.findall(r'line \d+', response.lower()))
            
            # Total precision references
            total_refs = file_refs + func_refs + line_refs
            
            # Scale to 0-1 range
            precision_score = min(1.0, total_refs / 10)  # Normalize; 10+ references = perfect score
            
            return precision_score
        
        except Exception as e:
            print(f"Error computing contextual precision: {e}")
            return 0.5  # Default value

    def compute_answer_relevance(self, reference, response):
        """
        Measures how well the response addresses key PR review concerns:
        - Summary of changes
        - File suggestions
        - Conflict predictions
        - Risk warnings
        - Test coverage
        """
        try:
            # Define key sections that should be present in a good PR review
            key_sections = [
                "summary", 
                "file", "suggest", "affect",
                "conflict", "prediction",
                "break", "risk", 
                "test", "coverage"
            ]
            
            # Check for presence of each section
            response_lower = response.lower()
            section_coverage = sum(1 for term in key_sections if term in response_lower) / len(key_sections)
            
            # Check for actionable content (specific suggestions)
            actionable_pattern = r'(should|could|must|recommend|suggest|consider)'
            actionable_score = min(1.0, len(re.findall(actionable_pattern, response_lower)) / 5)
            
            # Combine scores (70% section coverage, 30% actionable content)
            combined_score = (0.7 * section_coverage) + (0.3 * actionable_score)
            
            return combined_score
        
        except Exception as e:
            print(f"Error computing answer relevance: {e}")
            return 0.5  # Default value