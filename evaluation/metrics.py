import re
import asyncio
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from ragas.metrics import BleuScore, RougeScore
from ragas.dataset_schema import SingleTurnSample
from dotenv import load_dotenv
import os
import sys
import importlib.util

# Determine file paths dynamically using os.path
current_dir = os.path.dirname(os.path.abspath(__file__))  # metrics.py directory
project_root = os.path.abspath(os.path.join(current_dir, '..'))  # Go up one level to project root
sys.path.append(project_root)

try:
    # First try standard import (if package structure is correct)
    try:
        from RAGBasedAgent.FreeLLM_Wrapper import FreeLLMWrapper
        # print("Successfully imported FreeLLMWrapper via standard import")
    except ImportError:
        # Fall back to dynamic import if standard import fails
        wrapper_path = os.path.join(project_root, 'RAGBasedAgent', 'FreeLLM_Wrapper.py')
        if not os.path.exists(wrapper_path):
            raise ImportError(f"FreeLLM_Wrapper.py not found at {wrapper_path}")
            
        spec = importlib.util.spec_from_file_location("FreeLLM_Wrapper", wrapper_path)
        wrapper_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper_module)
        FreeLLMWrapper = wrapper_module.FreeLLMWrapper
        # print(f"Successfully imported FreeLLMWrapper via dynamic import from {wrapper_path}")
except Exception as e:
    print(f"Could not import FreeLLM_Wrapper: {e}")
    # Create dummy class as fallback
    class FreeLLMWrapper:
        def __init__(self):
            pass
        async def generate(self, prompts):
            return ["API not available"] * len(prompts)

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
        
        # SIMPLER APPROACH: Only initialize one LLM wrapper
        try:
            # Use direct import rather than dynamic import
            self.free_llm = FreeLLMWrapper()  # No model type needed now
            self.free_llm_available = True
            # print("Free LLM API (GPT-3.5-turbo) integration available")
        except Exception as e:
            print(f"Free LLM API not available: {e}")
            self.free_llm_available = False
            self.free_llm = None
        
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
        """Compute faithfulness based on content similarity"""
        try:
            # Apply basic similarity check
            if hasattr(self, '_clean_text'):
                clean_reference = self._clean_text(reference)
                clean_response = self._clean_text(response)
            else:
                # Fallback if _clean_text doesn't exist
                clean_reference = reference
                clean_response = response
        
            # Handle empty content
            if not clean_reference or not clean_response:
                return 0.6  # Default value
                
            # Extract significant words
            ref_words = set(self._get_significant_words(clean_reference))
            resp_words = set(self._get_significant_words(clean_response))
            
            # Avoid division by zero
            if not ref_words or not resp_words:
                return 0.6  # Default value
            
            # Calculate Jaccard similarity
            intersection = len(ref_words.intersection(resp_words))
            union = len(ref_words.union(resp_words))
            
            # Scale to make values more meaningful
            similarity = intersection / union
            scaled_score = 0.3 + similarity * 0.6  # Scale to 0.3-0.9 range
            
            return min(0.9, max(0.3, scaled_score))
            
        except Exception as e:
            print(f"Error in compute_faithfulness: {e}")
            return 0.6  # Return a reasonable default
        
    def compute_answer_relevance(self, reference, response):
        """Compute answer relevance score"""
        try:
            # Implement fallback relevance calculation
            # For simplicity, use jaccard similarity on significant words
            ref_words = set(self._get_significant_words(reference))
            resp_words = set(self._get_significant_words(response))
            
            # Avoid division by zero
            if not ref_words or not resp_words:
                return 0.5
                
            # Calculate Jaccard similarity
            intersection = len(ref_words.intersection(resp_words))
            union = len(ref_words.union(resp_words))
            
            # Scale similarity to 0.4-0.9 range
            similarity = intersection / union
            scaled_score = 0.4 + similarity * 0.5
            
            return min(0.9, max(0.4, scaled_score))
            
        except Exception as e:
            print(f"Error in compute_answer_relevance: {e}")
            return 0.6  # Return a reasonable default
        
    def _get_significant_words(self, text):
        """Extract significant words from text"""
        # Simple implementation to get important words
        words = text.lower().split()
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'by', 'with'}
        return [word for word in words if word not in stop_words and len(word) > 2]
        
    def _embed_text(self, text):
        """Generate a simple embedding for text"""
        try:
            # Fallback to TF-IDF if available
            if hasattr(self, 'embedding_function') and self.embedding_function:
                return self.embedding_function([text])[0]
                
            # Very simple fallback - word frequency vector
            words = text.lower().split()
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
                
            # Create a simple vector
            vector = [count for word, count in sorted(word_freq.items())]
            
            # Normalize
            magnitude = sum(v*v for v in vector) ** 0.5
            if magnitude > 0:
                return [v/magnitude for v in vector]
            return vector
            
        except:
            return None
        
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        try:
            # Handle different length vectors
            if len(vec1) != len(vec2):
                # Pad the shorter one
                if len(vec1) < len(vec2):
                    vec1 = vec1 + [0] * (len(vec2) - len(vec1))
                else:
                    vec2 = vec2 + [0] * (len(vec1) - len(vec2))
                    
            # Calculate dot product
            dot_product = sum(a*b for a, b in zip(vec1, vec2))
            
            # Calculate magnitudes
            mag1 = sum(a*a for a in vec1) ** 0.5
            mag2 = sum(b*b for b in vec2) ** 0.5
            
            # Avoid division by zero
            if mag1 * mag2 == 0:
                return 0
                
            return dot_product / (mag1 * mag2)
            
        except:
            return 0.5  # Default similarity

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

    # Simplify the RAGAS faithfulness method
    async def get_ragas_faithfulness(self, reference, response, use_custom=True):
        """Simplified RAGAS faithfulness using GPT-3.5-turbo"""
        # Default to custom implementation for reliability
        if use_custom or not self.free_llm_available:
            return self.compute_faithfulness(reference, response)
        
        # Safety check to prevent token limit errors
        max_length = 4000  # Keep well under token limits
        if len(reference) > max_length or len(response) > max_length:
            # print("Content too long for RAGAS faithfulness, using custom implementation")
            return self.compute_faithfulness(reference, response)
        
        try:
            # Try RAGAS faithfulness with simplified approach
            from ragas.metrics import Faithfulness
            
            if not self.faithfulness_metric and self.free_llm:
                self.faithfulness_metric = Faithfulness(llm=self.free_llm)
            
            if not self.faithfulness_metric:
                return self.compute_faithfulness(reference, response)
            
            # Create sample for RAGAS
            from ragas.dataset_schema import SingleTurnSample
            sample = SingleTurnSample(
                user_input="Review this PR",
                response=response,
                retrieved_contexts=[reference]
            )
            
            # Get score with timeout
            score = await asyncio.wait_for(
                self.faithfulness_metric.single_turn_ascore(sample),
                timeout=30  # Timeout after 30 seconds
            )
            return score
            
        except Exception as e:
            print(f"Error with RAGAS faithfulness: {e}")
            return self.compute_faithfulness(reference, response)

    def extract_relevant_pr_content(self, content, max_length=4000):
        """Extract the most relevant parts of PR content for evaluation"""
        if len(content) <= max_length:
            return content
            
        # Split content into chunks
        lines = content.split('\n')
        
        # 1. Extract file paths (critical for faithfulness evaluation)
        file_pattern = r'(?:src|test|lib)\/[\w\/\-\.]+\.\w+'
        file_matches = re.findall(file_pattern, content)
        file_section = "IMPORTANT FILES:\n" + "\n".join(set(file_matches)) + "\n\n"
        
        # 2. Extract code blocks (most relevant for review)
        code_blocks = []
        in_block = False
        current_block = []
        
        for line in lines:
            if line.strip().startswith('```') or line.strip().startswith('~~~'):
                if in_block:
                    # End of block
                    current_block.append(line)
                    code_blocks.append('\n'.join(current_block))
                    current_block = []
                    in_block = False
                else:
                    # Start of block
                    in_block = True
                    current_block = [line]
            elif in_block:
                current_block.append(line)
                
        code_section = "CODE BLOCKS:\n" + "\n\n".join(code_blocks[:3]) + "\n\n"  # Keep top 3 code blocks
        
        # 3. Extract PR summary (usually at beginning)
        summary_length = min(1000, max_length // 4)
        summary = '\n'.join(lines[:20])[:summary_length]
        summary_section = "PR SUMMARY:\n" + summary + "\n\n"
        
        # 4. Add key lines with functions, classes, etc.
        function_pattern = r'(def|class|function|method) [\w\d_]+'
        function_matches = []
        for line in lines:
            if re.search(function_pattern, line):
                function_matches.append(line.strip())
        
        function_section = "KEY FUNCTIONS/CLASSES:\n" + "\n".join(function_matches[:10]) + "\n\n"
        
        # Combine sections within max_length constraint
        combined = summary_section + file_section + function_section + code_section
        
        # Final truncation if still too long
        if len(combined) > max_length:
            return combined[:max_length] + "\n...[content intelligently truncated]"
        
        return combined
    
    def _clean_text(self, text):
        """
        Clean and normalize text for better comparison
        
        Args:
            text: The text string to clean
        
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
            
        # Convert to string if needed
        text = str(text)
        
        try:
            # Remove HTML/markdown syntax
            text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
            text = re.sub(r'#{1,6}\s+', ' ', text)  # Remove markdown headers
            text = re.sub(r'[\*\_]{1,2}([^\*\_]+)[\*\_]{1,2}', r'\1', text)  # Remove bold/italic
            text = re.sub(r'`{1,3}([^`]+)`{1,3}', r'\1', text)  # Remove code blocks
            
            # Remove special characters but keep periods, question marks, etc.
            text = re.sub(r'[^\w\s\.\?\!\,\:\;\-\(\)]', ' ', text)
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            print(f"Error in _clean_text: {e}")
            return text