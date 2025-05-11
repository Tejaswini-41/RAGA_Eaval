import os
from typing import Dict, Tuple, List
from review_evaluator import ReviewEvaluator
from models.groq_models import GroqModelFactory
import uuid
import json
from datetime import datetime

# Adjust these parameters in your chunking strategy
CHUNK_SIZE = 1200  # Increase from default (likely 400-800)
CHUNK_OVERLAP = 200  # Increase overlap for better context preservation
MIN_CHUNK_SIZE = 300  # Avoid tiny chunks that lack context

class ChunkingStrategyTester:
    """Tests and compares different chunking strategies for RAG system"""
    
    def __init__(self):
        self.evaluator = ReviewEvaluator()
        self.model = GroqModelFactory.create_model("llama")  # Using Llama model
        
    async def test_chunking_strategies(self, pr_data: Dict, session_id: str = None) -> Tuple[str, Dict]:
        """Run comparison tests between default and chunked strategies"""
        print("\n🧪 Testing Chunking Strategies...")
        
        if not pr_data.get("current_pr_changes"):
            return "No PR data available for testing", {}
        
        # Check if we have cached results for this session
        if session_id:
            cached_results = self._load_cached_results(session_id)
            if cached_results:
                print("\n✅ Using cached chunking test results")
                return cached_results["comparison"], cached_results["metrics"]
        
        # Generate reviews using both strategies
        default_review = await self._generate_default_review(pr_data)
        chunked_review = await self._generate_chunked_review(pr_data)
        
        if not default_review or not chunked_review:
            return "Failed to generate reviews for comparison", {}
            
        # Format comparison results
        comparison = self._format_comparison(default_review, chunked_review)
        
        # Calculate metrics
        metrics = {
            "default_length": len(default_review),
            "chunked_length": len(chunked_review),
            "default_points": len([line for line in default_review.split('\n') if line.strip()]),
            "chunked_points": len([line for line in chunked_review.split('\n') if line.strip()]),
            "improvement": len(chunked_review) / max(1, len(default_review))
        }
        
        # Save results
        self._save_test_results(comparison, metrics, session_id)
        
        return comparison, metrics
    
    async def _generate_default_review(self, pr_data: Dict) -> str:
        """Generate review using default strategy"""
        print("\n📍 Generating review with default strategy...")
        prompt = self._create_review_prompt(pr_data["current_pr_changes"], 
                                         pr_data.get("similar_prs_changes", []))
        return self.model.generate_response(prompt)
    
    async def _generate_chunked_review(self, pr_data: Dict) -> str:
        """Generate review using hybrid semantic chunking strategy"""
        print("\n📍 Generating review with hybrid semantic chunking strategy...")
        
        # Handle both string and list inputs
        current_changes = pr_data.get("current_pr_changes", "")
        similar_changes = pr_data.get("similar_prs_changes", [])
        
        # Apply hybrid semantic chunking to PR content
        chunked_current = self._apply_hybrid_semantic_chunking(current_changes)
        chunked_similar = self._apply_hybrid_semantic_chunking(similar_changes)
        
        # Generate review for each chunk pair
        reviews = []
        max_chunks = max(len(chunked_current), len(chunked_similar))
        
        for i in range(max_chunks):
            print(f"\n📄 Processing chunk {i+1}/{max_chunks}...")
            current_chunk = chunked_current[i] if i < len(chunked_current) else ""
            similar_chunk = chunked_similar[i] if i < len(chunked_similar) else ""
            
            prompt = self._create_review_prompt(current_chunk, similar_chunk)
            chunk_review = self.model.generate_response(prompt)
            reviews.append(chunk_review)
            
        # Combine chunk reviews
        return self._combine_chunk_reviews(reviews)
    
    def _create_review_prompt(self, current_changes: str, similar_changes: str) -> str:
        """Create prompt for review generation"""
        return f"""Please review these code changes and provide specific feedback:

Current Changes:
{current_changes}

Similar Changes for Context:
{similar_changes}

Focus on:
1. Code quality and best practices
2. Potential issues or improvements
3. Consistency with similar changes
"""
    
    def _apply_hybrid_semantic_chunking(self, content: str | list) -> list:
        """
        Apply hybrid semantic chunking strategy to content
        
        Args:
            content: Either a string of code/docs or a list of changes
            
        Returns:
            list: List of semantically chunked content
        """
        if not content:
            return []
            
        # Convert list to string if needed
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
            
        chunks = []
        lines = content.splitlines()
        current_chunk = []
        current_size = 0
        
        # Config parameters
        chunk_size = CHUNK_SIZE  # Increase from 400
        min_overlap = CHUNK_OVERLAP   # Increase from 50
        
        # Track code structure context
        in_function = False
        in_class = False
        indentation_level = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            is_code = not stripped.startswith(('#', '//', '/*', '*'))
            
            # Detect semantic boundaries
            if is_code:
                # Function/method/class detection
                if any(pattern in stripped for pattern in ['def ', 'function ', 'class ']):
                    if current_chunk and current_size > chunk_size / 2:
                        # Complete previous chunk if it's substantial
                        chunks.append('\n'.join(current_chunk))
                        # Keep context from previous chunk
                        context_lines = min(5, len(current_chunk))
                        current_chunk = current_chunk[-context_lines:]
                        current_size = sum(len(line) for line in current_chunk)
                    
                    in_function = 'def ' in stripped or 'function ' in stripped
                    in_class = 'class ' in stripped
                
                # Track indentation for block structure
                if stripped:
                    leading_spaces = len(line) - len(line.lstrip())
                    if leading_spaces < indentation_level and in_function and current_size > chunk_size / 2:
                        # Function end detected by dedent, complete chunk
                        chunks.append('\n'.join(current_chunk))
                        context_lines = min(5, len(current_chunk)) 
                        current_chunk = current_chunk[-context_lines:]
                        current_size = sum(len(line) for line in current_chunk)
                        in_function = False
                    indentation_level = leading_spaces
            else:
                # For comments/docs, look for paragraph breaks
                if not stripped and current_size > chunk_size:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
            
            # Add the current line to chunk
            current_chunk.append(line)
            current_size += len(line)
            
            # Check if we've reached chunk size and at a good break point
            if current_size >= chunk_size:
                if not in_function or not is_code:  # Don't break in middle of function
                    chunks.append('\n'.join(current_chunk))
                    # Adaptive overlap based on content type
                    overlap_lines = min(10 if is_code else 5, len(current_chunk))
                    current_chunk = current_chunk[-overlap_lines:]
                    current_size = sum(len(line) for line in current_chunk)
        
        # Add remaining content as final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks
    
    def _combine_chunk_reviews(self, chunk_reviews: list) -> str:
        """Combine reviews with improved context preservation"""
        if not chunk_reviews:
            return ""
            
        combined = "# Combined Review from Hybrid Semantic Chunking\n\n"
        
        # Extract key sections
        sections = {
            "Summary": [],
            "Issues": [],
            "Improvements": [],
            "Code Quality": []
        }
        
        # Aggregate points by section
        for review in chunk_reviews:
            for line in review.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                # Categorize points
                if "bug" in line.lower() or "error" in line.lower() or "issue" in line.lower():
                    sections["Issues"].append(line)
                elif "improve" in line.lower() or "enhanc" in line.lower() or "better" in line.lower():
                    sections["Improvements"].append(line)
                elif "quality" in line.lower() or "practic" in line.lower() or "standard" in line.lower():
                    sections["Code Quality"].append(line)
                else:
                    sections["Summary"].append(line)
        
        # Add sections to combined review
        for section, points in sections.items():
            if points:
                combined += f"## {section}\n"
                # Deduplicate and sort by length (shorter points first for readability)
                unique_points = sorted(set(points), key=len)
                combined += "\n".join(f"- {point}" for point in unique_points[:20])
                combined += "\n\n"
        
        return combined
    
    def _format_comparison(self, default_review: str, chunked_review: str) -> str:
        """Format the comparison results"""
        return f"""# 📊 Chunking Strategy Comparison Results

## Default Strategy Review
{default_review}

## Hybrid Semantic Chunking Strategy Review
{chunked_review}

## Analysis
- Default Review Length: {len(default_review)} characters
- Chunked Review Length: {len(chunked_review)} characters
- Number of points in default: {len(default_review.split('\n'))}
- Number of points in chunked: {len(chunked_review.split('\n'))}
"""
    
    def _save_test_results(self, results: str, metrics: Dict, session_id: str = None):
        """Save test results to file with session tracking"""
        os.makedirs("test_results", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())[:8]
        
        # Save markdown comparison
        markdown_file = f"test_results/chunking_comparison_{session_id}_{timestamp}.md"
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(results)
        
        # Save metrics and results for caching
        cache_file = f"test_results/chunking_cache_{session_id}.json"
        cache_data = {
            "session_id": session_id,
            "timestamp": timestamp,
            "comparison": results,
            "metrics": metrics
        }
        
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"\n✅ Test results saved to {markdown_file}")
        print(f"✅ Chunking cache saved to {cache_file}")
    
    def _load_cached_results(self, session_id: str) -> Dict:
        """Load cached results for a session if available"""
        cache_file = f"test_results/chunking_cache_{session_id}.json"
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading cached results: {e}")
        
        return None