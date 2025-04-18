import os
from typing import Dict, Tuple
from review_evaluator import ReviewEvaluator
from models.groq_models import GroqModelFactory

class ChunkingStrategyTester:
    """Tests and compares different chunking strategies for RAG system"""
    
    def __init__(self):
        self.evaluator = ReviewEvaluator()
        self.model = GroqModelFactory.create_model("llama")  # Using Llama model
        
    async def test_chunking_strategies(self, pr_data: Dict) -> Tuple[str, Dict]:
        """Run comparison tests between default and chunked strategies"""
        print("\n🧪 Testing Chunking Strategies...")
        
        if not pr_data.get("current_pr_changes"):
            return "No PR data available for testing", {}
        
        # Generate reviews using both strategies
        default_review = await self._generate_default_review(pr_data)
        chunked_review = await self._generate_chunked_review(pr_data)
        
        if not default_review or not chunked_review:
            return "Failed to generate reviews for comparison", {}
            
        # Format comparison results
        comparison = self._format_comparison(default_review, chunked_review)
        
        # Save results
        self._save_test_results(comparison)
        
        return comparison, {}
    
    async def _generate_default_review(self, pr_data: Dict) -> str:
        """Generate review using default strategy"""
        print("\n📍 Generating review with default strategy...")
        prompt = self._create_review_prompt(pr_data["current_pr_changes"], 
                                         pr_data["similar_prs_changes"])
        return self.model.generate_response(prompt)
    
    async def _generate_chunked_review(self, pr_data: Dict) -> str:
        """Generate review using chunked strategy"""
        print("\n📍 Generating review with chunking strategy...")
        
        # Handle both string and list inputs
        current_changes = pr_data.get("current_pr_changes", "")
        similar_changes = pr_data.get("similar_prs_changes", [])
        
        # Apply chunking to PR content
        chunked_current = self._apply_chunking(current_changes)
        chunked_similar = self._apply_chunking(similar_changes)
        
        # Generate review for each chunk pair
        reviews = []
        max_chunks = max(len(chunked_current), len(chunked_similar))
        
        for i in range(max_chunks):
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
    
    def _apply_chunking(self, content: str | list) -> list:
        """
        Apply chunking strategy to content based on recommendations
        
        Args:
            content: Either a string of code/docs or a list of changes
            
        Returns:
            list: List of chunked content
        """
        if not content:
            return []
            
        # Convert list to string if needed
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
            
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_size = 512  # As per recommendations
        overlap = 50  # ~10% overlap for context preservation
        
        for line in content.splitlines():
            is_code = not line.strip().startswith(('#', '//', '/*', '*'))
            
            # Add line to current chunk
            current_chunk.append(line)
            current_size += len(line)
            
            # Check if chunk is complete
            if current_size >= chunk_size:
                # Save current chunk
                chunks.append('\n'.join(current_chunk))
                # Keep overlap for context
                current_chunk = current_chunk[-overlap:]
                current_size = sum(len(line) for line in current_chunk)
                
        # Add remaining content
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks
    
    def _combine_chunk_reviews(self, chunk_reviews: list) -> str:
        """Combine reviews from different chunks into a cohesive review"""
        if not chunk_reviews:
            return ""
            
        combined = "# Combined Review from Chunked Analysis\n\n"
        
        # Aggregate similar points and remove duplicates
        points = set()
        for review in chunk_reviews:
            for line in review.split('\n'):
                if line.strip() and not line.startswith('#'):
                    points.add(line.strip())
        
        # Format combined review
        combined += "\n".join(sorted(points))
        return combined
    
    def _format_comparison(self, default_review: str, chunked_review: str) -> str:
        """Format the comparison results"""
        return f"""# 📊 Chunking Strategy Comparison Results

## Default Strategy Review
{default_review}

## Chunked Strategy Review
{chunked_review}

## Analysis
- Default Review Length: {len(default_review)} characters
- Chunked Review Length: {len(chunked_review)} characters
- Number of points in default: {len(default_review.split('\n'))}
- Number of points in chunked: {len(chunked_review.split('\n'))}
"""
    
    def _save_test_results(self, results: str):
        """Save test results to file"""
        os.makedirs("test_results", exist_ok=True)
        results_file = "test_results/chunking_comparison.md"
        
        with open(results_file, "w", encoding="utf-8") as f:
            f.write(results)
        
        print(f"\n✅ Test results saved to {results_file}")