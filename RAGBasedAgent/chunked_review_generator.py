import os
import sys
from typing import Dict, List, Optional, Union, Tuple
import uuid
from datetime import datetime

# Import existing components
from review_generator import generate_review
from prompts.review_prompts import ReviewPrompts
from review_evaluator import ReviewEvaluator  # Reuse the existing evaluator
from hybrid_chunker import HybridSemanticChunker
from semantic_chunker import SemanticChunker
from hierarchical_chunker import HierarchicalChunker
from fixed_size_chunker import FixedSizeChunker
from models.model_factory import ModelFactory  # Import the factory

class ChunkedReviewGenerator:
    """
    Generates reviews based on different chunking strategies and RAG
    """
    
    def __init__(self, 
                 chunk_collection_name=None,
                 chunking_strategy="hybrid"):
        """Initialize with optional collection name and chunking strategy"""
        # Ensure valid collection name that follows ChromaDB requirements
        if chunk_collection_name:
            # Validate collection name - must be 3-512 chars of [a-zA-Z0-9._-]
            # and start/end with alphanumeric
            if not chunk_collection_name[0].isalnum() or not chunk_collection_name[-1].isalnum() or len(chunk_collection_name) < 3:
                # Generate a fallback name if provided one is invalid
                fallback_name = f"chunks_{uuid.uuid4().hex[:8]}"
                print(f"⚠️ Invalid collection name '{chunk_collection_name}', using '{fallback_name}' instead")
                chunk_collection_name = fallback_name
        else:
            # Generate a valid default collection name
            chunk_collection_name = f"chunks_{uuid.uuid4().hex[:8]}"
        
        # Initialize chunkers with appropriate collection names    
        self.hybrid_chunker = HybridSemanticChunker(collection_name=f"hybrid_{chunk_collection_name}")
        self.semantic_chunker = SemanticChunker(collection_name=f"semantic_{chunk_collection_name}")
        self.hierarchical_chunker = HierarchicalChunker(collection_name=f"hierarchical_{chunk_collection_name}")
        self.fixed_chunker = FixedSizeChunker(collection_name=f"fixed_{chunk_collection_name}")
        
        # Set default chunking strategy
        self.set_chunking_strategy(chunking_strategy)
        
        # Other components
        self.model_factory = ModelFactory()
        self.evaluator = ReviewEvaluator()  # Reuse the existing evaluator
    
    def set_chunking_strategy(self, strategy: str):
        """Set the chunking strategy to use"""
        valid_strategies = ["hybrid", "semantic", "hierarchical", "fixed"]
        
        if strategy not in valid_strategies:
            print(f"⚠️ Invalid chunking strategy '{strategy}'. Using 'hybrid' instead.")
            self.chunking_strategy = "hybrid"
            self.active_chunker = self.hybrid_chunker
        else:
            self.chunking_strategy = strategy
            if strategy == "hybrid":
                self.active_chunker = self.hybrid_chunker
            elif strategy == "semantic":
                self.active_chunker = self.semantic_chunker
            elif strategy == "hierarchical":
                self.active_chunker = self.hierarchical_chunker
            elif strategy == "fixed":
                self.active_chunker = self.fixed_chunker
    
    async def process_pr_with_chunking(self, 
                                     current_pr_changes: str,
                                     similar_prs_changes: List[Dict],
                                     pr_number: int,
                                     model_name: str=None,
                                     chunking_strategy: str=None) -> Dict:
        """
        Process a PR using selected chunking strategy and generate a review
        
        Args:
            current_pr_changes: Current PR changes text
            similar_prs_changes: List of similar PR changes
            pr_number: PR number
            model_name: Model to use for review generation (if None, will evaluate and select best model)
            chunking_strategy: Override chunking strategy
            
        Returns:
            Dict with generated review and chunking stats
        """
        # Update chunking strategy if specified
        if chunking_strategy:
            self.set_chunking_strategy(chunking_strategy)
        
        print(f"\n🔄 Processing PR with {self.chunking_strategy} chunking...")
        
        # Step 1: Chunk and store PR content
        chunk_result = self.active_chunker.chunk_and_store(
            current_pr_changes,
            similar_prs_changes,
            pr_number
        )
        
        if not chunk_result.get("success"):
            print(f"❌ Chunking failed: {chunk_result.get('error')}")
            return {"success": False, "error": chunk_result.get('error')}
        
        print(f"✅ Chunked PR content into {chunk_result.get('current_pr_chunks')} chunks")
        print(f"✅ Chunked similar PRs into {chunk_result.get('similar_pr_chunks')} chunks")
        
        # Step 2: Use provided model or evaluate to find best model
        best_model = model_name
        model_metrics = {}
        
        if not best_model:
            print("\n🔍 Evaluating models to select best one for chunked review...")
            # Use existing ReviewEvaluator to evaluate models
            evaluator = self.evaluator
            
            # Get sample content for evaluation (first chunk)
            current_chunks = self.active_chunker.get_chunks_by_pr_number(
                pr_number,
                chunk_type="current_pr"
            )
            
            if not current_chunks.get("success") or not current_chunks.get("chunks"):
                print("⚠️ Could not retrieve chunks for model evaluation, using default model 'gemini'")
                best_model = "gemini"
            else:
                # Use the first chunk as sample content
                sample_chunk = current_chunks.get("chunks")[0]
                
                # Get related similar PR chunks
                similar_chunks = self.active_chunker.query_relevant_chunks(
                    sample_chunk,
                    filter_criteria={"chunk_type": "similar_pr"},
                    n_results=2
                )
                
                # Format sample data for evaluation
                sample_similar_changes = []
                for i, chunk in enumerate(similar_chunks.get("chunks", [])):
                    if i < len(similar_chunks.get("metadatas", [])):
                        pr_num = similar_chunks.get("metadatas")[i].get("pr_number", 0)
                        sample_similar_changes.append({
                            "pr_number": pr_num,
                            "changes": chunk
                        })
                
                # Evaluate models on sample chunk
                best_model, model_metrics = await evaluator.evaluate_models(
                    sample_chunk,
                    sample_similar_changes
                )
                
                print(f"\n✅ Best model for {self.chunking_strategy} chunked review: {best_model} (Score: {model_metrics[best_model]['Overall']:.3f})")
        
        # Step 3: Generate review with best model
        print(f"\n🤖 Generating chunked review with {best_model} using {self.chunking_strategy} chunking...")
        chunked_review = await self._generate_chunked_review(
            pr_number,
            best_model
        )
        
        # Format review for better readability
        formatted_review = f"# PR Review with {self.chunking_strategy.capitalize()} Chunking\n\n{chunked_review}"
        
        return {
            "success": True,
            "chunked_review": formatted_review,
            "chunking_stats": chunk_result,
            "best_model": best_model,
            "model_metrics": model_metrics,
            "chunking_strategy": self.chunking_strategy
        }
    
    async def process_all_chunking_strategies(self,
                                           current_pr_changes: str,
                                           similar_prs_changes: List[Dict],
                                           pr_number: int,
                                           model_name: str=None) -> Dict:
        """
        Process a PR using all chunking strategies for comparison
        
        Args:
            current_pr_changes: Current PR changes text
            similar_prs_changes: List of similar PR changes
            pr_number: PR number
            model_name: Model to use for review generation
            
        Returns:
            Dict with results for each chunking strategy
        """
        strategies = ["hybrid", "semantic", "hierarchical", "fixed"]
        results = {}
        
        for strategy in strategies:
            print(f"\n\n{'='*60}")
            print(f"📊 Testing {strategy.upper()} chunking strategy")
            print(f"{'='*60}")
            
            # Process with this strategy
            result = await self.process_pr_with_chunking(
                current_pr_changes,
                similar_prs_changes,
                pr_number,
                model_name,
                strategy
            )
            
            results[strategy] = result
            print(f"\n✅ Completed {strategy} chunking strategy evaluation")
        
        return results
    
    async def _generate_chunked_review(self, pr_number: int, model_name: str) -> str:
        """
        Generate review by retrieving and processing chunks
        
        Args:
            pr_number: PR number to review
            model_name: Model to use for generation
            
        Returns:
            Generated review text
        """
        # Get chunks for current PR
        current_chunks = self.active_chunker.get_chunks_by_pr_number(
            pr_number,
            chunk_type="current_pr"
        )
        
        if not current_chunks.get("success"):
            print(f"❌ Failed to retrieve chunks: {current_chunks.get('error')}")
            return f"Failed to generate chunked review with {self.chunking_strategy} chunking due to retrieval error"
        
        # Handle the case of hierarchical chunking differently
        is_hierarchical = self.chunking_strategy == "hierarchical"
        
        # Get template for chunk review
        review_template, system_prompt = ReviewPrompts.get_current_prompt()
        
        # Process chunks and generate mini-reviews
        print(f"\n🧩 Processing {len(current_chunks.get('chunks', []))} chunks...")
        chunk_reviews = []
        
        for i, chunk in enumerate(current_chunks.get("chunks", [])):
            # For hierarchical chunking, only process parent chunks or chunks without level metadata
            if is_hierarchical:
                # Get metadata for this chunk
                chunk_metadata = current_chunks.get("metadatas", [])[i] if i < len(current_chunks.get("metadatas", [])) else {}
                # Skip child chunks for hierarchical processing (we'll get their context from parents)
                if chunk_metadata.get("chunk_level") == "child":
                    continue
            
            print(f"\n📄 Processing chunk {i+1}/{len(current_chunks.get('chunks', []))}...")
            
            # Find related similar PR chunks
            similar_chunks = self.active_chunker.query_relevant_chunks(
                chunk,  # Use current chunk as query
                filter_criteria={"chunk_type": "similar_pr"},
                n_results=3
            )
            
            # Format prompt for this chunk
            prompt = review_template.format(
                similar_prs="\n".join(similar_chunks.get("chunks", [])),
                current_pr=chunk
            )
            
            # Generate mini-review for this chunk using existing model_factory
            try:
                chunk_review = self.model_factory.generate_response_with_prompt(
                    model_name,
                    prompt,
                    system_prompt
                )
                chunk_reviews.append(chunk_review)
                print(f"✅ Generated review for chunk {i+1}")
            except Exception as e:
                print(f"⚠️ Error generating review for chunk {i+1}: {e}")
                continue
        
        # Combine chunk reviews into final review
        combined_review = self._combine_chunk_reviews(chunk_reviews)
        
        return combined_review
    
    def _combine_chunk_reviews(self, chunk_reviews: List[str]) -> str:
        """
        Intelligently combine reviews from multiple chunks
        
        Args:
            chunk_reviews: List of reviews from chunks
            
        Returns:
            Combined review text
        """
        if not chunk_reviews:
            return "No chunk reviews available to combine."
        
        # Define key sections that should be present in the review
        sections = {
            "Summary": [],
            "File Changes": [],
            "Conflict Predictions": [],
            "Breakage Risks": [],
            "Test Coverage": [],
            "Code Quality": []
        }
        
        # Function to identify which section a line belongs to
        def identify_section(line):
            line_lower = line.lower()
            if any(term in line_lower for term in ["summary", "overview"]):
                return "Summary"
            elif any(term in line_lower for term in ["file", "change", "suggestion"]):
                return "File Changes"
            elif any(term in line_lower for term in ["conflict", "prediction"]):
                return "Conflict Predictions"
            elif any(term in line_lower for term in ["break", "risk"]):
                return "Breakage Risks"
            elif any(term in line_lower for term in ["test", "coverage"]):
                return "Test Coverage"
            elif any(term in line_lower for term in ["quality", "smell", "duplication"]):
                return "Code Quality"
            return None
        
        # Process each chunk review
        for review in chunk_reviews:
            current_section = None
            
            for line in review.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Check if this is a section header
                if line.startswith('#'):
                    section = identify_section(line)
                    if section:
                        current_section = section
                    continue
                
                # Add content to appropriate section if we're in a section
                if current_section and line and not line.startswith('#'):
                    if line not in sections[current_section]:
                        sections[current_section].append(line)
        
        # Build the combined review
        combined = f"## Chunking Strategy: {self.chunking_strategy.capitalize()}\n\n"
        
        # Add sections to combined review
        for section, points in sections.items():
            if points:
                combined += f"## {section}\n"
                # Deduplicate points and sort by importance (assuming longer points have more detail)
                unique_points = sorted(set(points), key=len, reverse=True)
                # Format as bullet points if they aren't already
                formatted_points = [
                    point if point.startswith(('-', '*', '•')) 
                    else f"- {point}" 
                    for point in unique_points[:15]  # Limit to top 15 points
                ]
                combined += "\n".join(formatted_points)
                combined += "\n\n"
        
        return combined