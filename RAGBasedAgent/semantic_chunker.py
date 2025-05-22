import os
import uuid
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import chromadb
import re

# Import existing functionality
from embedding_store import TFIDFEmbeddingFunction
from similarity_query import text_embedding

# Parameters for semantic chunking
SEMANTIC_MIN_CHUNK_SIZE = 200  # Minimum chunk size
SEMANTIC_MAX_CHUNK_SIZE = 1000  # Maximum chunk size

class SemanticChunker:
    """
    Handles pure semantic chunking of PR content based on textual meaning
    rather than structure, with focus on content similarity and natural breaks
    """
    
    def __init__(self, collection_name=None):
        """Initialize chunker with optional collection name"""
        self.collection_name = collection_name or f"semantic_chunks_{str(uuid.uuid4())[:8]}"
        self.embedding_function = TFIDFEmbeddingFunction()
        self.client = None
        self.collection = None
    
    def apply_semantic_chunking(self, content: Union[str, list]) -> list:
        """
        Apply semantic chunking strategy to content based on semantic meaning
        
        Args:
            content: Either a string of code/docs or a list of changes
            
        Returns:
            list: List of semantically chunked content
        """
        if not content:
            return []
            
        # Convert list to string if needed
        if isinstance(content, list):
            if all(isinstance(item, dict) and 'changes' in item for item in content):
                combined_content = ""
                for item in content:
                    combined_content += f"\nPR #{item.get('pr_number', 'unknown')}:\n"
                    combined_content += item.get('changes', '')
                content = combined_content
            else:
                content = "\n".join(str(item) for item in content)
        
        # Split content into paragraphs or logical sections
        paragraphs = self._split_into_semantic_units(content)
        
        # Group related paragraphs into coherent chunks
        chunks = self._group_by_semantic_similarity(paragraphs)
        
        return chunks
        
    def _split_into_semantic_units(self, content: str) -> List[str]:
        """
        Split content into semantic units (paragraphs, code blocks, etc.)
        """
        # Split by blank lines to identify paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        
        # Further process each paragraph to ensure clean units
        units = []
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
                
            # For code blocks with block-level comments, keep them together
            if p.startswith(('/**', '/*', '///', '#')):
                units.append(p)
            else:
                # Check if this is a code block with multiple statements
                lines = p.split('\n')
                if len(lines) > 5:  # Longer blocks that may need further splitting
                    # Try to identify logical breaks like function definitions
                    current_block = []
                    for line in lines:
                        current_block.append(line)
                        # If we detect a semantic boundary, create a new unit
                        if (line.strip().endswith('{') and 
                            any(keyword in line for keyword in ['function', 'class', 'def', 'if', 'for', 'while'])):
                            if len(current_block) > 1:  # Avoid tiny blocks
                                units.append('\n'.join(current_block))
                                current_block = []
                    
                    # Add any remaining lines
                    if current_block:
                        units.append('\n'.join(current_block))
                else:
                    units.append(p)
        
        return units
    
    def _group_by_semantic_similarity(self, units: List[str]) -> List[str]:
        """
        Group semantic units into coherent chunks based on content similarity
        """
        if not units:
            return []
            
        chunks = []
        current_chunk = []
        current_size = 0
        
        for unit in units:
            unit_size = len(unit)
            
            # If adding this unit would exceed max chunk size, finalize the current chunk
            if current_chunk and (current_size + unit_size > SEMANTIC_MAX_CHUNK_SIZE):
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Add current unit to chunk
            current_chunk.append(unit)
            current_size += unit_size
            
            # If we have a substantial chunk, consider finalizing it at semantic boundaries
            if current_size >= SEMANTIC_MIN_CHUNK_SIZE:
                # Look for semantic boundary indicators
                text = unit.lower()
                if (any(text.endswith(boundary) for boundary in ['}', ';', '.', ':', ')']) and 
                    not any(marker in text for marker in ['todo:', 'fixme:', 'note:', '/*', '//', '#'])):
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
        
        # Add any remaining content
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def initialize_chunk_store(self, chroma_path="chroma_semantic_chunks/"):
        """Initialize ChromaDB for storing chunks"""
        # Create directory if it doesn't exist
        os.makedirs(chroma_path, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        try:
            # Try to get existing collection or create a new one
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                print(f"✅ Using existing semantic chunk collection '{self.collection_name}'")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"},
                )
                print(f"✅ Created new semantic chunk collection '{self.collection_name}'")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing chunk store: {e}")
            return False
    
    def chunk_and_store(self, 
                        current_pr_changes: str, 
                        similar_prs_changes: List[Dict]=None, 
                        pr_number: int=None, 
                        metadata: Dict=None) -> Dict:
        """
        Chunk PR content and store in ChromaDB with embeddings
        
        Args:
            current_pr_changes: Current PR changes text
            similar_prs_changes: List of similar PR changes dicts
            pr_number: Current PR number for metadata
            metadata: Additional metadata to store with chunks
        
        Returns:
            Dict with stats about chunking and storage
        """
        # Initialize DB if not done already
        if not self.collection:
            if not self.initialize_chunk_store():
                return {"success": False, "error": "Failed to initialize chunk store"}
        
        # Generate chunks for current PR
        current_chunks = self.apply_semantic_chunking(current_pr_changes)
        
        # Generate chunks for similar PRs if provided
        similar_chunks = []
        if similar_prs_changes:
            for pr_data in similar_prs_changes:
                chunks = self.apply_semantic_chunking(pr_data.get('changes', ''))
                similar_chunks.extend([(chunk, pr_data.get('pr_number')) for chunk in chunks])
        
        # Prepare for storage
        all_chunks = []
        all_ids = []
        all_metadatas = []
        
        # Add current PR chunks
        for i, chunk in enumerate(current_chunks):
            chunk_id = f"semantic_pr_{pr_number}_chunk_{i}"
            chunk_metadata = {
                "pr_number": pr_number,
                "chunk_index": i,
                "chunk_type": "current_pr",
                "total_chunks": len(current_chunks),
                "chunker_type": "semantic"
            }
            if metadata:
                chunk_metadata.update(metadata)
            
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append(chunk_metadata)
        
        # Add similar PR chunks
        for i, (chunk, similar_pr_number) in enumerate(similar_chunks):
            chunk_id = f"semantic_pr_{similar_pr_number}_chunk_{i}"
            chunk_metadata = {
                "pr_number": similar_pr_number,
                "chunk_index": i,
                "chunk_type": "similar_pr",
                "reference_pr": pr_number,
                "chunker_type": "semantic"
            }
            
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append(chunk_metadata)
        
        # Store chunks with embeddings
        if all_chunks:
            try:
                self.collection.add(
                    documents=all_chunks,
                    ids=all_ids,
                    metadatas=all_metadatas
                )
                
                return {
                    "success": True,
                    "current_pr_chunks": len(current_chunks),
                    "similar_pr_chunks": len(similar_chunks),
                    "total_chunks": len(all_chunks),
                    "chunker_type": "semantic"
                }
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            return {"success": False, "error": "No chunks to store"}
    
    def query_relevant_chunks(self, query_text: str, 
                         filter_criteria: Dict=None,
                         n_results: int=5) -> Dict:
        """
        Query for relevant chunks based on search text
        
        Args:
            query_text: Text to search for
            filter_criteria: Optional filter for specific PR numbers or chunk types
            n_results: Number of results to return
            
        Returns:
            Dict with query results including chunks and metadata
        """
        if not self.collection:
            if not self.initialize_chunk_store():
                return {"success": False, "error": "Failed to initialize chunk store"}
    
        try:
            # Generate embedding for query text
            query_embedding = text_embedding(query_text)
            
            # Format filter criteria for ChromaDB if provided
            chroma_filter = None
            if filter_criteria:
                if 'chunk_type' in filter_criteria and 'pr_number' not in filter_criteria:
                    chroma_filter = {"chunk_type": {"$eq": filter_criteria['chunk_type']}}
                elif 'pr_number' in filter_criteria and 'chunk_type' not in filter_criteria:
                    chroma_filter = {"pr_number": {"$eq": filter_criteria['pr_number']}}
                # Handle combined filters
                elif 'chunk_type' in filter_criteria and 'pr_number' in filter_criteria:
                    chroma_filter = {
                        "$and": [
                            {"chunk_type": {"$eq": filter_criteria['chunk_type']}},
                            {"pr_number": {"$eq": filter_criteria['pr_number']}}
                        ]
                    }
        
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=chroma_filter
            )
            
            # Handle empty results safely
            if not results["documents"] or not results["documents"][0]:
                return {
                    "success": True,
                    "chunks": [],
                    "metadatas": [],
                    "distances": [],
                    "ids": []
                }
            
            # Format results for easier consumption
            formatted_results = {
                "success": True,
                "chunks": results["documents"][0],
                "metadatas": results["metadatas"][0],
                "distances": results["distances"][0],
                "ids": results["ids"][0]
            }
            
            return formatted_results
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_chunks_by_pr_number(self, pr_number: int, chunk_type: str=None) -> Dict:
        """
        Get all chunks for a specific PR number
        
        Args:
            pr_number: PR number to filter by
            chunk_type: Optional filter for chunk type ('current_pr' or 'similar_pr')
            
        Returns:
            Dict with chunks and metadata
        """
        if not self.collection:
            if not self.initialize_chunk_store():
                return {"success": False, "error": "Failed to initialize chunk store"}
    
        try:
            # Build filter criteria - use ChromaDB's where filter format
            if chunk_type:
                filter_criteria = {
                    "$and": [
                        {"pr_number": {"$eq": pr_number}},
                        {"chunk_type": {"$eq": chunk_type}}
                    ]
                }
            else:
                filter_criteria = {"pr_number": {"$eq": pr_number}}
            
            # Query the collection
            results = self.collection.get(
                where=filter_criteria
            )
            
            # Format results
            return {
                "success": True,
                "chunks": results["documents"],
                "metadatas": results["metadatas"],
                "ids": results["ids"],
                "chunker_type": "semantic"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}