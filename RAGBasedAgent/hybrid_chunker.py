import os
import uuid
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import chromadb

# Import existing functionality
from embedding_store import TFIDFEmbeddingFunction
from similarity_query import text_embedding

# Adjust these parameters in your chunking strategy
CHUNK_SIZE = 1200  # Increase from default (likely 400-800)
CHUNK_OVERLAP = 200  # Increase overlap for better context preservation
MIN_CHUNK_SIZE = 300  # Avoid tiny chunks that lack context

class HybridSemanticChunker:
    """
    Handles hybrid semantic chunking of PR content and integrates with embedding store
    """
    
    def __init__(self, collection_name=None):
        """Initialize chunker with optional collection name"""
        self.collection_name = collection_name or f"hybrid_chunks_{str(uuid.uuid4())[:8]}"
        self.embedding_function = TFIDFEmbeddingFunction()
        self.client = None
        self.collection = None
    
    def apply_hybrid_semantic_chunking(self, content: Union[str, list]) -> list:
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
            if all(isinstance(item, dict) and 'changes' in item for item in content):
                # Handle list of PR change dicts
                combined_content = ""
                for item in content:
                    combined_content += f"\nPR #{item.get('pr_number', 'unknown')}:\n"
                    combined_content += item.get('changes', '')
                content = combined_content
            else:
                # Handle other list types
                content = "\n".join(str(item) for item in content)
            
        chunks = []
        lines = content.splitlines()
        current_chunk = []
        current_size = 0
        
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
                    if current_chunk and current_size > CHUNK_SIZE / 2:
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
                    if leading_spaces < indentation_level and in_function and current_size > CHUNK_SIZE / 2:
                        # Function end detected by dedent, complete chunk
                        chunks.append('\n'.join(current_chunk))
                        context_lines = min(5, len(current_chunk)) 
                        current_chunk = current_chunk[-context_lines:]
                        current_size = sum(len(line) for line in current_chunk)
                        in_function = False
                    indentation_level = leading_spaces
            else:
                # For comments/docs, look for paragraph breaks
                if not stripped and current_size > CHUNK_SIZE:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
            
            # Add the current line to chunk
            current_chunk.append(line)
            current_size += len(line)
            
            # Check if we've reached chunk size and at a good break point
            if current_size >= CHUNK_SIZE:
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
    
    def initialize_chunk_store(self, chroma_path="chroma_chunks/"):
        """Initialize ChromaDB for storing chunks"""
        # Create directory if it doesn't exist
        os.makedirs(chroma_path, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        try:
            # Try to get existing collection or create a new one
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                print(f"✅ Using existing chunk collection '{self.collection_name}'")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"},
                )
                print(f"✅ Created new chunk collection '{self.collection_name}'")
            
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
        current_chunks = self.apply_hybrid_semantic_chunking(current_pr_changes)
        
        # Generate chunks for similar PRs if provided
        similar_chunks = []
        if similar_prs_changes:
            for pr_data in similar_prs_changes:
                chunks = self.apply_hybrid_semantic_chunking(pr_data.get('changes', ''))
                similar_chunks.extend([(chunk, pr_data.get('pr_number')) for chunk in chunks])
        
        # Prepare for storage
        all_chunks = []
        all_ids = []
        all_metadatas = []
        
        # Add current PR chunks
        for i, chunk in enumerate(current_chunks):
            chunk_id = f"pr_{pr_number}_chunk_{i}"
            chunk_metadata = {
                "pr_number": pr_number,
                "chunk_index": i,
                "chunk_type": "current_pr",
                "total_chunks": len(current_chunks)
            }
            if metadata:
                chunk_metadata.update(metadata)
            
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append(chunk_metadata)
        
        # Add similar PR chunks
        for i, (chunk, similar_pr_number) in enumerate(similar_chunks):
            chunk_id = f"pr_{similar_pr_number}_chunk_{i}"
            chunk_metadata = {
                "pr_number": similar_pr_number,
                "chunk_index": i,
                "chunk_type": "similar_pr",
                "reference_pr": pr_number
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
                    "total_chunks": len(all_chunks)
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
                where=chroma_filter  # Apply any filters
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
                # Format properly for ChromaDB
                
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
                "ids": results["ids"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}