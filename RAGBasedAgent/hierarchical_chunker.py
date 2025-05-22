import os
import uuid
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import chromadb
import re

# Import existing functionality
from embedding_store import TFIDFEmbeddingFunction
from similarity_query import text_embedding

# Parameters for hierarchical chunking
PARENT_CHUNK_SIZE = 1500  # Parent chunk size (larger for context)
CHILD_CHUNK_SIZE = 500    # Child chunk size (smaller for detailed retrieval)

class HierarchicalChunker:
    """
    Handles hierarchical chunking of PR content using multi-level chunking strategy,
    with parent chunks for context and child chunks for detailed retrieval
    """
    
    def __init__(self, collection_name=None):
        """Initialize chunker with optional collection name"""
        self.collection_name = collection_name or f"hierarchical_chunks_{str(uuid.uuid4())[:8]}"
        self.embedding_function = TFIDFEmbeddingFunction()
        self.client = None
        self.collection = None
    
    def apply_hierarchical_chunking(self, content: Union[str, list]) -> List[Dict]:
        """
        Apply hierarchical chunking strategy to content using multi-level chunking
        
        Args:
            content: Either a string of code/docs or a list of changes
            
        Returns:
            List[Dict]: List of hierarchical chunks with parent-child relationships
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
                
        hierarchical_chunks = []
        
        # Split first into logical sections (files, classes, or major blocks)
        parent_sections = self._split_into_parent_sections(content)
        
        for parent_idx, parent_content in enumerate(parent_sections):
            # Create parent chunk
            parent_id = f"parent_{parent_idx}"
            parent_chunk = {
                "id": parent_id,
                "content": parent_content,
                "level": "parent",
                "children": []
            }
            
            # Create child chunks from parent content
            child_chunks = self._create_child_chunks(parent_content)
            
            for child_idx, child_content in enumerate(child_chunks):
                child_id = f"parent_{parent_idx}_child_{child_idx}"
                child_chunk = {
                    "id": child_id,
                    "content": child_content,
                    "level": "child",
                    "parent_id": parent_id
                }
                parent_chunk["children"].append(child_id)
                hierarchical_chunks.append(child_chunk)
            
            hierarchical_chunks.append(parent_chunk)
            
        return hierarchical_chunks
    
    def _split_into_parent_sections(self, content: str) -> List[str]:
        """
        Split content into major logical sections (parent chunks)
        """
        if not content:
            return []
        
        # First try to split by file sections (common in PR diffs)
        file_sections = re.split(r'((?:^|\n)(?:diff --git|\+\+\+|---).*?\n)', content)
        
        # Filter and join relevant sections
        sections = []
        current_section = ""
        
        for section in file_sections:
            section = section.strip()
            if not section:
                continue
                
            if section.startswith(('diff --git', '+++', '---')):
                # Start of a new file section
                if current_section:
                    sections.append(current_section)
                current_section = section
            else:
                current_section += "\n" + section
                
                # If section is getting too large, split it
                if len(current_section) > PARENT_CHUNK_SIZE * 1.5:
                    sections.append(current_section)
                    current_section = ""
        
        # Add the last section if it exists
        if current_section:
            sections.append(current_section)
        
        # If no file sections were found, use a simpler approach
        if not sections:
            sections = self._split_by_size(content, PARENT_CHUNK_SIZE)
        
        return sections
    
    def _create_child_chunks(self, parent_content: str) -> List[str]:
        """
        Split parent content into smaller, more specific chunks (child chunks)
        """
        # Split by code blocks, functions, or paragraphs
        lines = parent_content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Track code structures
        in_function = False
        indentation_level = 0
        
        for line in lines:
            stripped = line.strip()
            is_code = not stripped.startswith(('#', '//', '/*', '*'))
            
            # Check for structural boundaries
            if is_code:
                # Function/class detection for better chunking
                if any(pattern in stripped for pattern in ['def ', 'function ', 'class ']):
                    if current_chunk and current_size > CHILD_CHUNK_SIZE / 2:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = []
                        current_size = 0
                    in_function = 'def ' in stripped or 'function ' in stripped
            
            # Add line to current chunk
            current_chunk.append(line)
            current_size += len(line)
            
            # Check size threshold
            if current_size >= CHILD_CHUNK_SIZE:
                # Try not to break functions
                if not in_function or not is_code:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
        
        # Add any remaining content
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        # If no chunks were created, just return the parent content as a single chunk
        if not chunks:
            return [parent_content]
            
        return chunks
    
    def _split_by_size(self, content: str, max_size: int) -> List[str]:
        """
        Split text by size while trying to preserve logical boundaries
        """
        if len(content) <= max_size:
            return [content]
            
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            if current_size + len(line) > max_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
                
            current_chunk.append(line)
            current_size += len(line)
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks
    
    def initialize_chunk_store(self, chroma_path="chroma_hierarchical_chunks/"):
        """Initialize ChromaDB for storing chunks"""
        # Create directory if it doesn't exist
        os.makedirs(chroma_path, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        try:
            # Try to get existing collection or create a new one
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                print(f"✅ Using existing hierarchical chunk collection '{self.collection_name}'")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"},
                )
                print(f"✅ Created new hierarchical chunk collection '{self.collection_name}'")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing hierarchical chunk store: {e}")
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
        
        # Generate hierarchical chunks for current PR
        current_hierarchical_chunks = self.apply_hierarchical_chunking(current_pr_changes)
        
        # Generate chunks for similar PRs if provided
        similar_hierarchical_chunks = []
        if similar_prs_changes:
            for pr_data in similar_prs_changes:
                chunks = self.apply_hierarchical_chunking(pr_data.get('changes', ''))
                similar_hierarchical_chunks.append((chunks, pr_data.get('pr_number')))
        
        # Prepare for storage
        all_chunks = []
        all_ids = []
        all_metadatas = []
        
        # Add current PR chunks
        parent_count = 0
        child_count = 0
        
        for chunk_data in current_hierarchical_chunks:
            level = chunk_data["level"]
            content = chunk_data["content"]
            
            if level == "parent":
                chunk_id = f"hierarchical_pr_{pr_number}_parent_{parent_count}"
                parent_count += 1
            else:  # child
                chunk_id = f"hierarchical_pr_{pr_number}_child_{child_count}"
                child_count += 1
                
            chunk_metadata = {
                "pr_number": pr_number,
                "chunk_level": level,
                "chunk_type": "current_pr", 
                "chunker_type": "hierarchical"
            }
            
            # Add parent-child relationship
            if level == "child" and "parent_id" in chunk_data:
                chunk_metadata["parent_id"] = chunk_data["parent_id"]
            elif level == "parent" and "children" in chunk_data:
                chunk_metadata["children"] = ",".join(chunk_data["children"])
                
            if metadata:
                chunk_metadata.update(metadata)
            
            all_chunks.append(content)
            all_ids.append(chunk_id)
            all_metadatas.append(chunk_metadata)
        
        # Add similar PR chunks
        for chunks_list, similar_pr_number in similar_hierarchical_chunks:
            s_parent_count = 0
            s_child_count = 0
            
            for chunk_data in chunks_list:
                level = chunk_data["level"]
                content = chunk_data["content"]
                
                if level == "parent":
                    chunk_id = f"hierarchical_pr_{similar_pr_number}_parent_{s_parent_count}"
                    s_parent_count += 1
                else:  # child
                    chunk_id = f"hierarchical_pr_{similar_pr_number}_child_{s_child_count}"
                    s_child_count += 1
                    
                chunk_metadata = {
                    "pr_number": similar_pr_number,
                    "chunk_level": level,
                    "chunk_type": "similar_pr",
                    "reference_pr": pr_number,
                    "chunker_type": "hierarchical"
                }
                
                # Add parent-child relationship
                if level == "child" and "parent_id" in chunk_data:
                    chunk_metadata["parent_id"] = chunk_data["parent_id"]
                elif level == "parent" and "children" in chunk_data:
                    chunk_metadata["children"] = ",".join(chunk_data["children"])
                
                all_chunks.append(content)
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
                    "current_pr_chunks": parent_count + child_count,
                    "similar_pr_chunks": len(all_chunks) - (parent_count + child_count),
                    "total_chunks": len(all_chunks),
                    "parent_chunks": parent_count,
                    "child_chunks": child_count,
                    "chunker_type": "hierarchical"
                }
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            return {"success": False, "error": "No chunks to store"}
    
    def query_relevant_chunks(self, query_text: str, 
                         filter_criteria: Dict=None,
                         n_results: int=5) -> Dict:
        """
        Query for relevant chunks based on search text using hierarchical approach:
        1. First find most relevant parent chunks
        2. Then include their children for more detailed context
        
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
            
            # Format filter criteria for ChromaDB
            chroma_filter = None
            if filter_criteria:
                if 'chunk_type' in filter_criteria and 'pr_number' not in filter_criteria:
                    chroma_filter = {"chunk_type": {"$eq": filter_criteria['chunk_type']}}
                elif 'pr_number' in filter_criteria and 'chunk_type' not in filter_criteria:
                    chroma_filter = {"pr_number": {"$eq": filter_criteria['pr_number']}}
                elif 'chunk_type' in filter_criteria and 'pr_number' in filter_criteria:
                    chroma_filter = {
                        "$and": [
                            {"chunk_type": {"$eq": filter_criteria['chunk_type']}},
                            {"pr_number": {"$eq": filter_criteria['pr_number']}}
                        ]
                    }
            
            # First, search for child chunks directly - they have more specific content
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=chroma_filter
            )
            
            # If no results, return empty
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
                "ids": results["ids"][0],
                "chunker_type": "hierarchical"
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
                "chunker_type": "hierarchical"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}