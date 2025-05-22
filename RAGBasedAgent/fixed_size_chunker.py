import os
import uuid
from typing import Dict, List, Tuple, Union, Optional
import chromadb

# Import existing functionality
from embedding_store import TFIDFEmbeddingFunction
from similarity_query import text_embedding

# Parameters for fixed size chunking
FIXED_CHUNK_SIZE = 800  # Fixed chunk size in characters
FIXED_CHUNK_OVERLAP = 100  # Overlap between chunks

class FixedSizeChunker:
    """
    Handles fixed size chunking of PR content, with consistent chunk
    size and optional overlap between chunks
    """
    
    def __init__(self, collection_name=None):
        """Initialize chunker with optional collection name"""
        self.collection_name = collection_name or f"fixed_chunks_{str(uuid.uuid4())[:8]}"
        self.embedding_function = TFIDFEmbeddingFunction()
        self.client = None
        self.collection = None
    
    def apply_fixed_size_chunking(self, content: Union[str, list]) -> list:
        """
        Apply fixed size chunking strategy to content
        
        Args:
            content: Either a string of code/docs or a list of changes
            
        Returns:
            list: List of fixed size chunks
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
            
        # Split content into fixed size chunks with overlap
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_size = 0
        
        for line in lines:
            # Add the line to current chunk
            current_chunk.append(line)
            current_size += len(line) + 1  # +1 for newline
            
            # Check if we've reached the chunk size
            if current_size >= FIXED_CHUNK_SIZE:
                # Add current chunk to chunks list
                chunks.append('\n'.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_lines = min(len(current_chunk) // 2, FIXED_CHUNK_OVERLAP // 30)
                current_chunk = current_chunk[-overlap_lines:] if overlap_lines > 0 else []
                current_size = sum(len(line) + 1 for line in current_chunk)
        
        # Add any remaining content as final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks
    
    def initialize_chunk_store(self, chroma_path="chroma_fixed_chunks/"):
        """Initialize ChromaDB for storing chunks"""
        # Create directory if it doesn't exist
        os.makedirs(chroma_path, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        try:
            # Try to get existing collection or create a new one
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                print(f"✅ Using existing fixed size chunk collection '{self.collection_name}'")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"},
                )
                print(f"✅ Created new fixed size chunk collection '{self.collection_name}'")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing fixed size chunk store: {e}")
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
        current_chunks = self.apply_fixed_size_chunking(current_pr_changes)
        
        # Generate chunks for similar PRs if provided
        similar_chunks = []
        if similar_prs_changes:
            for pr_data in similar_prs_changes:
                chunks = self.apply_fixed_size_chunking(pr_data.get('changes', ''))
                similar_chunks.extend([(chunk, pr_data.get('pr_number')) for chunk in chunks])
        
        # Prepare for storage
        all_chunks = []
        all_ids = []
        all_metadatas = []
        
        # Add current PR chunks
        for i, chunk in enumerate(current_chunks):
            chunk_id = f"fixed_pr_{pr_number}_chunk_{i}"
            chunk_metadata = {
                "pr_number": pr_number,
                "chunk_index": i,
                "chunk_type": "current_pr",
                "total_chunks": len(current_chunks),
                "chunk_size": FIXED_CHUNK_SIZE,
                "chunk_overlap": FIXED_CHUNK_OVERLAP,
                "chunker_type": "fixed"
            }
            if metadata:
                chunk_metadata.update(metadata)
            
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append(chunk_metadata)
        
        # Add similar PR chunks
        for i, (chunk, similar_pr_number) in enumerate(similar_chunks):
            chunk_id = f"fixed_pr_{similar_pr_number}_chunk_{i}"
            chunk_metadata = {
                "pr_number": similar_pr_number,
                "chunk_index": i,
                "chunk_type": "similar_pr",
                "reference_pr": pr_number,
                "chunk_size": FIXED_CHUNK_SIZE,
                "chunk_overlap": FIXED_CHUNK_OVERLAP,
                "chunker_type": "fixed"
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
                    "chunker_type": "fixed",
                    "chunk_size": FIXED_CHUNK_SIZE,
                    "chunk_overlap": FIXED_CHUNK_OVERLAP
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
                "ids": results["ids"][0],
                "chunker_type": "fixed"
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
                "chunker_type": "fixed"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}