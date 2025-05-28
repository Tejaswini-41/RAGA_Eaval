import os
import uuid
from typing import Dict, List, Union
import chromadb

from embedding_store import TFIDFEmbeddingFunction
from similarity_query import text_embedding

class BaseChunker:
    """
    Base chunker class that implements common functionality for all chunking strategies
    """
    
    def __init__(self, collection_name=None, chunker_type="base"):
        """Initialize chunker with optional collection name"""
        self.chunker_type = chunker_type
        self.collection_name = collection_name or f"{chunker_type}_chunks_{str(uuid.uuid4())[:8]}"
        self.embedding_function = TFIDFEmbeddingFunction()
        self.client = None
        self.collection = None
    
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
                print(f"✅ Using existing {self.chunker_type} chunk collection '{self.collection_name}'")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"},
                )
                print(f"✅ Created new {self.chunker_type} chunk collection '{self.collection_name}'")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing {self.chunker_type} chunk store: {e}")
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
        current_chunks = self.apply_chunking(current_pr_changes)
        
        # Generate chunks for similar PRs if provided
        similar_chunks = []
        if similar_prs_changes:
            for pr_data in similar_prs_changes:
                chunks = self.apply_chunking(pr_data.get('changes', ''))
                similar_chunks.extend([(chunk, pr_data.get('pr_number')) for chunk in chunks])
        
        # Prepare for storage
        all_chunks = []
        all_ids = []
        all_metadatas = []
        
        # Add current PR chunks
        for i, chunk in enumerate(current_chunks):
            chunk_id = f"{self.chunker_type}_pr_{pr_number}_chunk_{i}"
            chunk_metadata = {
                "pr_number": pr_number,
                "chunk_index": i,
                "chunk_type": "current_pr",
                "total_chunks": len(current_chunks),
                "chunker_type": self.chunker_type
            }
            
            # Add strategy-specific metadata
            self.add_strategy_metadata(chunk_metadata)
            
            if metadata:
                chunk_metadata.update(metadata)
            
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append(chunk_metadata)
        
        # Add similar PR chunks
        for i, (chunk, similar_pr_number) in enumerate(similar_chunks):
            chunk_id = f"{self.chunker_type}_pr_{similar_pr_number}_chunk_{i}"
            chunk_metadata = {
                "pr_number": similar_pr_number,
                "chunk_index": i,
                "chunk_type": "similar_pr",
                "reference_pr": pr_number,
                "chunker_type": self.chunker_type
            }
            
            # Add strategy-specific metadata
            self.add_strategy_metadata(chunk_metadata)
            
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
                
                result = {
                    "success": True,
                    "current_pr_chunks": len(current_chunks),
                    "similar_pr_chunks": len(similar_chunks),
                    "total_chunks": len(all_chunks),
                    "chunker_type": self.chunker_type
                }
                
                # Add strategy-specific results
                self.add_strategy_results(result)
                
                return result
                
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
                "chunker_type": self.chunker_type
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
                "chunker_type": self.chunker_type
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def apply_chunking(self, content: Union[str, list]) -> list:
        """
        Apply chunking strategy to content (to be implemented by subclasses)
        
        Args:
            content: Either a string of code/docs or a list of changes
            
        Returns:
            list: List of chunked content
        """
        raise NotImplementedError("Subclasses must implement apply_chunking method")
    
    def add_strategy_metadata(self, metadata: Dict) -> None:
        """
        Add strategy-specific metadata to chunk metadata (to be implemented by subclasses)
        """
        pass
    
    def add_strategy_results(self, results: Dict) -> None:
        """
        Add strategy-specific data to results (to be implemented by subclasses)
        """
        pass