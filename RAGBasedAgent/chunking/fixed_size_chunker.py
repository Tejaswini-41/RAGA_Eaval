from typing import Dict, List, Union
from chunking.base_chunker import BaseChunker

# Parameters for fixed size chunking
FIXED_CHUNK_SIZE = 800  # Fixed chunk size in characters
FIXED_CHUNK_OVERLAP = 100  # Overlap between chunks

class FixedSizeChunker(BaseChunker):
    """
    Handles fixed size chunking of PR content, with consistent chunk
    size and optional overlap between chunks
    """
    
    def __init__(self, collection_name=None):
        """Initialize chunker with optional collection name"""
        super().__init__(collection_name=collection_name, chunker_type="fixed")
    
    def apply_chunking(self, content: Union[str, list]) -> list:
        """Apply fixed size chunking strategy to content"""
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
    
    def add_strategy_metadata(self, metadata: Dict) -> None:
        """Add fixed-size specific metadata"""
        metadata.update({
            "chunk_size": FIXED_CHUNK_SIZE,
            "chunk_overlap": FIXED_CHUNK_OVERLAP
        })
    
    def add_strategy_results(self, results: Dict) -> None:
        """Add fixed-size specific results"""
        results.update({
            "chunk_size": FIXED_CHUNK_SIZE,
            "chunk_overlap": FIXED_CHUNK_OVERLAP
        })