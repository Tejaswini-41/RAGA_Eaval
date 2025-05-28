from typing import Dict, List, Union
from chunking.base_chunker import BaseChunker

# Parameters for hybrid chunking
CHUNK_SIZE = 1200  # Increase from default (likely 400-800)
CHUNK_OVERLAP = 200  # Increase overlap for better context preservation
MIN_CHUNK_SIZE = 300  # Avoid tiny chunks that lack context

class HybridSemanticChunker(BaseChunker):
    """
    Handles hybrid semantic chunking of PR content and integrates with embedding store
    """
    
    def __init__(self, collection_name=None):
        """Initialize chunker with optional collection name"""
        super().__init__(collection_name=collection_name, chunker_type="hybrid")
    
    def apply_chunking(self, content: Union[str, list]) -> list:
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
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = []
                        current_size = 0
                    
                    in_function = 'def ' in stripped or 'function ' in stripped
                    in_class = 'class ' in stripped
                
                # Track indentation for block structure
                if stripped:
                    leading_spaces = len(line) - len(line.lstrip())
                    if leading_spaces < indentation_level and in_function and current_size > CHUNK_SIZE / 2:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = []
                        current_size = 0
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
    
    def add_strategy_metadata(self, metadata: Dict) -> None:
        """Add hybrid strategy-specific metadata"""
        metadata.update({
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "min_chunk_size": MIN_CHUNK_SIZE
        })
    
    def add_strategy_results(self, results: Dict) -> None:
        """Add hybrid strategy-specific results"""
        results.update({
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "min_chunk_size": MIN_CHUNK_SIZE
        })