import re
from typing import Dict, List, Union
from chunking.base_chunker import BaseChunker

# Parameters for semantic chunking
SEMANTIC_MIN_CHUNK_SIZE = 200  # Minimum chunk size
SEMANTIC_MAX_CHUNK_SIZE = 1000  # Maximum chunk size

class SemanticChunker(BaseChunker):
    """
    Handles pure semantic chunking of PR content based on textual meaning
    rather than structure, with focus on content similarity and natural breaks
    """
    
    def __init__(self, collection_name=None):
        """Initialize chunker with optional collection name"""
        super().__init__(collection_name=collection_name, chunker_type="semantic")
    
    def apply_chunking(self, content: Union[str, list]) -> list:
        """Apply semantic chunking strategy to content"""
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
        """Split content into semantic units (paragraphs, code blocks, etc.)"""
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
        """Group semantic units into coherent chunks based on content similarity"""
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
    
    def add_strategy_metadata(self, metadata: Dict) -> None:
        """Add semantic-specific metadata"""
        metadata.update({
            "min_chunk_size": SEMANTIC_MIN_CHUNK_SIZE,
            "max_chunk_size": SEMANTIC_MAX_CHUNK_SIZE
        })