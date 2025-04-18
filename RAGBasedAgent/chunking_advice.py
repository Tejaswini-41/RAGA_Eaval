from models.groq_models import GroqModelFactory
from typing import Dict, List

class ChunkingAdvisor:
    """Provides intelligent chunking advice based on PR content analysis"""
    
    def __init__(self):
        # Initialize with Llama model from GroqModelFactory
        self.model = GroqModelFactory.create_model("llama")
        
    async def get_chunking_advice(self, pr_content: Dict) -> str:
        """Analyze PR content and suggest optimal chunking strategies"""
        content_summary = self._generate_content_summary(pr_content)
        chunking_prompt = self._get_chunking_prompt(content_summary)
        
        # Use the model's generate_response method
        response = self.model.generate_response(chunking_prompt)
        return self._format_advice(response)
    
    def _get_chunking_prompt(self, content_summary: str) -> str:
        return f"""As an expert in document chunking and RAG systems, analyze this pull request content and provide specific chunking recommendations.

Content Summary:
{content_summary}

Provide recommendations in the following areas:

1. Optimal Chunking Strategy:
   - Which chunking method is most appropriate (fixed-size, semantic, or hybrid)?
   - How should code sections be handled differently from documentation?

2. Implementation Guidelines:
   - Specific chunk size recommendations (in tokens/characters)
   - Handling of code blocks and documentation sections
   - Preservation of context between chunks

3. Technical Considerations:
   - Code structure preservation
   - Comment and documentation handling
   - Special delimiters or markers

Please provide concrete, implementable recommendations."""

    def _generate_content_summary(self, pr_content: Dict) -> str:
        """Generate a summary of PR content for analysis"""
        summary = []
        
        if pr_content.get("current_pr_changes"):
            total_lines = len(pr_content["current_pr_changes"].splitlines())
            code_lines = sum(1 for line in pr_content["current_pr_changes"].splitlines() 
                           if not line.strip().startswith(('#', '//', '/', '')))
            
            summary.extend([
                f"Total lines: {total_lines}",
                f"Code lines: {code_lines}",
                f"Documentation lines: {total_lines - code_lines}"
            ])
            
        if pr_content.get("pr_files"):
            file_types = {}
            for file in pr_content["pr_files"]:
                ext = file.filename.split('.')[-1] if '.' in file.filename else 'no_ext'
                file_types[ext] = file_types.get(ext, 0) + 1
            
            summary.append("\nFile distribution:")
            for ext, count in file_types.items():
                summary.append(f"- .{ext}: {count} files")
                
        return "\n".join(summary)
    
    def _format_advice(self, raw_advice: str) -> str:
        """Format the chunking advice in a clear, structured way"""
        return f"""
# 📊 Chunking Strategy Recommendations

{raw_advice}

## Implementation Notes
- Always validate chunk integrity before processing
- Monitor chunk processing performance
- Adjust strategies based on observed results
"""