from dotenv import load_dotenv
import os
import sys

# Path to the models directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Gemini model
from models.gemini_model import GeminiModel

def generate_review(current_pr_changes, similar_pr_changes):
    """Generate review suggestions based on PR changes using Gemini model"""
    # Load environment variables
    load_dotenv()
    
    # Build prompt
    prompt = f"""Compare these pull requests:
    
Similar PR:
{similar_pr_changes}

Current PR:
{current_pr_changes}

Please provide a detailed code review including:
1. Summary of the changes
2. Potential issues or bugs
3. Suggestions for improvement
4. Any patterns you notice from the similar PR that could be applied here
"""
    
    try:
        # Initialize Gemini model with system prompt
        system_prompt = "You are an expert code reviewer. Analyze the pull request and provide detailed, constructive feedback."
        gemini = GeminiModel(system_prompt=system_prompt)
        
        print("\n🤖 Generating review with Gemini model...")
        
        # Generate review using Gemini
        review = gemini.generate_response(prompt)
        
        print("\n✅ Generated AI Review:")
        print(review[:500] + "..." if len(review) > 500 else review)
        
        return review
        
    except Exception as e:
        print(f"Error using Gemini model: {e}")
        print("Falling back to rule-based review generation...")
        
        # Fallback: Create a simple review based on pattern matching
        review = "# Code Review Summary\n\n"
        
        # Count lines added/removed
        added_count = current_pr_changes.count("After:")
        removed_count = current_pr_changes.count("Before:")
        
        review += f"## Changes Overview\n"
        review += f"- Modified {added_count} file(s)\n"
        
        # Look for common patterns
        if "null" in current_pr_changes.lower() or "None" in current_pr_changes:
            review += "- Added null checks, improving code robustness\n"
            
        if "test" in current_pr_changes.lower():
            review += "- Modified tests, improving code coverage\n"
            
        if "fix" in current_pr_changes.lower() or "bug" in current_pr_changes.lower():
            review += "- Fixed bugs in existing functionality\n"
            
        # Add suggestions
        review += "\n## Suggestions\n"
        review += "- Consider adding unit tests for the changes\n"
        review += "- Ensure error handling is comprehensive\n"
        
        # Compare with similar PR
        review += "\n## Comparison with Similar PR\n"
        review += "- Both PRs modify similar components\n"
        if similar_pr_changes.lower().count("test") > current_pr_changes.lower().count("test"):
            review += "- The similar PR had more test coverage, consider adding tests\n"
        
        print("\n✅ Generated fallback rule-based review:")
        print(review[:500] + "..." if len(review) > 500 else review)
        
        return review