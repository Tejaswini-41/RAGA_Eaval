from dotenv import load_dotenv
import os
import sys
from datetime import datetime
import csv

# Path to the models directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def save_review_to_csv(pr_number, review_type, review_content, current_pr_file, similar_pr_number=None):
    """Save review to CSV file for analysis"""
    # Create reviews directory if it doesn't exist
    if not os.path.exists("reviews"):
        os.makedirs("reviews")
    
    csv_file = "reviews/pr_reviews.csv"
    file_exists = os.path.isfile(csv_file)
    
    # Current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract sections from review
    sections = {
        "Summary": [],
        "Issues": [],
        "Suggestions": [],
        "Comparison": []
    }
    
    current_section = "Summary"
    
    for line in review_content.split('\n'):
        lower_line = line.lower()
        
        # Check if this line is a section header
        if lower_line.startswith('#') or ('**' in lower_line and any(x in lower_line for x in ['1.', '2.', '3.', '4.', 'summary', 'issue', 'bug', 'suggest', 'improvement', 'compar'])):
            
            if any(term in lower_line for term in ["summary", "overview", "changes", "1."]):
                current_section = "Summary"
            elif any(term in lower_line for term in ["issue", "bug", "problem", "error", "2."]):
                current_section = "Issues"
            elif any(term in lower_line for term in ["suggest", "improvement", "recommend", "enhance", "3."]):
                current_section = "Suggestions"
            elif any(term in lower_line for term in ["compar", "similar", "relation", "pattern", "4."]):
                current_section = "Comparison"
        else:
            # Add non-empty lines to current section content
            if line.strip():
                sections[current_section].append(line.strip())
    
    # Join each section with line breaks to preserve formatting
    formatted_sections = {
        key: "\n".join(value) for key, value in sections.items()
    }
    
    # Prepare the row data
    row_data = {
        'PR Number': pr_number,
        'Timestamp': timestamp,
        'Review Type': review_type,
        'File Modified': current_pr_file,
        'Similar PR': similar_pr_number,
        'Summary': formatted_sections["Summary"],
        'Issues': formatted_sections["Issues"],
        'Suggestions': formatted_sections["Suggestions"],
        'Comparison': formatted_sections["Comparison"],
        'Full Review': review_content
    }
    
    # Write to CSV
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=row_data.keys())
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row_data)
    
    print(f"\n💾 Review saved to CSV: {csv_file}")

def generate_review(current_pr_changes, similar_pr_changes, pr_number=None, similar_pr_number=None, current_pr_file=None, model_name="gemini"):
    """Generate review suggestions based on PR changes using specified model"""
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
        # Import model factory
        from models.model_factory import ModelFactory
        
        # Custom system prompt for code review
        system_prompt = "You are an expert code reviewer. Analyze the pull request and provide detailed, constructive feedback."
        
        # Get model factory
        model_factory = ModelFactory()
        
        print(f"Generating review with {model_name} model...")
        
        # Generate review using the specified model
        review = model_factory.generate_response_with_prompt(
            model_name, prompt, system_prompt
        )
        
        print("\n✅ Generated AI Review:")
        print(review[:500] + "..." if len(review) > 500 else review)
        
        # Save review to CSV if PR number is provided
        if pr_number:
            save_review_to_csv(pr_number, model_name, review, current_pr_file, similar_pr_number)
        
        return review
        
    except Exception as e:
        print(f"Error using {model_name} model: {e}")
        print("Falling back to rule-based review generation...")
        
        # Simple rule-based review generation as fallback
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
        
        return review