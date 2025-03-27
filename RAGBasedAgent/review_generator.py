from dotenv import load_dotenv
import os
import sys
import csv
from datetime import datetime

# Path to the models directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Gemini model
from models.gemini_model import GeminiModel

def save_review_to_csv(pr_number, review_type, review_content, current_pr_file, similar_pr_number=None):
    """Save the review to a CSV file"""
    # Create directory if it doesn't exist
    csv_dir = os.path.join(os.path.dirname(__file__), 'reviews')
    os.makedirs(csv_dir, exist_ok=True)
    
    # Define the CSV file path
    csv_file = os.path.join(csv_dir, 'pr_reviews.csv')
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_file)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract sections from review content for better CSV structure
    sections = {
        "Summary": [],
        "Issues": [],
        "Suggestions": [],
        "Comparison": []
    }
    
    # More robust extraction based on both # headers and **numbered sections**
    current_section = "Summary"
    
    for line in review_content.split('\n'):
        lower_line = line.lower().strip()
        
        # Check for various section header formats (both # and **)
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

def generate_review(current_pr_changes, similar_pr_changes, pr_number=None, similar_pr_number=None, current_pr_file=None):
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
        # print(review)  # Print the full review without truncation
        
        # Save to CSV
        save_review_to_csv(pr_number, "Gemini", review, current_pr_file, similar_pr_number)
        
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
        # print(review)
        
        # Save to CSV
        save_review_to_csv(pr_number, "Fallback", review, current_pr_file, similar_pr_number)
        
        return review