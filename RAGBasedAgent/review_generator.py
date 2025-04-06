from dotenv import load_dotenv
import os
import sys
from datetime import datetime
import csv

# Path to the models directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def save_review_to_csv(pr_number, review_type, review_content, current_pr_file, similar_pr_number=None, current_pr_changes=None, similar_pr_changes=None):
    """Save review to CSV file for analysis with complete PR comparison data"""
    # Create reviews directory if it doesn't exist
    if not os.path.exists("reviews"):
        os.makedirs("reviews")
    
    csv_file = "reviews/pr_reviews.csv"
    file_exists = os.path.isfile(csv_file)
    
    # Current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract sections from review using the 6-section structure from the prompt
    sections = extract_review_sections(review_content)
    
    # Prepare the row data with clean, structured fields
    row_data = {
        'PR_Number': str(pr_number),
        'Timestamp': timestamp,
        'Model': review_type,
        'Files_Modified': clean_text_field(current_pr_file),
        'Similar_PR': str(similar_pr_number) if similar_pr_number else "",
        'Summary': clean_text_field(sections["Summary"]),
        'File_Suggestions': clean_text_field(sections["File_Suggestions"]),
        'Conflict_Predictions': clean_text_field(sections["Conflict_Predictions"]),
        'Breakage_Risks': clean_text_field(sections["Breakage_Risks"]),
        'Test_Coverage': clean_text_field(sections["Test_Coverage"]),
        'Code_Quality': clean_text_field(sections["Code_Quality"])
        # Store full changes and review in separate files to keep CSV clean
    }
    
    # Write to CSV
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=row_data.keys())
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row_data)
    
    # Store full PR changes and review in separate files for reference
    pr_data_dir = "reviews/pr_data"
    if not os.path.exists(pr_data_dir):
        os.makedirs(pr_data_dir)
        
    # Write full PR changes and review to separate files
    with open(f"{pr_data_dir}/PR_{pr_number}_current_changes.txt", "w", encoding="utf-8") as f:
        f.write(current_pr_changes or "")
        
    with open(f"{pr_data_dir}/PR_{pr_number}_similar_changes.txt", "w", encoding="utf-8") as f:
        f.write(similar_pr_changes or "")
        
    with open(f"{pr_data_dir}/PR_{pr_number}_review.txt", "w", encoding="utf-8") as f:
        f.write(review_content)
    
    print(f"\n💾 Review saved to CSV: {csv_file}")
    print(f"📄 Full PR data saved to {pr_data_dir}/PR_{pr_number}_*.txt")

def generate_review(current_pr_changes, similar_pr_changes, pr_number=None, similar_pr_number=None, current_pr_file=None, model_name="gemini"):
    """Generate review suggestions based on PR changes using specified model"""
    # Load environment variables
    load_dotenv()
    
    # Updated prompt with more specific sections
    prompt = f"""Compare these pull requests:
    
Similar PR:
{similar_pr_changes}

Current PR:
{current_pr_changes}

Please provide a detailed code review including:
1. Summary of the changes
2. File Change Suggestions - Identify files that might get affected based on changes
3. Conflict Prediction - Flag files changed in multiple PRs that could cause conflicts
4. Breakage Risk Warning - Note which changes might break existing functionality
5. Test Coverage Advice - Recommend test files that should be updated
6. Code Quality Suggestions - Point out potential code smells or duplication

Be specific with file names, function names, and line numbers when possible.
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
        # Display full review instead of truncating
        print(review)
        
        # Save review to CSV if PR number is provided
        if pr_number:
            save_review_to_csv(pr_number, model_name, review, current_pr_file, similar_pr_number, current_pr_changes, similar_pr_changes)
        
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

# Add these helper functions

def extract_review_sections(review_content):
    """Extract review sections into a structured dictionary"""
    sections = {
        "Summary": "",
        "File_Suggestions": "",
        "Conflict_Predictions": "",
        "Breakage_Risks": "",
        "Test_Coverage": "",
        "Code_Quality": ""
    }
    
    current_section = "Summary"
    section_content = []
    
    for line in review_content.split('\n'):
        lower_line = line.lower()
        
        # Check for section headers using the 6-section structure
        if lower_line.startswith('#') or ('**' in lower_line):
            if any(term in lower_line for term in ["summary", "overview", "changes", "1."]):
                if section_content:
                    sections[current_section] = "\n".join(section_content)
                current_section = "Summary"
                section_content = []
            elif any(term in lower_line for term in ["file", "suggest", "affect", "2."]):
                if section_content:
                    sections[current_section] = "\n".join(section_content)
                current_section = "File_Suggestions"
                section_content = []
            elif any(term in lower_line for term in ["conflict", "prediction", "3."]):
                if section_content:
                    sections[current_section] = "\n".join(section_content)
                current_section = "Conflict_Predictions"
                section_content = []
            elif any(term in lower_line for term in ["break", "risk", "warning", "4."]):
                if section_content:
                    sections[current_section] = "\n".join(section_content)
                current_section = "Breakage_Risks"
                section_content = []
            elif any(term in lower_line for term in ["test", "coverage", "5."]):
                if section_content:
                    sections[current_section] = "\n".join(section_content)
                current_section = "Test_Coverage"
                section_content = []
            elif any(term in lower_line for term in ["code", "quality", "smell", "6."]):
                if section_content:
                    sections[current_section] = "\n".join(section_content)
                current_section = "Code_Quality"
                section_content = []
        else:
            # Add non-empty lines to current section content
            if line.strip():
                section_content.append(line.strip())
    
    # Add the last section content
    if section_content:
        sections[current_section] = "\n".join(section_content)
    
    return sections

def clean_text_field(text):
    """Clean text for CSV storage by removing problematic characters"""
    if not text:
        return ""
    
    # Replace newlines with space and escape quotes
    cleaned = text.replace("\n", " ").replace("\r", " ")
    cleaned = cleaned.replace('"', '""')  # CSV escape for quotes
    
    # Truncate if too long (prevent CSV errors)
    if len(cleaned) > 32000:  # Excel has column limits
        cleaned = cleaned[:32000] + "... [truncated]"
        
    return cleaned