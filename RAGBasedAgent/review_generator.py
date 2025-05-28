from dotenv import load_dotenv
import os
import sys
from datetime import datetime
import csv

# Add import at top
from prompts.review_prompts import ReviewPrompts

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

def generate_review(current_pr_changes, similar_prs_changes, pr_number=None, similar_pr_number=None, current_pr_file=None, model_name="gemini"):
    """Generate review suggestions based on PR changes using specified model"""
    # Load environment variables
    load_dotenv()
    
    # Create combined similar PR changes for context
    combined_similar_changes = ""
    for i, pr_data in enumerate(similar_prs_changes):
        combined_similar_changes += f"\n--- Similar PR #{pr_data['pr_number']} ---\n"
        combined_similar_changes += pr_data['changes']
    
    # If empty, handle gracefully
    if not combined_similar_changes:
        combined_similar_changes = "No similar PR changes available."
    
    # Get current files as a comma-separated list
    current_files = current_pr_file
    if isinstance(current_files, list):
        current_files = ", ".join(current_files)
    elif not current_files:
        current_files = "Unknown files"
    
    # Create custom template with the 6 required sections
    custom_template = f"""As an expert code reviewer, analyze this PR and provide specific suggestions:

CURRENT PR DETAILS:
PR #{pr_number or 'Unknown'}
Changed Files: {current_files}

CODE CHANGES:
{{current_pr}}

SIMILAR PRS HISTORY:
{{similar_prs}}

PROVIDE THE FOLLOWING SECTIONS:
1. Summary of Changes - Brief overview of what this PR does
2. File Change Suggestions - Identify additional files that might need changes based on modified files
3. Conflict Prediction - Flag files with high change frequency that could cause conflicts
4. Breakage Risk Warning - Note which changes might break existing functionality
5. Test Coverage Advice - Recommend test files that should be updated
6. Code Quality Suggestions - Point out potential code smells or duplication

Be specific with file names, function names, and line numbers when possible.
"""

    # Create custom system prompt
    custom_system_prompt = """You are an expert code reviewer with deep expertise in software engineering best practices.
Your task is to thoroughly analyze pull requests and provide actionable, specific feedback."""
    
    # Safely handle template and system prompt updates
    try:
        # Check if methods exist before calling them
        if hasattr(ReviewPrompts, 'update_template') and callable(getattr(ReviewPrompts, 'update_template')):
            ReviewPrompts.update_template(custom_template)
        
        if hasattr(ReviewPrompts, 'update_system_prompt') and callable(getattr(ReviewPrompts, 'update_system_prompt')):
            ReviewPrompts.update_system_prompt(custom_system_prompt)
            
        # Try to get current prompt if method exists
        if hasattr(ReviewPrompts, 'get_current_prompt') and callable(getattr(ReviewPrompts, 'get_current_prompt')):
            review_template, system_prompt = ReviewPrompts.get_current_prompt()
        else:
            # Fall back to using custom prompts directly
            review_template = custom_template
            system_prompt = custom_system_prompt
    except Exception as e:
        print(f"Warning: Could not update prompt templates: {e}")
        # Use the custom templates directly as fallback
        review_template = custom_template
        system_prompt = custom_system_prompt
    
    # Format prompt with PR data
    try:
        prompt = review_template.format(
            similar_prs=combined_similar_changes,
            current_pr=current_pr_changes
        )
    except Exception as e:
        print(f"Warning: Error formatting prompt: {e}")
        # Fallback to a simple format if formatting fails
        prompt = f"""Review this PR:
PR #{pr_number or 'Unknown'}
Changed Files: {current_files}

CURRENT PR:
{current_pr_changes[:10000]}  # Limit size in case of very large PRs

SIMILAR PRS:
{combined_similar_changes[:5000]}  # Limit size
"""
    
    try:
        # Import model factory
        from models.model_factory import ModelFactory
        
        # Get model factory
        model_factory = ModelFactory()
        
        print(f"Generating review with {model_name} model...")
        
        # Generate review using the specified model
        response = model_factory.generate_response_with_prompt(
            model_name, 
            prompt,
            system_prompt
        )
        
        print("\n✅ Generated AI Review:")
        # Display full review instead of truncating
        print(response)
        
        # Make sure review has all required sections
        response = ensure_required_sections(response, pr_number, current_files)
        
        # Save review to CSV if PR number is provided
        if pr_number:
            # Use the most similar PR number for the CSV (first one)
            most_similar_pr = similar_prs_changes[0]['pr_number'] if similar_prs_changes else similar_pr_number
            save_review_to_csv(pr_number, model_name, response, current_pr_file, most_similar_pr, current_pr_changes, combined_similar_changes)
        
        return response
        
    except Exception as e:
        print(f"Error using {model_name} model: {e}")
        print("Falling back to rule-based review generation...")
        
        # Simple rule-based review generation as fallback but with all required sections
        review = "# Code Review Summary\n\n"
        
        # Section 1: Summary of Changes
        review += "## 1. Summary of Changes\n"
        added_count = current_pr_changes.count("After:")
        removed_count = current_pr_changes.count("Before:")
        review += f"- Modified approximately {added_count} file(s)\n"
        
        # Look for common patterns
        if "null" in current_pr_changes.lower() or "None" in current_pr_changes:
            review += "- Added null checks, improving code robustness\n"
            
        if "test" in current_pr_changes.lower():
            review += "- Modified tests, improving code coverage\n"
            
        if "fix" in current_pr_changes.lower() or "bug" in current_pr_changes.lower():
            review += "- Fixed bugs in existing functionality\n"
        
        # Section 2: File Change Suggestions
        review += "\n## 2. File Change Suggestions\n"
        review += "- Consider reviewing related configuration files\n"
        review += "- Check if documentation needs updates to reflect these changes\n"
        
        # Section 3: Conflict Prediction
        review += "\n## 3. Conflict Prediction\n"
        review += "- No specific conflicts predicted at this time\n"
        review += "- Monitor frequently changed files in this repository\n"
        
        # Section 4: Breakage Risk Warning
        review += "\n## 4. Breakage Risk Warning\n"
        review += "- Changes to core functionality may impact dependent modules\n"
        review += "- Consider thorough testing before merging\n"
        
        # Section 5: Test Coverage Advice
        review += "\n## 5. Test Coverage Advice\n"
        review += "- Consider adding unit tests for the changes\n"
        review += "- Ensure edge cases are properly tested\n"
        
        # Section 6: Code Quality Suggestions
        review += "\n## 6. Code Quality Suggestions\n"
        review += "- Ensure error handling is comprehensive\n"
        review += "- Check for code duplication and potential refactoring opportunities\n"
        
        return review

# Helper function to ensure all required sections are present
def ensure_required_sections(review_text, pr_number, current_files):
    """Ensure the review contains all the required sections"""
    required_sections = [
        {"name": "Summary of Changes", "number": "1"},
        {"name": "File Change Suggestions", "number": "2"},
        {"name": "Conflict Prediction", "number": "3"},
        {"name": "Breakage Risk Warning", "number": "4"},
        {"name": "Test Coverage Advice", "number": "5"},
        {"name": "Code Quality Suggestions", "number": "6"}
    ]
    
    # Check which sections are missing
    missing_sections = []
    for section in required_sections:
        if (section["name"].lower() not in review_text.lower() and 
            section["name"].lower().replace(" ", "") not in review_text.lower() and
            f"{section['number']}." not in review_text):
            missing_sections.append(section)
    
    # If all sections are present, return the original review
    if not missing_sections:
        return review_text
    
    # Otherwise, add missing sections with placeholder content
    formatted_review = review_text
    
    # Add missing sections to the end
    for section in missing_sections:
        if section["name"] == "Summary of Changes":
            formatted_review += f"\n\n## {section['number']}. Summary of Changes\n- This PR makes changes to {current_files}\n"
        elif section["name"] == "File Change Suggestions":
            formatted_review += f"\n\n## {section['number']}. File Change Suggestions\n- Consider reviewing related files that might need updates\n"
        elif section["name"] == "Conflict Prediction":
            formatted_review += f"\n\n## {section['number']}. Conflict Prediction\n- No specific conflicts predicted at this time\n"
        elif section["name"] == "Breakage Risk Warning":
            formatted_review += f"\n\n## {section['number']}. Breakage Risk Warning\n- Consider the impact of these changes on existing functionality\n"
        elif section["name"] == "Test Coverage Advice":
            formatted_review += f"\n\n## {section['number']}. Test Coverage Advice\n- Consider adding tests for the modified functionality\n"
        elif section["name"] == "Code Quality Suggestions":
            formatted_review += f"\n\n## {section['number']}. Code Quality Suggestions\n- Review code for potential improvements in readability and maintainability\n"
    
    return formatted_review

def extract_review_sections(review_content):
    # Fix section extraction for Code Quality
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
        
        # Improved section detection
        if lower_line.startswith('#') or ('**' in lower_line) or any(f"{i}." in lower_line for i in range(1, 7)):
            # Check for each section type
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
            elif any(term in lower_line for term in ["quality", "smell", "duplication", "6."]):
                if section_content:
                    sections[current_section] = "\n".join(section_content)
                current_section = "Code_Quality"
                section_content = []
        else:
            # Add non-empty lines to current section content
            if line.strip():
                section_content.append(line.strip())
    
    # Make sure the last section is added
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