from GithubAuth import get_pull_request

def get_file_changes(files):
    """Extract before/after changes from files in a PR"""
    final_changes = ""
    for file in files:
        if hasattr(file, 'patch') and file.patch:
            final_changes += f"File: {file.filename}\n"
            
            # Extract before and after sections
            before_lines = []
            after_lines = []
            
            for line in file.patch.split('\n'):
                if line.startswith('-') and not line.startswith('---'):
                    before_lines.append(line[1:])
                elif line.startswith('+') and not line.startswith('+++'):
                    after_lines.append(line[1:])
            
            # Add before section if there are removed lines
            if before_lines:
                final_changes += "Before:\n"
                final_changes += '\n'.join(before_lines)
                final_changes += "\n"
            
            # Add after section if there are added lines
            if after_lines:
                final_changes += "After:\n"
                final_changes += '\n'.join(after_lines)
                final_changes += "\n"
            
            final_changes += "\n"  # Add empty line between files
    
    return final_changes

def compare_pr_changes(pr_files, similar_pr_numbers, repo_owner, repo_name):
    """Compare changes between PR files and multiple similar PRs"""
    try:
        # Import directly from GithubAuth to avoid the missing module error
        from GithubAuth import get_pull_request
        
        current_pr_changes = get_file_changes(pr_files)
        all_similar_pr_changes = []
        
        # Process each similar PR
        for similar_pr_number in similar_pr_numbers:
            similar_pr = get_pull_request(similar_pr_number, repo_owner, repo_name)
            if not similar_pr:
                print(f"Error: Could not find PR #{similar_pr_number}")
                continue
                
            similar_files = similar_pr.get_files()
            
            # Get complete changes for similar PR
            similar_pr_changes = get_file_changes(similar_files)
            
            # Store changes along with PR number
            all_similar_pr_changes.append({
                'pr_number': similar_pr_number,
                'changes': similar_pr_changes
            })
            
            # Display comparison for this PR
            print(f"\n🔍 Changes in Similar PR #{similar_pr_number}:")
            print(similar_pr_changes)
        
        # Display changes in current PR
        print("\n🔍 Changes in Current PR:")
        print(current_pr_changes)
        
        return current_pr_changes, all_similar_pr_changes
        
    except Exception as e:
        print(f"Error comparing PR changes: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None