from GithubAuth import get_pull_request

def get_file_changes(files):
    """Extract before/after changes from files in a PR"""
    final_changes = ""
    for file in files:
        if file.patch:
            added_lines = [line[1:] for line in file.patch.split('\n') 
                          if line.startswith('+') and not line.startswith('+++')]
            removed_lines = [line[1:] for line in file.patch.split('\n') 
                            if line.startswith('-') and not line.startswith('---')]

            before_change = "\n".join(removed_lines)
            after_change = "\n".join(added_lines)

            final_changes += f"\nFile: {file.filename}\nBefore:\n{before_change}\nAfter:\n{after_change}\n"

    return final_changes

def compare_pr_changes(pr_files, similar_pr_number, repo_owner, repo_name):
    """Compare changes between PR files and a similar PR"""
    try:
        # Get similar PR
        similar_pr = get_pull_request(similar_pr_number, repo_owner, repo_name)
        if not similar_pr:
            return None, None
            
        similar_files = similar_pr.get_files()
        
        # Get changes for current PR - without truncation
        current_pr_changes = get_file_changes(pr_files)
        
        # Get changes for similar PR - without truncation
        similar_pr_changes = get_file_changes(similar_files)
        
        print("\n🔍 Changes in Current PR:")
        # Display full content instead of truncating
        print(current_pr_changes)
        
        print("\n🔍 Changes in Similar PR:")
        # Display full content instead of truncating
        print(similar_pr_changes)
        
        return current_pr_changes, similar_pr_changes
        
    except Exception as e:
        print(f"Error comparing PR changes: {e}")
        return None, None