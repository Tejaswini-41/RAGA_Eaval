import re
import os
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

# Import functionality from existing files
from similarity_query import calculate_cosine_similarity, text_embedding
from change_analyzer import get_file_changes

class ConfidenceScorer:
    """
    Adds confidence scores and risk assessments to PR reviews
    using existing RAG system components.
    """
    
    def __init__(self):
        """Initialize with default risk thresholds"""
        self.risk_levels = ["Low", "Medium", "High", "Critical"]
        
        # Define thresholds for different risk types
        self.functional_risk_thresholds = {
            "file_count": {
                "Low": 1,
                "Medium": 3,
                "High": 8,
                "Critical": 15
            },
            "code_change_percentage": {
                "Low": 5,
                "Medium": 15,
                "High": 30,
                "Critical": 50
            }
        }
        
        self.conflict_risk_thresholds = {
            "change_frequency": {
                "Low": 1,
                "Medium": 3,
                "High": 5,
                "Critical": 10
            },
            "overlap_percentage": {
                "Low": 10,
                "Medium": 30,
                "High": 60,
                "Critical": 80
            }
        }
        
        self.test_coverage_thresholds = {
            "test_to_code_ratio": {
                "Low": 0.8,
                "Medium": 0.5,
                "High": 0.2,
                "Critical": 0.0
            }
        }
    
    def analyze_pr(self, current_pr_changes, similar_prs_changes, pr_files=None, query_results=None):
        """
        Analyze PR changes and provide confidence scores
        
        Args:
            current_pr_changes: String content of current PR changes
            similar_prs_changes: List of similar PR changes
            pr_files: List of files changed in the PR
            query_results: Results from ChromaDB query (contains distances/similarities)
            
        Returns:
            Dict of confidence scores and explanations
        """
        # Extract similarity scores from query results if available
        similarity_scores = {}
        if query_results and "metadatas" in query_results and query_results["metadatas"]:
            for i, metadata in enumerate(query_results["metadatas"][0]):
                if "pr_number" in metadata and "distance" in query_results["distances"][0]:
                    pr_num = metadata["pr_number"]
                    # Convert distance to similarity score (distance of 0 = similarity of 1)
                    similarity_scores[pr_num] = 1 - min(query_results["distances"][0][i], 1.0)
        
        # Calculate metrics based on the PR changes
        metrics = self._calculate_metrics(current_pr_changes, similar_prs_changes, pr_files, similarity_scores)
        
        # Generate scores based on metrics
        scores = self._generate_scores(metrics)
        
        # Add detailed explanations
        explanations = self._generate_explanations(metrics, scores)
        
        return {
            "scores": scores,
            "explanations": explanations,
            "metrics": metrics
        }
    
    def _calculate_metrics(self, current_pr_changes, similar_prs_changes, pr_files=None, similarity_scores=None):
        """Calculate metrics from PR content using existing functionality"""
        metrics = {}
        
        # 1. Calculate file metrics
        if pr_files:
            metrics["file_count"] = len(pr_files)
            
            # Count test files vs code files
            test_files = sum(1 for file in pr_files if "test" in str(file.filename).lower())
            code_files = metrics["file_count"] - test_files
            metrics["test_files"] = test_files
            metrics["code_files"] = code_files
            metrics["test_to_code_ratio"] = test_files / max(code_files, 1)
            
            # Use existing functionality to analyze files
            if hasattr(pr_files[0], 'patch'):
                # Use the change analyzer's get_file_changes to extract details
                file_changes = get_file_changes(pr_files)
                metrics["total_changes"] = len(file_changes)
            
            # Track file types
            file_types = {}
            for file in pr_files:
                ext = os.path.splitext(str(file.filename))[1].lower()
                if ext:
                    file_types[ext] = file_types.get(ext, 0) + 1
            metrics["file_types"] = file_types
        else:
            # Estimate from changes text
            metrics["file_count"] = current_pr_changes.count("diff --git")
            metrics["test_to_code_ratio"] = 0.5  # Default assumption
        
        # 2. Analyze code complexity
        lines_added = len(re.findall(r'^\+[^+]', current_pr_changes, re.MULTILINE))
        lines_removed = len(re.findall(r'^-[^-]', current_pr_changes, re.MULTILINE))
        metrics["lines_added"] = lines_added
        metrics["lines_removed"] = lines_removed
        metrics["lines_changed"] = lines_added + lines_removed
        
        # 3. Analyze overlap with similar PRs
        metrics["similar_pr_count"] = len(similar_prs_changes)
        
        # Find modified files that appear in multiple PRs
        current_files = self._extract_file_paths(current_pr_changes)
        similar_files = []
        
        # Track similarity-weighted risks
        similarity_weighted_risk = 0
        
        for pr in similar_prs_changes:
            pr_files = self._extract_file_paths(pr.get('changes', ''))
            similar_files.extend(pr_files)
            
            # Calculate similarity risk if we have scores
            if similarity_scores and pr.get('pr_number') in similarity_scores:
                sim_score = similarity_scores[pr.get('pr_number')]
                # Higher similarity = higher risk of conflicts
                similarity_weighted_risk += sim_score
        
        metrics["similarity_risk"] = similarity_weighted_risk / max(len(similar_prs_changes), 1)
            
        # Calculate overlap metrics
        overlapping_files = set(current_files) & set(similar_files)
        metrics["overlapping_files"] = list(overlapping_files)
        metrics["overlapping_file_count"] = len(overlapping_files)
        metrics["overlap_percentage"] = (len(overlapping_files) / max(len(current_files), 1)) * 100 if current_files else 0
        
        # 4. Analyze functional complexity
        # Count complex patterns (functions, classes, conditionals)
        patterns = {
            "functions": r'(function\s+\w+|def\s+\w+)\(',
            "classes": r'class\s+\w+',
            "conditionals": r'(if\s+|else\s+|switch\s+|case\s+:)',
            "loops": r'(for\s+|while\s+|do\s+{)',
            "try_except": r'(try\s+{|catch\s+|except\s+)'
        }
        
        complexity_counts = {}
        for name, pattern in patterns.items():
            complexity_counts[name] = len(re.findall(pattern, current_pr_changes))
        
        metrics["complexity_counts"] = complexity_counts
        metrics["complexity_score"] = sum(complexity_counts.values())
        
        # Calculate code change percentage (approximation)
        total_lines = current_pr_changes.count('\n')
        metrics["code_change_percentage"] = (metrics["lines_changed"] / max(total_lines, 1)) * 100 if total_lines > 0 else 0
        
        # Calculate change frequency of files in similar PRs
        if similar_prs_changes:
            change_frequency = {}
            for pr in similar_prs_changes:
                pr_files = self._extract_file_paths(pr.get('changes', ''))
                for file in pr_files:
                    change_frequency[file] = change_frequency.get(file, 0) + 1
            
            metrics["change_frequency"] = max(change_frequency.values()) if change_frequency else 0
            metrics["frequently_changed_files"] = [file for file, count in change_frequency.items() if count > 1]
            
            # Calculate "hotspot" files (frequently changed AND in current PR)
            hotspot_files = [file for file in current_files if file in change_frequency and change_frequency[file] > 1]
            metrics["hotspot_file_count"] = len(hotspot_files)
            metrics["hotspot_files"] = hotspot_files
        else:
            metrics["change_frequency"] = 0
            metrics["frequently_changed_files"] = []
            metrics["hotspot_file_count"] = 0
            metrics["hotspot_files"] = []
        
        return metrics
    
    def _extract_file_paths(self, changes_text):
        """Extract file paths from git diff text"""
        if not changes_text:
            return []
        
        # Extract files from diff headers
        file_paths = []
        diff_headers = re.findall(r'File: (.*?)(?:\n|$)', changes_text)
        if diff_headers:
            file_paths = diff_headers
        else:
            # Try alternative pattern if the first doesn't work
            diff_headers = re.findall(r'diff --git a/(.?) b/(.)', changes_text)
            for header in diff_headers:
                if len(header) > 1 and header[1]:  # Use the 'b' path (new file)
                    file_paths.append(header[1])
        
        return file_paths
    
    def _generate_scores(self, metrics):
        """Generate confidence scores based on metrics"""
        scores = {}
        
        # 1. Functional Change Risk
        functional_risk_level = "Low"
        for threshold_name, thresholds in self.functional_risk_thresholds.items():
            if metrics.get(threshold_name, 0) > thresholds["Critical"]:
                functional_risk_level = "Critical"
                break
            elif metrics.get(threshold_name, 0) > thresholds["High"]:
                functional_risk_level = "High" if functional_risk_level != "Critical" else functional_risk_level
            elif metrics.get(threshold_name, 0) > thresholds["Medium"]:
                functional_risk_level = "Medium" if functional_risk_level not in ["Critical", "High"] else functional_risk_level
        
        # Adjust based on complexity score
        if metrics.get("complexity_score", 0) > 30:
            functional_risk_level = self._increase_risk_level(functional_risk_level)
        
        scores["Functional Change Risk"] = functional_risk_level
        
        # 2. Merge Conflict Risk
        conflict_risk_level = "Low"
        for threshold_name, thresholds in self.conflict_risk_thresholds.items():
            if metrics.get(threshold_name, 0) > thresholds["Critical"]:
                conflict_risk_level = "Critical"
                break
            elif metrics.get(threshold_name, 0) > thresholds["High"]:
                conflict_risk_level = "High" if conflict_risk_level != "Critical" else conflict_risk_level
            elif metrics.get(threshold_name, 0) > thresholds["Medium"]:
                conflict_risk_level = "Medium" if conflict_risk_level not in ["Critical", "High"] else conflict_risk_level
        
        # Adjust based on similarity risk
        if metrics.get("similarity_risk", 0) > 0.8:
            conflict_risk_level = self._increase_risk_level(conflict_risk_level)
        elif metrics.get("hotspot_file_count", 0) > 2:
            conflict_risk_level = self._increase_risk_level(conflict_risk_level)
            
        scores["Merge Conflict Risk"] = conflict_risk_level
        
        # 3. Test Coverage Sufficiency
        test_coverage_level = "Low"
        for threshold_name, thresholds in self.test_coverage_thresholds.items():
            if metrics.get(threshold_name, 0) >= thresholds["Low"]:
                test_coverage_level = "High"
                break
            elif metrics.get(threshold_name, 0) >= thresholds["Medium"]:
                test_coverage_level = "Medium"
                break
            elif metrics.get(threshold_name, 0) >= thresholds["High"]:
                test_coverage_level = "Low"
                break
            else:
                test_coverage_level = "Critical"
        
        scores["Test Coverage Sufficiency"] = test_coverage_level
        
        return scores
    
    def _increase_risk_level(self, current_level):
        """Increase the risk level by one step"""
        levels = ["Low", "Medium", "High", "Critical"]
        current_index = levels.index(current_level)
        if current_index < len(levels) - 1:
            return levels[current_index + 1]
        return current_level
    
    def _generate_explanations(self, metrics, scores):
        """Generate explanations for each risk score"""
        explanations = {}
        
        # Functional Change Risk explanation
        if scores["Functional Change Risk"] == "High" or scores["Functional Change Risk"] == "Critical":
            explanations["Functional Change Risk"] = (
                f"High complexity detected with {metrics.get('complexity_counts', {}).get('functions', 0)} functions, "
                f"{metrics.get('complexity_counts', {}).get('conditionals', 0)} conditionals, and "
                f"{metrics.get('lines_changed')} lines changed across {metrics.get('file_count', 0)} files."
            )
        else:
            explanations["Functional Change Risk"] = (
                f"Moderate changes with {metrics.get('lines_changed')} lines modified "
                f"across {metrics.get('file_count', 0)} files."
            )
        
        # Merge Conflict Risk explanation
        overlapping_files = metrics.get('overlapping_file_count', 0)
        frequently_changed = metrics.get('frequently_changed_files', [])
        hotspot_files = metrics.get('hotspot_files', [])
        
        frequent_files_str = ", ".join(frequently_changed[:3]) if frequently_changed else "none"
        hotspot_files_str = ", ".join(hotspot_files[:3]) if hotspot_files else "none"
        
        if scores["Merge Conflict Risk"] == "High" or scores["Merge Conflict Risk"] == "Critical":
            if hotspot_files:
                explanations["Merge Conflict Risk"] = (
                    f"{overlapping_files} files have been modified in similar PRs recently. "
                    f"Hot-spot files (high-risk for conflicts): {hotspot_files_str}"
                )
            else:
                explanations["Merge Conflict Risk"] = (
                    f"{overlapping_files} files have been modified in similar PRs recently. "
                    f"Frequently changed files include: {frequent_files_str}"
                )
        else:
            explanations["Merge Conflict Risk"] = (
                f"Only {overlapping_files} files overlap with recent PRs, with moderate change frequency."
            )
        
        # Test Coverage Sufficiency explanation
        test_ratio = metrics.get('test_to_code_ratio', 0)
        if scores["Test Coverage Sufficiency"] == "Low" or scores["Test Coverage Sufficiency"] == "Critical":
            explanations["Test Coverage Sufficiency"] = (
                f"Test to code ratio is low ({test_ratio:.2f}). Only {metrics.get('test_files', 0)} test files "
                f"for {metrics.get('code_files', 0)} code files. Consider adding more tests."
            )
        else:
            explanations["Test Coverage Sufficiency"] = (
                f"Test to code ratio is {test_ratio:.2f}, with {metrics.get('test_files', 0)} test files "
                f"for {metrics.get('code_files', 0)} code files."
            )
        
        return explanations

    def enhance_review_with_confidence(self, review_content, confidence_analysis):
        """
        Enhance an existing review with confidence scores
        
        Args:
            review_content: Original review content
            confidence_analysis: Dict containing scores, explanations, and metrics
            
        Returns:
            Enhanced review with confidence scores
        """
        scores = confidence_analysis["scores"]
        explanations = confidence_analysis["explanations"]
        
        # Create a confidence section to insert
        confidence_section = "\n## 🎯 Confidence Assessment\n\n"
        
        # Add each risk score with its color indicator and explanation
        for risk_name, risk_level in scores.items():
            # Add color emoji indicator
            if risk_level == "Critical":
                emoji = "🔴"
            elif risk_level == "High":
                emoji = "🟠"
            elif risk_level == "Medium":
                emoji = "🟡"
            else:  # Low
                emoji = "🟢"
            
            confidence_section += f"### {emoji} {risk_name}: {risk_level}\n"
            confidence_section += f"{explanations[risk_name]}\n\n"
        
        # Find where to insert the confidence section (after summary if it exists)
        if "# Summary" in review_content or "## Summary" in review_content:
            # Insert after the summary section
            pattern = r'(#+ Summary.*?)(\n#+ |$)'
            enhanced_review = re.sub(pattern, r'\1\n' + confidence_section + r'\2', review_content, flags=re.DOTALL)
        else:
            # Insert at the beginning if no summary section
            enhanced_review = confidence_section + review_content
        
        return enhanced_review

def enhance_review_with_confidence_scores(current_pr_changes, similar_prs_changes, review_content, 
                                         pr_files=None, query_results=None):
    """
    Standalone function to enhance review with confidence scores using existing RAG components
    
    Args:
        current_pr_changes: String content of current PR changes
        similar_prs_changes: List of similar PR changes
        review_content: Original review content
        pr_files: List of files changed in the PR
        query_results: Results from similar PR query (contains similarity information)
    
    Returns:
        Review content enhanced with confidence scores
    """
    scorer = ConfidenceScorer()
    confidence_analysis = scorer.analyze_pr(
        current_pr_changes, 
        similar_prs_changes, 
        pr_files,
        query_results
    )
    enhanced_review = scorer.enhance_review_with_confidence(review_content, confidence_analysis)
    return enhanced_review