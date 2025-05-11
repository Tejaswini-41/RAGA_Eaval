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
            "file_count": {"Low": 1, "Medium": 3, "High": 8, "Critical": 15},
            "lines_changed": {"Low": 10, "Medium": 50, "High": 200, "Critical": 500},
        }
        
        self.conflict_risk_thresholds = {
            "overlap_percentage": {"Low": 10, "Medium": 30, "High": 60, "Critical": 80},
        }
        
        self.test_coverage_thresholds = {
            "test_to_code_ratio": {"Low": 0.8, "Medium": 0.5, "High": 0.2, "Critical": 0.0},
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
        # Calculate metrics based on the PR changes
        metrics = self._calculate_metrics(current_pr_changes, similar_prs_changes, pr_files)
        
        # Generate scores based on metrics
        scores = self._generate_scores(metrics)
        
        # Add detailed explanations
        explanations = self._generate_explanations(metrics, scores)
        
        return {
            "scores": scores,
            "explanations": explanations,
            "metrics": metrics
        }
    
    def _calculate_metrics(self, current_pr_changes, similar_prs_changes, pr_files=None):
        """Calculate metrics from PR content"""
        metrics = {}

        # 1. Calculate file metrics
        if pr_files:
            metrics["file_count"] = len(pr_files)

            # Count test files vs code files
            test_files = sum(1 for file in pr_files if "test" in file.lower() or file.endswith(("_test.js", "_test.py", "_spec.ts")))
            code_files = metrics["file_count"] - test_files
            metrics["test_files"] = test_files
            metrics["code_files"] = code_files
            metrics["test_to_code_ratio"] = test_files / max(code_files, 1)
        else:
            metrics["file_count"] = 0
            metrics["test_files"] = 0
            metrics["code_files"] = 0
            metrics["test_to_code_ratio"] = 0.0

        # 2. Analyze code changes
        lines_added = len(re.findall(r'^\+[^+]', current_pr_changes, re.MULTILINE))
        lines_removed = len(re.findall(r'^-[^-]', current_pr_changes, re.MULTILINE))
        metrics["lines_added"] = lines_added
        metrics["lines_removed"] = lines_removed
        metrics["lines_changed"] = lines_added + lines_removed

        # 3. Analyze overlap with similar PRs
        current_files = self._extract_file_paths(current_pr_changes)
        similar_files = []
        for pr in similar_prs_changes:
            similar_files.extend(self._extract_file_paths(pr.get('changes', '')))
        
        overlapping_files = set(current_files) & set(similar_files)
        metrics["overlapping_file_count"] = len(overlapping_files)
        metrics["overlap_percentage"] = (len(overlapping_files) / max(len(current_files), 1)) * 100 if current_files else 0

        return metrics
    
    def _extract_file_paths(self, changes_text):
        """Extract file paths from git diff text"""
        if not changes_text:
            return []
        
        # Extract files from diff headers
        file_paths = re.findall(r'diff --git a/(.*?) b/', changes_text)
        return file_paths
    
    def _generate_scores(self, metrics):
        """Generate confidence scores based on metrics"""
        scores = {}

        # Functional Change Risk
        lines_changed = metrics.get("lines_changed", 0)
        file_count = metrics.get("file_count", 0)
        if lines_changed > self.functional_risk_thresholds["lines_changed"]["Critical"] or \
           file_count > self.functional_risk_thresholds["file_count"]["Critical"]:
            scores["Functional Change Risk"] = "Critical"
        elif lines_changed > self.functional_risk_thresholds["lines_changed"]["High"] or \
             file_count > self.functional_risk_thresholds["file_count"]["High"]:
            scores["Functional Change Risk"] = "High"
        elif lines_changed > self.functional_risk_thresholds["lines_changed"]["Medium"] or \
             file_count > self.functional_risk_thresholds["file_count"]["Medium"]:
            scores["Functional Change Risk"] = "Medium"
        else:
            scores["Functional Change Risk"] = "Low"

        # Merge Conflict Risk
        overlap_percentage = metrics.get("overlap_percentage", 0)
        if overlap_percentage > self.conflict_risk_thresholds["overlap_percentage"]["Critical"]:
            scores["Merge Conflict Risk"] = "Critical"
        elif overlap_percentage > self.conflict_risk_thresholds["overlap_percentage"]["High"]:
            scores["Merge Conflict Risk"] = "High"
        elif overlap_percentage > self.conflict_risk_thresholds["overlap_percentage"]["Medium"]:
            scores["Merge Conflict Risk"] = "Medium"
        else:
            scores["Merge Conflict Risk"] = "Low"

        # Test Coverage Sufficiency
        test_to_code_ratio = metrics.get("test_to_code_ratio", 0)
        if test_to_code_ratio >= self.test_coverage_thresholds["test_to_code_ratio"]["Low"]:
            scores["Test Coverage Sufficiency"] = "Low"
        elif test_to_code_ratio >= self.test_coverage_thresholds["test_to_code_ratio"]["Medium"]:
            scores["Test Coverage Sufficiency"] = "Medium"
        elif test_to_code_ratio >= self.test_coverage_thresholds["test_to_code_ratio"]["High"]:
            scores["Test Coverage Sufficiency"] = "High"
        else:
            scores["Test Coverage Sufficiency"] = "Critical"

        return scores
    
    def _generate_explanations(self, metrics, scores):
        """Generate explanations for each risk score"""
        explanations = {}

        # Functional Change Risk explanation
        explanations["Functional Change Risk"] = (
            f"{metrics.get('lines_changed', 0)} lines changed across {metrics.get('file_count', 0)} files."
        )

        # Merge Conflict Risk explanation
        explanations["Merge Conflict Risk"] = (
            f"{metrics.get('overlapping_file_count', 0)} files overlap with recent PRs, "
            f"with an overlap percentage of {metrics.get('overlap_percentage', 0):.2f}%."
        )

        # Test Coverage Sufficiency explanation
        explanations["Test Coverage Sufficiency"] = (
            f"Test to code ratio is {metrics.get('test_to_code_ratio', 0):.2f}, "
            f"with {metrics.get('test_files', 0)} test files and {metrics.get('code_files', 0)} code files."
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