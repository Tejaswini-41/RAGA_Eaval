from models.groq_models import GroqModel

def generate_review(current_pr_changes, similar_pr_changes, use_local_model=False):
    """Generate review suggestions based on PR changes"""
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
        if use_local_model:
            # Use a local model with transformers
            from transformers import pipeline
            
            # Load an open-source model
            review_model = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")
            
            # Generate response
            response = review_model(prompt, max_length=512)
            review = response[0]["generated_text"]
        else:
            # Use Groq with Alibaba Qwen model
            system_prompt = "You are an expert code reviewer. Analyze the pull request and provide# filepath: t:\RAGA_Eaval\RAGBasedAgent\generate_review.py"
from models.groq_models import GroqModel

def generate_review(current_pr_changes, similar_pr_changes, use_local_model=False):
    """Generate review suggestions based on PR changes"""
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
        if use_local_model:
            # Use a local model with transformers
            from transformers import pipeline
            
            # Load an open-source model
            review_model = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")
            
            # Generate response
            response = review_model(prompt, max_length=512)
            review = response[0]["generated_text"]
        else:
            # Use Groq with Alibaba Qwen model
            system_prompt = "You are an expert code reviewer. Analyze the pull request and provide"