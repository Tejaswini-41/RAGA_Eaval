import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

class AIRecommender:
    """Uses Gemini to provide intelligent recommendations for improving responses"""
    
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-flash")
    
    def get_ai_recommendations(self, question, reference_response, model_response, scores):
        """
        Generate AI-powered recommendations based on evaluation results
        
        Args:
            question: The original question
            reference_response: The reference/ground truth response
            model_response: The response being evaluated
            scores: Dict of evaluation metrics scores
            
        Returns:
            Dictionary containing AI-generated recommendations
        """
        try:
            # Format the prompt for Gemini
            prompt = self._create_analysis_prompt(question, reference_response, 
                                                 model_response, scores)
            
            # Get recommendations from Gemini
            response = self.model.generate_content(prompt)
            recommendations = self._parse_gemini_response(response.text)
            
            return recommendations
        
        except Exception as e:
            print(f"Error generating AI recommendations: {e}")
            return {"error": str(e)}
    
    def _create_analysis_prompt(self, question, reference_response, model_response, scores):
        """Create a detailed prompt for the analysis"""
        metrics_description = "\n".join([f"- {metric}: {score}" for metric, score in scores.items()])
        
        prompt = f"""As an AI evaluation expert, analyze this response and provide specific recommendations to improve it.

QUESTION: 
{question}

REFERENCE RESPONSE (GROUND TRUTH): 
{reference_response}

MODEL RESPONSE: 
{model_response}

EVALUATION SCORES:
{metrics_description}

TASK:
Based on the scores and response content, provide 3-5 specific recommendations to improve the model's response. Focus on:

1. Relevance - how well the response addresses the question
2. Accuracy - factual correctness of the information
3. Groundedness - how well the response relates to the reference response
4. Completeness - whether the response covers all key points
5. BLEU/ROUGE - text similarity to reference

For each recommendation:
- Identify the specific issue
- Suggest a concrete improvement
- Explain why this would improve the relevant metrics

FORMAT YOUR RESPONSE AS JSON:
{{
  "recommendations": [
    {{
      "issue": "Issue description",
      "suggestion": "Specific suggestion",
      "target_metrics": ["Metric1", "Metric2"],
      "expected_improvement": "Why this would help"
    }},
    ...
  ]
}}
"""
        return prompt
    
    def _parse_gemini_response(self, response_text):
        """Parse the Gemini response to extract recommendations"""
        try:
            # Try to extract JSON content
            import json
            import re
            
            # Look for JSON pattern in the response
            json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                recommendations = json.loads(json_str)
                return recommendations
            else:
                # If no JSON found, do best-effort parsing
                return {"recommendations": [{"suggestion": response_text}]}
                
        except Exception as e:
            print(f"Error parsing AI recommendations: {e}")
            return {"recommendations": [{"suggestion": response_text}]}