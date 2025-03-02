import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

class GeminiModel:
    """Handler for Gemini API interactions"""
    
    def __init__(self, model_name="gemini-1.5-flash", system_prompt=None):
        """
        Initialize Gemini model
        
        Args:
            model_name (str): Name of the Gemini model to use
            system_prompt (str, optional): System prompt to use with all queries
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Gemini client"""
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(self.model_name)
    
    def generate_response(self, question):
        """
        Generate a response from the Gemini model
        
        Args:
            question (str): Question to answer
            
        Returns:
            str: Generated response
        """
        try:
            # Format prompt with system prompt if available
            if self.system_prompt:
                prompt = f"{self.system_prompt}\nQuestion: {question}"
            else:
                prompt = question
                
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error with Gemini ({self.model_name}): {e}")
            return f"Error: {str(e)}"
    
    def get_name(self):
        """Return the name of this model"""
        return "gemini"