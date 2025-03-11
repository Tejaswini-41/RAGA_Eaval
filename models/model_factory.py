import os
from dotenv import load_dotenv
from .gemini_model import GeminiModel
from .groq_models import GroqModelFactory

# Load environment variables
load_dotenv()

# System prompt for all models
SYSTEM_PROMPT = """You are a helpful AI assistant. Please follow these guidelines:
- Provide concise and accurate answers
- Focus on factual information
- Use clear and simple language
- Avoid speculation or uncertain statements
- Keep responses brief but complete
- If unsure, acknowledge limitations
- Maintain professional tone"""

class ModelFactory:
    def __init__(self):
        # Initialize models
        self.gemini_model = GeminiModel(system_prompt=SYSTEM_PROMPT)
        self.groq_models = GroqModelFactory.create_all_models(system_prompt=SYSTEM_PROMPT)
        
        # Combine all models
        self.all_models = {
            "gemini": self.gemini_model,
            **self.groq_models
        }
    
    def get_model_names(self):
        """Return list of available model names"""
        return list(self.all_models.keys())
    
    def generate_response(self, model_name, question):
        """Generate response from specified model"""
        if model_name not in self.all_models:
            raise ValueError(f"Unknown model: {model_name}")
        return self.all_models[model_name].generate_response(question)
    
    def generate_all_responses(self, question):
        """Generate responses from all models for a question"""
        responses = {}
        for model_name, model in self.all_models.items():
            print(f"Generating response from {model_name}...")
            responses[model_name] = model.generate_response(question)
        return responses
    
    def get_model(self, model_name):
        """Get a specific model by name"""
        if model_name not in self.all_models:
            raise ValueError(f"Unknown model: {model_name}")
        return self.all_models[model_name]

    def generate_response_with_prompt(self, model_name, question, system_prompt):
        """Generate response from a specific model using a custom system prompt"""
        try:
            model = self.get_model(model_name)
            
            # Store the original system prompt
            original_prompt = model.system_prompt
            
            # Set the new system prompt
            model.system_prompt = system_prompt
            
            # Generate response
            response = model.generate_response(question)
            
            # Restore the original system prompt
            model.system_prompt = original_prompt
            
            return response
            
        except Exception as e:
            print(f"Error generating response from {model_name}: {e}")
            return f"Error: {str(e)}"

    # Add a getter for the system prompt
    def get_system_prompt(self):
        """Return the current system prompt"""
        return SYSTEM_PROMPT
