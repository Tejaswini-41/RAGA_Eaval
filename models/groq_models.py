import os
from dotenv import load_dotenv
import groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define available models
MODEL_CONFIGS = {
    "deepseek": {
        "groq_id": "deepseek-r1-distill-qwen-32b",
        "display_name": "DeepSeek"
    },
    "llama": {
        "groq_id": "llama3-70b-8192", 
        "display_name": "Llama3"
    },
    "alibaba": {
        "groq_id": "qwen-2.5-32b",
        "display_name": "Alibaba Qwen"
    }
    # "mixtral": {
    #     "groq_id": "mixtral-8x7b-32768",
    #     "display_name": "Mixtral"
    # }
}

class GroqModel:
    """Handler for Groq API interactions"""
    
    def __init__(self, model_id, system_prompt=None):
        """
        Initialize Groq model
        
        Args:
            model_id (str): ID of model to use ("deepseek", "llama", etc.)
            system_prompt (str, optional): System prompt to use with all queries
        """
        if model_id not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model ID: {model_id}")
        
        self.model_id = model_id
        self.model_config = MODEL_CONFIGS[model_id]
        self.groq_model_name = self.model_config["groq_id"]
        self.display_name = self.model_config["display_name"]
        self.system_prompt = system_prompt
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Groq client"""
        self.client = groq.Client(api_key=GROQ_API_KEY)
    
    def generate_response(self, question, max_tokens=500):
        """
        Generate a response from the model
        
        Args:
            question (str): Question to answer
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            str: Generated response
        """
        try:
            messages = []
            
            # Add system prompt if available
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
                
            # Add user question
            messages.append({"role": "user", "content": question})
            
            # Generate completion
            response = self.client.chat.completions.create(
                model=self.groq_model_name,
                messages=messages,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error with {self.display_name} ({self.groq_model_name}): {e}")
            return f"Error: {str(e)}"
    
    def get_name(self):
        """Return the name of this model"""
        return self.model_id

class GroqModelFactory:
    """Factory for creating Groq model instances"""
    
    @staticmethod
    def create_model(model_id, system_prompt=None):
        """Create a GroqModel instance for the specified model ID"""
        return GroqModel(model_id, system_prompt)
    
    @staticmethod
    def get_available_models():
        """Get list of available Groq model IDs"""
        return list(MODEL_CONFIGS.keys())
    
    @staticmethod
    def create_all_models(system_prompt=None):
        """Create instances of all available Groq models"""
        return {
            model_id: GroqModel(model_id, system_prompt)
            for model_id in MODEL_CONFIGS
        }