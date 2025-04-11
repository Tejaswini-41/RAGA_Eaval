from typing import List, Any
import httpx
import asyncio
import json
import os
from dotenv import load_dotenv

class FreeLLMWrapper:
    """Wrapper for free LLM APIs compatible with RAGAS"""
    
    def __init__(self, model_type="claude"):
        """
        Initialize with preferred model type
        model_type options: "claude", "llama", "gpt4", "mixtral"
        """
        load_dotenv()
        self.model_type = model_type
        self.client = httpx.AsyncClient(timeout=60.0)
        
        # Load API keys from .env
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        
        # API endpoints from cheahjs/free-llm-api-resources
        self.endpoints = {
            "claude": "https://api.anthropic.com/v1/messages",
            "llama": "https://api.groq.com/openai/v1/chat/completions",
            "gpt4": "https://4.ai.faked.x10.mx/v1/chat/completions",
            "mixtral": "https://api.groq.com/openai/v1/chat/completions"
        }
        
        # Model mappings
        self.models = {
            "claude": "claude-instant-1.2",  # Try smaller Claude model
            "llama": "llama-2-7b-chat",       # Try smaller Llama model
            "gpt4": "gpt-3.5-turbo",         # Use GPT-3.5 instead of GPT-4
            "mixtral": "mixtral-8x7b"        # Use shorter context Mixtral
        }

    async def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses using free LLM APIs - compatible with RAGAS"""
        responses = []
        
        for prompt in prompts:
            if self.model_type == "claude":
                response = await self._generate_with_claude(prompt)
            elif self.model_type == "llama" or self.model_type == "mixtral":
                response = await self._generate_with_groq(prompt)
            elif self.model_type == "gpt4":
                response = await self._generate_with_gpt4_proxy(prompt)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
            responses.append(response)
        
        return responses
    
    async def _generate_with_claude(self, prompt: str) -> str:
        """Generate response using Claude API"""
        try:
            headers = {
                "x-api-key": self.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            payload = {
                "model": self.models["claude"],
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1000
            }
            
            response = await self.client.post(
                self.endpoints["claude"], 
                headers=headers, 
                json=payload
            )
            data = response.json()
            
            # Improved error handling for Claude response format
            if "content" in data:
                if isinstance(data["content"], list) and len(data["content"]) > 0:
                    if "text" in data["content"][0]:
                        return data["content"][0]["text"]
            
            # If we can't extract using the expected format, look for alternatives
            if "completion" in data:
                return data["completion"]
            elif "choices" in data and len(data["choices"]) > 0:
                if "message" in data["choices"][0]:
                    return data["choices"][0]["message"]["content"]
            
            # Return raw response for debugging
            print(f"Claude API response structure: {list(data.keys())}")
            return str(data)[:1000]  # Return part of raw response for debugging
            
        except Exception as e:
            print(f"Error with Claude API: {e}")
            return f"API Error: {str(e)}"
    
    async def _generate_with_groq(self, prompt: str) -> str:
        """Generate response using Groq API (for Llama and Mixtral)"""
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.models[self.model_type],
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            response = await self.client.post(
                self.endpoints[self.model_type], 
                headers=headers, 
                json=payload
            )
            data = response.json()
            
            # Extract content from OpenAI-compatible response format
            return data["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"Error with Groq API: {e}")
            return f"API Error: {str(e)}"
    
    async def _generate_with_gpt4_proxy(self, prompt: str) -> str:
        """Generate response using GPT-4 proxy endpoint"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.models["gpt4"],
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            response = await self.client.post(
                self.endpoints["gpt4"], 
                headers=headers, 
                json=payload
            )
            data = response.json()
            
            # Extract content from OpenAI-compatible response format
            return data["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"Error with GPT-4 proxy API: {e}")
            return f"API Error: {str(e)}"