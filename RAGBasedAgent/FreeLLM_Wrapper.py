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
            "claude": "claude-3-haiku-20240307",
            "llama": "llama-3-8b-8192",
            "gpt4": "gpt-4o",
            "mixtral": "mixtral-8x7b-32768"
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
            
            # Extract content from Claude response format
            return data["content"][0]["text"]
            
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