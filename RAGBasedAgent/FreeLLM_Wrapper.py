from typing import List, Any
import httpx
import asyncio
import os
from dotenv import load_dotenv

class FreeLLMWrapper:
    """Simple wrapper for free LLM API"""
    
    def __init__(self):
        """Initialize with GPT-3.5-turbo endpoint"""
        load_dotenv()
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Use a reliable free GPT-3.5-turbo proxy
        self.endpoint = "https://free.churchless.tech/v1/chat/completions"
        
    async def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses using GPT-3.5"""
        responses = []
        
        for prompt in prompts:
            # Truncate prompt to avoid token limit issues
            truncated_prompt = prompt[:4000] if len(prompt) > 4000 else prompt
            
            try:
                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": truncated_prompt}],
                    "temperature": 0.1,
                    "max_tokens": 500
                }
                
                response = await self.client.post(
                    self.endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0]["message"]["content"]
                        responses.append(content)
                    else:
                        responses.append("API response missing expected content")
                else:
                    responses.append(f"API error: Status {response.status_code}")
                    
            except Exception as e:
                responses.append(f"Error: {str(e)}")
        
        return responses