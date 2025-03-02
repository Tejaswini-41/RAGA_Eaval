# Configuration constants

# Model configurations
MODELS = {
    "gemini": "gemini-1.5-flash",
    "deepseek": "deepseek-r1-distill-qwen-32b",
    "llama": "llama3-70b-8192",
    "alibaba": "qwen-2.5-32b",
    "mixtral": "mixtral-8x7b-32768"
}

# Evaluation weights
METRIC_WEIGHTS = {
    "Relevance": 0.20,
    "Accuracy": 0.25,
    "Groundedness": 0.20, 
    "Completeness": 0.15,
    "BLEU": 0.10,
    "ROUGE": 0.10
}

# System prompt
SYSTEM_PROMPT = """You are a helpful AI assistant. Please follow these guidelines:
- Provide concise and accurate answers
- Focus on factual information
- Use clear and simple language
- Avoid speculation or uncertain statements
- Keep responses brief but complete
- If unsure, acknowledge limitations
- Maintain professional tone"""