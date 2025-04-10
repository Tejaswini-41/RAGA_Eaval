# 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

"""Run this model in Python

> pip install openai
"""
import os
from openai import OpenAI

# To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings. 
# Create your PAT token by following instructions here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key="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",
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "",
        },
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
    model="gpt-4o",
    temperature=1,
    max_tokens=4096,
    top_p=1
)

print(response.choices[0].message.content)
