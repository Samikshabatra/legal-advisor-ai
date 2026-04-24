import os
import requests

API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}

response = requests.post(API_URL, headers=headers, json={
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "model": "meta-llama/Meta-Llama-3-8B-Instruct:novita"
})

print(response.json()["choices"][0]["message"])