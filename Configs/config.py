# config.py
import os
from dotenv import load_dotenv
import together

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
MODEL_NAME = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

def get_together_client():
    if not TOGETHER_API_KEY:
        raise ValueError("API key not found")
    together.api_key = TOGETHER_API_KEY
    return together

def test_connection():
    try:
        client = get_together_client()
        response = together.Complete.create(
            prompt="Say 'Hello'",
            model=MODEL_NAME,
            max_tokens=5
        )
        return "Connected"
    except Exception as e:
        return f" Failed: {e}"

print(test_connection())