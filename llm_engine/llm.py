import requests
import os

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

def generate_answer(code, context):

    prompt = f"""
    You are a strict compiler-like coding assistant.

    Rules:
    - Only report REAL errors
    - If no errors, say exactly:
    Errors: None
    Explanation: Code is correct
    Fix: No fix needed

    Code:
    {code}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model":"qwen2.5-coder",
            "prompt":prompt,
            "stream":False
        }
    )

    print("STATUS:", response.status_code)
    print("TEXT:", response.text)

    return response.json().get("response", "No response")