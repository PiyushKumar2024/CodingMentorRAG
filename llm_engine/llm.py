import requests
import os

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

def generate_answer(code, context):

    prompt = f"""
    You are an expert programming mentor.

    Your job is to analyze code carefully and ONLY report real issues.

    STRICT RULES:
    - Do NOT assume errors.
    - Do NOT hallucinate.
    - If the code is correct, clearly say: "No issues found."
    - Only point out errors that actually exist.
    - Be precise and short.

    RESPONSE FORMAT:

    Errors:
    - (List real errors, or write "None")

    Explanation:
    - (Explain briefly why it's an error or why code is correct)

    Fix:
    - (Give corrected code ONLY if there is an error, otherwise write "No fix needed")

    Code:
    {code}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model":"mistral",
            "prompt":prompt,
            "stream":False
        }
    )

    print("STATUS:", response.status_code)
    print("TEXT:", response.text)

    return response.json().get("response", "No response")