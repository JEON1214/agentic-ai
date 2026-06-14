import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

def run_agent_client():
    # 1. Load environment variables
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file!")

    # 2. Initialize client
    client = genai.Client(api_key=api_key)

    # 3. Initialize the model identity
    identity = """
You are a car mechanic and you know everything about cars, engines, and automotive repair.

    CONSTRAINTS:
    1. Always answer in a concise manner.
    2. Always answer in a helpful manner.
    3. Always answer in an absolute manner.
"""

    # Test identity (returns the generated text)
    response = client.models.generate_content(
        model='gemini-3.1-flash-lite',
        contents = "What is the top of the line car during 2015?",
        config=types.GenerateContentConfig(
            system_instruction=identity
        )
    )
    print(f"Agent Response: {response.text}")
    return response.text


if __name__ == "__main__":
    run_agent_client()
