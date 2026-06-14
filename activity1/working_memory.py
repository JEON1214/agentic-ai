import os
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

# Initialize client
client = genai.Client(api_key=api_key)

identity = """
You are a car mechanic and you know everything about cars, engines, and automotive repair.

    CONSTRAINTS:
    1. Always answer in a concise manner.
    2. Always answer in a helpful manner.
    3. Always answer in an absolute manner.
"""

# Create a persistent chat session
chat = client.chats.create(
    model="gemini-2.5-flash"
)

def agent_loop(user_input):
    prompt = f"{identity}\n\nUser: {user_input}"
    response = chat.send_message(prompt)
    return response.text

def main():
    print("\n--- Agent is active. Type 'exit' to quit. ---")

    while True:
        try:
            user_msg = input("\nUser: ")

            if user_msg.lower() == "exit":
                print("Defeat")
                break

            if not user_msg.strip():
                continue

            response = agent_loop(user_msg)
            print(f"Agent: {response}")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break


if __name__ == "__main__":
    main()
