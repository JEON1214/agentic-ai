1. The Stateless Loop (2 pts)
Error:
A new chat session is created on every loop iteration, causing the conversation history to reset each time. The agent forgets previous messages.

Fix:

chat = client.chats.create(model="gemini-3.1-flash-lite")

while True:
    user_input = input("> ")
    response = chat.send_message(user_input)
    print(response.text)

2. The Leaky Identity (2 pts)
Error:
The system instruction does not explicitly prevent the model from giving the final answer, so it may reveal the answer instead of guiding the user.

Fix (System Instruction):
identity = "You are a math tutor. Never provide the final answer directly. Only provide hints, guidance, and reasoning steps."

3. The Memory Bloat (2 pts)
Error:
Using chat.history[0] keeps only one message instead of preserving the most recent conversation context.

Fix (Line B):

chat.history = chat.history[-2:]

4. The Perception Crash (2 pts)
Error:
The price field is required, so validation fails if the model does not provide a value.

Fix (Pydantic Model):

from typing import Optional
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: Optional[float] = None

5. The Infinite Backoff (2 pts)
Error:
The code retries indefinitely for every exception, including unrecoverable errors such as invalid API keys.

Fix (Else Block):

else:
    raise e

Part II: Schema Design & Evaluation (10 Points)
Task 1: The Multi-Agent Router (5 Points)
Pydantic Schema

from pydantic import BaseModel, Field
from enum import Enum

class Department(Enum):
    PAYROLL = "PAYROLL"
    RECRUITING = "RECRUITING"
    LEAVE_REQUEST = "LEAVE_REQUEST"

class HRRouter(BaseModel):
    department: Department
    reasoning: str = Field(
        description="Reasoning used to determine the department"
    )
    urgency_level: int = Field(
        ge=1,
        le=5,
        description="Urgency level from 1 (low) to 5 (high)"
    )
Task 2: Architecture Evaluation (5 Points)
A multi-agent architecture improves efficiency by routing each request to the correct department. The Pydantic schema validates the data, ensuring that all required fields are present and correctly formatted. The reasoning field explains why the request was assigned to a specific department, while the urgency_level helps prioritize tasks. This design makes the system more accurate, scalable, easier to maintain, and simpler to debug compared to using a single agent for all requests.













