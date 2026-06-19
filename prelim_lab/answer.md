1. The Stateless Loop (2 pts)
Error:A new chat session is created on every loop iteration, so conversation history is reset and the agent forgets previous messages.
chat = client.chats.create(model="gemini-3.1-flash-lite")

while True:
    user_input = input("> ")
    response = chat.send_message(user_input)
    print(response.text)


2. The Leaky Identity (2 pts)
Error:The system instruction does not explicitly prohibit giving the answer, so the model may prioritize helpfulness and reveal it.
Answer:
identity = "You are a math tutor. Never provide the final answer directly. Only provide hints, guidance, and reasoning steps."


3. The Memory Bloat (2 pts)
Error: chat.history[0] keeps only a single message instead of the last two messages.
[chat.history = chat.history[-2:]]

Fix (Line B):
[Answer here]

4. The Perception Crash (2 pts)
Error: The schema requires price, so validation fails if the model omits that field.
from typing import Optional

class Item(BaseModel):
    name: str
    price: Optional[float] = None]



5. The Infinite Backoff (2 pts)
Error:The code retries every exception forever, including invalid API keys and other unrecoverable errors.


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
[Answer here]













