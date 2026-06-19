import json
import os
import re
from enum import Enum
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv, find_dotenv
from google import genai

load_dotenv(find_dotenv())
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Make sure .env is present and contains a valid key.")
client = genai.Client(api_key=api_key)

class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class Symptom(BaseModel):
    symptom_name: str
    severity: Severity
    duration_days: int = Field(ge=0)

class MedicalIntake(BaseModel):
    symptoms: list[Symptom]
    allergies: list[str]
    urgency_rating: int = Field(ge=1, le=10)
    clinical_reasoning: str

SYSTEM_PROMPT = """
You are a clinical intake assistant. Parse the patient description into a structured JSON medical intake record.
Use only the following schema:
{
  "symptoms": [
    {
      "symptom_name": "string",
      "severity": "LOW|MEDIUM|HIGH",
      "duration_days": 0
    }
  ],
  "allergies": ["string"],
  "urgency_rating": 1,
  "clinical_reasoning": "string"
}
Severity must be one of LOW, MEDIUM, HIGH. urgency_rating must be an integer between 1 and 10.
Output only valid JSON with no markdown or extra commentary.
"""

MAX_RETRIES = 3


def extract_json(text: str):
    trimmed = text.strip()
    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", trimmed, re.S)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def call_model(messages):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config={"system_instruction": SYSTEM_PROMPT},
            contents=messages
        )
        if response and response.text:
            return response.text.strip()
        return ""
    except Exception as error:
        error_text = str(error)
        if "RESOURCE_EXHAUSTED" in error_text or "quota exceeded" in error_text.lower():
            raise RuntimeError(
                "Gemini quota exhausted. Please check your API plan or billing and retry after the suggested delay. "
                f"Error details: {error_text}"
            ) from error
        raise RuntimeError(f"Model call failed: {error_text}") from error


def process_intake(patient_input: str) -> MedicalIntake:
    messages = [patient_input]
    last_feedback = None

    for attempt in range(1, MAX_RETRIES + 1):
        prompt_messages = messages.copy()
        if last_feedback:
            prompt_messages.append(last_feedback)

        raw_output = call_model(prompt_messages)
        if not raw_output:
            raise RuntimeError("Model returned no response.")

        print(f"\nAttempt {attempt}: model output:\n{raw_output}\n")

        parsed = extract_json(raw_output)
        if parsed is None:
            last_feedback = (
                "The previous response was not valid JSON. "
                "Please return only valid JSON matching the schema."
            )
            print("Validation issue: model output could not be parsed as JSON.")
            continue

        try:
            record = MedicalIntake.model_validate(parsed)
            return record
        except ValidationError as error:
            last_feedback = (
                "The JSON did not validate against the medical intake schema. "
                f"Validation errors:\n{error}\n"
                "Please correct the JSON and return only valid JSON."
            )
            print(f"ValidationError on attempt {attempt}:\n{error}\n")
            continue

    raise RuntimeError("Failed to produce a valid MedicalIntake record after 3 attempts.")


if __name__ == "__main__":
    test_input = (
        "My stomach is cramping incredibly badly since last night! "
        "The pain is unbearable, definitely an urgency of 15 out of 10! "
        "I don't think I have allergies."
    )

    try:
        record = process_intake(test_input)
        print("\n--- Validated Intake Record ---")
        print(record.model_dump_json(indent=2))
    except Exception as e:
        print(f"Failed: {e}")