import os
import re
from dotenv import load_dotenv
from google import genai

# =====================================================
# SETUP
# =====================================================

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# =====================================================
# DATABASE
# =====================================================

HOTEL_DATABASE = {
    "tokyo": [
        {"name": "Shibuya Grand", "price_per_night": 180},
        {"name": "Imperial Palace Stay", "price_per_night": 450},
        {"name": "Capsule Capsule", "price_per_night": 45}
    ],
    "paris": [
        {"name": "Hotel de L'Opera", "price_per_night": 220},
        {"name": "Ritz Paris", "price_per_night": 950},
        {"name": "Montmartre Hostel", "price_per_night": 70}
    ]
}

BUDGET = 200.0

# =====================================================
# SYSTEM PROMPT (STRICT TOOL ENFORCEMENT)
# =====================================================

SYSTEM_PROMPT = """
You are SkyLuxe Agent, a friendly high-end travel booking assistant.
When the user asks about hotel search or booking, respond with only a TOOL command.
When the user asks anything else, answer directly as a helpful concierge.
Use the tool formats exactly like:
TOOL: search_hotels(paris)
TOOL: book_hotel(Montmartre Hostel)
Do not include any additional text around the TOOL command.
"""

# =====================================================
# SAFETY GUARD
# =====================================================

def is_safe(text: str) -> bool:
    blocked = ["free room", "override price", "ignore rules", "bypass validation"]
    return not any(b in text.lower() for b in blocked)

# =====================================================
# TOOLS
# =====================================================

def search_hotels(city: str, budget: float = BUDGET) -> str:
    city = city.lower().strip()
    hotels = HOTEL_DATABASE.get(city)

    if not hotels:
        return f"No hotels found in {city}."

    available = [h for h in hotels if h["price_per_night"] <= budget]

    if not available:
        return f"No hotels within budget (${budget}) in {city.title()}."

    result = [f"Hotels within budget (${budget}) in {city.title()}:"]
    result += [f"- {hotel['name']} (${hotel['price_per_night']})" for hotel in available]

    return "\n".join(result)


def book_hotel(name: str, budget: float = BUDGET) -> str:
    for hotels in HOTEL_DATABASE.values():
        for hotel in hotels:
            if hotel["name"].lower() == name.lower():
                price = hotel["price_per_night"]
                if price > budget:
                    return (
                        f"Booking failed. Price of {hotel['name']} (${price}) "
                        f"exceeds budget (${budget}). Suggest an alternative within budget."
                    )
                return f"Booking confirmed for {hotel['name']} at ${price}/night."

    return f"Booking failed. Hotel '{name}' not found."


def extract_hotel_name(text: str):
    text_lower = text.lower()
    for hotels in HOTEL_DATABASE.values():
        for hotel in hotels:
            name = hotel["name"]
            if name.lower() in text_lower:
                return name
    return None


def is_booking_intent(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in ["book", "reserve", "check in", "check-in", "stay", "room"])


def is_balance_query(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in ["balance", "budget", "remaining"])

# =====================================================
# TOOL PARSER (ROBUST + FINAL)
# =====================================================

def parse_tool(text: str):

    if not text:
        return None

    # TOOL: search_hotels(paris)
    m1 = re.search(r"TOOL:\s*search_hotels\((.*?)\)", text, re.I)
    if m1:
        return ("search_hotels", m1.group(1).strip().strip("'\""))

    # search_hotels(city='paris')
    m2 = re.search(r"search_hotels\(city=['\"](.*?)['\"]\)", text, re.I)
    if m2:
        return ("search_hotels", m2.group(1))

    # TOOL: book_hotel(name)
    m3 = re.search(r"TOOL:\s*book_hotel\((.*?)\)", text, re.I)
    if m3:
        return ("book_hotel", m3.group(1).strip().strip("'\""))

    # book_hotel(hotel_name='x')
    m4 = re.search(r"book_hotel\(hotel_name=['\"](.*?)['\"]\)", text, re.I)
    if m4:
        return ("book_hotel", m4.group(1))

    return None

# =====================================================
# MODEL CALL (SAFE)
# =====================================================

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

    except Exception:
        return ""

# =====================================================
# AGENT LOOP (REACT ENGINE)
# =====================================================

def agent_loop():

    history = []
    destination = "paris"
    remaining_budget = BUDGET

    print("\n✈️ SkyLuxe Agent Ready\n")

    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            print("\nGoodbye.")
            break

        if not user_input.strip():
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye.")
            break

        if not is_safe(user_input):
            print("Blocked unsafe input.")
            continue

        if "tokyo" in user_input.lower():
            destination = "tokyo"
        elif "paris" in user_input.lower():
            destination = "paris"

        history = history[-4:]
        context = f"[CONTEXT: Destination={destination}, Budget=${BUDGET}, Remaining=${remaining_budget}]"
        messages = history + [context, f"User: {user_input}"]

        # =================================================
        # STEP 1: MODEL CALL
        # =================================================

        response = call_model(messages)
        tool = parse_tool(response)

        if not response or not tool:
            hotel_name = extract_hotel_name(user_input)
            if is_booking_intent(user_input) and hotel_name:
                response = f"TOOL: book_hotel({hotel_name})"
            elif is_balance_query(user_input):
                direct_answer = f"Your remaining budget is ${remaining_budget:.2f} per night."
                print("\nSkyLuxe Agent:", direct_answer)
                history.append(f"User: {user_input}")
                history.append(f"Assistant: {direct_answer}")
                continue
            else:
                response = f"TOOL: search_hotels({destination})"

        print("\nSkyLuxe Agent:", response)

        tool = parse_tool(response)

        # =================================================
        # STEP 2: TOOL EXECUTION
        # =================================================

        if tool:
            name, arg = tool
            arg = arg.strip().strip("'\"")

            if name == "search_hotels":
                observation = "OBSERVATION:\n" + search_hotels(arg, remaining_budget)
            elif name == "book_hotel":
                observation = "OBSERVATION:\n" + book_hotel(arg, remaining_budget)
                if observation.startswith("OBSERVATION:\nBooking confirmed"):
                    for hotels in HOTEL_DATABASE.values():
                        for hotel in hotels:
                            if hotel["name"].lower() == arg.lower():
                                remaining_budget -= hotel["price_per_night"]
                                remaining_budget = max(0.0, remaining_budget)
                                break
                        else:
                            continue
                        break
            else:
                observation = "OBSERVATION: Unknown tool"

            followup = history + [context, f"User: {user_input}", f"Assistant: {response}", observation]
            final = call_model(followup)
            if not final or "TOOL:" in final:
                final_text = observation
            else:
                final_text = final

            print("\nSkyLuxe Agent:", final_text)
            history.append(f"User: {user_input}")
            history.append(f"Assistant: {final_text}")
        else:
            print("\nSkyLuxe Agent:", response)
            history.append(f"User: {user_input}")
            history.append(f"Assistant: {response}")


if __name__ == "__main__":
    agent_loop()