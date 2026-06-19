# Activity 4 Reflection

## 1. Did the agent enforce the budget and tool rules correctly?
Yes. The SkyLuxe agent uses local Python guardrails to intercept hotel booking requests and reject any hotel above the $200 per night budget. It also blocks unsafe prompt attempts like "free room" or "override price" before calling the model.

## 2. How does the agent keep state after pruning history?
The agent keeps only the last 4 messages in chat history and prepends an explicit context string each turn: `[CONTEXT: Destination=..., Budget=$200, Remaining=$...]`. This preserves destination and budget state even when the sliding window drops older turns.

## 3. What challenge did the simulated tool loop solve?
It separates reasoning from execution. The model decides which tool to call, the script runs that tool against the local hotel database, and the tool output is returned as an observation so the model can produce a final user-facing response.

---

## Notes
- The agent supports both search and booking tool semantics: `TOOL: search_hotels(city)` and `TOOL: book_hotel(hotel_name)`.
- Successful bookings deduct from the remaining budget, and balance queries report the current session budget.
- If the model fails to output a valid tool command, fallback logic handles booking or balance intents locally.
