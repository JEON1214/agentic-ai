# Routing Report Week 5

## Section 1 — Routing Results Table

| Query | Expected Tool | Actual Tool | Correct? |
|-------|---------------|-------------|----------|
| What is ReAct? | search_documents | search_documents | ✓ |
| What is the travel budget? | search_documents | search_documents | ✓ |
| What did we learn about chunking? | search_documents | search_documents | ✓ |
| Calculate 45 * 12 | calculate | calculate | ✓ |
| What is 15% of 3000? | calculate | calculate | ✓ |
| What is 2 to the power of 10? | calculate | calculate | ✓ |
| Help me with that thing | clarify | clarify | ✓ |
| Do the stuff I asked | clarify | clarify | ✓ |
| Hello! | None | None | ✓ |
| Thank you! | None | None | ✓ |

## Section 2 — Failure Analysis

The initial version had two routing issues during testing:

1. "What is 2 to the power of 10?" was routed to `search_documents` instead of `calculate`.
   - Cause: the fallback logic treated the phrase as a general factual question before the arithmetic pattern was recognized.
2. "Help me with that thing" was not routed to `clarify`.
   - Cause: the ambiguous phrase was being mistaken for a greeting because the matching logic used broad substring checks.

These issues were resolved by adding explicit math-pattern handling and word-boundary matching for clarification phrases.

## Section 3 — Reflection

1. The hardest query type to route was ambiguous prompts such as "Help me with that thing" because they can be interpreted as either a knowledge lookup or a clarification request.
2. The final tool descriptions were made more explicit by adding concrete use cases such as "Use this for any numeric computation" and "Use this as a fallback when no other tool fits". The routing logic was also adjusted to handle arithmetic expressions and ambiguous prompts more reliably.
3. A fourth tool such as `get_current_date` would be described as: "Return the current date or time for time-sensitive questions. Use this when the user asks for today's date, the current time, or a date-based calculation."
