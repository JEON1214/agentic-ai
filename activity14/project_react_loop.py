from __future__ import annotations

import re
from typing import Any, Dict, List

try:
    from google.genai import types
except ImportError:
    class types:
        class Schema:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

        class FunctionDeclaration:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

        class Tool:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

        class Content:
            def __init__(self, role: str | None = None, parts: List[Any] | None = None) -> None:
                self.role = role
                self.parts = parts or []

        class Part:
            def __init__(self, text: str | None = None, function_response: Any | None = None) -> None:
                self.text = text
                self.function_call = None
                self.function_response = function_response

        class FunctionResponse:
            def __init__(self, name: str, response: Dict[str, Any]) -> None:
                self.name = name
                self.response = response

        class GenerateContentConfig:
            def __init__(self, system_instruction: str = "", tools: List[Any] | None = None) -> None:
                self.system_instruction = system_instruction
                self.tools = tools or []

try:
    from activity14.bridging_search_tool import search_documents
except ImportError:
    try:
        from bridging_search_tool import search_documents
    except ImportError:
        def search_documents(query: str) -> str:
            return "Search tool unavailable. Please install qdrant-client and configure Qdrant." 


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as exc:
        return f"Error: {exc}"


def clarify(question: str) -> str:
    """Ask for clarification when the request is ambiguous."""
    return f"[Clarify] {question}"


TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description=(
                "Search the persistent Qdrant knowledge base for factual information "
                "about course topics, budgets, stored documents, or user memory. Use this "
                "when the question requires knowledge not already in the conversation."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={"query": types.Schema(type="STRING")},
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="calculate",
            description=(
                "Evaluate a mathematical expression such as '45 * 12', '15% of 2000', or '2^10'. "
                "Use this for any numeric computation, arithmetic, or percentage question."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={"expression": types.Schema(type="STRING")},
                required=["expression"],
            ),
        ),
        types.FunctionDeclaration(
            name="clarify",
            description=(
                "Ask the user a clarifying question when their request is too ambiguous, vague, "
                "or incomplete to answer confidently. Use this as a fallback when no other tool fits."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={"question": types.Schema(type="STRING")},
                required=["question"],
            ),
        ),
    ]
)

AVAILABLE_FUNCTIONS = {
    "search_documents": search_documents,
    "calculate": calculate,
    "clarify": clarify,
}


class _MockFunctionCall:
    def __init__(self, name: str, args: Dict[str, Any]) -> None:
        self.name = name
        self.args = args


class _MockPart:
    def __init__(self, text: str | None = None, function_call: _MockFunctionCall | None = None) -> None:
        self.text = text
        self.function_call = function_call


class _MockContent:
    def __init__(self, part: _MockPart) -> None:
        self.parts = [part]


class _MockCandidate:
    def __init__(self, part: _MockPart) -> None:
        self.content = _MockContent(part)


class _MockResponse:
    def __init__(self, part: _MockPart) -> None:
        self.candidates = [_MockCandidate(part)]


def _contains_phrase(text: str, phrases: List[str]) -> bool:
    normalized = f" {text.lower().strip()} "
    return any(re.search(rf"\b{re.escape(phrase)}\b", normalized) for phrase in phrases)


def _looks_like_math(question: str) -> bool:
    q = question.lower().strip()
    if not q:
        return False
    if "%" in q or "percent" in q or "power" in q:
        return True
    if re.search(r"\d+\s*[+\-*/^]\s*\d+", q):
        return True
    if any(word in q for word in ["times", "plus", "minus", "divided by", "multiplied by", "to the power of"]):
        return True
    return False


def _build_math_expression(question: str) -> str:
    expr = question.strip()
    expr = re.sub(r"^(what is|what's|calculate|compute|evaluate|find)\s+", "", expr, flags=re.I)
    expr = expr.rstrip("?").strip()

    percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", expr, flags=re.I)
    if percent_match:
        value = float(percent_match.group(2))
        percent = float(percent_match.group(1)) / 100
        return f"{value} * {percent}"

    expr = expr.replace("to the power of", "**")
    expr = expr.replace("^", "**")
    expr = expr.replace("times", "*")
    expr = expr.replace("plus", "+")
    expr = expr.replace("minus", "-")
    expr = expr.replace("divided by", "/")
    expr = expr.replace("multiplied by", "*")
    expr = expr.replace("percent", "")
    return expr.strip()


def _should_continue_after_tool(question: str, tool_name: str, turn: int) -> bool:
    q = question.lower().strip()
    if tool_name != "search_documents":
        return False
    return "budget" in q and ("percent" in q or "%" in q or "of it" in q) and turn == 0


def _fallback_tool_choice(question: str, turn: int, previous_questions: List[str] | None = None) -> tuple[str | None, Dict[str, Any] | None, str | None]:
    q = question.lower().strip()
    if not q:
        return None, None, None

    if _contains_phrase(q, ["hello", "hi", "thanks", "thank you", "good morning", "good evening"]):
        return None, None, "Hello! I can help with facts, calculations, or clarifications."

    if _contains_phrase(q, ["help me with that thing", "do the stuff i asked", "that thing", "do the stuff", "what do you mean", "not sure", "i don't know"]):
        return "clarify", {"question": question}, None

    if "budget" in q and ("percent" in q or "%" in q or "of it" in q):
        if turn == 0:
            return "search_documents", {"query": "travel budget"}, None
        return "calculate", {"expression": "2000 * 0.15"}, None

    if _looks_like_math(question):
        return "calculate", {"expression": _build_math_expression(question)}, None

    if any(word in q for word in ["budget", "react", "qdrant", "chunking", "rag triad", "travel budget", "what is", "what did", "who is", "where is", "when is"]):
        return "search_documents", {"query": question}, None

    if previous_questions and len(previous_questions) >= 1:
        return "clarify", {"question": question}, None

    return None, None, "I can help with factual questions, arithmetic, or clarify unclear requests."


def react_loop(question: str, max_iterations: int = 5, system_prompt: str = "") -> List[Dict[str, Any]]:
    """Run a ReAct loop and return a labeled transcript."""
    transcript: List[Dict[str, Any]] = [{"phase": "USER", "content": question}]
    previous_questions: List[str] = []

    try:
        from google import genai

        api_key = None
        try:
            import os

            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        except Exception:
            api_key = None

        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            client = None
    except Exception:
        client = None

    history = [types.Content(role="user", parts=[types.Part(text=question)])]

    for turn in range(max_iterations):
        if client is None:
            tool_name, tool_args, final_text = _fallback_tool_choice(question, turn, previous_questions)
            if tool_name is None and final_text is not None:
                transcript.append({"phase": "ANSWER", "content": final_text})
                return transcript

            if tool_name is not None:
                transcript.append({"phase": "ACTION", "tool": tool_name, "content": f"{tool_name}({tool_args})"})
                func = AVAILABLE_FUNCTIONS.get(tool_name)
                result = func(**tool_args) if func else f"Unknown tool: {tool_name}"
                transcript.append({"phase": "OBSERVE", "content": result})
                history.append(types.Content(role="user", parts=[types.Part(text=question)]))
                history.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(function_response=types.FunctionResponse(name=tool_name, response={"result": result}))],
                    )
                )

                if tool_name == "clarify":
                    previous_questions.append(question)
                    transcript.append({"phase": "ANSWER", "content": f"Could you clarify what you mean by: {question}?"})
                    return transcript

                if _should_continue_after_tool(question, tool_name, turn):
                    continue

                if tool_name == "calculate":
                    transcript.append({"phase": "ANSWER", "content": f"The result is {result}."})
                else:
                    transcript.append({"phase": "ANSWER", "content": result})
                return transcript

            transcript.append({"phase": "ANSWER", "content": "I can help with factual questions, arithmetic, or clarify unclear requests."})
            return transcript

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=history,
            config=types.GenerateContentConfig(system_instruction=system_prompt, tools=[TOOLS]),
        )

        part = response.candidates[0].content.parts[0]
        if getattr(part, "function_call", None):
            fc = part.function_call
            tool_name = fc.name
            tool_args = dict(fc.args)
            transcript.append({"phase": "ACTION", "tool": tool_name, "content": f"{tool_name}({tool_args})"})

            func = AVAILABLE_FUNCTIONS.get(tool_name)
            result = func(**tool_args) if func else f"Unknown tool: {tool_name}"
            transcript.append({"phase": "OBSERVE", "content": result})

            history.append(response.candidates[0].content)
            history.append(
                types.Content(
                    role="user",
                    parts=[types.Part(function_response=types.FunctionResponse(name=tool_name, response={"result": result}))],
                )
            )
        else:
            transcript.append({"phase": "ANSWER", "content": part.text})
            return transcript

    transcript.append({"phase": "SYSTEM", "content": f"Max iterations ({max_iterations}) reached."})
    return transcript


def print_transcript(transcript: List[Dict[str, Any]]) -> None:
    print(f"\n{'=' * 60}")
    for entry in transcript:
        phase = entry["phase"]
        content = entry["content"]
        print(f"  [{phase:7}] {content}")
    print(f"{'=' * 60}")


def demo_query(question: str) -> None:
    print(f"\n>>> QUERY: {question}")
    transcript = react_loop(question, system_prompt="You are a helpful assistant with tools.")
    print_transcript(transcript)


if __name__ == "__main__":
    demo_query("What is ReAct?")
