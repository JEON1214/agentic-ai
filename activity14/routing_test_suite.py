from project_react_loop import react_loop

ROUTING_TESTS = [
    ("What is ReAct?", "search_documents"),
    ("What is the travel budget?", "search_documents"),
    ("What did we learn about chunking?", "search_documents"),
    ("Calculate 45 * 12", "calculate"),
    ("What is 15% of 3000?", "calculate"),
    ("What is 2 to the power of 10?", "calculate"),
    ("Help me with that thing", "clarify"),
    ("Do the stuff I asked", "clarify"),
    ("Hello!", None),
    ("Thank you!", None),
]


def test_routing_accuracy() -> None:
    correct = 0
    for query, expected in ROUTING_TESTS:
        transcript = react_loop(query)
        actual_tool = None
        for entry in transcript:
            if entry.get("phase") == "ACTION":
                actual_tool = entry.get("tool")
                break
        match = actual_tool == expected
        correct += 1 if match else 0
        status = "✓" if match else "✗"
        print(f"{status} expected={str(expected):20s} got={str(actual_tool):20s} | {query}")

    total = len(ROUTING_TESTS)
    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.0f}%)")


if __name__ == "__main__":
    test_routing_accuracy()
