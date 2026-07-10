import unittest

from activity14.project_react_loop import react_loop


class ReactLoopConversationTests(unittest.TestCase):
    def test_ambiguous_request_asks_for_clarification(self) -> None:
        transcript = react_loop("Help me with that thing")
        answers = [entry.get("content", "") for entry in transcript if entry.get("phase") == "ANSWER"]

        self.assertTrue(answers)
        self.assertTrue(any("clarify" in answer.lower() or "clarification" in answer.lower() for answer in answers))


if __name__ == "__main__":
    unittest.main()
