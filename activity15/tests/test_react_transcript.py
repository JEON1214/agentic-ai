import asyncio
import unittest

from activity15.backend.main import chat, store


class ReActTranscriptTests(unittest.TestCase):
    def setUp(self) -> None:
        store.db.clear()
        store.add(
            "demo",
            [
                "Qdrant is a vector database for long-term agent memory.",
                "ReAct stands for reasoning and acting, with a thought-action-observe-answer loop.",
            ],
            [{"user_id": "u1", "source": "demo.txt", "collection": "demo"}] * 2,
        )

    def test_ambiguity_clarification_returns_transcript(self) -> None:
        response = asyncio.run(
            chat({"query": "Help me with that thing", "collection": "demo", "user_id": "u1"})
        )

        self.assertIn("transcript", response)
        self.assertEqual(response["answer"],
                         "I need a bit more detail to answer that. Can you rephrase the question with the specific topic or object?")
        self.assertIsNone(response["triad"])
        phases = [entry.get("phase") for entry in response["transcript"]]
        self.assertEqual(phases, ["USER", "ACTION", "OBSERVE", "ANSWER"])


if __name__ == "__main__":
    unittest.main()
