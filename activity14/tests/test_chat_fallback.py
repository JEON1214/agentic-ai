import asyncio
import unittest

from activity14.backend.main import chat, store


class ChatFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        store.db.clear()

    def test_returns_answer_when_query_matches_uploaded_transcript(self) -> None:
        store.add(
            "demo",
            ["Java is a programming language used to build software applications."],
            [{"user_id": "u1", "source": "demo.txt", "collection": "demo"}],
        )

        response = asyncio.run(
            chat({"query": "What is Java?", "collection": "demo", "user_id": "u1"})
        )

        self.assertNotEqual(
            response["answer"],
            "I don't know — this topic is not covered in the uploaded transcript.",
        )
        self.assertTrue(response["sources"])

    def test_returns_fallback_when_query_has_no_relevant_content(self) -> None:
        store.add(
            "demo",
            ["Java is a programming language used to build software applications."],
            [{"user_id": "u1", "source": "demo.txt", "collection": "demo"}],
        )

        response = asyncio.run(
            chat({"query": "What is quantum physics?", "collection": "demo", "user_id": "u1"})
        )

        self.assertEqual(
            response["answer"],
            "I don't know — this topic is not covered in the uploaded transcript.",
        )

    def test_prefers_the_most_relevant_sentence_for_topic_questions(self) -> None:
        store.add(
            "demo",
            ["Java is a programming language used to build software applications. Primitive types include int, char, long, double, and boolean."],
            [{"user_id": "u1", "source": "demo.txt", "collection": "demo"}],
        )

        response = asyncio.run(
            chat({"query": "What are primitive types in Java?", "collection": "demo", "user_id": "u1"})
        )

        self.assertIn("Primitive types include", response["answer"])

    def test_answers_broad_questions_when_the_transcript_is_relevant(self) -> None:
        store.add(
            "demo",
            ["Java is a programming language used to build software applications."],
            [{"user_id": "u1", "source": "demo.txt", "collection": "demo"}],
        )

        response = asyncio.run(
            chat({"query": "What is this transcript about?", "collection": "demo", "user_id": "u1"})
        )

        self.assertNotEqual(
            response["answer"],
            "I don't know — this topic is not covered in the uploaded transcript.",
        )


if __name__ == "__main__":
    unittest.main()
