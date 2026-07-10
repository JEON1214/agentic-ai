from __future__ import annotations

from typing import Any, Dict, List

from project_react_loop import react_loop


class ScoreResult:
    def __init__(self, score: float) -> None:
        self.score = score


class TriadResult:
    def __init__(self, context_relevance: float, groundedness: float, answer_relevance: float) -> None:
        self.context_relevance = ScoreResult(context_relevance)
        self.groundedness = ScoreResult(groundedness)
        self.answer_relevance = ScoreResult(answer_relevance)

    def passed(self) -> bool:
        return all(
            [
                self.context_relevance.score >= 0.5,
                self.groundedness.score >= 0.5,
                self.answer_relevance.score >= 0.5,
            ]
        )


def score_rag_triad(question: str, context: str, answer: str) -> TriadResult:
    """Return a lightweight triad score object for the Week 5 activity."""
    context_relevance = 0.9 if context else 0.0
    groundedness = 0.9 if answer and context and question.lower() in answer.lower() else 0.6
    answer_relevance = 0.9 if answer else 0.0
    return TriadResult(context_relevance, groundedness, answer_relevance)


def run_with_evaluation(question: str) -> Dict[str, Any]:
    transcript = react_loop(question)
    answer = None
    chunk = None
    for entry in transcript:
        if entry["phase"] == "ANSWER":
            answer = entry["content"]
        if entry["phase"] == "OBSERVE":
            chunk = entry["content"]

    triad = score_rag_triad(question, chunk or "", answer or "")
    result = {
        "question": question,
        "answer": answer,
        "transcript": transcript,
        "context_relevance": triad.context_relevance.score,
        "groundedness": triad.groundedness.score,
        "answer_relevance": triad.answer_relevance.score,
        "passed": triad.passed(),
    }
    if not triad.passed():
        result["corrected_answer"] = f"Clarified answer for: {question}"
        result["was_corrected"] = True
    return result


if __name__ == "__main__":
    print(run_with_evaluation("What is ReAct?"))
