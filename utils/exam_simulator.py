import time


def initialize_exam(questions: list[dict], duration_minutes: int) -> dict:
    """Initialize exam state."""
    return {
        "questions": questions,
        "answers": {},
        "start_time": time.time(),
        "duration_seconds": duration_minutes * 60,
        "completed": False,
        "score": None,
    }


def get_remaining_time(exam_state: dict) -> int:
    """Get remaining time in seconds."""
    elapsed = time.time() - exam_state["start_time"]
    remaining = exam_state["duration_seconds"] - elapsed
    return max(0, int(remaining))


def submit_exam(exam_state: dict) -> dict:
    """Calculate and return exam results."""
    questions = exam_state["questions"]
    answers = exam_state["answers"]
    correct = 0
    results = []
    for i, q in enumerate(questions):
        user_answer = answers.get(i)
        is_correct = user_answer == q["correct"]
        if is_correct:
            correct += 1
        results.append({
            "question": q["question"],
            "options": q["options"],
            "user_answer": user_answer,
            "correct_answer": q["correct"],
            "is_correct": is_correct,
            "explanation": q["explanation"],
        })
    score_pct = (correct / len(questions) * 100) if questions else 0
    return {
        "total": len(questions),
        "correct": correct,
        "score_pct": score_pct,
        "results": results,
    }
