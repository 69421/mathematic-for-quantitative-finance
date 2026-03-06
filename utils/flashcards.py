import json
import os

FLASHCARDS_FILE = "flashcards.json"


def save_flashcards(cards: list[dict]) -> None:
    """Save flashcards to a JSON file."""
    existing = load_all_flashcards()
    existing.extend(cards)
    with open(FLASHCARDS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def load_all_flashcards() -> list[dict]:
    """Load all saved flashcards."""
    if os.path.exists(FLASHCARDS_FILE):
        with open(FLASHCARDS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def delete_flashcards_by_topic(topic: str) -> None:
    """Delete all flashcards for a given topic."""
    cards = load_all_flashcards()
    cards = [c for c in cards if c.get("topic") != topic]
    with open(FLASHCARDS_FILE, "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)


def get_topics() -> list[str]:
    """Get all unique topics from saved flashcards."""
    cards = load_all_flashcards()
    return list({c.get("topic", "Général") for c in cards})
