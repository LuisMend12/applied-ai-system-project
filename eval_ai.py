"""
Test harness / evaluation script for the AI recommendation system.

Runs 5 predefined queries through the full agentic pipeline and checks:
  1. Response contains at least one recommended song
  2. Confidence meets the minimum threshold for the test case
  3. mood_label is a valid value (Hype, Chill, or Mixed)
  4. No hallucinated song titles (all recs exist in the catalog)

Usage:
    python eval_ai.py

Requires ANTHROPIC_API_KEY to be set (in .env or environment).
Prints a pass/fail summary with avg confidence at the end.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from ai_assistant import get_ai_recommendation, get_client
from playlist_logic import DEFAULT_PROFILE

# ---------------------------------------------------------------------------
# Minimal test catalog — defined here to avoid importing Streamlit via app.py
# ---------------------------------------------------------------------------

TEST_CATALOG = [
    {"title": "Lo-fi Rain", "artist": "DJ Calm", "genre": "lofi", "energy": 2, "tags": ["study"]},
    {"title": "Thunderstruck", "artist": "AC/DC", "genre": "rock", "energy": 9, "tags": ["classic", "guitar"]},
    {"title": "Night Drive", "artist": "Neon Echo", "genre": "electronic", "energy": 6, "tags": ["synth"]},
    {"title": "Soft Piano", "artist": "Sleep Sound", "genre": "ambient", "energy": 1, "tags": ["sleep"]},
    {"title": "Bohemian Rhapsody", "artist": "Queen", "genre": "rock", "energy": 8, "tags": ["classic", "opera"]},
    {"title": "Blinding Lights", "artist": "The Weeknd", "genre": "pop", "energy": 8, "tags": ["synth", "dance"]},
    {"title": "Take Five", "artist": "Dave Brubeck", "genre": "jazz", "energy": 4, "tags": ["classic", "instrumental"]},
    {"title": "Strobe", "artist": "Deadmau5", "genre": "electronic", "energy": 7, "tags": ["progressive", "long"]},
    {"title": "Weightless", "artist": "Marconi Union", "genre": "ambient", "energy": 1, "tags": ["relax", "sleep"]},
    {"title": "Smells Like Teen Spirit", "artist": "Nirvana", "genre": "rock", "energy": 9, "tags": ["grunge", "90s"]},
    {"title": "Levitating", "artist": "Dua Lipa", "genre": "pop", "energy": 8, "tags": ["dance", "party"]},
    {"title": "So What", "artist": "Miles Davis", "genre": "jazz", "energy": 3, "tags": ["trumpet", "cool"]},
    {"title": "Midnight City", "artist": "M83", "genre": "electronic", "energy": 7, "tags": ["indie", "dream"]},
    {"title": "Gymnopedie No.1", "artist": "Erik Satie", "genre": "ambient", "energy": 1, "tags": ["piano", "calm"]},
    {"title": "Sandstorm", "artist": "Darude", "genre": "electronic", "energy": 10, "tags": ["trance", "meme"]},
    {"title": "Uptown Funk", "artist": "Mark Ronson ft. Bruno Mars", "genre": "pop", "energy": 9, "tags": ["funk", "dance"]},
    {"title": "Feeling Good", "artist": "Nina Simone", "genre": "jazz", "energy": 6, "tags": ["soul", "vocal"]},
    {"title": "Fly Me to the Moon", "artist": "Frank Sinatra", "genre": "jazz", "energy": 5, "tags": ["vocal", "swing"]},
]

VALID_MOODS = {"Hype", "Chill", "Mixed"}

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "name": "Study session",
        "query": "studying late at night, need calm focus music",
        "min_confidence": 0.6,
        "expected_mood": "Chill",
    },
    {
        "name": "Gym workout",
        "query": "intense gym workout, pump up high energy music",
        "min_confidence": 0.65,
        "expected_mood": "Hype",
    },
    {
        "name": "Sleep / wind down",
        "query": "I want to sleep, something very calm and quiet",
        "min_confidence": 0.6,
        "expected_mood": "Chill",
    },
    {
        "name": "Night drive",
        "query": "driving alone at night, synthy atmospheric electronic",
        "min_confidence": 0.55,
        "expected_mood": None,  # any valid mood is acceptable
    },
    {
        "name": "Jazz background",
        "query": "jazz instrumental background music for a coffee shop",
        "min_confidence": 0.65,
        "expected_mood": "Chill",
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_checks(case: dict, result: dict, catalog_titles: set) -> list:
    """Return list of (label, passed) tuples for this test case."""
    checks = []
    recs = result.get("recommended_songs", [])
    conf = result.get("confidence", 0.0)
    mood = result.get("mood_label", "")

    checks.append(("Has at least one recommendation", len(recs) >= 1))
    checks.append((f"Confidence >= {case['min_confidence']:.0%}", conf >= case["min_confidence"]))
    checks.append(("Valid mood label", mood in VALID_MOODS))
    no_hallucinations = all(r.lower() in catalog_titles for r in recs)
    checks.append(("No hallucinated song titles", no_hallucinations))
    if case["expected_mood"]:
        checks.append((f"Mood is '{case['expected_mood']}'", mood == case["expected_mood"]))

    return checks


def main():
    client = get_client()
    if client is None:
        print("ERROR: ANTHROPIC_API_KEY is not set. Add it to .env or your environment.")
        sys.exit(1)

    catalog_titles = {s["title"].lower() for s in TEST_CATALOG}
    passed_total = 0
    run_total = 0
    confidences = []

    print("=" * 65)
    print("  Playlist Chaos AI — Evaluation Harness")
    print(f"  Catalog: {len(TEST_CATALOG)} songs  |  Test cases: {len(TEST_CASES)}")
    print("=" * 65)

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\nTest {i}: {case['name']}")
        print(f'  Query: "{case["query"]}"')

        result = get_ai_recommendation(
            case["query"], TEST_CATALOG, DEFAULT_PROFILE, client
        )

        if result.get("error"):
            print(f"  FAIL — system error: {result['error']}")
            continue

        conf = result.get("confidence", 0.0)
        mood = result.get("mood_label", "?")
        recs = result.get("recommended_songs", [])
        kb = result.get("knowledge_used", [])
        confidences.append(conf)

        print(f"  mood={mood}  confidence={conf:.0%}  recs={recs}")
        if kb:
            print(f"  knowledge base used: {kb}")

        checks = run_checks(case, result, catalog_titles)
        case_passed = all(ok for _, ok in checks)
        if case_passed:
            passed_total += 1
        run_total += 1

        for label, ok in checks:
            symbol = "PASS" if ok else "FAIL"
            print(f"    [{symbol}] {label}")

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    print("\n" + "=" * 65)
    print(f"  Results : {passed_total}/{run_total} test cases fully passed")
    print(f"  Avg confidence : {avg_conf:.0%}")
    print("=" * 65)
    sys.exit(0 if passed_total == run_total else 1)


if __name__ == "__main__":
    main()
