"""
Music knowledge base — second retrieval source for multi-source RAG.
The RAG retriever queries both the live song catalog AND these expert
descriptions before prompting Claude, giving the model richer context.
"""

from typing import Dict, List

KNOWLEDGE_BASE: List[Dict[str, object]] = [
    {
        "id": "studying_focus",
        "topics": ["study", "focus", "concentrate", "homework", "work", "read", "coding"],
        "text": (
            "For studying and deep focus, lo-fi, ambient, and classical music are ideal. "
            "These genres feature steady non-distracting rhythms, minimal or no lyrics, "
            "and low energy levels (1-4/10). The consistency helps maintain concentration "
            "without causing mental fatigue. Jazz instrumentals also work well."
        ),
    },
    {
        "id": "workout_exercise",
        "topics": ["workout", "gym", "exercise", "run", "running", "training", "pump", "fitness", "intense"],
        "text": (
            "Workout and exercise music needs high energy (7-10/10), fast tempo, and driving "
            "rhythms. Rock, electronic/EDM, and trance dominate gym playlists. Tracks with "
            "intensity and momentum help maintain physical effort and push through fatigue "
            "during intense exercise."
        ),
    },
    {
        "id": "sleep_rest",
        "topics": ["sleep", "rest", "wind down", "bedtime", "insomnia", "nap", "sleeping"],
        "text": (
            "Sleep and wind-down music should have very low energy (1-2/10), slow tempo, "
            "and minimal sudden changes. Ambient soundscapes, gentle piano, and soft "
            "instrumental pieces are most effective. Avoid anything with lyrics or "
            "sudden dynamic shifts."
        ),
    },
    {
        "id": "party_social",
        "topics": ["party", "social", "dance", "celebration", "night out", "club", "friends", "dancing"],
        "text": (
            "Party and social music thrives on high energy (7-10/10), strong beats, and "
            "familiar hooks. Pop, electronic dance music, and funk keep crowds engaged. "
            "Songs with recognizable choruses and dance-friendly rhythms work best for "
            "social settings."
        ),
    },
    {
        "id": "night_drive",
        "topics": ["drive", "driving", "road", "night", "car", "commute", "cruise"],
        "text": (
            "Driving music benefits from medium-to-high energy (5-8/10), steady momentum, "
            "and atmospheric soundscapes. Electronic, synth-wave, and rock with steady "
            "rhythms keep drivers alert. Night drives pair well with moody, introspective "
            "electronic or ambient rock."
        ),
    },
    {
        "id": "sad_emotional",
        "topics": ["sad", "cry", "melancholy", "heartbreak", "emotional", "moody", "down", "blue"],
        "text": (
            "For emotional or melancholic moods, low-energy ambient, slow jazz, and "
            "minimalist piano create space for reflection. Energy levels 1-4/10 and "
            "simple harmonic structures resonate most deeply during introspective moments."
        ),
    },
    {
        "id": "morning_wakeup",
        "topics": ["morning", "wake up", "breakfast", "coffee", "start day", "energize", "sunrise"],
        "text": (
            "Morning music should gently increase energy — starting calm (3-5/10) and "
            "building upward. Light jazz, upbeat pop, and positive acoustic tracks ease "
            "the transition from sleep to activity without being jarring."
        ),
    },
    {
        "id": "chill_relax",
        "topics": ["chill", "relax", "unwind", "cozy", "comfortable", "lazy", "mellow", "lounge"],
        "text": (
            "Chill and relaxation music sits in the 2-5/10 energy range. Lo-fi, ambient, "
            "and cool jazz are staples. The goal is a comfortable background presence — "
            "engaging enough to notice, unobtrusive enough to ignore when needed."
        ),
    },
]


def retrieve_knowledge_context(query: str, top_k: int = 2) -> List[Dict]:
    """
    Search the knowledge base for entries relevant to the user's query.
    Returns up to top_k entries ranked by topic-keyword overlap with the query.
    """
    query_lower = query.lower()
    scored = []
    for entry in KNOWLEDGE_BASE:
        score = sum(1 for topic in entry["topics"] if topic in query_lower)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]
