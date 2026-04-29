"""AI assistant module: RAG retrieval + Claude-powered music recommendations."""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import anthropic

from playlist_logic import Song

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("ai_interaction.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ai_assistant")

# Mood-to-genre/tag keyword mapping used during RAG scoring
MOOD_KEYWORDS: Dict[str, List[str]] = {
    "study": ["lofi", "ambient", "calm", "instrumental", "focus", "piano"],
    "focus": ["lofi", "instrumental", "ambient", "jazz", "classical"],
    "workout": ["rock", "electronic", "hype", "pump", "intense", "trance"],
    "gym": ["rock", "electronic", "hype", "pump", "intense"],
    "sleep": ["ambient", "calm", "sleep", "relax", "piano", "classical"],
    "relax": ["ambient", "jazz", "chill", "calm", "lofi"],
    "party": ["pop", "dance", "electronic", "funk", "hype"],
    "dance": ["pop", "electronic", "funk", "dance"],
    "sad": ["ambient", "chill", "calm", "piano"],
    "happy": ["pop", "funk", "dance", "upbeat"],
    "drive": ["rock", "electronic", "pop", "synth"],
    "morning": ["pop", "jazz", "calm", "upbeat"],
    "night": ["electronic", "ambient", "synth", "dream"],
    "chill": ["lofi", "ambient", "jazz", "chill"],
    "hype": ["rock", "electronic", "pop", "punk", "dance"],
}


def retrieve_relevant_songs(
    query: str, songs: List[Song], top_k: int = 8
) -> List[Song]:
    """
    RAG retrieval step: score every song in the catalog by relevance to the
    user's free-text query, then return the top_k candidates.
    """
    query_lower = query.lower()
    scored: List[Tuple[int, Song]] = []

    for song in songs:
        score = 0
        genre = str(song.get("genre", "")).lower()
        tags = [str(t).lower() for t in song.get("tags", [])]
        energy = int(song.get("energy", 5))
        title = str(song.get("title", "")).lower()
        artist = str(song.get("artist", "")).lower()

        for mood_word, related in MOOD_KEYWORDS.items():
            if mood_word in query_lower:
                if genre in related:
                    score += 3
                for tag in tags:
                    if tag in related:
                        score += 1

        if genre in query_lower:
            score += 4
        for tag in tags:
            if tag in query_lower:
                score += 2

        if artist in query_lower:
            score += 5
        if title in query_lower:
            score += 5

        if any(w in query_lower for w in ["high energy", "pump", "intense", "loud", "fast"]):
            score += max(0, energy - 5)
        if any(w in query_lower for w in ["calm", "chill", "slow", "quiet", "soft"]):
            score += max(0, 6 - energy)

        scored.append((score, song))

    scored.sort(key=lambda x: x[0], reverse=True)
    retrieved = [s for _, s in scored[:top_k]]
    logger.info(
        "RAG retrieval: query=%r → %d/%d songs (top scores: %s)",
        query,
        len(retrieved),
        len(songs),
        [sc for sc, _ in scored[:3]],
    )
    return retrieved


def _catalog_text(songs: List[Song]) -> str:
    lines = []
    for s in songs:
        tags_str = ", ".join(str(t) for t in s.get("tags", []))
        lines.append(
            f'- "{s["title"]}" by {s["artist"]}'
            f" | genre: {s['genre']} | energy: {s['energy']}/10 | tags: {tags_str}"
        )
    return "\n".join(lines)


def get_ai_recommendation(
    user_query: str,
    songs: List[Song],
    profile: Dict,
    client: anthropic.Anthropic,
) -> Dict:
    """
    Agentic workflow:
      1. Retrieve relevant songs via RAG scoring
      2. Build a context-rich prompt from the retrieved catalog slice
      3. Call Claude to reason over candidates and pick the best matches
      4. Parse and validate the JSON response
      5. Clamp / compute the confidence score

    Returns a dict with keys: recommended_songs, reasoning, mood_label,
    confidence, retrieved_count, query, error (on failure).
    """
    logger.info(
        "AI recommendation request: query=%r profile=%s",
        user_query,
        profile.get("name"),
    )

    # Step 1 — RAG retrieval
    retrieved = retrieve_relevant_songs(user_query, songs, top_k=8)
    if not retrieved:
        logger.warning("Catalog is empty; cannot recommend.")
        return {
            "error": "No songs in catalog to recommend from.",
            "recommended_songs": [],
            "reasoning": "The song catalog is empty.",
            "mood_label": "Mixed",
            "confidence": 0.0,
            "retrieved_count": 0,
            "query": user_query,
        }

    # Step 2 — Build prompt
    catalog = _catalog_text(retrieved)
    system_prompt = (
        "You are a music recommendation assistant with access to a song catalog. "
        "Given a user's mood or activity, recommend songs from the provided catalog. "
        "Respond ONLY with valid JSON matching this exact schema:\n"
        '{"recommended_songs": ["Title1", "Title2"], '
        '"reasoning": "one or two sentences", '
        '"mood_label": "Hype or Chill or Mixed", '
        '"confidence": 0.85}\n'
        "Rules:\n"
        "- Only recommend songs whose titles appear verbatim in the catalog.\n"
        "- Recommend 2–4 songs.\n"
        "- confidence is 0.0–1.0; reflect how well the catalog matched the request.\n"
        "- Output raw JSON only — no markdown fences, no extra text."
    )
    user_prompt = (
        f'User request: "{user_query}"\n\n'
        f"Catalog ({len(retrieved)} most relevant songs):\n{catalog}\n\n"
        f"User's favorite genre: {profile.get('favorite_genre', 'any')}\n\n"
        "Respond with JSON only."
    )

    # Step 3 — Claude API call
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()
        logger.info("Claude response (truncated): %s", raw[:300])
    except anthropic.APIError as exc:
        logger.error("Claude API error: %s", exc)
        return {
            "error": f"API error: {exc}",
            "recommended_songs": [],
            "reasoning": "",
            "mood_label": "Mixed",
            "confidence": 0.0,
            "retrieved_count": len(retrieved),
            "query": user_query,
        }

    # Step 4 — Parse JSON (strip markdown fences if present)
    text = raw
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    try:
        result: Dict = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse error: %s | raw=%r", exc, raw)
        return {
            "error": "Could not parse AI response.",
            "recommended_songs": [],
            "reasoning": raw,
            "mood_label": "Mixed",
            "confidence": 0.0,
            "retrieved_count": len(retrieved),
            "query": user_query,
        }

    # Step 5 — Validate songs exist in the full catalog
    title_map = {s["title"].lower(): s["title"] for s in songs}
    validated = [
        title_map[t.lower()]
        for t in result.get("recommended_songs", [])
        if t.lower() in title_map
    ]
    for t in result.get("recommended_songs", []):
        if t.lower() not in title_map:
            logger.warning("AI hallucinated song not in catalog: %r", t)

    result["recommended_songs"] = validated
    result["retrieved_count"] = len(retrieved)
    result["query"] = user_query

    conf = float(result.get("confidence", 0.7))
    if not validated:
        conf = 0.0
    elif len(validated) < 2:
        conf = min(conf, 0.5)
    result["confidence"] = round(max(0.0, min(1.0, conf)), 2)

    logger.info(
        "Final recommendation: songs=%s confidence=%.2f",
        validated,
        result["confidence"],
    )
    return result


def get_client() -> Optional[anthropic.Anthropic]:
    """Return an Anthropic client using ANTHROPIC_API_KEY, or None if unset."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set; AI features disabled.")
        return None
    return anthropic.Anthropic(api_key=api_key)
