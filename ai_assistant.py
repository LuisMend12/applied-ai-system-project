"""
AI assistant: multi-source RAG retrieval + Claude-powered recommendations.

Agentic workflow (each step is recorded and returned for UI display):
  1. RAG — score and rank songs from the catalog
  2. RAG — retrieve relevant context from the music knowledge base
  3. Build prompt with few-shot examples for the selected personality mode
  4. Call Claude Haiku
  5. Parse and validate JSON response
  6. Compute and clamp confidence score
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import anthropic

from knowledge_base import retrieve_knowledge_context
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

# ---------------------------------------------------------------------------
# Personality modes — few-shot examples that specialize Claude's tone/style
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES: Dict[str, str] = {
    "standard": (
        'Example — input: "something calm for studying"\n'
        'Output: {"recommended_songs": ["Lo-fi Rain", "Take Five"], '
        '"reasoning": "Lo-fi and jazz instrumentals provide calm, non-distracting '
        'background sound ideal for sustained concentration.", '
        '"mood_label": "Chill", "confidence": 0.9}'
    ),
    "dj": (
        'Example — input: "something calm for studying"\n'
        'Output: {"recommended_songs": ["Lo-fi Rain", "Take Five"], '
        '"reasoning": "Dropping these smooth low-key cuts — the lo-fi vibes and '
        'cool jazz will keep your brain locked in without killing the vibe.", '
        '"mood_label": "Chill", "confidence": 0.9}'
    ),
    "wellness": (
        'Example — input: "something calm for studying"\n'
        'Output: {"recommended_songs": ["Gymnopedie No.1", "Lo-fi Rain"], '
        '"reasoning": "These calming, low-energy pieces support cognitive focus and '
        'reduce cortisol, creating an optimal neurological environment for deep work.", '
        '"mood_label": "Chill", "confidence": 0.9}'
    ),
}

PERSONALITY_SYSTEM_SUFFIX: Dict[str, str] = {
    "standard": "Respond in a clear, friendly tone.",
    "dj": (
        "Respond as an enthusiastic DJ. Use casual, high-energy language with "
        "DJ slang (e.g., 'dropping', 'vibes', 'locked in', 'fire track'). "
        "Keep reasoning energetic and fun."
    ),
    "wellness": (
        "Respond as a music therapist / wellness coach. Use calm, scientific-sounding "
        "language. Reference how music affects mood, focus, or energy physiologically. "
        "Keep reasoning thoughtful and health-oriented."
    ),
}

# ---------------------------------------------------------------------------
# Mood-to-genre/tag keyword mapping for catalog scoring (RAG step 1)
# ---------------------------------------------------------------------------

MOOD_KEYWORDS: Dict[str, List[str]] = {
    "study": ["lofi", "ambient", "calm", "instrumental", "focus", "piano"],
    "focus": ["lofi", "instrumental", "ambient", "jazz", "classical"],
    "workout": ["rock", "electronic", "hype", "pump", "intense", "trance"],
    "gym": ["rock", "electronic", "pump", "intense"],
    "sleep": ["ambient", "calm", "sleep", "relax", "piano", "classical"],
    "relax": ["ambient", "jazz", "chill", "calm", "lofi"],
    "party": ["pop", "dance", "electronic", "funk"],
    "dance": ["pop", "electronic", "funk", "dance"],
    "sad": ["ambient", "chill", "calm", "piano"],
    "happy": ["pop", "funk", "dance"],
    "drive": ["rock", "electronic", "pop", "synth"],
    "morning": ["pop", "jazz", "calm"],
    "night": ["electronic", "ambient", "synth", "dream"],
    "chill": ["lofi", "ambient", "jazz"],
    "hype": ["rock", "electronic", "pop", "punk"],
}


def retrieve_relevant_songs(
    query: str, songs: List[Song], top_k: int = 8
) -> Tuple[List[Song], List[Tuple[str, int]]]:
    """
    RAG step 1: score every catalog song by keyword overlap with the query.
    Returns (top_k songs, [(title, score), ...] for the step trace).
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
    top_scores = [(str(s.get("title", "")), sc) for sc, s in scored[:3]]
    logger.info(
        "RAG catalog: query=%r → %d/%d songs (top: %s)",
        query, len(retrieved), len(songs), top_scores,
    )
    return retrieved, top_scores


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
    personality: str = "standard",
) -> Dict:
    """
    Full agentic workflow — returns a result dict including a 'steps' list so
    the UI can display exactly what the agent did at each stage.

    Keys in returned dict:
      recommended_songs, reasoning, mood_label, confidence,
      retrieved_count, knowledge_used, query, steps, error (on failure)
    """
    steps: List[str] = []
    logger.info("Request: query=%r personality=%s profile=%s", user_query, personality, profile.get("name"))

    # ── Step 1: RAG — catalog retrieval ────────────────────────────────────
    retrieved, top_scores = retrieve_relevant_songs(user_query, songs, top_k=8)
    top_str = ", ".join(f'"{t}" (score {s})' for t, s in top_scores)
    steps.append(
        f"Step 1 [RAG — Catalog]: Scored {len(songs)} songs → "
        f"selected top {len(retrieved)}. Highest matches: {top_str}."
    )

    if not retrieved:
        steps.append("Step 1 aborted — catalog is empty.")
        return _error("No songs in catalog to recommend from.", steps, user_query)

    # ── Step 2: RAG — knowledge base retrieval ─────────────────────────────
    kb_entries = retrieve_knowledge_context(user_query, top_k=2)
    if kb_entries:
        kb_ids = ", ".join(e["id"] for e in kb_entries)
        steps.append(
            f"Step 2 [RAG — Knowledge Base]: Found {len(kb_entries)} relevant "
            f"music knowledge entries: {kb_ids}."
        )
    else:
        steps.append("Step 2 [RAG — Knowledge Base]: No matching knowledge entries for this query.")
    knowledge_used = [e["id"] for e in kb_entries]

    # ── Step 3: Build prompt ────────────────────────────────────────────────
    catalog = _catalog_text(retrieved)
    kb_context = (
        "\n\nMusic domain knowledge:\n"
        + "\n".join(f'- {e["text"]}' for e in kb_entries)
        if kb_entries
        else ""
    )
    few_shot = FEW_SHOT_EXAMPLES.get(personality, FEW_SHOT_EXAMPLES["standard"])
    persona_suffix = PERSONALITY_SYSTEM_SUFFIX.get(personality, PERSONALITY_SYSTEM_SUFFIX["standard"])

    system_prompt = (
        f"You are a music recommendation assistant. {persona_suffix}\n"
        "Given a user's mood or activity, recommend songs from the provided catalog.\n"
        "Respond ONLY with valid JSON matching this exact schema:\n"
        '{"recommended_songs": ["Title1", "Title2"], '
        '"reasoning": "one or two sentences", '
        '"mood_label": "Hype or Chill or Mixed", '
        '"confidence": 0.85}\n'
        "Rules:\n"
        "- Only recommend songs whose titles appear verbatim in the catalog.\n"
        "- Recommend 2-4 songs.\n"
        "- confidence is 0.0-1.0; reflect how well the catalog matched the request.\n"
        "- Output raw JSON only — no markdown fences, no extra text.\n\n"
        f"Few-shot example:\n{few_shot}"
    )
    user_prompt = (
        f'User request: "{user_query}"\n\n'
        f"Catalog ({len(retrieved)} most relevant songs):\n{catalog}"
        f"{kb_context}\n\n"
        f"User's favorite genre: {profile.get('favorite_genre', 'any')}\n\n"
        "Respond with JSON only."
    )
    steps.append(
        f"Step 3 [Prompt Built]: personality={personality}, "
        f"catalog_songs={len(retrieved)}, knowledge_entries={len(kb_entries)}."
    )

    # ── Step 4: Claude API call ─────────────────────────────────────────────
    steps.append(f"Step 4 [Claude API]: Calling claude-haiku-4-5 (personality={personality})…")
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()
        logger.info("Claude raw (truncated): %s", raw[:300])
        steps.append(f"Step 4 [Claude API]: Response received ({len(raw)} chars).")
    except anthropic.APIError as exc:
        logger.error("Claude API error: %s", exc)
        steps.append(f"Step 4 [Claude API]: ERROR — {exc}")
        return _error(f"API error: {exc}", steps, user_query)

    # ── Step 5: Parse and validate JSON ────────────────────────────────────
    text = raw
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    try:
        result: Dict = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse error: %s | raw=%r", exc, raw)
        steps.append(f"Step 5 [Validate]: JSON parse failed — {exc}")
        return _error("Could not parse AI response.", steps, user_query)

    title_map = {s["title"].lower(): s["title"] for s in songs}
    raw_recs = result.get("recommended_songs", [])
    validated = [title_map[t.lower()] for t in raw_recs if t.lower() in title_map]
    hallucinated = [t for t in raw_recs if t.lower() not in title_map]

    if hallucinated:
        logger.warning("Hallucinated titles dropped: %s", hallucinated)
    steps.append(
        f"Step 5 [Validate]: {len(validated)}/{len(raw_recs)} songs validated against catalog"
        + (f"; dropped hallucinations: {hallucinated}" if hallucinated else "") + "."
    )

    # ── Step 6: Confidence scoring ──────────────────────────────────────────
    conf = float(result.get("confidence", 0.7))
    if not validated:
        conf = 0.0
    elif len(validated) < 2:
        conf = min(conf, 0.5)
    conf = round(max(0.0, min(1.0, conf)), 2)
    steps.append(
        f"Step 6 [Result]: mood={result.get('mood_label', '?')} "
        f"confidence={conf:.0%} songs={validated}."
    )

    logger.info("Done: songs=%s confidence=%.2f", validated, conf)
    return {
        "recommended_songs": validated,
        "reasoning": result.get("reasoning", ""),
        "mood_label": result.get("mood_label", "Mixed"),
        "confidence": conf,
        "retrieved_count": len(retrieved),
        "knowledge_used": knowledge_used,
        "query": user_query,
        "steps": steps,
    }


def _error(msg: str, steps: List[str], query: str) -> Dict:
    return {
        "error": msg,
        "recommended_songs": [],
        "reasoning": "",
        "mood_label": "Mixed",
        "confidence": 0.0,
        "retrieved_count": 0,
        "knowledge_used": [],
        "query": query,
        "steps": steps,
    }


def get_client() -> Optional[anthropic.Anthropic]:
    """Return an Anthropic client using ANTHROPIC_API_KEY, or None if unset."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set; AI features disabled.")
        return None
    return anthropic.Anthropic(api_key=api_key)
