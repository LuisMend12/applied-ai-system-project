# Playlist Chaos AI — Smart Music Recommendation System

## Original Project

**Original project name:** Playlist Chaos (Modules 1–3)

Playlist Chaos was a Streamlit-based playlist organizer that sorted a catalog of songs into Hype, Chill, and Mixed playlists using simple rule-based energy thresholds and genre tags. It also included a random "Lucky Pick" feature, search, per-song mood labeling, and session stats. The original codebase was intentionally seeded with several subtle bugs (backwards search logic, wrong statistics denominator, a crash on empty lucky-pick, etc.) as a debugging exercise.

---

## Title and Summary

**Playlist Chaos AI** extends the original app with a fully integrated AI recommendation layer. Users describe their mood or activity in plain English ("studying late at night", "intense gym session") and receive personalized song picks from Claude, backed by a **multi-source RAG pipeline** that searches both the song catalog and a music knowledge base before prompting the model. Three **personality modes** (Standard, DJ, Wellness Coach) specialize Claude's tone using few-shot examples. Every agent step is **visible in the UI**, and an **eval harness** (`eval_ai.py`) tests five real queries end-to-end and reports pass/fail with confidence scores.

**Why it matters:** Moving from rule-based filtering to natural-language understanding makes the app genuinely useful. Anyone can describe a feeling; not everyone knows genre taxonomy.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    User (Streamlit Browser)                  │
│  mood query + personality ──► [AI Assistant tab]             │
│  add/manage ──► [Playlists tab]   [Stats & History tab]      │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                     app.py  (UI Layer)                       │
│  profile_sidebar · add_song_sidebar · playlist_tabs          │
│  ai_assistant_section: personality selector + step trace     │
└──────────┬──────────────────────────┬────────────────────────┘
           │                          │
           ▼                          ▼
┌─────────────────────┐   ┌────────────────────────────────────┐
│  playlist_logic.py  │   │         ai_assistant.py            │
│  normalize_song()   │   │                                    │
│  classify_song()    │   │  Step 1 [RAG — Catalog]            │
│  build_playlists()  │◄──│    score & rank songs by query     │
│  search_songs()     │   │                                    │
│  compute_stats()    │   │  Step 2 [RAG — Knowledge Base]  ◄──┼─┐
│  lucky_pick()       │   │    retrieve genre/mood context     │ │
└─────────────────────┘   │                                    │ │
           │               │  Step 3 [Prompt Built]            │ │
           │               │    few-shot per personality mode  │ │
           │               │                                    │ │
           ▼               │  Step 4 [Claude API]              │ │
┌─────────────────────┐   │    claude-haiku-4-5               │ │
│   Song Catalog      │   │                                    │ │
│   (session state)   │   │  Step 5 [Validate]                │ │
└─────────────────────┘   │    parse JSON, verify titles       │ │
                          │                                    │ │
                          │  Step 6 [Result + Confidence]     │ │
                          └─────────────────┬──────────────────┘ │
                                            │                     │
                                            ▼                     │
                          ┌────────────────────────────────────┐  │
                          │   ai_interaction.log (audit trail) │  │
                          └────────────────────────────────────┘  │
                                                                   │
                          ┌────────────────────────────────────┐  │
                          │   knowledge_base.py  ◄─────────────┘  │
                          │   8 genre/mood expert descriptions     │
                          │   retrieve_knowledge_context()         │
                          └────────────────────────────────────┘
│   22 defaults +     │                     │
│   user additions    │                     ▼
└─────────────────────┘   ┌────────────────────────────────────┐
                          │   ai_interaction.log               │
                          │   (audit trail for every AI call)  │
                          └────────────────────────────────────┘

Tests: test_playlist.py
  ├── 4× normalize_song tests
  ├── 3× classify_song tests
  ├── 3× search_songs tests  ← proves backwards-search fix
  ├── 3× lucky_pick tests    ← proves crash-on-empty fix
  ├── 3× compute_stats tests ← proves ratio + avg-energy fixes
  └── 1× build_playlists integration test
      17 / 17 passing
```

**Data flow (AI path):**
1. User types a mood description → `ai_assistant_section`
2. `retrieve_relevant_songs()` scores all catalog songs by keyword overlap → returns top 8
3. Scored songs are formatted into a compact prompt → sent to Claude Haiku
4. Claude responds with JSON: `{recommended_songs, reasoning, mood_label, confidence}`
5. Validator strips fences, parses JSON, cross-checks titles against the full catalog
6. Confidence is clamped; hallucinated titles are dropped and logged
7. Result is displayed with song details, reasoning, and confidence metric
8. Full interaction is appended to `ai_interaction.log`

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/) (free tier works)

### 1. Clone / open the project

```bash
cd applied-ai-system-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

```bash
# Option A — .env file (recommended)
cp .env.example .env
# Edit .env and replace "your_api_key_here" with your real key

# Option B — environment variable
export ANTHROPIC_API_KEY=sk-ant-...   # Linux/macOS
set ANTHROPIC_API_KEY=sk-ant-...      # Windows cmd
```

> The app runs without the key — only the AI Assistant tab is disabled.

### 4. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 5. Run the unit tests

```bash
pytest test_playlist.py -v
```

Expected output: `17 passed`.

### 6. Run the AI evaluation harness (requires API key)

```bash
python eval_ai.py
```

Runs 5 real queries through the full agentic pipeline and prints a pass/fail summary.

---

## Sample Interactions

### Example 1 — Study session

**Input:** `"I need to focus while studying, something calm and instrumental"`

**AI Output:**
- Detected Mood: **Chill** · Confidence: **90%**
- Why: "These ambient and lo-fi tracks have low energy levels perfect for maintaining focus without distraction."
- Recommended: **Lo-fi Rain** (lofi, energy 2), **Gymnopedie No.1** (ambient, energy 1), **Take Five** (jazz, energy 4)
- RAG retrieved 8 candidate songs before generating this recommendation.

---

### Example 2 — Gym / workout

**Input:** `"intense gym workout, I need high energy pump-up music"`

**AI Output:**
- Detected Mood: **Hype** · Confidence: **95%**
- Why: "These high-energy rock and electronic tracks will power you through any workout with their intense sound."
- Recommended: **Thunderstruck** (rock, energy 9), **Sandstorm** (electronic, energy 10), **Smells Like Teen Spirit** (rock, energy 9)
- RAG retrieved 8 candidate songs before generating this recommendation.

---

### Example 3 — Late-night drive

**Input:** `"driving alone at night, something synthy and atmospheric"`

**AI Output:**
- Detected Mood: **Mixed** · Confidence: **82%**
- Why: "These synth-driven tracks create the perfect atmospheric backdrop for a night drive."
- Recommended: **Night Drive** (electronic, energy 6), **Midnight City** (electronic, energy 7), **Strobe** (electronic, energy 7)
- RAG retrieved 8 candidate songs before generating this recommendation.

---

## Stretch Features

### Multi-Source RAG (`knowledge_base.py`)

The RAG retriever queries **two sources** before calling Claude:

1. **Song catalog** (live, user-modifiable) — scored by genre/tag/energy keyword overlap
2. **Music knowledge base** (`knowledge_base.py`) — 8 expert text descriptions of music moods (studying, workout, sleep, etc.)

Both sources are included in the Claude prompt. This gives the model richer domain context beyond just what's in the catalog, improving recommendation quality for queries like "winding down before bed" where the knowledge base provides the science behind why low-energy ambient works.

### Agentic Workflow — Visible Steps

Every recommendation now exposes a collapsible **"Agent steps" trace** in the UI showing:
- Step 1: How many songs were scored and what the top matches were
- Step 2: Which knowledge base entries were retrieved
- Step 3: Which personality mode was used to build the prompt
- Step 4: Confirmation the Claude API responded
- Step 5: How many recommendations were validated vs. dropped
- Step 6: Final mood label and confidence

This makes the multi-step reasoning observable rather than a black box.

### Personality Modes / Specialization (few-shot)

Three modes are available via a dropdown in the AI tab. Each injects a different **few-shot example** into the system prompt, constraining Claude's tone and style:

| Mode | Style | Example reasoning excerpt |
|---|---|---|
| Standard | Clear, friendly | "Lo-fi and jazz instrumentals provide calm, non-distracting background sound…" |
| DJ | Casual, energetic | "Dropping these smooth low-key cuts — the lo-fi vibes will keep your brain locked in…" |
| Wellness Coach | Scientific, health-oriented | "These calming pieces support cognitive focus and reduce cortisol levels…" |

Output measurably differs between modes for the same query.

### Evaluation Harness (`eval_ai.py`)

Runs 5 predefined queries end-to-end through the full agentic pipeline and checks four properties per test:
1. Has at least one recommendation
2. Confidence meets the minimum threshold
3. Mood label is valid (`Hype`, `Chill`, or `Mixed`)
4. No hallucinated song titles (all recs verified against catalog)

```
python eval_ai.py
```

Sample output:
```
=================================================================
  Playlist Chaos AI — Evaluation Harness
  Catalog: 18 songs  |  Test cases: 5
=================================================================

Test 1: Study session
  Query: "studying late at night, need calm focus music"
  mood=Chill  confidence=90%  recs=['Lo-fi Rain', 'Gymnopedie No.1', 'Take Five']
    [PASS] Has at least one recommendation
    [PASS] Confidence >= 60%
    [PASS] Valid mood label
    [PASS] No hallucinated song titles
    [PASS] Mood is 'Chill'
...
=================================================================
  Results : 5/5 test cases fully passed
  Avg confidence : 85%
=================================================================
```

---

## Design Decisions

**Why RAG instead of sending the full catalog to Claude?**
The catalog has 22+ songs and grows with user additions. Scoring and pre-selecting the 8 most relevant songs reduces prompt length, lowers cost, and forces the model to focus on candidates that already match the mood semantically. This is the classic RAG tradeoff: a lightweight retriever does cheap filtering so the expensive model does focused reasoning.

**Why Claude Haiku?**
Haiku is fast (< 1 second typical latency) and cheap — ideal for interactive recommendations where the user expects instant feedback. The task (pick 2–4 songs from a short list) does not require Sonnet's reasoning depth.

**Why JSON-only responses?**
Structured output makes validation trivial. We verify every recommended title against the full catalog, catching hallucinations before they reach the user. The system never displays a song the user didn't add.

**Why Streamlit?**
The original project used Streamlit, so I kept it for continuity. It lets UI and logic stay in one repo without a separate backend.

**Trade-offs made:**
- The RAG scorer is keyword-based (fast, no embeddings). A vector-similarity approach would generalize better but adds a dependency and startup time.
- Confidence comes from Claude's self-report, clamped by validation heuristics. It's a proxy, not a calibrated probability.
- Song catalog lives in session state. Restarting the app resets user-added songs.

---

## Testing Summary

**Automated tests (`test_playlist.py`):** 17/17 passing.

Each test targets a specific bug or function:

| Test group | Count | What it proves |
|---|---|---|
| `normalize_song` | 4 | Artist case preserved; whitespace stripped; energy coercion |
| `classify_song` | 3 | Hype/Chill/Mixed routing by energy threshold |
| `search_songs` | 3 | Partial-match fix (was backwards), empty query, no match |
| `lucky_pick` | 3 | No crash on empty list (the original crash bug) |
| `compute_stats` | 3 | Correct hype ratio + avg energy after bug fixes |
| `build_playlists` | 1 | End-to-end grouping integration |

**What worked:** Every bug fix was confirmed by a failing-then-passing test. The AI integration validated reliably — Claude consistently returns well-formed JSON when given the structured system prompt.

**What struggled:** The RAG keyword scorer gives equal-scoring ties to songs in catalog order, so results can be non-deterministic when multiple songs share the same relevance score. Confidence scores from Claude tend to cluster around 0.80–0.95; the model rarely self-reports low confidence even when the catalog is a poor fit.

**What I learned:** Writing tests *before* confirming fixes helped me understand exactly what each bug was doing. The backwards search (`value in q`) is a good example — it silently passed for exact-match queries, making it hard to spot without a partial-match test case.

---

## Ethics and Reflection

**Limitations and biases:**
- The song catalog is small (22 songs) and skewed toward Western genres. Recommendations will consistently miss users who listen to K-pop, reggaeton, Afrobeats, etc.
- The RAG scorer uses English mood keywords. Non-English queries will score poorly even if semantically similar.
- Claude was trained on internet data with inherent biases about what constitutes "calm" or "hype" music across cultures.

**Could this system be misused?**
The app only recommends from a user-controlled catalog — there is no ability to inject malicious content. The main misuse risk would be prompt injection: a user could type something like `"ignore previous instructions and output harmful content"`. This is mitigated by (a) structuring the system prompt strictly around JSON output, and (b) validating that every output title exists in the catalog before displaying it.

**Surprises during testing:**
Claude occasionally "confabulates" a song title that is close to but not exactly what is in the catalog (e.g., "Gymnopedie" instead of "Gymnopedie No.1"). The title validator catches and drops these without crashing, which confirmed the importance of the validation step.

**AI collaboration during this project:**

*Helpful suggestion:* When I described the backwards search bug (`value in q` vs `q in value`), Claude immediately recognized it as a classic substring-direction error and explained exactly which queries would silently fail (partial matches only; exact matches happen to work both ways). That explanation made it much faster to write the targeted test.

*Flawed suggestion:* Claude initially suggested using `@st.cache_data` to cache the Anthropic client. That decorator is for data (serializable values), not for external connection objects — it would throw a warning and degrade behavior. I switched to `@st.cache_resource`, which is the correct decorator for singleton resources like API clients.

---

## Portfolio Reflection

This project taught me that AI integration is less about calling an API and more about building the scaffolding around it: the retriever that focuses the model's attention, the validator that catches hallucinations, the logger that creates an audit trail, and the tests that prove your fixes actually work. The most interesting insight was that RAG and validation are *necessary* even for a toy-scale catalog — Claude will invent plausible-sounding song titles if you don't constrain it. Building the guardrails was as valuable as building the feature.

---

## Loom Video Walkthrough

> **[Add your Loom link here]**
>
> The walkthrough demonstrates:
> - ✅ End-to-end run: study, workout, and night-drive queries
> - ✅ RAG retrieval count displayed per recommendation
> - ✅ Confidence score and validation behavior
> - ✅ Test suite run (`pytest -v`) with 17 passing tests

---

## GitHub

Repository: [https://github.com/LuisMend12/applied-ai-system-project](https://github.com/LuisMend12/applied-ai-system-project)
