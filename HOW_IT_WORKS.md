# How the Flik AI Training Planner Works

A plain-English explanation of the end-to-end flow — from a coach typing a question to a structured training plan appearing on screen.

---

## Overview

The planner has four stages:

```
Coach input  →  5-layer RAG retrieval  →  LLM plan generation  →  Rendered output
                      ↑
             (inline Q&A bypasses this)
```

Each stage is independent: the intake conversation collects context, the RAG layer finds relevant Flikulti content, and the LLM turns that content into a structured plan. General questions asked mid-conversation are answered immediately without triggering a full plan.

---

## Stage 1 — Intake (Conversational Context Gathering)

**File:** `app.py` → `/chat` route
**Model:** `gpt-4o-mini`

When the coach opens the app, a short AI-driven conversation collects:

| Field | Example |
|---|---|
| `topic` | "breaking the mark" |
| `level` | "intermediate club team" |
| `team_size` | "18 players" |
| `duration` | "90 minutes" |
| `context` | "playing a cup zone team next weekend" |

The intake model asks one question at a time. Once it has at least a **topic** and **level**, it stops asking and outputs a single JSON object — nothing else. The frontend detects this JSON with a regex and automatically moves to Stage 2.

**Inline Q&A:** If a message looks like a question (ends in `?` or starts with a question word like *how, what, why*), the app answers it immediately using RAG — without interrupting the intake flow. The answer appears with a blue "📚 Flikulti Knowledge Base" badge and the conversation continues naturally.

**Why a conversation instead of a form?**
Coaches describe their needs in natural language. A form would miss context like *"they panic under pressure"* or *"we're preparing for a specific opponent"* — which meaningfully changes what the plan should contain.

---

## Stage 2 — 5-Layer RAG Retrieval

**File:** `retriever.py` → `retrieve_layered()`
**Vector store:** Pinecone (`flikulti-theory` index, 891 chunks)
**Embedding model:** `text-embedding-3-small` (OpenAI, 1536 dimensions)

The retrieval pipeline runs five layers in sequence, with real-time status updates streaming to the browser as it works.

### Layer 1 — LLM Query Expansion

`gpt-4o-mini` generates **10 sport-aware queries** covering every angle of the coaching need:

| Angle | Example query |
|---|---|
| Core theory | "breaking the mark tactics and positioning" |
| Beginner drill | "simplest beginner drill for breaking the mark" |
| Intermediate drill | "intermediate decision-making drill break" |
| Advanced drill | "game-speed advanced break throw drill" |
| Common errors | "mistakes players make when breaking the mark" |
| **Video tutorial** | "video demonstration of break throw technique" |
| **Game footage** | "real match footage breaking the mark analysis" |
| Prerequisite skill | "foundational throwing mechanics prerequisite" |
| Session plan | "full practice session plan break throws" |
| Visual/step-by-step | "drills with images step-by-step visual guide" |

**Why expand queries?** A single query returns semantically similar chunks (mostly drills *or* mostly theory). 10 diverse queries — including dedicated video and game footage angles — ensure the LLM gets the full progression arc plus any visual resources available.

### Layer 2 — Multi-Query Vector Search

All 10 queries are **batch-embedded in one API call**, then each is searched against Pinecone. Results are merged by URL using:

```
combined_score = best_score × (1 + 0.08 × hit_count)
```

Pages that appear across multiple queries are ranked higher (frequency bonus). Results below **0.30 cosine similarity** are dropped.

### Layer 3 — Full-Page Reconstruction

Pinecone metadata only stores the first 1000 characters of each chunk. Layer 3 fetches the **complete page text** (up to 3500 characters) for the top-ranked URLs:

1. **Fast path:** Check `scraped_pages.json` (local cache of all scraped pages) — returns full article text
2. **Slow path:** Stitch all Pinecone chunks for that URL back together in order (`chunk_index`)

Up to 22 pages are reconstructed, prioritising theory, drills, analysis, session, S&C, and video pages.

### Layer 4 — Drill Link Extraction

Theory articles on flikulti.com often link to specific drills with `flikulti.com/drills/...` URLs. Layer 4 scans all retrieved theory pages with a regex, finds those linked drill pages, and **fetches up to 8 additional drills** not already in the result set.

This ensures the plan uses drills the theory article itself recommends — not just whatever ranked highest in the vector search.

### Layer 5 — LLM Relevance Curation

`gpt-4o-mini` reviews all retrieved pages and:

- Scores each **0–10 for relevance** to the coaching need
- Classifies each drill as **beginner / intermediate / advanced**
- Provides a one-line reason for its score

Pages scoring below 4/10 are filtered out. The remainder are bucketed by type and formatted for the plan generator:

| Bucket | Content | Max items |
|---|---|---|
| Theory & Tactics | Concept and strategy articles | 2 |
| Drills — Beginner | Entry-level exercises | 3 |
| Drills — Intermediate | Decision-making drills | 3 |
| Drills — Advanced | Game-speed / complex drills | 2 |
| **Video Tutorials** | Video pages from `/video/` section | 3 |
| Game Analysis | Match footage descriptions | 2 |
| Existing Session Plans | Flikulti session structures | 1 |
| Strength & Conditioning | Physical preparation | 1 |

**Fallback:** If the full 5-layer pipeline throws an exception, the app falls back to a simpler 6-query single-pass retrieval automatically.

---

## Stage 3 — Plan Generation (LLM + Structured Prompt)

**File:** `app.py` → `/generate` route
**Model:** `gpt-4o` (streamed)

The curated content is passed to GPT-4o with a tightly constrained system prompt. Every drill block must include all of the following:

```
### Drill [N]: [Drill Name] ([X] min)
🔗 [flikulti.com/drills/drill-name](URL)

**Why this drill:** One sentence on what isolated skill it builds.

**Setup:**
- Number of players and roles
- Field space and equipment needed
- Starting positions

**How to run it (step by step):**
1. Who moves where, what the throw is, where they go next
2. Second action
3. Continue until one rep is complete
4. How to reset for the next rep

**Coaching cues** *(what to say out loud on the pitch):*
- "..."

**Common mistakes & fixes:**
| Mistake | Fix |
|---------|-----|
| What goes wrong | What to tell the player |

**Progression:** Once comfortable, add [specific constraint].
**How to know it's working:** [Observable success indicator]
```

The plan also includes:

| Section | What it contains |
|---|---|
| Session goal | One sentence on what players will achieve |
| Warm-up | Purpose-built for the session theme |
| Conditioned game | Rule tweak that forces the skill in a game context |
| **Video Tutorials** | 🎬 links to relevant Flikulti video pages (if found) |
| In Real Play | How this skill appears in match footage |
| Further Reading | Links to key theory articles |

The prompt enforces:
- **Drill ordering:** simple → complex across sessions
- **Source grounding:** exact flikulti.com URLs only — no invented content
- **3-session arc:** Session 1 (fundamentals) → Session 2 (add pressure) → Session 3 (live application)
- **Time allocations** that add up to the requested session length

The response streams back token-by-token via Server-Sent Events (SSE), so the plan renders live in the browser as it's written.

---

## How the Files Connect

```
app.py                Flask backend — routes, intake, inline Q&A, streaming
retriever.py          5-layer RAG pipeline (query expansion → curation)
query.py              Pinecone search interface (from the workshop setup)
planner.py            Standalone CLI version (simple 6-query retrieval)
templates/
  index.html          Full 3-column web UI (History | Chat | Plan)
  widget.html         Compact embeddable version (phase tabs)
.env                  API keys (never committed)
scraped_pages.json    Full page text cache used by Layer 3 (not in git)
```

**Note:** `planner.py` is the original CLI version and uses a simpler 6-bucket single-pass retrieval. The full 5-layer pipeline only runs through the web app (`app.py` + `retriever.py`).

---

## Request Flow (Full Example)

1. Coach types: *"We're playing a cup zone team next weekend, how do we prepare?"*
2. Intake model asks: *"What's your team's level — beginner, intermediate, or advanced?"*
3. Coach replies: *"Intermediate club"*
4. Intake model outputs JSON: `{"topic": "defending cup zone", "level": "intermediate", ...}`
5. Frontend detects JSON → calls `POST /generate`
6. **Layer 1:** LLM generates 10 targeted queries (including video tutorial + game footage angles)
7. **Layer 2:** All queries batch-embedded and searched → ~30 unique pages found, merged by URL
8. **Layer 3:** Full text (up to 3500 chars) reconstructed for top 22 pages
9. **Layer 4:** Linked drill URLs extracted from theory articles → up to 8 extra drills fetched
10. **Layer 5:** LLM scores and classifies all pages → bucketed into 8 typed sections
11. GPT-4o streams a 3-session plan with fully detailed drill blocks and video links where available
12. Plan renders live in the right panel, saved to localStorage history
13. Coach clicks **Download .md** and walks onto the pitch with it

**Mid-conversation question example:** Coach asks *"What is a cup zone?"* at any point → RAG answers immediately with a blue badge, conversation continues, no plan triggered.

---

## UI Features

| Feature | How to use |
|---|---|
| **History sidebar** | Click any past plan to reload it instantly |
| **Regenerate** | Re-runs retrieval and generation with the same session context |
| **Copy** | Copies the full plan markdown to clipboard |
| **Download .md** | Saves the plan as a markdown file named after the topic |
| **New Chat** | Resets the conversation to plan a new session (history stays intact) |

---

## Embedding on the Flikulti Website

The widget at `/widget` is designed for iframe embedding:

```html
<iframe
  src="https://your-server.com/widget"
  width="100%"
  height="700"
  style="border:none; border-radius:12px">
</iframe>
```

The widget has the same full pipeline but in a compact single-column layout with three phase tabs (Your session → Build plan → Your plan).

**What the Flikulti team needs to run it:**

- One server running `python app.py` (any cloud VM or PaaS like Render/Railway)
- Three environment variables: `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`
- Running cost: approximately $0.05–0.15 per plan generated (query expansion + multi-search embeddings + curation + GPT-4o generation)

---

## Updating the Knowledge Base

If Flikulti publishes new content, re-run the ingestion pipeline:

```bash
python scraper.py        # re-scrape the site (~45–60 min)
python clean_scraped.py  # strip nav noise
python ingest.py         # re-embed and upload to Pinecone
```

Upserts are idempotent — existing vectors are overwritten, new ones are added. No downtime required. After re-ingestion, also replace `scraped_pages.json` so Layer 3 full-page reconstruction uses the updated content.
