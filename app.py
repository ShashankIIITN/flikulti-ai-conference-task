"""
app.py — Flask web UI for the Flik AI Training Planner
Run: python app.py
"""

import os
import json
import re
import queue
import threading
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from openai import OpenAI
from retriever import retrieve_layered
from query import search
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Intake (conversational context gathering)
# ---------------------------------------------------------------------------

INTAKE_SYSTEM = """You are a helpful assistant for Ultimate Frisbee coaches.
Your job is to gather key details for a training session plan through a short conversation.

Collect (conversationally, not as a form):
1. What they want to work on (topic / skill / problem)
2. Team skill level (beginner / intermediate / advanced / mixed)
3. Team size and session length
4. Any extra context (upcoming opponent, specific struggle)

IMPORTANT RULES:
- Ask ONE short follow-up question at a time.
- Once you have at least the topic AND skill level, you MUST stop asking and output ONLY the JSON below.
- Do NOT generate a training plan. Do NOT add any text before or after the JSON.
- Output exactly this, on one line, with no other text:
{"topic": "...", "level": "...", "team_size": "...", "duration": "...", "context": "..."}"""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/widget")
def widget():
    return render_template("widget.html")


QUESTION_WORDS = ("why", "what", "how", "when", "where", "who", "which",
                  "can", "could", "does", "do", "is", "are", "should", "will")

QA_SYSTEM = """You are a knowledgeable Ultimate Frisbee coach assistant backed by the Flikulti knowledge base.
Answer the coach's question concisely and practically using the provided Flikulti content.
- Keep your answer under 180 words.
- Use bullet points for lists of 3+ items.
- End with a natural one-sentence bridge back to session planning
  (e.g. "Want me to build a session plan around this?").
- If the content doesn't cover the question, say so honestly."""


def _is_question(text: str) -> bool:
    """Heuristic: treat message as a general question if it looks like one."""
    t = text.strip().lower()
    if t.endswith("?"):
        return True
    if any(t.startswith(w + " ") for w in QUESTION_WORDS):
        return True
    return False


def _rag_answer(question: str) -> str:
    """Quick RAG search + LLM answer for a general question."""
    matches = search(question, top_k=5)
    good    = [m for m in matches if m.score >= 0.30]

    if not good:
        return ("I don't have specific Flikulti content on that topic. "
                "What would you like to work on in your next training session?")

    context = "\n\n".join(
        f"[{m.metadata.get('title', '')}]\n{m.metadata.get('text', '')[:700]}"
        for m in good
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"{QA_SYSTEM}\n\nFlikulti content:\n{context}"},
            {"role": "user",   "content": question},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


@app.route("/chat", methods=["POST"])
def chat():
    """Handle one turn of the intake conversation, with inline Q&A support."""
    data     = request.json
    messages = data.get("messages", [])

    # Last user message
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )

    # If it looks like a general question, answer it with RAG directly
    if _is_question(last_user):
        answer = _rag_answer(last_user)
        return jsonify({"reply": answer, "context": None, "is_answer": True})

    # Otherwise run the normal intake flow
    full_messages = [{"role": "system", "content": INTAKE_SYSTEM}] + messages

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=full_messages,
        temperature=0.4,
    )
    reply = response.choices[0].message.content.strip()

    context_data = None
    json_match = re.search(r'\{[^{}]*"topic"[^{}]*\}', reply, re.DOTALL)
    if json_match:
        try:
            context_data = json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return jsonify({"reply": reply, "context": context_data, "is_answer": False})


# ---------------------------------------------------------------------------
# Plan generation (streaming)
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You are an expert Ultimate Frisbee coach writing a session plan a coach can walk onto the pitch with and run immediately — without reading any other material.

You have been given full-text content from flikulti.com. Use it as your source of truth.

OUTPUT STRUCTURE — follow this exactly:

## Session [N] — [Title]
*Goal: one sentence stating what players will be able to do by the end.*
*Total time: X min | Players: Y | Space: describe field area needed*

---

### Warm-up ([X] min)
Brief description of what to do and why it prepares for the main theme.

---

### Drill [N]: [Drill Name] ([X] min)
🔗 [flikulti.com/drills/drill-name](URL)

**Why this drill:** One sentence on what isolated skill it builds.

**Setup:**
- Number of players and roles (e.g. thrower, cutter, marker, cone positions)
- Field space and equipment needed
- Starting positions

**How to run it (step by step):**
1. [First action — be specific: who moves where, what the throw is, where they go next]
2. [Second action]
3. [Continue until one repetition is complete]
4. [How to reset for the next rep]

**Coaching cues** *(what to say out loud on the pitch):*
- "..."
- "..."
- "..."

**Common mistakes & fixes:**
| Mistake | Fix |
|---------|-----|
| [what goes wrong] | [what to tell the player] |
| [what goes wrong] | [what to tell the player] |

**Progression:** Once players are comfortable, add [specific constraint or variation].

**How to know it's working:** [One observable success indicator — what you see when the drill is going well]

---

[Repeat the drill block for each drill in the session]

### Conditioned Game ([X] min)
*Rule tweak that forces players to apply the session theme in a game-like context.*
How to set it up, the rule, and what to watch for.

---

[Repeat ## Session block for Sessions 2 and 3]

---

## Video Tutorials
If any VIDEO TUTORIALS content was provided, list each one as:
🎬 [Title](URL) — one sentence on what the video demonstrates and why it's worth watching.
If no video content was provided, omit this section entirely.

## In Real Play
Reference any game analysis content found. Show how this skill appears in actual match footage and why it matters.

## Further Reading
- [Title](URL) — one-line description

---

STRICT RULES:
- Every drill MUST include all sections: Setup, How to run it, Coaching cues, Common mistakes, Progression, How to know it's working.
- Use the exact flikulti.com URLs from the provided content. Never invent URLs.
- If you don't have enough content for a drill block, say so and suggest the coach search flikulti.com instead of making things up.
- Time allocations must add up to the total session length.
- Drills must be ordered simple → complex across sessions.
- Include Video Tutorials section only if VIDEO TUTORIALS content was provided. Never invent video links."""


def _simple_retrieve(ctx: dict) -> str:
    """Fallback: original single-pass retrieval used if layered pipeline fails."""
    topic   = ctx.get("topic", "")
    level   = ctx.get("level", "")
    context = ctx.get("context", "")
    queries = {
        "Theory":     f"theory and tactics: {topic}",
        "Drills":     f"drills exercises {topic} {level}",
        "Beginner":   f"beginner prerequisite drills {topic}",
        "Advanced":   f"advanced progression drills {topic}",
        "Analysis":   f"game analysis real match {topic}",
        "Sessions":   f"practice session plan {topic}",
    }
    if context:
        queries["Context"] = context

    seen, blocks = set(), []
    for label, q in queries.items():
        matches = [m for m in search(q, top_k=5) if m.score >= 0.30]
        if not matches:
            continue
        lines = [f"\n--- {label.upper()} ---"]
        for m in matches:
            url = m.metadata.get("url", "")
            if url in seen:
                continue
            seen.add(url)
            lines.append(
                f"Title: {m.metadata.get('title', '')}\n"
                f"URL: {url}\n"
                f"Content: {m.metadata.get('text', '')[:800]}\n"
            )
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


@app.route("/generate", methods=["POST"])
def generate():
    """
    Stream the training plan back to the client using SSE.
    Two event types:
      data: {"type": "status", "text": "..."}  — retrieval progress messages
      data: {"type": "chunk",  "text": "..."}  — plan markdown tokens
      data: [DONE]
    """
    ctx      = request.json
    topic    = ctx.get("topic", "the topic")
    level    = ctx.get("level", "club")
    size     = ctx.get("team_size", "unknown")
    duration = ctx.get("duration", "90 minutes")
    extra    = ctx.get("context", "")

    def stream():
        # ── Retrieval phase ────────────────────────────────────────────────
        status_q: queue.Queue = queue.Queue()

        def status_cb(msg: str):
            status_q.put(("status", msg))

        rag_result: list[str] = []
        exc_holder: list[Exception] = []

        def run_retrieval():
            try:
                rag_result.append(retrieve_layered(ctx, status_cb=status_cb))
            except Exception as e:
                exc_holder.append(e)
            finally:
                status_q.put(None)  # sentinel

        t = threading.Thread(target=run_retrieval, daemon=True)
        t.start()

        # Stream status events while retrieval runs
        while True:
            item = status_q.get()
            if item is None:
                break
            _, msg = item
            yield f"data: {json.dumps({'type': 'status', 'text': msg})}\n\n"

        if exc_holder:
            err = str(exc_holder[0])
            print(f"[generate] Layered retrieval failed: {err}. Falling back to simple retrieval.")
            yield f"data: {json.dumps({'type': 'status', 'text': f'⚠️ Advanced retrieval failed, using fallback… ({err[:80]})'})}\n\n"
            # Fall back to the simple single-pass retrieval
            rag_context = _simple_retrieve(ctx)
        else:
            rag_context = rag_result[0] if rag_result else ""

        if not rag_context.strip():
            yield f"data: {json.dumps({'type': 'error', 'text': 'No relevant content found in the Flikulti knowledge base for this topic. Try rephrasing your request.'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # ── Generation phase: stream plan tokens ───────────────────────────
        yield f"data: {json.dumps({'type': 'status', 'text': '✍️ Writing training plan…'})}\n\n"

        user_prompt = f"""Coach's request:
- Topic: {topic}
- Team level: {level}
- Team size: {size}
- Session length: {duration}
- Additional context: {extra if extra else "None"}

Flikulti content retrieved (multi-layer RAG, full page text, LLM-curated):
{rag_context}

Generate the full 2-3 session training plan now."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.5,
            stream=True,
        )
        for chunk in response:
            text = chunk.choices[0].delta.content
            if text:
                yield f"data: {json.dumps({'type': 'chunk', 'text': text})}\n\n"

        yield "data: [DONE]\n\n"

    return Response(stream_with_context(stream()), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
