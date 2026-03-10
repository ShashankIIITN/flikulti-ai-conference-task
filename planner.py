"""
planner.py

AI-powered training planner for Ultimate Frisbee coaches.
Uses the Flikulti vector store to retrieve relevant content,
then generates a structured 2-3 session training plan with coaching notes.

Usage:
    python planner.py
"""

import os
from openai import OpenAI
from query import search
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Step 1: Gather trainer context via a short conversation
# ---------------------------------------------------------------------------

INTAKE_SYSTEM = """You are a helpful assistant for Ultimate Frisbee coaches.
Your job is to gather the key details needed to build a great training session plan.

Ask the coach for the following — but keep it conversational, not a form:
1. What they want to work on (topic / skill / problem)
2. The team's skill level (beginner / intermediate / advanced / mixed)
3. Approximate team size and session length
4. Any other context (e.g. upcoming opponent, specific struggle they're seeing)

Once you have enough to work with (you don't need all four perfectly answered),
output a JSON object on a single line in this exact format and nothing else after it:

{"topic": "...", "level": "...", "team_size": "...", "duration": "...", "context": "..."}

Keep asking follow-up questions until you have at least the topic and level."""

def gather_context() -> dict:
    """Chat with the coach to collect training context. Returns a dict."""
    messages = [{"role": "system", "content": INTAKE_SYSTEM}]
    print("\nFlik AI Training Planner")
    print("=" * 50)
    print("Tell me what you want to train and I'll build a session plan.\n")

    # Opening question
    messages.append({
        "role": "assistant",
        "content": "What do you want to work on in your next session?"
    })
    print("Assistant: What do you want to work on in your next session?\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
        )
        reply = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": reply})

        # Check if the model produced the final JSON
        if reply.startswith("{") and "topic" in reply:
            import json
            try:
                ctx = json.loads(reply)
                print(f"\n[Context captured: {ctx}]\n")
                return ctx
            except json.JSONDecodeError:
                pass  # keep chatting

        print(f"\nAssistant: {reply}\n")


# ---------------------------------------------------------------------------
# Step 2: Multi-angle RAG retrieval
# ---------------------------------------------------------------------------

def retrieve_content(ctx: dict) -> dict:
    """Run targeted searches and return bucketed results."""
    topic   = ctx.get("topic", "")
    level   = ctx.get("level", "")
    context = ctx.get("context", "")

    base = f"{topic} ultimate frisbee"

    queries = {
        "theory":   f"theory and tactics: {topic}",
        "drills":   f"drills and exercises for {topic} {level}",
        "beginner": f"beginner prerequisite drills for {topic}",
        "advanced": f"advanced progression drills {topic}",
        "analysis": f"game analysis real match footage {topic}",
        "session":  f"practice session plan {topic}",
    }

    if context:
        queries["context"] = context

    print("Searching Flikulti knowledge base...")
    results = {}
    for bucket, q in queries.items():
        matches = search(q, top_k=4)
        # Only keep matches with a reasonable score
        results[bucket] = [m for m in matches if m.score >= 0.35]
        print(f"  [{bucket}] {len(results[bucket])} results")

    return results


# ---------------------------------------------------------------------------
# Step 3: Build the prompt context string
# ---------------------------------------------------------------------------

def format_results(results: dict) -> str:
    """Turn the bucketed search results into a readable context block."""
    sections = []
    seen_urls = set()

    labels = {
        "theory":   "THEORY & TACTICS",
        "drills":   "DRILLS (relevant level)",
        "beginner": "DRILLS (simpler / prerequisites)",
        "advanced": "DRILLS (harder / progressions)",
        "analysis": "GAME ANALYSIS",
        "session":  "EXISTING SESSION PLANS",
        "context":  "ADDITIONAL CONTEXT",
    }

    for bucket, matches in results.items():
        if not matches:
            continue
        lines = [f"\n--- {labels.get(bucket, bucket.upper())} ---"]
        for m in matches:
            url = m.metadata.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            lines.append(
                f"Title: {m.metadata.get('title', '')}\n"
                f"URL: {url}\n"
                f"Content: {m.metadata.get('text', '')[:600]}\n"
            )
        sections.append("\n".join(lines))

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Step 4: Generate the training plan
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You are an expert Ultimate Frisbee coach and curriculum designer.
You have been given a collection of content from flikulti.com — the leading online resource
for Ultimate Frisbee training.

Your task is to create a practical, structured training plan based on the coach's request
and the provided Flikulti content.

Rules:
- Ground EVERY drill and drill coaching note in the actual Flikulti content provided.
  Include the exact flikulti.com URL for each drill.
- Order drills from simple → complex. Always start with the simplest form of the skill.
- Include realistic time allocations that add up to the session length.
- For each drill write:
    * Why this drill (what skill it isolates)
    * Things to emphasise (2-3 specific coaching cues)
    * How to know it's going well (observable success indicator)
- Include a brief "In real play" section referencing any game analysis you found.
- Suggest 2-3 sessions total: Session 1 (fundamentals), Session 2 (add pressure/decisions),
  Session 3 (live application / game-like scenarios).
- End with a "Further Reading" section linking the key theory articles.
- Do NOT invent drill names or URLs. Only use what is in the provided content.
  If you don't have a drill for a step, say so and suggest the coach search flikulti.com.
- Write in a direct, practical tone — the output should be usable on the pitch."""


def generate_plan(ctx: dict, rag_context: str) -> str:
    topic    = ctx.get("topic", "the requested topic")
    level    = ctx.get("level", "club")
    size     = ctx.get("team_size", "unknown")
    duration = ctx.get("duration", "90 minutes")
    extra    = ctx.get("context", "")

    user_prompt = f"""Coach's request:
- Topic: {topic}
- Team level: {level}
- Team size: {size}
- Session length: {duration}
- Additional context: {extra if extra else "None"}

Flikulti content retrieved for this topic:
{rag_context}

Please generate the full 2-3 session training plan now."""

    print("\nGenerating training plan...\n")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.5,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Gather context
    ctx = gather_context()

    # 2. Retrieve content
    results = retrieve_content(ctx)
    rag_context = format_results(results)

    # 3. Generate plan
    plan = generate_plan(ctx, rag_context)

    # 4. Print and save
    print("\n" + "=" * 60)
    print(plan)
    print("=" * 60)

    # Save to file
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", ctx.get("topic", "plan").lower()).strip("-")
    filename = f"plan_{slug}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Training Plan: {ctx.get('topic', '')}\n\n")
        f.write(f"*Level: {ctx.get('level', '')} | Duration: {ctx.get('duration', '')}*\n\n")
        f.write(plan)
    print(f"\nPlan saved to: {filename}")


if __name__ == "__main__":
    main()
