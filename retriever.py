"""
retriever.py — Multi-layer RAG retrieval pipeline for the Flik Training Planner

Layer 1: LLM Query Expansion
  — Use gpt-4o-mini to generate 10 sport-aware search queries from coaching context

Layer 2: Multi-Query Vector Search
  — Batch-embed all queries, run against Pinecone, group and score by URL

Layer 3: Full-Page Reconstruction
  — For top URLs, retrieve complete page text (scraped_pages.json → Pinecone chunks fallback)
    Pinecone metadata only stores text[:1000]; full pages can be 3000+ words

Layer 4: Drill Link Extraction + Secondary Search
  — Parse theory article text for flikulti.com/drills/ URLs (linked drills)
  — Fetch those specific drill pages if not already retrieved

Layer 5: LLM-Based Curation
  — Score each page 0-10 for relevance to the coaching need
  — Classify drill difficulty: beginner / intermediate / advanced
  — Structure output into typed buckets for the plan generator
"""

import json
import os
import re
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_pc     = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
_index  = _pc.Index(os.getenv("PINECONE_INDEX_NAME", "flikulti-theory"))

EMBED_MODEL   = "text-embedding-3-small"
MIN_SCORE     = 0.30          # drop results below this cosine similarity
MAX_PAGES     = 22            # cap on total pages passed to curation
FULL_TEXT_CAP = 3500          # chars of full text per page passed to the LLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_url(url: str) -> str:
    if "/drills/" in url:        return "drill"
    if "/sessions/" in url:      return "session"
    if "/analysis/" in url:      return "analysis"
    if "/sc-dashboard/" in url:  return "sc"
    if "/theory/" in url:        return "theory"
    if "/video/" in url:         return "video"
    return "other"


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in a single OpenAI call."""
    resp = _client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def _load_local_cache() -> dict[str, str]:
    """Load scraped_pages.json as url→full_text if it exists."""
    path = "scraped_pages.json"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            pages = json.load(f)
        return {p["url"]: p.get("text", "") for p in pages}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Layer 1: LLM Query Expansion
# ---------------------------------------------------------------------------

def expand_queries(ctx: dict) -> list[str]:
    """Use gpt-4o-mini to generate 10 sport-aware queries from coaching context."""
    topic   = ctx.get("topic", "")
    level   = ctx.get("level", "")
    context = ctx.get("context", "")

    prompt = f"""You are an expert Ultimate Frisbee coach searching a training knowledge base.

Generate 10 diverse search queries to find ALL relevant content for this coaching need:
- Topic: {topic}
- Team level: {level}
- Extra context: {context or "none"}

The knowledge base contains: theory articles, drill descriptions, session plans, game analysis.

Cover these angles:
1. Core concept / tactical theory for this topic
2. Simplest beginner drill for this skill
3. Intermediate drill introducing decision-making
4. Advanced or game-speed drill
5. Sport-specific terminology coaches use for this
6. Common errors or struggles players have with this
7. Video tutorial or visual demonstration of this skill
8. Game footage analysis showing this skill in a real match
9. Foundational prerequisite skill needed first
10. Full practice session plan incorporating this
11. Relevant strength/conditioning or physical preparation
12. How to start practicing this skill from scratch
13. Drills with images or step-by-step visual guides

Return ONLY a JSON array of 10 strings. No preamble.
["query 1", "query 2", ...]"""

    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    raw = resp.choices[0].message.content.strip()

    # Strip markdown fences if model added them
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        queries = json.loads(raw)
        if isinstance(queries, list) and queries:
            return [str(q) for q in queries[:10]]
    except Exception:
        pass

    # Fallback: hard-coded buckets
    return [
        f"theory tactics {topic}",
        f"beginner drill {topic}",
        f"intermediate drill {topic}",
        f"advanced drill {topic} {level}",
        f"{topic} ultimate frisbee coaching cues",
        f"common mistakes {topic}",
        f"game analysis {topic}",
        f"prerequisite foundational skill {topic}",
        f"practice session plan {topic}",
        context or f"{topic} pressure decision making",
    ]


# ---------------------------------------------------------------------------
# Layer 2: Multi-Query Vector Search
# ---------------------------------------------------------------------------

def multi_search(queries: list[str], top_k: int = 6) -> dict[str, dict]:
    """
    Embed all queries in one batch, search Pinecone, merge results by URL.
    Returns url → {best_score, hit_count, title, text_preview, type, ...}
    """
    embeddings = _embed(queries)

    url_map: dict[str, dict] = {}

    for query, embedding in zip(queries, embeddings):
        results = _index.query(vector=embedding, top_k=top_k, include_metadata=True)
        for match in results.matches:
            if match.score < MIN_SCORE:
                continue
            meta = match.metadata or {}
            url  = meta.get("url", "")
            if not url:
                continue

            if url not in url_map:
                url_map[url] = {
                    "url":          url,
                    "title":        meta.get("title", ""),
                    "text_preview": meta.get("text", ""),
                    "type":         _classify_url(url),
                    "best_score":   match.score,
                    "hit_count":    1,
                }
            else:
                entry = url_map[url]
                entry["hit_count"] += 1
                if match.score > entry["best_score"]:
                    entry["best_score"]   = match.score
                    entry["text_preview"] = meta.get("text", entry["text_preview"])

    # Re-rank: score × frequency bonus
    for entry in url_map.values():
        entry["combined_score"] = entry["best_score"] * (1 + 0.08 * entry["hit_count"])

    return dict(sorted(url_map.items(), key=lambda x: x[1]["combined_score"], reverse=True))


# ---------------------------------------------------------------------------
# Layer 3: Full-Page Reconstruction
# ---------------------------------------------------------------------------

def reconstruct_pages(top_urls: list[str], dummy_embedding: list[float], url_map: Optional[dict] = None) -> dict[str, str]:
    """
    Get complete page text for each URL.
    Priority: scraped_pages.json (complete) → Pinecone chunks (partial but better than 1000-char preview).
    """
    cache    = _load_local_cache()
    full     = {}

    for url in top_urls:
        # Fast path: local JSON has the full scraped text
        if url in cache and cache[url]:
            full[url] = cache[url]
            continue

        # Slow path: stitch chunks from Pinecone (filter may not be available on all plans)
        try:
            results = _index.query(
                vector=dummy_embedding,
                top_k=50,
                include_metadata=True,
                filter={"url": {"$eq": url}},
            )
            if results.matches:
                chunks = sorted(results.matches,
                                key=lambda m: m.metadata.get("chunk_index", 0))
                full[url] = "\n\n".join(m.metadata.get("text", "") for m in chunks)
        except Exception as e:
            # Filter not supported — fall back to text_preview from url_map
            print(f"  [layer3] Pinecone filter unavailable for {url}: {e}. Using preview.")
            preview = (url_map or {}).get(url, {}).get("text_preview", "")
            full[url] = preview

    return full


# ---------------------------------------------------------------------------
# Layer 4: Drill Link Extraction + Secondary Search
# ---------------------------------------------------------------------------

DRILL_URL_RE = re.compile(
    r'https?://(?:www\.)?flikulti\.com/drills/[^\s\)\]\"\'\,\.]+'
)

def extract_and_fetch_linked_drills(
    url_map: dict[str, dict],
    full_texts: dict[str, str],
    dummy_embedding: list[float],
) -> tuple[dict[str, dict], dict[str, str]]:
    """
    Find flikulti.com/drills/ URLs embedded in theory article text.
    Fetch any that aren't already in our result set.
    Returns updated url_map and full_texts.
    """
    linked_drill_urls: set[str] = set()

    for url, text in full_texts.items():
        if "/theory/" not in url:
            continue
        for match in DRILL_URL_RE.finditer(text):
            drill_url = match.group().rstrip(".,;)")
            linked_drill_urls.add(drill_url)

    new_urls = [u for u in linked_drill_urls if u not in url_map]
    print(f"  [layer4] Found {len(linked_drill_urls)} linked drills, "
          f"{len(new_urls)} not yet retrieved")

    for drill_url in new_urls[:8]:   # cap secondary fetches
        try:
            results = _index.query(
                vector=dummy_embedding,
                top_k=30,
                include_metadata=True,
                filter={"url": {"$eq": drill_url}},
            )
            if not results.matches:
                continue

            chunks = sorted(results.matches,
                            key=lambda m: m.metadata.get("chunk_index", 0))
            title = results.matches[0].metadata.get("title", "")

            url_map[drill_url] = {
                "url":          drill_url,
                "title":        title,
                "text_preview": results.matches[0].metadata.get("text", ""),
                "type":         "drill",
                "best_score":   0.52,   # linked → treat as relevant
                "hit_count":    1,
                "combined_score": 0.52,
            }
            full_texts[drill_url] = "\n\n".join(
                m.metadata.get("text", "") for m in chunks
            )
        except Exception:
            continue

    return url_map, full_texts


# ---------------------------------------------------------------------------
# Layer 5: LLM-Based Curation
# ---------------------------------------------------------------------------

def curate(url_map: dict[str, dict], full_texts: dict[str, str], ctx: dict) -> list[dict]:
    """
    Ask gpt-4o-mini to score each page for relevance and classify drill difficulty.
    Returns a list of enriched page dicts, sorted by relevance desc.
    """
    topic = ctx.get("topic", "")
    level = ctx.get("level", "")

    # Pick candidates: top by combined_score, capped to MAX_PAGES
    candidates = sorted(url_map.values(),
                        key=lambda x: x.get("combined_score", 0),
                        reverse=True)[:MAX_PAGES]

    items_for_scoring = [
        {
            "url":     c["url"],
            "title":   c["title"],
            "type":    c["type"],
            "preview": full_texts.get(c["url"], c.get("text_preview", ""))[:400],
        }
        for c in candidates
    ]

    prompt = f"""You are an expert Ultimate Frisbee coach reviewing content for relevance.

Coach's need: "{topic}" for a {level} team.

Score each item 0–10 for usefulness in building a training plan.
For drills, also classify difficulty: "beginner", "intermediate", or "advanced".

Items:
{json.dumps(items_for_scoring, indent=2)}

Return ONLY a JSON array (no markdown fences):
[{{"url": "...", "relevance": 8, "difficulty": "intermediate", "reason": "one-line reason"}}, ...]"""

    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    ratings_by_url: dict[str, dict] = {}
    try:
        ratings = json.loads(raw)
        for r in ratings:
            ratings_by_url[r["url"]] = r
    except Exception:
        pass

    enriched = []
    for c in candidates:
        url    = c["url"]
        rating = ratings_by_url.get(url, {})
        enriched.append({
            **c,
            "relevance":  rating.get("relevance",  int(c.get("combined_score", 0.5) * 10)),
            "difficulty": rating.get("difficulty", "intermediate"),
            "reason":     rating.get("reason",     ""),
            "full_text":  full_texts.get(url, c.get("text_preview", "")),
        })

    return sorted(enriched, key=lambda x: x["relevance"], reverse=True)


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

def _fmt_section(items: list[dict], label: str, max_items: int = 3) -> str:
    if not items:
        return ""
    lines = [f"\n{'=' * 64}", f"  {label}", f"{'=' * 64}"]
    for item in items[:max_items]:
        text = item.get("full_text", item.get("text_preview", ""))[:FULL_TEXT_CAP]
        lines += [
            f"\nTitle: {item.get('title', '')}",
            f"URL:   {item.get('url', '')}",
            f"Score: {item.get('relevance', '?')}/10 — {item.get('reason', '')}",
            f"\n{text}",
            "\n" + "-" * 48,
        ]
    return "\n".join(lines)


def format_context(curated: list[dict]) -> str:
    """Bucket curated pages by type/difficulty and format for the plan LLM."""
    def pick(type_val: str, diff: Optional[str] = None, min_rel: int = 4):
        return [
            i for i in curated
            if i["type"] == type_val
            and (diff is None or i.get("difficulty") == diff)
            and i.get("relevance", 0) >= min_rel
        ]

    sections = [
        _fmt_section(pick("theory"),                            "THEORY & TACTICS",             max_items=2),
        _fmt_section(pick("drill", "beginner"),                 "DRILLS — BEGINNER / SIMPLE",   max_items=3),
        _fmt_section(pick("drill", "intermediate"),             "DRILLS — INTERMEDIATE",        max_items=3),
        _fmt_section(pick("drill", "advanced"),                 "DRILLS — ADVANCED / COMPLEX",  max_items=2),
        _fmt_section(pick("video",  min_rel=3),                 "VIDEO TUTORIALS",              max_items=3),
        _fmt_section(pick("analysis", min_rel=3),               "GAME ANALYSIS",                max_items=2),
        _fmt_section(pick("session",  min_rel=3),               "EXISTING SESSION PLANS",       max_items=1),
        _fmt_section(pick("sc",       min_rel=3),               "STRENGTH & CONDITIONING",      max_items=1),
    ]
    return "\n".join(s for s in sections if s)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def retrieve_layered(ctx: dict, status_cb=None) -> str:
    """
    Full 5-layer retrieval. Returns a rich context string ready for the plan LLM.
    status_cb: optional callable(str) for streaming status updates to the UI.
    """
    def log(msg: str):
        print(f"[retriever] {msg}")
        if status_cb:
            status_cb(msg)

    log("Layer 1: Expanding queries with LLM…")
    queries = expand_queries(ctx)
    log(f"  → {len(queries)} queries generated")

    log("Layer 2: Multi-query vector search…")
    url_map = multi_search(queries, top_k=6)
    log(f"  → {len(url_map)} unique pages found")

    # Select top URLs per type for full reconstruction
    by_type: dict[str, list[str]] = {}
    for url, info in url_map.items():
        by_type.setdefault(info["type"], []).append(url)

    top_urls: list[str] = []
    for t in ("theory", "drill", "analysis", "session", "sc", "video"):
        top_urls.extend(by_type.get(t, [])[:5])
    top_urls = list(dict.fromkeys(top_urls))[:MAX_PAGES]

    log(f"Layer 3: Reconstructing full text for {len(top_urls)} pages…")
    dummy_emb  = _embed([ctx.get("topic", "ultimate frisbee training")])[0]
    full_texts = reconstruct_pages(top_urls, dummy_emb, url_map=url_map)
    log(f"  → Full text retrieved for {len(full_texts)} pages")

    log("Layer 4: Extracting linked drill URLs from theory articles…")
    url_map, full_texts = extract_and_fetch_linked_drills(url_map, full_texts, dummy_emb)

    log("Layer 5: LLM relevance curation…")
    curated = curate(url_map, full_texts, ctx)
    high_rel = [i for i in curated if i.get("relevance", 0) >= 5]
    log(f"  → {len(high_rel)}/{len(curated)} pages scored ≥5/10")

    return format_context(curated)
