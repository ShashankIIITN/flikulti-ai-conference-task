# Flikulti Knowledge Base – Technical Details

This document covers the full technical setup: how the Flik website was scraped, how the content was cleaned and stored in the vector database, and how to query it in your application.

---

## What is this?

The Flikulti knowledge base is a **vector store** containing 891 chunks of content scraped from flikulti.com, covering:

| Section | Content |
|---------|---------|
| `/theory` | Tactics, strategy, coaching, analysis |
| `/drills` | Individual drill descriptions |
| `/sessions` | Full practice session plans |
| `/video` | Video tutorial descriptions |
| `/sc-dashboard` | Strength & conditioning |

When you search, you get back the **most semantically similar** content to your query — not just keyword matches. So searching "how to beat a cup" also finds content about "breaking zone defence" even if those exact words aren't in the query.

---

## Part 1: Querying the vector store

This is what most workshop participants need. The vector store is already built — you just need to query it.

### Setup

**Prerequisites:** Python 3.11+, access to the project folder (`rag-demo/`)

```bash
pip install pinecone openai python-dotenv
```

Add the following to a `.env` file in the project root — ask Maartje for the values:

```
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=flikulti-theory
OPENAI_API_KEY=...
```

### Option A: Command line

```bash
# Single query
python query.py "how does vertical stack work?"

# Interactive mode
python query.py
```

**Example output:**
```
Query: how does vertical stack work?
============================================================

[1] Score: 0.61
    URL:   https://www.flikulti.com/theory/vertical-stack/how-does-it-work
    Title: How does Vertical Stack work?
    Text:  The vertical stack is an offensive formation where...

[2] Score: 0.58
    URL:   https://www.flikulti.com/theory/vertical-stack/pros
    ...
```

- **Score**: 0.0–1.0. Above 0.5 is a good match. Above 0.65 is excellent.
- **URL**: The original flikulti.com page — click it to read the full article.
- **Text**: A 300-char preview of the matching chunk.

### Option B: Use the `search()` function in your own code

```python
from query import search

results = search("how to defend a handler", top_k=5)

for match in results:
    print(match.score)                    # relevance score
    print(match.metadata["url"])          # link to source
    print(match.metadata["title"])        # page title
    print(match.metadata["text"])         # chunk text (up to 1000 chars)
```

You can increase `top_k` to get more results (default is 5, max ~20 is useful).

### Example queries to try

```bash
python query.py "what is the role of the marker in cup zone?"
python query.py "drills for improving cutting timing"
python query.py "how to coach beginners on throwing"
python query.py "give and go movement in handler offence"
python query.py "workout plan for ultimate players"
python query.py "how to beat a force middle defence"
python query.py "practice plan for zone offence"
python query.py "spirit of the game rules"
```

### Tips for good queries

| Do | Don't |
|----|-------|
| Use natural questions: *"how do I..."* | Use only single keywords: *"cup"* |
| Be specific: *"backhand break throw technique"* | Be too vague: *"throwing"* |
| Use coaching language from the sport | Use unrelated terminology |
| Try rephrasing if results are weak | Give up on the first attempt |

### Minimal RAG chatbot example

```python
from openai import OpenAI
from query import search

client = OpenAI()

def ask(question: str) -> str:
    results = search(question, top_k=5)
    context = "\n\n".join(m.metadata["text"] for m in results)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Answer using this Flikulti content:\n\n{context}"},
            {"role": "user", "content": question},
        ]
    )
    return response.choices[0].message.content

print(ask("What are the key principles of cup zone defence?"))
```

---

## Part 2: How the knowledge base was built

This section is for anyone who wants to rebuild the vector store from scratch — for example if the Flik website gets new content, or if you want to adapt this pipeline for a different website.

### Overview

The pipeline has three steps:

```
flikulti.com  →  scraper.py  →  scraped_pages.json
                                       ↓
                              clean_scraped.py
                                       ↓
                               ingest.py  →  Pinecone
```

### Step 1: Scraping the website (`scraper.py`)

The Flik website requires a login to access most content. The scraper:

1. Opens a headless Chromium browser (via Playwright)
2. Navigates to the WordPress login page and fills in the credentials
3. Uses the authenticated browser session to crawl all pages under the configured sections
4. Follows links breadth-first, staying within the allowed URL prefixes
5. Saves all pages as `scraped_pages.json`

**What you need:**
- A Flik account with access to the full content library (Maartje can add you as a team member)
- The credentials stored in `.env`:

```
FLIKULTI_EMAIL=you@example.com
FLIKULTI_PASSWORD=yourpassword
```

**Install the browser engine (one-time):**

```bash
pip install crawl4ai playwright
playwright install chromium
```

**Run the scraper:**

```bash
python scraper.py
```

This takes approximately 45–60 minutes and produces `scraped_pages.json` (~5MB, 768 pages).

**How it works internally:**

The scraper uses `crawl4ai`, which wraps Playwright. It logs in using JavaScript executed in the browser context, then reuses the authenticated session (identified by `session_id="flikulti"`) for all subsequent page requests. Links are discovered by parsing all `href` attributes in the HTML and filtering to only follow URLs under the configured sections.

**Sections scraped:**

```python
START_URLS = [
    "https://www.flikulti.com/theory/",
    "https://www.flikulti.com/sessions/",
    "https://www.flikulti.com/drills/",
    "https://www.flikulti.com/video/",
    "https://www.flikulti.com/sc-dashboard/",
]
```

To add or remove sections, edit this list in `scraper.py`.

---

### Step 2: Cleaning the scraped text (`clean_scraped.py`)

The raw scraped pages include navigation noise — menus, language switchers, footers — that appears on every page before the actual article content. This noise would pollute the vector embeddings and reduce search quality.

`clean_scraped.py` strips this noise using two patterns specific to the flikulti.com WordPress theme:

- **Header nav:** The language switcher block ends with a quoted language name (e.g. `"Ukrainian")`) followed by the article's H1 heading. Everything before that transition is removed.
- **Footer:** Content after markers like `Previous article` or `Linked Drills:` is removed.

**Run the cleaner:**

```bash
python clean_scraped.py
```

This modifies `scraped_pages.json` in place. Typical result: removes ~40% of raw text per page, leaving only the article body.

> **Note:** If flikulti.com redesigns their site or changes their WordPress theme, these patterns may need updating. The relevant regex is `NAV_END_RE` in `clean_scraped.py`.

---

### Step 3: Embedding and uploading to Pinecone (`ingest.py`)

This step takes the cleaned text, splits it into overlapping chunks, creates vector embeddings using OpenAI, and upserts everything into Pinecone.

**What you need:**
- A Pinecone account (free tier works) — create one at pinecone.io
- An OpenAI account — create one at platform.openai.com
- Both API keys in `.env`:

```
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=flikulti-theory
OPENAI_API_KEY=...
```

**Run ingestion:**

```bash
python ingest.py
```

**How it works:**

| Setting | Value | Notes |
|---------|-------|-------|
| Embedding model | `text-embedding-3-small` | OpenAI, 1536 dimensions |
| Chunk size | 500 words | With 50-word overlap between chunks |
| Pinecone metric | cosine | Standard for semantic search |
| Pinecone cloud | AWS us-east-1 | Serverless, free tier |

The Pinecone index is created automatically if it doesn't exist. Each vector is stored with metadata: `url`, `title`, `chunk_index`, and `text` (first 1000 chars of the chunk).

Upserts are idempotent — running `ingest.py` again with updated content will overwrite existing vectors with the same ID (based on a hash of `url + chunk_index`).

**Cost estimate:**
- Embedding 891 chunks with `text-embedding-3-small`: < $0.01
- Pinecone free tier: up to 100k vectors, no cost
- Ongoing query cost: ~$0.0001 per query (one embedding call)

---

### Re-running the full pipeline

If the Flik website gets new content:

```bash
# 1. Re-scrape (45-60 min)
PYTHONUTF8=1 python scraper.py

# 2. Clean nav noise
PYTHONUTF8=1 python clean_scraped.py

# 3. Re-embed and upload
PYTHONUTF8=1 python ingest.py
```

> **Windows note:** Always prefix commands with `PYTHONUTF8=1` to avoid encoding errors with special characters in the scraped content.

---

## Files overview

| File | Purpose |
|------|---------|
| `scraper.py` | Logs into flikulti, crawls all sections, saves `scraped_pages.json` |
| `clean_scraped.py` | Strips nav/footer noise from scraped text |
| `ingest.py` | Chunks text, creates embeddings, upserts to Pinecone |
| `query.py` | Search interface — use this in your app |
| `scraped_pages.json` | Raw scraped content (local cache, not in git) |
| `.env` | API keys and credentials — never share or commit this |
| `requirements.txt` | All Python dependencies |
