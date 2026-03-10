"""
ingest.py

Reads scraped_pages.json, chunks the text, creates OpenAI embeddings,
and upserts everything into a Pinecone index.

Usage:
    python ingest.py
"""

import json
import os
import time
import hashlib

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
INDEX_NAME         = os.getenv("PINECONE_INDEX_NAME", "flikulti-theory")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
INPUT_FILE         = "scraped_pages.json"
EMBEDDING_MODEL    = "text-embedding-3-small"
EMBEDDING_DIM      = 1536
CHUNK_SIZE         = 500    # words per chunk
CHUNK_OVERLAP      = 50     # words overlap between chunks
UPSERT_BATCH_SIZE  = 100

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 50]  # skip tiny chunks


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with OpenAI."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Pinecone setup
# ---------------------------------------------------------------------------

def get_or_create_index(pc: Pinecone) -> object:
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"[ingest] Creating Pinecone index '{INDEX_NAME}' ...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print("[ingest] Waiting for index to be ready...")
            time.sleep(2)
    else:
        print(f"[ingest] Using existing index '{INDEX_NAME}'")
    return pc.Index(INDEX_NAME)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load scraped data
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)
    print(f"[ingest] Loaded {len(pages)} pages from {INPUT_FILE}")

    # Init Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = get_or_create_index(pc)

    # Process pages
    all_vectors = []
    for page in pages:
        url   = page["url"]
        title = page.get("title", "")
        text  = page.get("text", "")

        if not text.strip():
            continue

        chunks = chunk_text(text)
        print(f"[ingest] {url}  ->  {len(chunks)} chunks")

        # Embed in batches of 20 (OpenAI rate limit friendly)
        for i in range(0, len(chunks), 20):
            batch = chunks[i : i + 20]
            embeddings = embed_texts(batch)

            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                chunk_id = hashlib.md5(f"{url}_{i+j}".encode()).hexdigest()
                all_vectors.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {
                        "url": url,
                        "title": title,
                        "chunk_index": i + j,
                        "text": chunk[:1000],   # Pinecone metadata limit
                    },
                })

    # Upsert to Pinecone in batches
    print(f"\n[ingest] Upserting {len(all_vectors)} vectors to Pinecone ...")
    for i in range(0, len(all_vectors), UPSERT_BATCH_SIZE):
        batch = all_vectors[i : i + UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch)
        print(f"  Upserted {min(i + UPSERT_BATCH_SIZE, len(all_vectors))}/{len(all_vectors)}")

    print(f"\n[ingest] Done. {len(all_vectors)} vectors stored in '{INDEX_NAME}'.")


if __name__ == "__main__":
    main()
