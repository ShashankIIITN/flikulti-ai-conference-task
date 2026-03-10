"""
query.py

Search your Pinecone vector store with a natural language question.

Usage:
    python query.py "What is the theory of harmony?"
    python query.py  (interactive mode)
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME", "flikulti-theory")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL  = "text-embedding-3-small"
TOP_K            = 5

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


def search(query: str, top_k: int = TOP_K) -> list[dict]:
    # Embed the query
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    query_vector = response.data[0].embedding

    # Query Pinecone
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    return results.matches


def print_results(matches: list, query: str):
    print(f"\nQuery: {query}")
    print("=" * 60)
    for i, match in enumerate(matches, 1):
        meta = match.metadata
        print(f"\n[{i}] Score: {match.score:.4f}")
        print(f"    URL:   {meta.get('url', '')}")
        print(f"    Title: {meta.get('title', '')}")
        print(f"    Text:  {meta.get('text', '')[:300]}...")
    print()


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        matches = search(query)
        print_results(matches, query)
    else:
        print("Flikulti Theory Search  (type 'quit' to exit)\n")
        while True:
            query = input("Search: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            matches = search(query)
            print_results(matches, query)


if __name__ == "__main__":
    main()
