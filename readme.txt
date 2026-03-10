================================================================
  FLIK ULTIMATE – AI TRAINING PLANNER  |  Workshop Project
================================================================

START HERE: Read WORKSHOP_ASSIGNMENT.md first.
It explains the challenge, what you need to build, and how to get started.

----------------------------------------------------------------
FOLDER OVERVIEW
----------------------------------------------------------------

WORKSHOP_ASSIGNMENT.md   <-- START HERE
  The challenge brief. Written as customer requirements.
  Explains the problem, what to build, and the bonus goal.

TECHNICAL_DETAILS.md
  Everything technical: how to query the vector store,
  how the website was scraped, and how to rebuild the
  knowledge base from scratch if needed.

query.py
  The search interface. Import search() into your app,
  or run it directly from the command line.

  Usage:
    python query.py "how does cup defence work?"
    python query.py   (interactive mode)

ingest.py
  Chunks, embeds, and uploads content to Pinecone.
  Only needed if you are rebuilding the vector store.

scraper.py
  Crawls flikulti.com (requires login) and saves all
  training material to scraped_pages.json.
  Only needed if you are rebuilding the vector store.

clean_scraped.py
  Strips navigation noise from scraped pages.
  Run after scraper.py, before ingest.py.

requirements.txt
  All Python dependencies. Install with:
    pip install -r requirements.txt

.env  (not shared — ask Maartje)
  API keys for Pinecone and OpenAI, and Flik login credentials.
  Required for both querying and rebuilding the knowledge base.

scraped_pages.json  (not in git)
  Local cache of all scraped Flik content (768 pages, ~5MB).
  Used as input for ingest.py.

----------------------------------------------------------------
QUICK START FOR WORKSHOP PARTICIPANTS
----------------------------------------------------------------

1. Read WORKSHOP_ASSIGNMENT.md
2. Ask Maartje for the .env file (API keys)
3. Install dependencies:  pip install pinecone openai python-dotenv
4. Test the vector store:  python query.py "how does cup defence work?"
5. Read TECHNICAL_DETAILS.md if you want to understand how it works
6. Build your solution on top of the search() function in query.py

----------------------------------------------------------------
