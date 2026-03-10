"""
clean_scraped.py

Post-processes scraped_pages.json to strip navigation noise from each page's text.
Overwrites scraped_pages.json in place.

Usage:
    python clean_scraped.py
"""

import json
import re

INPUT_FILE = "scraped_pages.json"

# On flikulti pages, a language-switcher block (lots of flag image links) always
# precedes the article. It ends with a quoted language name like "Ukrainian")\n
# followed by the article's H1 heading. We find the LAST such transition.
NAV_END_RE = re.compile(r'"[A-Z][a-z]+"(?:\)|\))\n+#')

# Footer noise starts at one of these markers
FOOTER_CUTOFF_PATTERNS = [
    r"\nPrevious article\n",
    r"\n\*\*Previous article",
    r"\nLinked Drills:",
    r"\n---\n.*?Facebook",
]


def strip_nav(text: str) -> str:
    """Remove navigation header and footer noise from a flikulti page."""
    # --- Strip header nav ---
    # Find the last language-flag → heading transition
    best = None
    for m in NAV_END_RE.finditer(text):
        best = m
    if best:
        # Keep from the '#' character (start of the article heading)
        text = text[best.end() - 1:].lstrip("\n")

    # --- Strip footer noise ---
    cut_end = len(text)
    for pattern in FOOTER_CUTOFF_PATTERNS:
        m = re.search(pattern, text, re.DOTALL)
        if m and m.start() < cut_end:
            cut_end = m.start()

    text = text[:cut_end].rstrip()
    return text


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)

    print(f"[cleaner] Processing {len(pages)} pages ...")

    before_total = sum(len(p["text"]) for p in pages)
    cleaned = 0

    for page in pages:
        original_len = len(page["text"])
        page["text"] = strip_nav(page["text"])
        if len(page["text"]) < original_len:
            cleaned += 1

    after_total = sum(len(p["text"]) for p in pages)
    removed_kb = (before_total - after_total) / 1024

    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print(f"[cleaner] Cleaned {cleaned}/{len(pages)} pages")
    print(f"[cleaner] Removed {removed_kb:.1f} KB of nav noise")
    print(f"[cleaner] Saved -> {INPUT_FILE}")


if __name__ == "__main__":
    main()
