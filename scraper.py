"""
scraper.py

Logs into flikulti.com, discovers all subpages across configured sections,
and scrapes their content.
Output: scraped_pages.json  (list of {url, title, text})

Usage:
    python scraper.py
"""

import asyncio
import json
import os
import re
from urllib.parse import urljoin, urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from dotenv import load_dotenv

load_dotenv()

EMAIL = os.getenv("FLIKULTI_EMAIL")
PASSWORD = os.getenv("FLIKULTI_PASSWORD")
OUTPUT_FILE = "scraped_pages.json"

# All sections to crawl
START_URLS = [
    "https://www.flikulti.com/theory/",
    "https://www.flikulti.com/sessions/",
    "https://www.flikulti.com/drills/",
    "https://www.flikulti.com/video/",
    "https://www.flikulti.com/sc-dashboard/",
]

# URL path prefixes that are allowed to be followed
ALLOWED_PREFIXES = tuple(
    urlparse(u).path.rstrip("/") for u in START_URLS
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_allowed_url(url: str) -> bool:
    """Only follow links under the configured sections."""
    parsed = urlparse(url)
    return (
        parsed.netloc == "www.flikulti.com"
        and any(parsed.path.startswith(prefix) for prefix in ALLOWED_PREFIXES)
    )


def extract_links(html: str, base_url: str) -> set[str]:
    """Pull all href values from raw HTML."""
    urls = set()
    for match in re.finditer(r'href=["\']([^"\']+)["\']', html):
        raw = match.group(1)
        full = urljoin(base_url, raw).split("#")[0].rstrip("/")
        if is_allowed_url(full):
            urls.add(full)
    return urls


# ---------------------------------------------------------------------------
# Login JS  (adjust selectors if flikulti uses different field names)
# ---------------------------------------------------------------------------

LOGIN_JS = f"""
(async () => {{
    // Wait for login form
    await new Promise(r => setTimeout(r, 1500));

    // Flikulti uses WordPress login fields
    const emailField = document.querySelector('#flik_user_login');
    const passField  = document.querySelector('#flik_user_pass, input[name="flik_user_pass"], input[type="password"]');
    const submit     = document.querySelector('#wp-submit, input[type="submit"], button[type="submit"]');

    if (emailField) emailField.value = {json.dumps(EMAIL)};
    if (passField)  passField.value  = {json.dumps(PASSWORD)};
    if (submit)     submit.click();

    // Wait for redirect after login
    await new Promise(r => setTimeout(r, 3000));
}})();
"""

# ---------------------------------------------------------------------------
# Main crawler
# ---------------------------------------------------------------------------

async def scrape() -> list[dict]:
    browser_cfg = BrowserConfig(headless=True, verbose=False)

    pages: list[dict] = []
    visited: set[str] = set()
    queue: list[str] = [u.rstrip("/") for u in START_URLS]

    # Flikulti WordPress login page
    login_url = "https://www.flikulti.com/wp-login.php"

    async with AsyncWebCrawler(config=browser_cfg) as crawler:

        # --- Step 1: log in (reuse the same browser session via session_id) ---
        print(f"[scraper] Logging in as {EMAIL} ...")
        login_result = await crawler.arun(
            url=login_url,
            session_id="flikulti",
            js_code=LOGIN_JS,
            config=CrawlerRunConfig(wait_until="domcontentloaded", page_timeout=30000),
        )
        # Verify login by checking the page we land on
        if login_result.url and "wp-login" in login_result.url:
            print("[scraper] WARNING: Still on login page — check credentials or login selectors.")
        else:
            print(f"[scraper] Login redirected to: {login_result.url}")
        print("[scraper] Login done. Starting crawl ...")

        # --- Step 2: BFS crawl of all pages across configured sections ---
        while queue:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            print(f"[scraper] Scraping: {url}")
            result = await crawler.arun(
                url=url,
                session_id="flikulti",          # reuse authenticated session
                config=CrawlerRunConfig(
                    wait_until="domcontentloaded",  # more reliable than networkidle
                    page_timeout=30000,
                    excluded_selector="nav, header, footer, .sidebar, .menu, .navigation, .widget, #gtranslate_wrapper, .language-switcher",
                    excluded_tags=["nav", "header", "footer", "script", "style", "noscript"],
                ),
            )

            if not result.success:
                print(f"  [!] Failed: {result.error_message}")
                continue

            # Store the page
            pages.append({
                "url": url,
                "title": result.metadata.get("title", ""),
                "text": result.markdown or result.cleaned_html or "",
            })

            # Discover new links
            new_links = extract_links(result.html or "", url) - visited
            queue.extend(new_links)
            print(f"  [+] Found {len(new_links)} new links. Queue size: {len(queue)}")

    return pages


async def main():
    if not EMAIL or EMAIL == "your@email.com":
        print("[!] Set FLIKULTI_EMAIL and FLIKULTI_PASSWORD in your .env file first.")
        return

    pages = await scrape()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print(f"\n[scraper] Done. Scraped {len(pages)} pages -> {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
