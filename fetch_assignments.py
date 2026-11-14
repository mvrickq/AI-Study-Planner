import asyncio
import json
import os
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.async_api import async_playwright

# Load environment variables from .env
load_dotenv()

BASE_URL = os.getenv("BLACKBOARD_BASE_URL")
AUTH_STATE = "auth_state.json"

DATA_DIR = Path("data")
ASSIGNMENTS_JSON = DATA_DIR / "assignments.json"
DEBUG_HTML = DATA_DIR / "debug_stream.html"


async def fetch_html() -> str:
    """
    Use Playwright with the saved auth_state.json to open the Blackboard
    Activity Stream page and return the full HTML content.
    """
    if not BASE_URL:
        raise ValueError("BLACKBOARD_BASE_URL not set in .env")
    if not os.path.exists(AUTH_STATE):
        raise FileNotFoundError(
            f"{AUTH_STATE} not found. Run save_blackboard_session.py first."
        )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(storage_state=AUTH_STATE)
        page = await context.new_page()

        # Ultra Activity Stream URL
        stream_url = f"{BASE_URL}/ultra/stream"
        print(f"Navigating to {stream_url} ...")
        await page.goto(stream_url)
        await page.wait_for_load_state("networkidle")

        # Scroll to force lazy-loaded items to render
        await page.mouse.wheel(0, 4000)
        await page.wait_for_timeout(2000)

        html = await page.content()
        await browser.close()
        return html


def parse_assignments(html: str):
    """
    Parse assignments from the Activity Stream HTML.

    This version is designed to match the structure in your screenshot:
    - Each assignment appears as a vertical block with:
        * relative time (e.g. "6 hours 5:50 PM")
        * course code (e.g. "COM-270-904 - FA 25-26")
        * "Due: <assignment title>"
        * "Due Date: <date/time>"

    We:
      * look at all <li> elements
      * keep only those containing 'Due:' or 'Due Date'
      * extract:
          - course (best guess)
          - title (between 'Due:' and 'Due Date')
          - due date (after 'Due Date:')
    """
    soup = BeautifulSoup(html, "lxml")

    # Just for sanity check: print page title
    page_title = soup.title.string.strip() if soup.title else "No <title>"
    print(f"Page <title>: {page_title}")

    items = []

    li_elements = soup.find_all("li")
    print(f"Found {len(li_elements)} <li> elements total")

    for li in li_elements:
        text = li.get_text(" ", strip=True)

        # We only care about list items that mention a due string
        if not any(keyword in text for keyword in ["Due:", "Due Date"]):
            continue

        # Basic debug snippet
        print("Potential assignment block:", text[:160])

        full_text = text

        # --- COURSE ---
        # Very rough heuristic: find the first token that looks like ABC-123-456
        course = None
        tokens = full_text.split()

        for tok in tokens:
            # Example: "COM-270-904"
            parts = tok.split("-")
            if len(parts) >= 2 and parts[0].isalpha() and parts[0].isupper():
                course = tok
                break

        if course is None:
            course = "Unknown Course"

        # --- TITLE ---
        title = "Untitled"
        if "Due:" in full_text:
            after_due = full_text.split("Due:", 1)[1]
            if "Due Date" in after_due:
                title = after_due.split("Due Date", 1)[0].strip()
            else:
                title = after_due.strip()

        # --- DUE DATE ---
        due = "Unknown"
        if "Due Date" in full_text:
            after_due_date = full_text.split("Due Date:", 1)[1]
            due = after_due_date.strip()

        items.append(
            {
                "course": course,
                "title": title,
                "due": due,
                "raw": full_text,  # keep raw text for debugging
            }
        )

    print(f"Parsed {len(items)} assignment-like items.")
    return items


async def main():
    DATA_DIR.mkdir(exist_ok=True)

    html = await fetch_html()

    # Save the raw HTML so you can open it in a browser for debugging
    with DEBUG_HTML.open("w", encoding="utf-8") as f:
        f.write(html)
    print(f"💾 Saved raw HTML to {DEBUG_HTML}")

    assignments = parse_assignments(html)

    with ASSIGNMENTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(assignments, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(assignments)} assignments to {ASSIGNMENTS_JSON}")


if __name__ == "__main__":
    asyncio.run(main())
