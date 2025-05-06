"""
scraper.py
==========
Fetches a job‑posting URL (Indeed, LinkedIn, or generic) and returns a tuple of
(raw_text, meta_dict).

* raw_text – long text we send to parser.parse() for NLP extraction
* meta_dict – immediate fields scraped via CSS selectors (company, title, …)

Add site extractors easily: write _extract_<site>(dom) returning (text, meta)
"""
from __future__ import annotations

import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urlparse

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
}


class ScrapeError(Exception):
    """Raised when a page cannot be fetched or parsed."""


# ---------------------------------------------------------------------------
# Network fetch
# ---------------------------------------------------------------------------
def fetch_page(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        raise ScrapeError(f"Failed to fetch {url}: {exc}") from exc


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def extract_text(html: str, url: str) -> tuple[str, dict]:
    dom = BeautifulSoup(html, "lxml")
    host = urlparse(url).netloc.lower()

    if "indeed." in host:
        return _extract_indeed(dom)
    elif "linkedin." in host:
        return _extract_linkedin(dom)
    # TODO: add more: glassdoor, lever, greenhouse, etc.

    # Generic fallback – just grab body text
    text = dom.get_text(" ", strip=True)
    return text, {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _text_of(el: Tag | None) -> str:
    return el.get_text(" ", strip=True) if el else ""


# ---------------- Indeed ----------------
def _extract_indeed(dom: BeautifulSoup) -> tuple[str, dict]:
    title = _text_of(dom.select_one("h1.jobsearch-JobInfoHeader-title"))
    company = _text_of(dom.select_one("div.jobsearch-InlineCompanyRating div:nth-child(1)"))
    location = _text_of(dom.select_one("div.jobsearch-InlineCompanyRating div:nth-child(3)"))
    body_el = dom.select_one("div#jobDescriptionText")
    body = _text_of(body_el)

    meta = {"company": company, "position": title, "location": location}
    return body or dom.get_text(" ", strip=True), meta


# ---------------- LinkedIn ----------------
def _extract_linkedin(dom: BeautifulSoup) -> tuple[str, dict]:
    title = _text_of(dom.select_one("h1.top-card-layout__title"))
    company = _text_of(dom.select_one("a.topcard__org-name-link"))
    location = _text_of(dom.select_one("span.topcard__flavor--bullet"))
    body_el = dom.select_one("div.description__text") or dom.select_one("div.show-more-less-html__markup")
    body = _text_of(body_el)

    meta = {"company": company, "position": title, "location": location}
    return body or dom.get_text(" ", strip=True), meta
