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

# ─── NLP helper that mines salary / deadline / start-date ──────────────
import re, spacy, dateparser
NLP = spacy.load("en_core_web_sm")

def extract_insights_from_description(desc: str) -> dict:
    insights = dict.fromkeys(["salary", "deadline", "timeframe", "start_date"])
    if not desc:
        return insights
    doc = NLP(desc)

    # salary
    sal = re.findall(r'[$£€₹]\\s?\\d{2,3}(?:[\\,\\d]{0,3})?', desc)
    if sal:
        nums = [int(re.sub(r'[^\\d]', '', s)) for s in sal][:2]
        insights["salary"] = (f"${nums[0]}–${nums[1]}" if len(nums) == 2 else f"${nums[0]}")
    # deadline
    for s in doc.sents:
        if any(k in s.text.lower() for k in ("apply by", "deadline")):
            d = dateparser.parse(s.text)
            if d: insights["deadline"] = d.strftime("%Y-%m-%d"); break
    # timeframe
    m = re.search(r'\\b(\\d{1,2}\\s?(?:weeks?|months?)|summer \\d{4}|fall \\d{4}|spring \\d{4})\\b', desc, re.I)
    if m: insights["timeframe"] = m.group(1)
    # start date
    # … (keep your regex logic)
    return insights

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
    title    = _text_of(dom.select_one("h1.top-card-layout__title"))
    company  = _text_of(dom.select_one("a.topcard__org-name-link"))
    location = _text_of(dom.select_one("span.topcard__flavor--bullet"))
    body_el  = (dom.select_one("div.description__text")
                or dom.select_one("div.show-more-less-html__markup"))
    body     = _text_of(body_el)

    # Run your NLP insight extractor on the description
    extra = extract_insights_from_description(body)

    meta = {
        "company":   company,
        "position":  title,
        "location":  location,
        **extra      # salary, deadline, timeframe, start_date
    }
    logger.debug("LinkedIn meta after NLP: %s", meta)
    return body or dom.get_text(" ", strip=True), meta