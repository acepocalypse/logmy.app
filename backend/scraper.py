"""
scraper.py
==========
Fetches a job‑posting URL and returns (raw_text, meta_dict).
Now accepts the loaded spaCy NLP model for its insight extraction.
"""
from __future__ import annotations

import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urlparse
import logging
import re
# import spacy # No longer load spacy here
import dateparser

logger = logging.getLogger(__name__) # Use Flask's app.logger or configure separately

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
}

class ScrapeError(Exception):
    """Raised when a page cannot be fetched or parsed."""

# NLP insight extraction now accepts the loaded nlp_model
def extract_insights_from_description(desc: str, nlp_model) -> dict:
    insights = {"salary": None, "deadline": None, "timeframe": None, "start_date": None}
    if not desc or not nlp_model:
        return insights
    
    doc = nlp_model(desc) # Use the passed-in nlp_model

    # Salary: Improved regex to capture ranges and common currency symbols
    # Looks for patterns like $50k-$60k, $70,000 - $80,000, €60K, £55000
    salary_patterns = [
        r'[\$€£]\s?\d{1,3}(?:[,\.\s]?\d{3})*\s?[Kk]?\s?(?:(?:to|-|–)\s?[\$€£]?\s?\d{1,3}(?:[,\.\s]?\d{3})*\s?[Kk]?)?', # e.g., $50k-$60k, $70,000 - $80,000
        r'\d{1,3}(?:[,\.\s]?\d{3})*\s?[Kk]?\s?[\$€£]\s?(?:(?:to|-|–)\s?\d{1,3}(?:[,\.\s]?\d{3})*\s?[Kk]?\s?[\$€£]?)?'  # e.g., 50k-60k USD
    ]
    raw_salaries = []
    for pattern in salary_patterns:
        raw_salaries.extend(re.findall(pattern, desc))
    
    if raw_salaries:
        # Simple approach: take the first found pattern. Could be refined.
        insights["salary"] = raw_salaries[0].strip()
        logger.debug(f"Found potential salary: {insights['salary']}")


    # Deadline: Using SpaCy entities and dateparser
    # Look for sentences containing deadline-related keywords
    deadline_keywords = ["apply by", "application deadline", "deadline", "applications close", "closing date"]
    potential_deadline_texts = []
    for sent in doc.sents:
        if any(keyword in sent.text.lower() for keyword in deadline_keywords):
            potential_deadline_texts.append(sent.text)
            # Also check for DATE entities within these sentences
            for ent in sent.ents:
                if ent.label_ == "DATE":
                    potential_deadline_texts.append(ent.text)
    
    if potential_deadline_texts:
        # Try to parse the found texts. dateparser is quite flexible.
        # Sort by length descending to try more specific phrases first.
        for text_to_parse in sorted(list(set(potential_deadline_texts)), key=len, reverse=True):
            try:
                # Give context that we are looking for a future date
                parsed_date = dateparser.parse(text_to_parse, settings={'PREFER_DATES_FROM': 'future', 'STRICT_PARSING': False})
                if parsed_date:
                    insights["deadline"] = parsed_date.strftime("%Y-%m-%d")
                    logger.debug(f"Parsed deadline: {insights['deadline']} from text: '{text_to_parse}'")
                    break # Take the first successfully parsed deadline
            except Exception as e:
                logger.warning(f"Dateparser failed for deadline string '{text_to_parse}': {e}")
    

    # Timeframe (e.g., "12 weeks", "Summer 2025", "3 months")
    timeframe_match = re.search(
        r'\b(\d{1,2}\s?(?:weeks?|months?)|(?:summer|fall|winter|spring)\s+\d{4})\b', 
        desc, 
        re.IGNORECASE
    )
    if timeframe_match:
        insights["timeframe"] = timeframe_match.group(1).strip()
        logger.debug(f"Found timeframe: {insights['timeframe']}")

    # Start Date: Similar to deadline, look for keywords and DATE entities
    start_date_keywords = ["start date", "starting on", "begins on", "available from"]
    potential_start_date_texts = []
    for sent in doc.sents:
        if any(keyword in sent.text.lower() for keyword in start_date_keywords):
            potential_start_date_texts.append(sent.text)
            for ent in sent.ents:
                if ent.label_ == "DATE":
                    potential_start_date_texts.append(ent.text)
    
    if potential_start_date_texts:
        for text_to_parse in sorted(list(set(potential_start_date_texts)), key=len, reverse=True):
            try:
                parsed_date = dateparser.parse(text_to_parse, settings={'PREFER_DATES_FROM': 'future', 'STRICT_PARSING': False})
                if parsed_date:
                    insights["start_date"] = parsed_date.strftime("%Y-%m-%d")
                    logger.debug(f"Parsed start date: {insights['start_date']} from text: '{text_to_parse}'")
                    break
            except Exception as e:
                logger.warning(f"Dateparser failed for start date string '{text_to_parse}': {e}")

    return insights

# ---------------------------------------------------------------------------
# Network fetch
# ---------------------------------------------------------------------------
def fetch_page(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        return resp.text
    except requests.exceptions.Timeout:
        logger.error(f"Timeout while fetching {url}")
        raise ScrapeError(f"Timeout: Could not fetch {url} within 15 seconds.")
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error fetching {url}: {http_err}")
        raise ScrapeError(f"HTTP error: {http_err.response.status_code} for {url}.")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception fetching {url}: {req_err}")
        raise ScrapeError(f"Network error: Failed to fetch {url} ({req_err}).")
    except Exception as exc:
        logger.error(f"Generic exception fetching {url}: {exc}", exc_info=True)
        raise ScrapeError(f"Failed to fetch {url}: {exc}")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def extract_text(html: str, url: str, nlp_model) -> tuple[str, dict]: # Accept nlp_model
    dom = BeautifulSoup(html, "lxml")
    host = urlparse(url).netloc.lower()

    # Site-specific extractors
    if "indeed." in host:
        return _extract_indeed(dom, nlp_model) # Pass nlp_model
    elif "linkedin." in host:
        return _extract_linkedin(dom, nlp_model) # Pass nlp_model
    
    # Generic fallback (less accurate metadata)
    logger.info(f"Using generic fallback extractor for {url}")
    body_text = ""
    # Try common main content tags
    main_content_selectors = ['article', 'main', '[role="main"]', '.job-description', '#jobDescriptionText', '.description__text']
    for selector in main_content_selectors:
        main_el = dom.select_one(selector)
        if main_el:
            body_text = _text_of(main_el)
            break
    if not body_text: # Fallback to whole body if specific selectors fail
        body_text = _text_of(dom.body) if dom.body else dom.get_text(" ", strip=True)
    
    # For generic fallback, meta will be minimal or rely on parser.py
    meta = {} 
    # Optionally, run a simplified version of extract_insights_from_description for generic pages
    if body_text:
         insights = extract_insights_from_description(body_text, nlp_model)
         meta.update(insights)

    return body_text, meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _text_of(el: Tag | None) -> str:
    return el.get_text(" ", strip=True) if el else ""


# ---------------- Indeed ----------------
def _extract_indeed(dom: BeautifulSoup, nlp_model) -> tuple[str, dict]: # Accept nlp_model
    logger.debug("Extracting content from Indeed page.")
    title = _text_of(dom.select_one("h1.jobsearch-JobInfoHeader-title"))
    # Indeed often has company and location in a specific div
    company_info_div = dom.select_one('div[data-testid="job- компании-name"]') # Using a more robust selector if available
    company = ""
    location = ""

    if company_info_div: # New selector for company
        company = _text_of(company_info_div)
    else: # Fallback to old selector
        company = _text_of(dom.select_one("div.jobsearch-InlineCompanyRating div:nth-child(1)"))
    
    location_div = dom.select_one('div[data-testid="job-location"]')
    if location_div: # New selector for location
        location = _text_of(location_div)
    else: # Fallback
        location_data_qa = dom.select_one('div[data-qa="job- ठंडा-location"]') # Another possible selector
        if location_data_qa:
            location = _text_of(location_data_qa.find('span', recursive=False)) # Get direct span text
        else:
            location = _text_of(dom.select_one("div.jobsearch-InlineCompanyRating div:nth-child(3)"))


    body_el = dom.select_one("div#jobDescriptionText")
    body = _text_of(body_el)

    meta = {"company": company.strip(), "position": title.strip(), "location": location.strip()}
    if body: # Only extract insights if body text is found
        insights = extract_insights_from_description(body, nlp_model)
        meta.update(insights)
        
    logger.debug(f"Indeed extracted meta: {meta}")
    return body or dom.get_text(" ", strip=True), meta


# ---------------- LinkedIn ----------------
def _extract_linkedin(dom: BeautifulSoup, nlp_model) -> tuple[str, dict]: # Accept nlp_model
    logger.debug("Extracting content from LinkedIn page.")
    title    = _text_of(dom.select_one("h1.top-card-layout__title, h1.job-title, .job-details-jobs-unified-top-card__job-title"))
    company  = _text_of(dom.select_one("a.topcard__org-name-link, a.job-card-container__company-name, .job-details-jobs-unified-top-card__company-name a"))
    location = _text_of(dom.select_one("span.topcard__flavor--bullet, .job-card-container__metadata-item, .job-details-jobs-unified-top-card__primary-description-container > div:nth-child(2)")) # More selectors for location

    body_el  = (dom.select_one("div.description__text")
                or dom.select_one("div.show-more-less-html__markup")
                or dom.select_one("#job-details") # Common container for job details
                or dom.select_one(".jobs-description__content > .jobs-box__html-content") # New LinkedIn UI
                )
    body = ""
    if body_el:
        # LinkedIn often has "Show more" buttons. This simple extraction won't click them.
        # For complex cases, Selenium/Playwright might be needed.
        body = _text_of(body_el)
    
    meta = {"company": company.strip(), "position": title.strip(), "location": location.strip()}
    if body:
        insights = extract_insights_from_description(body, nlp_model)
        meta.update(insights)

    logger.debug(f"LinkedIn extracted meta: {meta}")
    return body or dom.get_text(" ", strip=True), meta
