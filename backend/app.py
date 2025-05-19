# backend/app.py

from __future__ import annotations

import os
from datetime import datetime
from functools import wraps
import logging # Import logging

from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from dotenv import load_dotenv
import jwt
from jwt import InvalidTokenError

# Import your parser and scraper modules
from parser import parse as parse_job_text # Renamed to avoid conflict
from scraper import fetch_page, extract_text, ScrapeError

# For improved error handling on /submit
from postgrest.exceptions import APIError # Make sure this matches your supabase-py version's exception
import requests # If supabase-py uses requests and might raise its exceptions directly

# --- SpaCy Model Loading ---
# Load the SpaCy model ONCE when the application starts.
# This is a critical performance optimization.
import spacy
NLP_MODEL = None
try:
    NLP_MODEL = spacy.load("en_core_web_sm")
    logging.info("SpaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    logging.error(
        "Could not load SpaCy model 'en_core_web_sm'. "
        "Make sure it's downloaded (python -m spacy download en_core_web_sm) "
        "and the path is correct or it's in your virtual environment."
    )
    # You might want to exit or handle this more gracefully if NLP is essential
    # For now, the app will run but NLP features will fail if NLP_MODEL is None.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()
SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") # This should be the service_role key for backend operations
SUPABASE_JWT_SECRET  = os.getenv("SUPABASE_JWT_SECRET") # Used to decode JWT from frontend
JWT_ALG = "HS256"

if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_JWT_SECRET]):
    raise RuntimeError("Missing required environment variables for Supabase.")

# Initialize Supabase client with the service_role key for backend operations
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Configure Flask logging (optional, but good for seeing logs in Render)
if not app.debug:
    app.logger.setLevel(logging.INFO) # Log INFO and above in production
    # You can add handlers here, e.g., StreamHandler to see logs in console/Render
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    app.logger.addHandler(stream_handler)

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def get_current_user(fn):
    """Decorator to ensure request has a valid Supabase JWT."""
    @wraps(fn)
    def _wrapped(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            app.logger.warning("Auth error: Missing Bearer token")
            return jsonify({"error": "Missing or invalid Bearer token"}), 401
        token = auth_header.split(" ", 1)[1]
        try:
            # Decode using the JWT secret
            payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=[JWT_ALG], audience="authenticated")
            request.user_id = payload.get("sub")
            if not request.user_id:
                app.logger.warning(f"Auth error: No 'sub' (user ID) in JWT payload. Payload: {payload}")
                return jsonify({"error": "Invalid token payload"}), 401
        except InvalidTokenError as exc:
            app.logger.warning(f"Auth error: Invalid token: {exc}")
            return jsonify({"error": "Invalid or expired token", "detail": str(exc)}), 401
        except Exception as exc:
            app.logger.error(f"Auth error: Unexpected error during token decoding: {exc}", exc_info=True)
            return jsonify({"error": "Token processing error"}), 401
        return fn(*args, **kwargs)
    return _wrapped

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/parse", methods=["POST"]) # This endpoint seems unused based on frontend, but kept for completeness
def parse_endpoint():
    text = request.get_json(force=True).get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400
    if NLP_MODEL is None:
        app.logger.error("NLP model not loaded, cannot parse text.")
        return jsonify({"error": "NLP service not available"}), 503
    return jsonify(parse_job_text(text, NLP_MODEL)), 200 # Pass loaded NLP_MODEL


@app.route("/link-parse", methods=["POST"])
def link_parse_endpoint():
    url = request.get_json(force=True).get("url", "").strip()
    if not url:
        return jsonify({"error": "Missing 'url' field"}), 400

    if NLP_MODEL is None:
        app.logger.error("NLP model not loaded, cannot parse link.")
        return jsonify({"error": "NLP service not available"}), 503

    try:
        app.logger.info(f"Fetching page: {url}")
        html_content = fetch_page(url)
        app.logger.info(f"Extracting text from HTML for URL: {url}")
        # Pass NLP_MODEL to extract_text if it directly uses it, or to functions it calls
        raw_text, meta = extract_text(html_content, url, NLP_MODEL) # Pass NLP_MODEL
        
        app.logger.info(f"Parsing extracted text for URL: {url}")
        data = parse_job_text(raw_text, NLP_MODEL) # Pass NLP_MODEL
        
        # Combine metadata from scraping and parsing.
        # Ensure 'meta' (from scraping) doesn't overwrite more accurate 'data' (from parsing)
        # unless 'meta' has fields not present in 'data'.
        # A simple way is to prioritize 'data' and add missing keys from 'meta'.
        final_data = meta.copy() # Start with scraper's meta
        final_data.update(data)  # Update with parser's data (parser might be more detailed for some fields)
        # Or, more explicitly:
        # final_data = data.copy()
        # for key, value in meta.items():
        #     if key not in final_data or not final_data[key]: # If key not in parsed data or parsed data for key is empty
        #         final_data[key] = value

        final_data["job_url"] = url # Ensure job_url is always included
        app.logger.info(f"Successfully parsed link: {url}. Data: {final_data}")
        return jsonify(final_data), 200
    except ScrapeError as exc:
        app.logger.error(f"Scraping error for {url}: {exc}")
        return jsonify({"error": str(exc)}), 422 # Unprocessable Entity
    except Exception as exc:
        app.logger.error(f"Unexpected error during link parsing for {url}: {exc}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(exc)}"}), 500


@app.route("/submit", methods=["POST"])
@get_current_user # Ensures request.user_id is set
def submit_endpoint():
    data = request.get_json(force=True)
    app.logger.info(f"Submit endpoint called by user: {request.user_id} with data: {data}")

    # Prepare application data for Supabase
    application = {
        "user_id": request.user_id, # Use the authenticated user's ID
        "company": data.get("company") or None, # Use None for empty strings if column is nullable
        "position": data.get("position") or None,
        "location": data.get("location") or None,
        "job_type": data.get("job_type") if data.get("job_type") else None, # Ensure empty string becomes NULL
        "application_date": data.get("application_date") or datetime.utcnow().date().isoformat(),
        "deadline": data.get("deadline") if data.get("deadline") else None, # Ensure empty string becomes NULL
        "status": data.get("status") or "Applied",
        "job_url": data.get("job_url") or None,
        "notes": data.get("notes") or None,
        # created_at and updated_at are usually handled by Supabase default values or triggers
        # "created_at": datetime.utcnow().isoformat(),
        # "updated_at": datetime.utcnow().isoformat(),
    }
    # Filter out None values if your Supabase insert prefers not to have them for optional fields
    # application_payload = {k: v for k, v in application.items() if v is not None}
    # However, supabase-py usually handles None as NULL correctly.

    try:
        app.logger.info(f"Attempting to insert application for user {request.user_id}: {application}")
        # The user_id from the JWT is used here.
        # Ensure your RLS policies in Supabase allow this user to insert.
        # The service_role key bypasses RLS by default, but it's good practice
        # to design RLS assuming user context if you ever switch from service_role for inserts.
        response = supabase.table("applications").insert(application).execute()

        # More robust error checking for Supabase response
        # The structure of `response` can vary slightly between supabase-py versions.
        # Inspect `response` object in case of errors if this doesn't work.
        
        # For newer versions that use Postgrest APIError for failures:
        # The execute() call itself will raise APIError if status_code >= 400 (as seen in logs)
        # So, the primary check is the try-except block for APIError.

        # If execute() doesn't raise for some non-2xx codes or if there's an error object in the response
        if hasattr(response, 'error') and response.error:
            app.logger.error(f"Supabase insert error object: {response.error} for payload: {application}")
            error_message = response.error.message if hasattr(response.error, 'message') else "Unknown Supabase error"
            return jsonify({"error": error_message, "details": str(response.error)}), 500

        # Fallback check for status code if not an APIError and no response.error
        if hasattr(response, 'status_code') and response.status_code >= 300: # Check for any non-success (>=300)
            app.logger.error(f"Supabase insert returned status {response.status_code}: {response.data or response.text} for payload: {application}")
            error_data = response.data if hasattr(response, 'data') and response.data else response.text
            return jsonify({"error": "Failed to save to database.", "details": error_data}), response.status_code

        # Success case
        if response.data:
            app.logger.info(f"Successfully inserted application: {response.data[0]['id']}")
            return jsonify({"success": True, "data": response.data}), 201
        else:
            app.logger.warning(f"Supabase insert successful but no data returned. Response: {response} for payload: {application}")
            return jsonify({"success": True, "message": "Application saved, but no confirmation data."}), 201


    except APIError as e: # Specific to postgrest.exceptions.APIError
        app.logger.error(f"Supabase APIError during insert for user {request.user_id}: {e.message} (Code: {e.code}, Details: {e.details}, Hint: {e.hint}) for payload: {application}", exc_info=False) # Log exc_info=False as e contains details
        # Provide a more user-friendly message for constraint violations
        if e.code == '23514': # Check constraint violation
             # Try to parse the human-readable message if available
            error_message = e.message if e.message else "Data validation error."
            if "applications_job_type_check" in str(e.details) or "applications_job_type_check" in error_message:
                error_message = "Invalid 'Job Type' selected. Please choose a valid option."
            return jsonify({"error": error_message, "db_code": e.code}), 400 # Bad Request
        return jsonify({"error": f"Database error: {e.message}", "db_code": e.code}), 500 # Internal Server Error for other DB errors
    except requests.exceptions.RequestException as e: # Catch network errors if supabase-py uses requests directly
        app.logger.error(f"Network RequestException during Supabase insert for user {request.user_id}: {e}", exc_info=True)
        return jsonify({"error": "Network error connecting to database service."}), 504 # Gateway Timeout
    except Exception as e: # Catch any other unexpected errors
        app.logger.error(f"Unexpected error during Supabase insert for user {request.user_id}: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred while saving."}), 500


@app.route("/health")
def health():
    return "OK", 200

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Basic logging config for when running with `python app.py`
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False) # debug=False for production on Render
```python
# backend/parser.py
"""
parser.py – spaCy‑powered job‑posting parser
===========================================
Now accepts the loaded spaCy NLP model as an argument to parse().
"""
import re
import time
import logging

import dateparser
# import spacy # No longer load spacy here, it's passed in
# from spacy.language import Language # Not needed if nlp object is passed

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s [parser] %(message)s") # Configured in app.py
logger = logging.getLogger(__name__) # Use Flask's app.logger or configure separately if run standalone

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
COMPANY_RE  = re.compile(r"(?:Company|Employer)[:\-]?\s*(.+)", re.I)
POSITION_RE = re.compile(r"(?:Job Title|Title|Position|Role)[:\-]?\s*(.+)", re.I)
LOCATION_RE = re.compile(r"(?:Location)[:\-]?\s*(.+)", re.I)
DEADLINE_RE = re.compile(r"(?:Apply\s*by|Deadline)[:\-]?\s*([^\n]+)", re.I)

# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

def _extract_deadline(text: str) -> str:
    m = DEADLINE_RE.search(text)
    if m:
        try:
            parsed = dateparser.parse(m.group(1).strip(), settings={"PREFER_DATES_FROM": "future", "STRICT_PARSING": False})
            if parsed:
                return parsed.date().isoformat()
        except Exception as e:
            logger.warning(f"Dateparser failed for deadline string '{m.group(1).strip()}': {e}")
        # Fallback to the raw string if parsing fails or returns None
        return m.group(1).strip()
    return ""


def _extract_company(doc): # doc is now a spaCy Doc object
    m = COMPANY_RE.search(doc.text)
    if m:
        return m.group(1).strip()
    for ent in doc.ents:
        if ent.label_ == "ORG":
            return ent.text.strip()
    return ""


def _extract_location(doc): # doc is now a spaCy Doc object
    m = LOCATION_RE.search(doc.text)
    if m:
        return m.group(1).strip()
    # Prioritize GPE (Geopolitical Entity) over LOC (Location, less specific)
    gpe_entities = [ent.text.strip() for ent in doc.ents if ent.label_ == "GPE"]
    if gpe_entities:
        return ", ".join(gpe_entities) # Join if multiple GPEs found (e.g., "City, State")
    
    loc_entities = [ent.text.strip() for ent in doc.ents if ent.label_ == "LOC"]
    if loc_entities:
        return ", ".join(loc_entities)
    return ""


def _extract_position(doc): # doc is now a spaCy Doc object
    m = POSITION_RE.search(doc.text)
    if m:
        return m.group(1).strip()
    
    # Look for noun chunks that might be job titles, especially near the beginning
    # Consider patterns like "Job Title:", "Position:", or capitalized phrases
    # This is a simple heuristic, more advanced title extraction is complex
    
    # Try to find noun chunks that are likely titles (e.g., capitalized, contain keywords)
    possible_titles = []
    for chunk in doc[:min(100, len(doc))].noun_chunks: # Search in the first 100 tokens
        text = chunk.text.strip()
        if len(text.split()) <= 5 and any(word.istitle() or word.isupper() for word in text.split()): # Heuristic: up to 5 words, some capitalized
            if any(kw in text.lower() for kw in ["intern", "analyst", "engineer", "developer", "manager", "specialist", "coordinator"]):
                possible_titles.append(text)
    
    if possible_titles:
        # Prefer shorter, more specific titles if multiple are found
        return min(possible_titles, key=len)

    # Fallback to the largest noun chunk in the beginning if no better title found
    slice_doc = doc[: min(50, len(doc))] # Search in first 50 tokens
    noun_chunks = sorted([nc.text.strip() for nc in slice_doc.noun_chunks if len(nc.text.strip().split()) <= 5], key=len, reverse=True)
    if noun_chunks:
        return noun_chunks[0]
        
    return ""

# ---------------------------------------------------------------------------
# Public parse() function
# ---------------------------------------------------------------------------

def parse(text: str, nlp_model) -> dict: # Accept nlp_model as argument
    """Return dict with keys company, position, location, deadline."""
    if not nlp_model:
        logger.error("NLP model is None, cannot perform parsing.")
        return {"company": "", "position": "", "location": "", "deadline": ""}

    logger.debug(f"parser.parse() called with text length={len(text)} | preview={text[:120]!r}")
    start_time = time.perf_counter()
    
    doc = nlp_model(text) # Use the passed-in nlp_model
    
    result = {
        "company":  _extract_company(doc),
        "position": _extract_position(doc),
        "location": _extract_location(doc),
        "deadline": _extract_deadline(text), # Deadline parsing doesn't always need the full doc
    }
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.debug(f"Parser extracted {result} in {elapsed_ms:.1f} ms")
    return result
```python
# backend/scraper.py
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
        body = _text_of(body_el)
    
    meta = {"company": company.strip(), "position": title.strip(), "location": location.strip()}
    if body:
        insights = extract_insights_from_description(body, nlp_model)
        meta.update(insights)

    logger.debug(f"LinkedIn extracted meta: {meta}")
    return body or dom.get_text(" ", strip=True), meta
