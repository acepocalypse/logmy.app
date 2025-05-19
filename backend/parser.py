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