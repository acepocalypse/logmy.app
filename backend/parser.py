"""
parser.py – spaCy‑powered job‑posting parser
===========================================
Adds DEBUG‑level log statements so you can watch extraction steps in
Render’s logs or local console.  **No functional changes.**
"""
import re
import time
import logging

import dateparser
import spacy
from spacy.language import Language

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s [parser] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load spaCy English model once at import time
# ---------------------------------------------------------------------------
logger.debug("Loading spaCy model 'en_core_web_sm' …")
NLP: Language = spacy.load("en_core_web_sm")
logger.debug("spaCy model loaded.")

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
        parsed = dateparser.parse(m.group(1), settings={"PREFER_DATES_FROM": "future"})
        if parsed:
            return parsed.date().isoformat()
        return m.group(1).strip()
    return ""


def _extract_company(doc):
    m = COMPANY_RE.search(doc.text)
    if m:
        return m.group(1).strip()
    for ent in doc.ents:
        if ent.label_ == "ORG":
            return ent.text
    return ""


def _extract_location(doc):
    m = LOCATION_RE.search(doc.text)
    if m:
        return m.group(1).strip()
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            return ent.text
    return ""


def _extract_position(doc):
    m = POSITION_RE.search(doc.text)
    if m:
        return m.group(1).strip()
    slice_doc = doc[: min(50, len(doc))]
    noun_chunks = sorted(slice_doc.noun_chunks, key=lambda nc: -len(nc.text))
    if noun_chunks:
        return noun_chunks[0].text.strip()
    return ""

# ---------------------------------------------------------------------------
# Public parse() function
# ---------------------------------------------------------------------------

def parse(text: str) -> dict:
    """Return dict with keys company, position, location, deadline."""
    logger.debug(f"parse() called with text length={len(text)} | preview={text[:120]!r}")
    start = time.perf_counter()
    doc = NLP(text)
    result = {
        "company":  _extract_company(doc),
        "position": _extract_position(doc),
        "location": _extract_location(doc),
        "deadline": _extract_deadline(text),
    }
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.debug("Extracted %s in %.1f ms", result, elapsed_ms)
    return result
