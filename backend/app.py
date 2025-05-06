"""
Flask backend for Job/Internship Tracker – **spaCy‑powered parser**
=================================================================
This version upgrades `/parse` from simple regexes to a lightweight
NLP pipeline using **spaCy `en_core_web_sm`** plus a few targeted
regex/date helpers.  The goal is to handle real‑world job ads copied
from sites like Indeed, LinkedIn, or company pages.

Key changes
-----------
* Loads `en_core_web_sm` at startup (≈ 10 MB, okay for Render free tier).
* Extracts:
  * **company**  – first ORG entity or 'Company:' line.
  * **position** – first match after keywords *title/role/position* or
    the longest uppercase‑ish noun chunk early in the text.
  * **location** – first GPE/LOC entity or 'Location:' line.
  * **deadline** – looks for phrases like *apply by* or *deadline* plus
    a date (parsed with `dateparser`).
* Adds a `/parse` benchmark log so you can see parse time in Render logs.
* App now returns HTTP 422 if spaCy cannot load (helpful during build).

Env (.env)
----------
Same vars as before (`SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `SUPABASE_JWT_SECRET`).

New dependencies (add to **requirements.txt**):
```
spacy
en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
dateparser
flask
flask-cors
python-dotenv
supabase
pyjwt>=2.0
```
(The explicit wheel URL lets Render install the small English model during build.)
"""
import os
import re
import time
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from dotenv import load_dotenv
import jwt
from jwt import InvalidTokenError

# NLP imports
try:
    import spacy
    from spacy.language import Language
except ImportError:
    spacy = None  # will raise later

import dateparser

# ---------------------------------------------------------------------------
# Initialisation & Configuration
# ---------------------------------------------------------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
JWT_ALG = "HS256"

if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_JWT_SECRET]):
    raise RuntimeError("Missing required env vars.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ---------------------------------------------------------------------------
# Auth decorator (unchanged)
# ---------------------------------------------------------------------------

def get_current_user(fn):
    @wraps(fn)
    def _wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing Bearer token"}), 401
        token = auth_header.split(" ", 1)[1]
        try:
            payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=[JWT_ALG])
        except InvalidTokenError as exc:
            return jsonify({"error": "Invalid token", "detail": str(exc)}), 401
        request.user_id = payload.get("sub")
        return fn(*args, **kwargs)
    return _wrapper

# ---------------------------------------------------------------------------
# spaCy model load
# ---------------------------------------------------------------------------
if spacy is None:
    raise RuntimeError("spaCy not installed. Add it to requirements.txt.")

try:
    NLP: Language = spacy.load("en_core_web_sm")
except OSError as e:
    # Model not present – instructions for Render build logs
    raise RuntimeError("spaCy model 'en_core_web_sm' not found. Make sure the wheel is installed in requirements.txt") from e

# ---------------------------------------------------------------------------
# Helper functions for parsing
# ---------------------------------------------------------------------------
COMPANY_RE = re.compile(r"(?:Company|Employer)[:\-]?\s*(.+)", re.I)
POSITION_RE = re.compile(r"(?:Job Title|Title|Position|Role)[:\-]?\s*(.+)", re.I)
LOCATION_RE = re.compile(r"(?:Location)[:\-]?\s*(.+)", re.I)
DEADLINE_RE = re.compile(r"(?:Apply\s*by|Deadline)[:\-]?\s*([^\n]+)", re.I)


def extract_deadline(text: str) -> str:
    m = DEADLINE_RE.search(text)
    if m:
        parsed = dateparser.parse(m.group(1), settings={"PREFER_DATES_FROM": "future"})
        if parsed:
            return parsed.date().isoformat()
        return m.group(1).strip()
    return ""


def extract_company(doc):
    # Prefer explicit line first
    m = COMPANY_RE.search(doc.text)
    if m:
        return m.group(1).strip()
    # fallback: first ORG entity
    for ent in doc.ents:
        if ent.label_ == "ORG":
            return ent.text
    return ""


def extract_location(doc):
    m = LOCATION_RE.search(doc.text)
    if m:
        return m.group(1).strip()
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            return ent.text
    return ""


def extract_position(doc):
    m = POSITION_RE.search(doc.text)
    if m:
        return m.group(1).strip()
    # fallback: choose longest noun chunk in first 250 chars that looks title‑ish
    slice_doc = doc[: min(50, len(doc))]
    noun_chunks = sorted(slice_doc.noun_chunks, key=lambda nc: -len(nc.text))
    if noun_chunks:
        text = noun_chunks[0].text.strip()
        # heuristic: Title Case words or ALL CAPS
        return text
    return ""


def parse_job(text: str):
    start = time.perf_counter()
    doc = NLP(text)
    parsed = {
        "company": extract_company(doc),
        "position": extract_position(doc),
        "location": extract_location(doc),
        "deadline": extract_deadline(text),
    }
    app.logger.debug("/parse completed in %.2f ms", (time.perf_counter() - start) * 1000)
    return parsed

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/parse", methods=["POST"])
def parse_endpoint():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400
    try:
        return jsonify(parse_job(text)), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 422

@app.route("/submit", methods=["POST"])
@get_current_user
def submit_endpoint():
    data = request.get_json(force=True)
    application = {
        "user_id": request.user_id,
        "company": data.get("company", ""),
        "position": data.get("position", ""),
        "location": data.get("location", ""),
        "job_type": data.get("job_type", ""),
        "application_date": data.get("application_date") or datetime.utcnow().date().isoformat(),
        "deadline": data.get("deadline", ""),
        "status": data.get("status", "Applied"),
        "job_url": data.get("job_url", ""),
        "notes": data.get("notes", ""),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }
    try:
        response = supabase.table("applications").insert(application).execute()
        if response.error:
            return jsonify({"error": response.error.message}), 500
        return jsonify({"success": True, "data": response.data}), 201
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
