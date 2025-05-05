"""
Lightweight Flask backend for Job/Internship Application Tracker
--------------------------------------------------------------

Endpoints
---------
POST /parse
    Payload: {"text": "<raw job posting>"}
    → Returns minimal extracted fields for the form.

POST /submit
    Headers:  Authorization: Bearer <SUPABASE_JWT>
    Payload:  {
        "company": "...",
        "position": "...",
        "location": "...",
        "job_type": "Internship | Full‑Time",
        "application_date": "<ISO8601> | optional (defaults now)",
        "deadline": "<ISO8601 | string>",
        "status": "Applied | Interview | Offer | Rejected",
        "job_url": "https://…",
        "notes": "free‑text"
      }
    → Saves the record under the JWT user’s UID.

Run locally:
$ pip install -r requirements.txt  # see bottom of file
$ python app.py                    # runs on http://127.0.0.1:5000
"""

import os
import re
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify
from supabase import create_client, Client
from dotenv import load_dotenv
import jwt
from jwt import InvalidTokenError

# ---------------------------------------------------------------------------
# Initialisation & Configuration
# ---------------------------------------------------------------------------
load_dotenv()

SUPABASE_URL: str | None = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY: str | None = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_JWT_SECRET: str | None = os.getenv("SUPABASE_JWT_SECRET")
JWT_ALG = "HS256"

if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_JWT_SECRET]):
    raise RuntimeError("Environment variables SUPABASE_URL, SUPABASE_SERVICE_KEY, and SUPABASE_JWT_SECRET must be set.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def get_current_user(fn):
    """Decorator to validate Supabase JWT sent via Authorization header."""
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

        # Attach user_id to request context
        request.user_id = payload.get("sub")
        if not request.user_id:
            return jsonify({"error": "Token missing subject (sub) claim."}), 401
        return fn(*args, **kwargs)
    return _wrapper

# ---------------------------------------------------------------------------
# Naïve regex‑based parser (MVP)
# ---------------------------------------------------------------------------
PATTERNS = {
    "company":  re.compile(r"(?:Company|Employer)[:\-]?\s*(.+)", re.I),
    "position": re.compile(r"(?:Title|Position|Role)[:\-]?\s*(.+)", re.I),
    "location": re.compile(r"(?:Location)[:\-]?\s*(.+)", re.I),
    "deadline": re.compile(r"(?:Deadline|Apply\s*by)[:\-]?\s*(.+)", re.I),
}


def parse_job(text: str) -> dict:
    """Extract key fields from raw job posting text (very simple heuristics)."""
    result: dict[str, str] = {}
    for field, pattern in PATTERNS.items():
        m = pattern.search(text)
        result[field] = m.group(1).strip() if m else ""
    return result

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/parse", methods=["POST"])
def parse_endpoint():
    data = request.get_json(force=True)
    text: str = data.get("text", "")
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400
    return jsonify(parse_job(text)), 200


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
def health_check():
    return "OK", 200

# ---------------------------------------------------------------------------
# Local dev entry‑point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, port=port)
