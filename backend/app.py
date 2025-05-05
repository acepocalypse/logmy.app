"""
Lightweight Flask backend for Job/Internship Application Tracker.

Endpoints
---------
POST /parse
    Payload: {"text": "<raw job posting>"}
    Returns: {"company": "...", "position": "...", "location": "...", "deadline": "..."}

POST /submit
    Payload: {
        "user_id": "<uuid from Supabase auth>",
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
    Returns: {"success": true, "data": …}

Environment (set in a .env or server secrets)
---------------------------------------------
SUPABASE_URL           – https://<project>.supabase.co
SUPABASE_SERVICE_KEY   – service role or anon key (⚠️ service key must stay server‑side)

Run locally with:
$ pip install -r requirements.txt  # see requirements section below
$ uvicorn app:app --reload        # or python app.py

"""
import os
import re
from datetime import datetime

from flask import Flask, request, jsonify
from supabase import create_client, Client
from dotenv import load_dotenv

# --------------------------------------------------
# Initialise / Config
# --------------------------------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # keep secret

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY env vars must be set.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
app = Flask(__name__)

# --------------------------------------------------
# Naïve regex patterns for MVP parsing
# --------------------------------------------------
PATTERNS = {
    "company": re.compile(r"(?:Company|Employer)[:\-]?\s*(.+)", re.I),
    "position": re.compile(r"(?:Title|Position|Role)[:\-]?\s*(.+)", re.I),
    "location": re.compile(r"(?:Location)[:\-]?\s*(.+)", re.I),
    "deadline": re.compile(r"(?:Deadline|Apply\s*by)[:\-]?\s*(.+)", re.I),
}


def parse_job(text: str) -> dict:
    """Return a dict of parsed fields using simple regex heuristics."""
    parsed = {}
    for field, pattern in PATTERNS.items():
        match = pattern.search(text)
        parsed[field] = match.group(1).strip() if match else ""
    return parsed


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/parse", methods=["POST"])
def parse_endpoint():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400
    return jsonify(parse_job(text))


@app.route("/submit", methods=["POST"])
def submit_endpoint():
    data = request.get_json(force=True)
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing 'user_id'"}), 400

    application = {
        "user_id": user_id,
        "company": data.get("company", ""),
        "position": data.get("position", ""),
        "location": data.get("location", ""),
        "job_type": data.get("job_type", ""),
        "application_date": data.get("application_date") or datetime.utcnow().isoformat(),
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
        return jsonify({"success": True, "data": response.data}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# --------------------------------------------------
# Dev entry‑point
# --------------------------------------------------
if __name__ == "__main__":
    # Do *not* use debug=True in production.
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))