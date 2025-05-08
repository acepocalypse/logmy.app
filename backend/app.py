"""
backend/app.py – Flask API for Job/Internship Tracker
====================================================
Provides endpoints to:
  • `/parse`       – parse raw job‑posting text
  • `/link-parse`  – fetch a job‑posting URL, scrape + parse
  • `/submit`      – save an application row to Supabase (JWT auth required)
  • `/health`      – simple health‑check for uptime monitoring

NOTE: Render requires the server to bind to **0.0.0.0:$PORT**.  The
`app.run()` line at the bottom now explicitly sets `host="0.0.0.0"` so
Render detects the open port (fixes the "No open ports detected" error).
"""
from __future__ import annotations

import os
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from dotenv import load_dotenv
import jwt
from jwt import InvalidTokenError

from parser import parse as parse_job
from scraper import fetch_page, extract_text, ScrapeError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()
SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_JWT_SECRET  = os.getenv("SUPABASE_JWT_SECRET")
JWT_ALG = "HS256"

if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_JWT_SECRET]):
    raise RuntimeError("Missing required environment variables for Supabase.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def get_current_user(fn):
    """Decorator to ensure request has a valid Supabase JWT."""
    @wraps(fn)
    def _wrapped(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing Bearer token"}), 401
        token = auth_header.split(" ", 1)[1]
        try:
            payload = jwt.decode(token,SUPABASE_JWT_SECRET,algorithms=["HS256"], audience="authenticated")
        except InvalidTokenError as exc:
            return jsonify({"error": "Invalid token", "detail": str(exc)}), 401
        request.user_id = payload.get("sub")
        return fn(*args, **kwargs)
    return _wrapped

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/parse", methods=["POST"])
def parse_endpoint():
    text = request.get_json(force=True).get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400
    return jsonify(parse_job(text)), 200


@app.route("/link-parse", methods=["POST"])
def link_parse_endpoint():
    url = request.get_json(force=True).get("url", "").strip()
    if not url:
        return jsonify({"error": "Missing 'url' field"}), 400
    try:
        html = fetch_page(url)
        raw_text, meta = extract_text(html, url)
        data = parse_job(raw_text)
        data |= meta
        data["job_url"] = url
        return jsonify(data), 200
    except ScrapeError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/submit", methods=["POST"])
@get_current_user
def submit_endpoint():
    data = request.get_json(force=True)

    application = {
        "user_id":          request.user_id,
        "company":          data.get("company", ""),
        "position":         data.get("position", ""),
        "location":         data.get("location", ""),
        "job_type":         data.get("job_type", ""),
        "application_date": data.get("application_date") or datetime.utcnow().date().isoformat(),
        "deadline":         data.get("deadline") or None,
        "status":           data.get("status", "Applied"),
        "job_url":          data.get("job_url", ""),
        "notes":            data.get("notes", ""),
        "created_at":       datetime.utcnow().isoformat(),
        "updated_at":       datetime.utcnow().isoformat(),
    }
    response = supabase.table("applications").insert(application).execute()
    if response.error:
        return jsonify({"error": response.error.message}), 500
    return jsonify({"success": True, "data": response.data}), 201


@app.route("/health")
def health():
    return "OK", 200

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    # Bind to 0.0.0.0 so Render can detect the open port
    app.run(host="0.0.0.0", port=port, debug=False)
