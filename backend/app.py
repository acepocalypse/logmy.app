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
from parser import parse as parse_job_text
from scraper import fetch_page, extract_text, ScrapeError

# For improved error handling on /submit
from postgrest.exceptions import APIError
import requests

# --- SpaCy Model Loading ---
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

if not app.debug:
    app.logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    app.logger.addHandler(stream_handler)

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------
def get_current_user(fn):
    @wraps(fn)
    def _wrapped(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            app.logger.warning("Auth error: Missing Bearer token")
            return jsonify({"error": "Missing or invalid Bearer token"}), 401
        token = auth_header.split(" ", 1)[1]
        try:
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
@app.route("/parse", methods=["POST"])
def parse_endpoint():
    text = request.get_json(force=True).get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400
    if NLP_MODEL is None:
        app.logger.error("NLP model not loaded, cannot parse text.")
        return jsonify({"error": "NLP service not available"}), 503
    return jsonify(parse_job_text(text, NLP_MODEL)), 200

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
        raw_text, meta_from_scraper = extract_text(html_content, url, NLP_MODEL)
        app.logger.debug(f"Scraper meta for {url}: {meta_from_scraper}")
        
        app.logger.info(f"Parsing extracted text with general parser for URL: {url}")
        data_from_parser = parse_job_text(raw_text, NLP_MODEL)
        app.logger.debug(f"Parser data for {url}: {data_from_parser}")
        
        # Refined merge strategy to prioritize scraper's specific extractions
        final_data = {}

        # Core fields: Prioritize scraper's direct CSS selector extractions (already in meta_from_scraper)
        # and its NLP insights, then fall back to general parser.
        final_data['company'] = meta_from_scraper.get('company') or data_from_parser.get('company')
        final_data['position'] = meta_from_scraper.get('position') or data_from_parser.get('position')
        final_data['location'] = meta_from_scraper.get('location') or data_from_parser.get('location')

        # NLP-derived insights from scraper (extract_insights_from_description)
        final_data['salary'] = meta_from_scraper.get('salary')
        final_data['timeframe'] = meta_from_scraper.get('timeframe')
        final_data['start_date'] = meta_from_scraper.get('start_date')
        
        # Deadline: Prioritize scraper's NLP insight, then parser's regex/NLP.
        # (Both modules might attempt deadline extraction)
        scraper_deadline = meta_from_scraper.get('deadline')
        parser_deadline = data_from_parser.get('deadline')
        final_data['deadline'] = scraper_deadline if scraper_deadline else parser_deadline

        # Add any other fields that might be exclusively in one or the other,
        # ensuring that we don't overwrite good data with None or empty strings.
        # Example: if parser found a 'job_type' and scraper didn't.
        for key in data_from_parser:
            if key not in final_data or final_data[key] is None or final_data[key] == "":
                if data_from_parser[key]: # Only if parser has a non-empty value
                    final_data[key] = data_from_parser[key]
        
        for key in meta_from_scraper: # Ensure any unique keys from scraper are also included
            if key not in final_data or final_data[key] is None or final_data[key] == "":
                if meta_from_scraper[key]:
                     final_data[key] = meta_from_scraper[key]


        final_data["job_url"] = url # Ensure job_url is always included

        # Clean up None values to empty strings if preferred by frontend, or leave as None
        # for key, value in final_data.items():
        #     if value is None:
        #         final_data[key] = ""

        app.logger.info(f"Successfully parsed link: {url}. Combined Data: {final_data}")
        return jsonify(final_data), 200
        
    except ScrapeError as exc:
        app.logger.error(f"Scraping error for {url}: {exc}")
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:
        app.logger.error(f"Unexpected error during link parsing for {url}: {exc}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(exc)}"}), 500


@app.route("/submit", methods=["POST"])
@get_current_user
def submit_endpoint():
    data = request.get_json(force=True)
    app.logger.info(f"Submit endpoint called by user: {request.user_id} with data: {data}")
    application = {
        "user_id": request.user_id,
        "company": data.get("company") or None,
        "position": data.get("position") or None,
        "location": data.get("location") or None,
        "job_type": data.get("job_type") if data.get("job_type") else None,
        "application_date": data.get("application_date") or datetime.utcnow().date().isoformat(),
        "deadline": data.get("deadline") if data.get("deadline") else None,
        "status": data.get("status") or "Applied",
        "job_url": data.get("job_url") or None,
        "notes": data.get("notes") or None,
    }
    try:
        app.logger.info(f"Attempting to insert application for user {request.user_id}: {application}")
        response = supabase.table("applications").insert(application).execute()

        if hasattr(response, 'error') and response.error:
            app.logger.error(f"Supabase insert error object: {response.error} for payload: {application}")
            error_message = response.error.message if hasattr(response.error, 'message') else "Unknown Supabase error"
            return jsonify({"error": error_message, "details": str(response.error)}), 500
        if hasattr(response, 'status_code') and response.status_code >= 300:
            app.logger.error(f"Supabase insert returned status {response.status_code}: {response.data or response.text} for payload: {application}")
            error_data = response.data if hasattr(response, 'data') and response.data else response.text
            return jsonify({"error": "Failed to save to database.", "details": error_data}), response.status_code
        if response.data:
            app.logger.info(f"Successfully inserted application: {response.data[0]['id']}")
            return jsonify({"success": True, "data": response.data}), 201
        else:
            app.logger.warning(f"Supabase insert successful but no data returned. Response: {response} for payload: {application}")
            return jsonify({"success": True, "message": "Application saved, but no confirmation data."}), 201
    except APIError as e:
        app.logger.error(f"Supabase APIError during insert for user {request.user_id}: {e.message} (Code: {e.code}, Details: {e.details}, Hint: {e.hint}) for payload: {application}", exc_info=False)
        if e.code == '23514':
            error_message = e.message if e.message else "Data validation error."
            if "applications_job_type_check" in str(e.details) or "applications_job_type_check" in error_message:
                error_message = "Invalid 'Job Type' selected. Please choose a valid option."
            return jsonify({"error": error_message, "db_code": e.code}), 400
        return jsonify({"error": f"Database error: {e.message}", "db_code": e.code}), 500
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Network RequestException during Supabase insert for user {request.user_id}: {e}", exc_info=True)
        return jsonify({"error": "Network error connecting to database service."}), 504
    except Exception as e:
        app.logger.error(f"Unexpected error during Supabase insert for user {request.user_id}: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred while saving."}), 500

@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)