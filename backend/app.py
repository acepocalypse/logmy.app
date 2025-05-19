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

if __name__ == "__main__":
    # Basic logging config for when running with `python app.py`
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False) # debug=False for production on Render