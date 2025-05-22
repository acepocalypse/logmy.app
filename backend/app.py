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

# For improved error handling on /submit and /signup
from postgrest.exceptions import APIError as PostgrestAPIError # For data operations
from gotrue.errors import AuthApiError # For authentication operations
import requests

# --- Google AI Gemini/Gemma Model Loading ---
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Define the model ID you want to use (e.g., "gemma-2b", "gemma-7b", "gemini-1.0-pro")
# Make this configurable via an environment variable
GEMMA_MODEL_ID_FOR_API = os.getenv("GEMMA_MODEL_ID", "gemma-2b") # Defaulting to "gemma-2b"

GEMINI_MODEL_INSTANCE = None
try:
    if not GEMINI_API_KEY:
        logging.warning("GEMINI_API_KEY environment variable not set. Parsing features will be disabled if model loading fails.")
        # Attempting to configure without explicit key might work in some environments (e.g. Colab)
        # but is not reliable for typical server deployments.
    
    genai.configure(api_key=GEMINI_API_KEY) # Configure with key (or None if not set)
    
    GEMINI_MODEL_INSTANCE = genai.GenerativeModel(GEMMA_MODEL_ID_FOR_API)
    # Optional: A simple test to see if the model can be reached.
    # try:
    #     GEMINI_MODEL_INSTANCE.generate_content("test", generation_config=genai.types.GenerationConfig(candidate_count=1))
    #     logging.info(f"Google AI Gemini API configured and model '{GEMMA_MODEL_ID_FOR_API}' loaded and tested successfully.")
    # except Exception as test_e:
    #     logging.error(f"Model '{GEMMA_MODEL_ID_FOR_API}' loaded but failed test call: {test_e}")
    #     GEMINI_MODEL_INSTANCE = None # Disable if test fails
    # else:
    logging.info(f"Google AI Gemini API configured and model '{GEMMA_MODEL_ID_FOR_API}' instance created.")

except Exception as e:
    logging.error(f"Could not initialize Google AI Gemini API or create model instance for '{GEMMA_MODEL_ID_FOR_API}': {e}", exc_info=True)
    # GEMINI_MODEL_INSTANCE will remain None.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()
SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_JWT_SECRET  = os.getenv("SUPABASE_JWT_SECRET")
JWT_ALG = "HS256"
APP_FRONTEND_URL = os.getenv("APP_FRONTEND_URL", "http://localhost:5500/frontend/index.html")

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
else: 
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

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
@app.route("/signup", methods=["POST"])
def signup_endpoint():
    data = request.get_json(force=True)
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    try:
        app.logger.info(f"Attempting signup for email: {email}")
        redirect_url = f"{APP_FRONTEND_URL}" 
        
        res = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": { "email_redirect_to": redirect_url }
        })
        
        app.logger.info(f"Signup response for {email}: User - {'present' if res.user else 'absent'}, Session - {'present' if res.session else 'absent'}")

        if res.user:
            email_verified = res.user.email_confirmed_at is not None
            if not email_verified and hasattr(res.user, 'identities') and res.user.identities:
                primary_identity_data = next((id_data.get('identity_data', {}) for id_data in res.user.identities if id_data.get('user_id') == res.user.id), None)
                if primary_identity_data:
                    email_verified = primary_identity_data.get('email_verified', False)

            if not email_verified:
                app.logger.info(f"Signup successful for {email}. User created, email verification pending.")
                return jsonify({
                    "success": True,
                    "message": "Signup successful! Please check your email to verify your account.",
                    "user_id": res.user.id
                }), 201
            else: 
                app.logger.info(f"Signup successful for {email}. User created and email already confirmed.")
                return jsonify({
                    "success": True,
                    "message": "Signup successful! Your account is active.",
                    "user_id": res.user.id,
                    "session": res.session.model_dump_json() if res.session else None
                }), 201
        
        app.logger.error(f"Signup for {email} resulted in no user object in response, but no AuthApiError. Response: {res}")
        return jsonify({"error": "Signup failed. Unexpected response from auth service."}), 500

    except AuthApiError as e:
        app.logger.error(f"AuthApiError during signup for {email}: {e.message} (Status: {e.status})", exc_info=False)
        error_message = f"Signup failed: {e.message or 'Please try again.'}"
        status_code = e.status if isinstance(e.status, int) and 400 <= e.status < 600 else 500
        
        err_data_msg = str(getattr(e, 'data', {}).get('msg', '')).lower() if hasattr(e, 'data') else ""

        if "user already registered" in str(e.message).lower() or e.status == 422 or "already_exists" in err_data_msg:
            error_message = "This email is already registered. Please try logging in."
            status_code = 409
        elif "to disable email confirmation" in str(e.message).lower():
             error_message = "Signup requires email confirmation. Please check your Supabase project settings."
             status_code = 500
        elif "password should be at least 6 characters" in str(e.message).lower() or "characters" in err_data_msg:
            error_message = "Password should be at least 6 characters long."
            status_code = 400
        return jsonify({"error": error_message}), status_code
    except Exception as e:
        app.logger.error(f"Unexpected error during signup for {email}: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred during signup."}), 500

@app.route("/parse", methods=["POST"])
def parse_endpoint():
    text = request.get_json(force=True).get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    if GEMINI_MODEL_INSTANCE is None:
        app.logger.error("Google AI Gemma model not loaded, cannot parse text.")
        return jsonify({"error": "NLP service (Google AI Gemma) not available"}), 503
        
    return jsonify(parse_job_text(text, GEMINI_MODEL_INSTANCE)), 200

@app.route("/link-parse", methods=["POST"])
def link_parse_endpoint():
    url = request.get_json(force=True).get("url", "").strip()
    if not url:
        return jsonify({"error": "Missing 'url' field"}), 400
    
    if GEMINI_MODEL_INSTANCE is None:
        app.logger.error("Google AI Gemma model not loaded, cannot parse link.")
        return jsonify({"error": "NLP service (Google AI Gemma) not available"}), 503
        
    try:
        app.logger.info(f"Fetching page: {url}")
        html_content = fetch_page(url)
        
        app.logger.info(f"Extracting text from HTML for URL: {url}")
        raw_text, meta_from_scraper = extract_text(html_content, url, GEMINI_MODEL_INSTANCE)
        app.logger.debug(f"Scraper meta for {url}: {meta_from_scraper}")
        
        app.logger.info(f"Parsing extracted text with general parser for URL: {url}")
        data_from_parser = parse_job_text(raw_text, GEMINI_MODEL_INSTANCE)
        app.logger.debug(f"Parser data for {url}: {data_from_parser}")
        
        final_data = meta_from_scraper.copy() if meta_from_scraper else {}

        if not final_data.get('company') and data_from_parser.get('company'):
            final_data['company'] = data_from_parser.get('company')
        if not final_data.get('position') and data_from_parser.get('position'):
            final_data['position'] = data_from_parser.get('position')
        if not final_data.get('location') and data_from_parser.get('location'):
            final_data['location'] = data_from_parser.get('location')
        
        if not final_data.get('deadline') and data_from_parser.get('deadline'):
            final_data['deadline'] = data_from_parser.get('deadline')
        if meta_from_scraper and meta_from_scraper.get('deadline'): # Prefer scraper's deadline if available
            final_data['deadline'] = meta_from_scraper.get('deadline')
        
        # Ensure other insight fields from scraper (which includes job_type now) are present
        # These were already copied if final_data started with meta_from_scraper.copy()
        # This loop is more of a safeguard or for clarity if the copy logic was different.
        for key_insight in ['job_type', 'salary', 'timeframe', 'start_date']:
            if meta_from_scraper and meta_from_scraper.get(key_insight) and not final_data.get(key_insight):
                final_data[key_insight] = meta_from_scraper.get(key_insight)


        final_data["job_url"] = url
        
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
        
        if response.data and (not hasattr(response, 'error') or not response.error):
            app.logger.info(f"Successfully inserted application: {response.data[0]['id']}")
            return jsonify({"success": True, "data": response.data}), 201
        else:
            error_details = "Unknown error during submission."
            status_code_to_return = 500
            if hasattr(response, 'error') and response.error:
                error_details = response.error.message or "Failed to save due to Supabase error."
                app.logger.error(f"Supabase insert error: {error_details} for payload: {application}")
            elif hasattr(response, 'status_code') and response.status_code >= 300:
                 error_details = str(response.data) if hasattr(response, 'data') and response.data else str(response.text)
                 status_code_to_return = response.status_code
                 app.logger.error(f"Supabase insert returned status {status_code_to_return}: {error_details} for payload: {application}")
            else:
                app.logger.warning(f"Supabase insert attempt had issues or no data returned. Response: {response} for payload: {application}")
            return jsonify({"error": "Failed to save application.", "details": error_details}), status_code_to_return

    except PostgrestAPIError as e: 
        app.logger.error(f"Supabase PostgrestAPIError: {e.message} (Code: {e.code}, Details: {e.details}) for payload: {application}", exc_info=False)
        error_message = e.message or "Database constraint error."
        if e.code == '23514':
            if "applications_job_type_check" in str(e.details).lower():
                error_message = "Invalid 'Job Type' selected."
            elif "applications_status_check" in str(e.details).lower():
                 error_message = "Invalid 'Status' selected."
        return jsonify({"error": error_message, "db_code": e.code, "details": e.details}), 400
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Network RequestException (Supabase): {e}", exc_info=True)
        return jsonify({"error": "Network error connecting to database."}), 504
    except Exception as e:
        app.logger.error(f"Unexpected error during submission: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred."}), 500

@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    if app.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    
    port = int(os.getenv("PORT", 10000)) 
    is_production = os.getenv("RENDER", False) 
    app.run(host="0.0.0.0", port=port, debug=not is_production)