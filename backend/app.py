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

# For improved error handling on /submit and /signup
from postgrest.exceptions import APIError as PostgrestAPIError # For data operations
from gotrue.errors import AuthApiError # For authentication operations
import requests

# --- Google AI Gemini/Gemma Model Loading ---
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMMA_MODEL_ID_FOR_API = os.getenv("GEMMA_MODEL_ID", "gemma-3-27b-it") # Defaulting to "gemma-2b"

GEMINI_MODEL_INSTANCE = None
try:
    if not GEMINI_API_KEY:
        # If running in an environment where API key might not be needed (e.g. Vertex AI with service accounts),
        # this check might be adjusted. But for direct google-genai SDK use, key is typical.
        logging.warning("GEMINI_API_KEY environment variable not set. Attempting to configure without explicit key.")
        # Depending on the environment (e.g., Colab, some cloud environments),
        # google.generativeai might pick up credentials automatically.
        # If not, this will likely fail at the genai.GenerativeModel() call.
    
    # Configure the API key
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Create the GenerativeModel instance
    GEMINI_MODEL_INSTANCE = genai.GenerativeModel(GEMMA_MODEL_ID_FOR_API)
    # You could add a simple test call here if desired, e.g.,
    # GEMINI_MODEL_INSTANCE.generate_content("test prompt", generation_config=genai.types.GenerationConfig(candidate_count=1))
    logging.info(f"Google AI Gemini API configured and model '{GEMMA_MODEL_ID_FOR_API}' loaded successfully.")
except Exception as e:
    logging.error(f"Could not initialize Google AI Gemini API or load model '{GEMMA_MODEL_ID_FOR_API}': {e}", exc_info=True)
    # GEMINI_MODEL_INSTANCE will remain None, and your parsing functions will check for this.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()
SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_JWT_SECRET  = os.getenv("SUPABASE_JWT_SECRET")
JWT_ALG = "HS256"
# Frontend URL for email redirects - IMPORTANT for email verification
APP_FRONTEND_URL = os.getenv("APP_FRONTEND_URL", "http://localhost:5500/frontend/index.html") # Default for local dev

if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_JWT_SECRET]):
    raise RuntimeError("Missing required environment variables for Supabase.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True) # Adjust origins for production

if not app.debug:
    app.logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    app.logger.addHandler(stream_handler)
else: # Ensure basicConfig is called if in debug mode and not using app.logger exclusively
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
            "options": {
                "email_redirect_to": redirect_url
            }
        })
        
        app.logger.info(f"Signup response for {email}: User - {'present' if res.user else 'absent'}, Session - {'present' if res.session else 'absent'}")

        if res.user:
            # Check identities and email_verified status for new signups
            email_verified = res.user.email_confirmed_at is not None
            if not email_verified and res.user.identities:
                 # Attempt to get specific identity data if available
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
            else: # Email is confirmed
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
        if "User already registered" in e.message or e.status == 422 or (hasattr(e, 'data') and e.data and "already_exists" in e.data.get("msg","").lower()):
            error_message = "This email is already registered. Please try logging in."
            status_code = 409
        elif "To disable email confirmation" in e.message:
             error_message = "Signup requires email confirmation. Please check your Supabase project settings."
             status_code = 500
        elif "Password should be at least 6 characters" in e.message or (hasattr(e, 'data') and e.data and "characters" in e.data.get("msg","").lower()):
            error_message = "Password should be at least 6 characters long."
            status_code = 400
        else:
            error_message = f"Signup failed: {e.message or 'Please try again.'}"
            status_code = e.status if isinstance(e.status, int) and 400 <= e.status < 600 else 500
        return jsonify({"error": error_message}), status_code
    except Exception as e:
        app.logger.error(f"Unexpected error during signup for {email}: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred during signup."}), 500


@app.route("/parse", methods=["POST"])
def parse_endpoint():
    text = request.get_json(force=True).get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    if GEMINI_MODEL_INSTANCE is None: # Check if the Gemini model instance is available
        app.logger.error("Google AI Gemma model not loaded, cannot parse text.")
        return jsonify({"error": "NLP service (Google AI Gemma) not available"}), 503
        
    # Pass the GEMINI_MODEL_INSTANCE to parse_job_text
    return jsonify(parse_job_text(text, GEMINI_MODEL_INSTANCE)), 200

@app.route("/link-parse", methods=["POST"])
def link_parse_endpoint():
    url = request.get_json(force=True).get("url", "").strip()
    if not url:
        return jsonify({"error": "Missing 'url' field"}), 400
    
    if GEMINI_MODEL_INSTANCE is None: # Check if the Gemini model instance is available
        app.logger.error("Google AI Gemma model not loaded, cannot parse link.")
        return jsonify({"error": "NLP service (Google AI Gemma) not available"}), 503
        
    try:
        app.logger.info(f"Fetching page: {url}")
        html_content = fetch_page(url)
        
        app.logger.info(f"Extracting text from HTML for URL: {url}")
        # Pass the GEMINI_MODEL_INSTANCE to extract_text
        raw_text, meta_from_scraper = extract_text(html_content, url, GEMINI_MODEL_INSTANCE)
        app.logger.debug(f"Scraper meta for {url}: {meta_from_scraper}")
        
        app.logger.info(f"Parsing extracted text with general parser for URL: {url}")
        # Pass the GEMINI_MODEL_INSTANCE to parse_job_text
        data_from_parser = parse_job_text(raw_text, GEMINI_MODEL_INSTANCE)
        app.logger.debug(f"Parser data for {url}: {data_from_parser}")
        
        # Consolidate data (same logic as before)
        final_data = {}
        # Prioritize scraper's direct extractions for company, position, location if available
        final_data['company'] = meta_from_scraper.get('company') if meta_from_scraper.get('company') else data_from_parser.get('company')
        final_data['position'] = meta_from_scraper.get('position') if meta_from_scraper.get('position') else data_from_parser.get('position')
        final_data['location'] = meta_from_scraper.get('location') if meta_from_scraper.get('location') else data_from_parser.get('location')
        
        # For other fields, take scraper's insight if available, then parser's, then null
        final_data['salary'] = meta_from_scraper.get('salary') # Scraper is primary for salary
        final_data['timeframe'] = meta_from_scraper.get('timeframe') # Scraper is primary for timeframe
        final_data['start_date'] = meta_from_scraper.get('start_date') # Scraper is primary for start_date
        
        scraper_deadline = meta_from_scraper.get('deadline')
        parser_deadline = data_from_parser.get('deadline')
        final_data['deadline'] = scraper_deadline if scraper_deadline else parser_deadline
        
        # Fill any remaining empty fields from parser if scraper didn't provide them
        for key in data_from_parser:
            if not final_data.get(key) and data_from_parser.get(key):
                final_data[key] = data_from_parser[key]
        # Ensure meta_from_scraper fields (like salary, timeframe, start_date from insights) are in final_data if not already
        for key in ['salary', 'timeframe', 'start_date', 'deadline']: # ensure these specific insight keys are prioritized if from scraper
             if meta_from_scraper.get(key) and (not final_data.get(key) or final_data.get(key) == data_from_parser.get(key) ): # if scraper has it and final doesn't, or final has parser's version
                 final_data[key] = meta_from_scraper[key]


        final_data["job_url"] = url
        app.logger.info(f"Successfully parsed link: {url}. Combined Data: {final_data}")
        return jsonify(final_data), 200
        
    except ScrapeError as exc:
        app.logger.error(f"Scraping error for {url}: {exc}")
        return jsonify({"error": str(exc)}), 422 # Unprocessable Entity
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
        "job_type": data.get("job_type") if data.get("job_type") else None, # Ensure empty string becomes null
        "application_date": data.get("application_date") or datetime.utcnow().date().isoformat(),
        "deadline": data.get("deadline") if data.get("deadline") else None, # Ensure empty string becomes null
        "status": data.get("status") or "Applied",
        "job_url": data.get("job_url") or None,
        "notes": data.get("notes") or None,
    }
    try:
        app.logger.info(f"Attempting to insert application for user {request.user_id}: {application}")
        response = supabase.table("applications").insert(application).execute()
        
        # Proper error checking for Supabase response
        if response.data and not hasattr(response, 'error') or not response.error: # Check if data exists and no error attribute or error is None/falsey
            app.logger.info(f"Successfully inserted application: {response.data[0]['id']}")
            return jsonify({"success": True, "data": response.data}), 201
        else: # Handle potential errors or unexpected responses
            error_details = "Unknown error"
            status_code_to_return = 500
            if hasattr(response, 'error') and response.error:
                app.logger.error(f"Supabase insert error: {response.error.message if response.error.message else 'No message'} for payload: {application}")
                error_details = response.error.message or "Failed to save to database due to an unknown error from Supabase."
                # Try to get status code from Supabase error if available, else default
                # This part is tricky as Supabase client might not directly expose HTTP status for all errors
                # For PostgrestAPIError, it's handled below.
            elif hasattr(response, 'status_code') and response.status_code >= 300: # Check if it's an HTTP-like error structure
                 app.logger.error(f"Supabase insert returned status {response.status_code}: {response.data or response.text} for payload: {application}")
                 error_details = response.data if hasattr(response, 'data') and response.data else str(response.text)
                 status_code_to_return = response.status_code
            else:
                app.logger.warning(f"Supabase insert attempt had issues or no data returned. Response: {response} for payload: {application}")
            
            return jsonify({"error": "Failed to save to database.", "details": error_details}), status_code_to_return

    except PostgrestAPIError as e: 
        app.logger.error(f"Supabase PostgrestAPIError during insert for user {request.user_id}: {e.message} (Code: {e.code}, Details: {e.details}, Hint: {e.hint}) for payload: {application}", exc_info=False)
        error_message = e.message or "Database interaction error."
        if e.code == '23514': # Check constraint violation
            if "applications_job_type_check" in str(e.details) or "applications_job_type_check" in error_message:
                error_message = "Invalid 'Job Type' selected. Please choose a valid option."
            elif "applications_status_check" in str(e.details) or "applications_status_check" in error_message:
                 error_message = "Invalid 'Status' selected. Please choose a valid option."
        return jsonify({"error": error_message, "db_code": e.code, "details": e.details}), 400 # Usually client error for constraint violations
    except requests.exceptions.RequestException as e: # Network errors for Supabase client
        app.logger.error(f"Network RequestException during Supabase insert for user {request.user_id}: {e}", exc_info=True)
        return jsonify({"error": "Network error connecting to database service."}), 504 # Gateway Timeout
    except Exception as e:
        app.logger.error(f"Unexpected error during Supabase insert for user {request.user_id}: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred while saving."}), 500


@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    # Ensure logging is configured if not running with Flask's default app.logger setup (e.g. when app.debug is True)
    if app.debug: # or a more specific check if you are not relying on app.debug for this
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    
    port = int(os.getenv("PORT", 10000)) 
    # For Render, debug=False is typical. For local, you might want debug=True.
    # Render sets PORT env var.
    is_production = os.getenv("RENDER", False) # Render sets a RENDER env var.
    app.run(host="0.0.0.0", port=port, debug=not is_production)