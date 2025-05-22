import re
import time
import logging
import dateparser

# Import for Google AI (Gemini API / google-genai)
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold # For safety settings

logger = logging.getLogger(__name__)

# Regex helpers can still be useful as a fallback or to parse Gemma's output
COMPANY_RE  = re.compile(r"(?:Company|Employer)[:\-]?\s*(.+)", re.I)
POSITION_RE = re.compile(r"(?:Job Title|Title|Position|Role)[:\-]?\s*(.+)", re.I)
LOCATION_RE = re.compile(r"(?:Location)[:\-]?\s*(.+)", re.I)
DEADLINE_RE = re.compile(r"(?:Apply\s*by|Deadline)[:\-]?\s*([^\n]+)", re.I)


def _call_gemini_api_for_parsing(text: str, gemini_model_instance: genai.GenerativeModel) -> dict:
    """
    Helper function to interact with a Gemma model via the Google AI Gemini API
    and parse its structured output.
    """
    # The user's example conversation history.
    # You might want to make this more dynamic or keep it fixed if it helps guide the model.
    # For this general parsing task, a simpler, direct prompt is usually better.
    prompt = f"""Your task is to extract specific information from the job posting text provided below.

Job Posting Text:
---
{text}
---

Extract the following information:
- Company
- Position
- Location
- Deadline (Format as YYYY-MM-DD if possible. If not, provide the raw text for the deadline.)

Present the extracted information in a structured format, with each piece of information on a new line, like this:
Company: [Extracted Company Name]
Position: [Extracted Position Title]
Location: [Extracted Location]
Deadline: [Extracted Deadline]

If any piece of information cannot be found in the text, use the value "Not found" for that specific field.
"""
    
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=256, # Adjust as needed
        temperature=0.2,      # Adjust for creativity vs. factuality
        # top_p=0.8,            # Supported by some models, check Gemma model specifics
    )
    
    # Safety settings (adjust as per your application's needs)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    parsed_data = {
        "company": "",
        "position": "",
        "location": "",
        "deadline": ""
    }

    try:
        response = gemini_model_instance.generate_content(
            contents=[prompt], # Pass the direct prompt
            generation_config=generation_config,
            safety_settings=safety_settings,
            # stream=False, # Default is False for generate_content
        )

        # Check for safety blocks
        if not response.candidates: # If blocked, candidates list might be empty
             if response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.warning(f"Parsing call blocked by API. Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}")
                return parsed_data # Or raise specific error
             # Check if any candidate was blocked
             for candidate in response.candidates:
                 if candidate.finish_reason == genai.types.FinishReason.SAFETY:
                    logger.warning(f"Parsing stopped due to safety reasons. Candidate safety ratings: {candidate.safety_ratings}")
                    return parsed_data # Or specific error based on rating
        
        gemini_response_text = response.text # Accessing the text directly
        logger.debug(f"Google AI Gemma raw response for parsing: {gemini_response_text}")

        for line in gemini_response_text.split('\n'):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                
                if value.lower() == "not found":
                    value = "" 

                if key == "company":
                    parsed_data["company"] = value
                elif key == "position":
                    parsed_data["position"] = value
                elif key == "location":
                    parsed_data["location"] = value
                elif key == "deadline":
                    if value:
                        try:
                            # Attempt to parse with dateparser, keeping original if not parsable
                            parsed_date = dateparser.parse(value, settings={"PREFER_DATES_FROM": "future", "STRICT_PARSING": False})
                            if parsed_date:
                                parsed_data["deadline"] = parsed_date.date().isoformat()
                            else:
                                parsed_data["deadline"] = value # Keep raw if dateparser returns None
                        except Exception:
                            parsed_data["deadline"] = value # Keep raw on any parsing error
                    else:
                        parsed_data["deadline"] = ""
        
    except Exception as e:
        logger.error(f"Error during Google AI Gemma interaction or parsing: {e}", exc_info=True)
        # Fallback to regex if Gemma fails entirely
        parsed_data["company"] = (COMPANY_RE.search(text).group(1).strip() if COMPANY_RE.search(text) else "")
        parsed_data["position"] = (POSITION_RE.search(text).group(1).strip() if POSITION_RE.search(text) else "")
        parsed_data["location"] = (LOCATION_RE.search(text).group(1).strip() if LOCATION_RE.search(text) else "")
        parsed_data["deadline"] = _extract_deadline_regex(text)
        return parsed_data

    # Regex fallbacks for fields Gemma might have missed or if output format is off
    if not parsed_data.get("company"):
        m = COMPANY_RE.search(text)
        if m: parsed_data["company"] = m.group(1).strip()
    if not parsed_data.get("position"):
        m = POSITION_RE.search(text)
        if m: parsed_data["position"] = m.group(1).strip()
    if not parsed_data.get("location"):
        m = LOCATION_RE.search(text)
        if m: parsed_data["location"] = m.group(1).strip()
    if not parsed_data.get("deadline") and parsed_data.get("deadline", None) is None: # Check if not found AND not empty string from "Not found"
        parsed_data["deadline"] = _extract_deadline_regex(text)

    return parsed_data


def _extract_deadline_regex(text: str) -> str:
    m = DEADLINE_RE.search(text)
    if m:
        try:
            parsed = dateparser.parse(m.group(1).strip(), settings={"PREFER_DATES_FROM": "future", "STRICT_PARSING": False})
            if parsed:
                return parsed.date().isoformat()
        except Exception as e:
            logger.warning(f"Dateparser failed for deadline string '{m.group(1).strip()}': {e}")
        return m.group(1).strip() # Fallback to raw string
    return ""


def parse(text: str, gemini_model_instance: genai.GenerativeModel) -> dict:
    """Return dict with keys company, position, location, deadline using Gemma via Google AI Gemini API."""
    
    if not gemini_model_instance:
        logger.error("Google AI Gemma model instance is None, cannot perform parsing.")
        return {"company": "", "position": "", "location": "", "deadline": ""}

    logger.debug(f"parser.parse() called with text length={len(text)} | preview={text[:120]!r}")
    start_time = time.perf_counter()

    result = _call_gemini_api_for_parsing(text, gemini_model_instance)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.debug(f"Google AI Gemma Parser extracted {result} in {elapsed_ms:.1f} ms")
    return result