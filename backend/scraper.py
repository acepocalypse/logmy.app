from __future__ import annotations

import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urlparse
import logging
import re
import dateparser

# Import for Google AI (Gemini API / google-genai)
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold # For safety settings


logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
}

class ScrapeError(Exception):
    """Raised when a page cannot be fetched or parsed."""


def _call_gemini_api_for_insights(description: str, gemini_model_instance: genai.GenerativeModel) -> dict:
    """
    Helper function to interact with Gemma model via Google AI Gemini API for extracting insights.
    """
    prompt = f"""Your task is to extract specific details from the job description text provided below.

Job Description Text:
---
{description}
---

Extract the following details if available:
- Salary (e.g., "$50k-$60k", "€70,000 per year")
- Deadline (application deadline, format as YYYY-MM-DD if possible. If not, provide the raw text.)
- Timeframe (duration of the job/internship, e.g., "12 weeks", "Summer 2025", "3 months")
- Start Date (Format as YYYY-MM-DD if possible. If not, provide the raw text.)

Present the extracted information in a structured format, with each piece of information on a new line:
Salary: [Extracted Salary Information]
Deadline: [Extracted Deadline]
Timeframe: [Extracted Timeframe]
Start Date: [Extracted Start Date]

If any piece of information cannot be found, use the value "Not found" for that specific field.
"""
    insights = {"salary": None, "deadline": None, "timeframe": None, "start_date": None}

    generation_config = genai.types.GenerationConfig(
        max_output_tokens=200, # Adjust as needed
        temperature=0.2,
    )
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    
    try:
        response = gemini_model_instance.generate_content(
            contents=[prompt],
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        if not response.candidates:
             if response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.warning(f"Insights call blocked by API. Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}")
                return insights
             for candidate in response.candidates: # Should not happen if candidates list is empty
                 if candidate.finish_reason == genai.types.FinishReason.SAFETY:
                    logger.warning(f"Insights extraction stopped due to safety reasons. Ratings: {candidate.safety_ratings}")
                    return insights

        gemini_response_text = response.text
        logger.debug(f"Google AI Gemma raw response for insights: {gemini_response_text}")

        for line in gemini_response_text.split('\n'):
            if ":" in line:
                key, value = line.split(":", 1)
                key_formatted = key.strip().lower().replace(" ", "_") # e.g., "start date" -> "start_date"
                value_stripped = value.strip()

                if value_stripped.lower() == "not found":
                    value_stripped = None # Store None if explicitly "Not found"
                
                if key_formatted == "salary" and value_stripped:
                    insights["salary"] = value_stripped
                elif key_formatted == "timeframe" and value_stripped:
                    insights["timeframe"] = value_stripped
                elif (key_formatted == "deadline" or key_formatted == "start_date") and value_stripped:
                    try:
                        parsed_date = dateparser.parse(value_stripped, settings={'PREFER_DATES_FROM': 'future', 'STRICT_PARSING': False})
                        if parsed_date:
                            insights[key_formatted] = parsed_date.strftime("%Y-%m-%d")
                        else:
                            insights[key_formatted] = value_stripped # Keep raw if not parsable
                    except Exception:
                        insights[key_formatted] = value_stripped # Keep raw on error
        
    except Exception as e:
        logger.error(f"Error during Google AI Gemma interaction for insights: {e}", exc_info=True)
        # Minimal fallback if Gemma fails
        return insights

    # Fallback regex for salary if Gemma didn't find it
    if not insights.get("salary"): # Check if salary is None or empty
        salary_patterns = [
            r'[\$€£]\s?\d{1,3}(?:[,\.\s]?\d{3})*\s?[Kk]?\s?(?:(?:to|-|–)\s?[\$€£]?\s?\d{1,3}(?:[,\.\s]?\d{3})*\s?[Kk]?)?',
            r'\d{1,3}(?:[,\.\s]?\d{3})*\s?[Kk]?\s?[\$€£]\s?(?:(?:to|-|–)\s?\d{1,3}(?:[,\.\s]?\d{3})*\s?[Kk]?\s?[\$€£]?)?'
        ]
        raw_salaries = []
        for pattern in salary_patterns:
            match = re.search(pattern, description) # Search instead of findall to get first good match
            if match:
                raw_salaries.append(match.group(0))
        if raw_salaries:
            insights["salary"] = raw_salaries[0].strip()
            logger.debug(f"Found potential salary via regex fallback: {insights['salary']}")
            
    return insights


def extract_insights_from_description(desc: str, gemini_model_instance: genai.GenerativeModel) -> dict:
    insights = {"salary": None, "deadline": None, "timeframe": None, "start_date": None}
    if not desc or not gemini_model_instance:
        return insights # Return empty insights if no description or model
    
    insights = _call_gemini_api_for_insights(desc, gemini_model_instance)
    
    logger.debug(f"Google AI Gemma extracted insights: {insights}")
    return insights

# --- Network fetch --- (This function remains unchanged)
def fetch_page(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status() 
        return resp.text
    except requests.exceptions.Timeout:
        logger.error(f"Timeout while fetching {url}")
        raise ScrapeError(f"Timeout: Could not fetch {url} within 15 seconds.")
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error fetching {url}: {http_err}")
        raise ScrapeError(f"HTTP error: {http_err.response.status_code} for {url}.")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception fetching {url}: {req_err}")
        raise ScrapeError(f"Network error: Failed to fetch {url} ({req_err}).")
    except Exception as exc:
        logger.error(f"Generic exception fetching {url}: {exc}", exc_info=True)
        raise ScrapeError(f"Failed to fetch {url}: {exc}")


# --- Dispatcher ---
# The `gemma_model_vertex` parameter is now `gemini_model_instance`
def extract_text(html: str, url: str, gemini_model_instance: genai.GenerativeModel) -> tuple[str, dict]:
    dom = BeautifulSoup(html, "lxml")
    host = urlparse(url).netloc.lower()

    if "indeed." in host:
        return _extract_indeed(dom, gemini_model_instance)
    elif "linkedin." in host:
        return _extract_linkedin(dom, gemini_model_instance)
    
    logger.info(f"Using generic fallback extractor for {url}")
    body_text = ""
    main_content_selectors = ['article', 'main', '[role="main"]', '.job-description', '#jobDescriptionText', '.description__text']
    for selector in main_content_selectors:
        main_el = dom.select_one(selector)
        if main_el:
            body_text = _text_of(main_el)
            break
    if not body_text:
        body_text = _text_of(dom.body) if dom.body else dom.get_text(" ", strip=True)
    
    meta = {} 
    if body_text and gemini_model_instance: 
         insights = extract_insights_from_description(body_text, gemini_model_instance)
         meta.update(insights)

    return body_text, meta


# --- Helpers --- (This function remains unchanged)
def _text_of(el: Tag | None) -> str:
    return el.get_text(" ", strip=True) if el else ""


# --- Indeed ---
# The `gemma_model_vertex` parameter is now `gemini_model_instance`
def _extract_indeed(dom: BeautifulSoup, gemini_model_instance: genai.GenerativeModel) -> tuple[str, dict]:
    logger.debug("Extracting content from Indeed page.")
    title = _text_of(dom.select_one("h1.jobsearch-JobInfoHeader-title"))
    # Updated selectors based on typical Indeed structure
    company_info_div = dom.select_one('div[data-testid="job-info-header"] div[data-company-name="true"], div.jobsearch-CompanyInfoContainer') 
    company_raw = _text_of(company_info_div.find('a') if company_info_div and company_info_div.find('a') else company_info_div) # Prefer link text if available
    company = company_raw.split('-')[0].strip() if company_raw else ""


    location_div = dom.select_one('div[data-testid="job-info-header"] div[data-testid="inlineHeader-companyLocation"], div.jobsearch-CompanyInfoContainer div:nth-child(2)') # trying to get location
    location = _text_of(location_div).split('Location')[1].strip() if location_div and 'Location' in _text_of(location_div) else _text_of(location_div)
    location = location.split('\n')[0].strip() # Often location is followed by other details in new lines


    body_el = dom.select_one("div#jobDescriptionText")
    body = _text_of(body_el)

    meta = {"company": company.strip(), "position": title.strip(), "location": location.strip()}
    if body and gemini_model_instance: 
        insights = extract_insights_from_description(body, gemini_model_instance)
        meta.update(insights)
        
    logger.debug(f"Indeed extracted meta: {meta}")
    return body or dom.get_text(" ", strip=True), meta


# --- LinkedIn ---
# The `gemma_model_vertex` parameter is now `gemini_model_instance`
def _extract_linkedin(dom: BeautifulSoup, gemini_model_instance: genai.GenerativeModel) -> tuple[str, dict]:
    logger.debug("Extracting content from LinkedIn page.")
    title    = _text_of(dom.select_one("h1.top-card-layout__title, h1.job-title, .job-details-jobs-unified-top-card__job-title, .topcard__title"))
    company  = _text_of(dom.select_one("a.topcard__org-name-link, span.topcard__flavor, a.job-card-container__company-name, .job-details-jobs-unified-top-card__company-name a, .topcard__flavor:first-child"))
    location = _text_of(dom.select_one("span.topcard__flavor--bullet, span.topcard__flavor:last-child, .job-card-container__metadata-item, .job-details-jobs-unified-top-card__primary-description-container > div:nth-child(2) span.tvm__text, .topcard__flavor-label + .topcard__flavor--bullet"))


    body_el  = (dom.select_one("div.description__text section.show-more-less-html") # Prioritize the actual description section
                or dom.select_one("div.description__text")
                or dom.select_one("div.show-more-less-html__markup")
                or dom.select_one("#job-details") 
                or dom.select_one(".jobs-description__content .jobs-box__html-content") 
                )
    body = ""
    if body_el:
        body = _text_of(body_el)
    
    meta = {"company": company.strip(), "position": title.strip(), "location": location.strip()}
    if body and gemini_model_instance: 
        insights = extract_insights_from_description(body, gemini_model_instance)
        meta.update(insights)

    logger.debug(f"LinkedIn extracted meta: {meta}")
    return body or dom.get_text(" ", strip=True), meta