# backend/scraper.py

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
    prompt = f"""Your task is to carefully analyze the job description text provided below and extract specific details. Focus on conciseness and accuracy for each field.

Job Description Text:
---
{description}
---

Extract the following details if available. Provide only the direct information for each field.

- Salary: The compensation offered, including currency and pay period if mentioned (e.g., "$50k-$60k per year", "€20/hour").
- Deadline: The application deadline. Format as YYYY-MM-DD if possible. If not, provide the raw text describing the deadline.
- Timeframe: The duration of the job or internship, if specified (e.g., "12 weeks", "Summer 2025", "3 months", "Permanent").
- Start Date: The anticipated start date for the position. Format as YYYY-MM-DD if possible. If not, provide the raw text.
- Job Type: The type of employment (e.g., "Full-time", "Part-time", "Internship", "Contract", "Temporary").

Present the extracted information in a structured format, with each piece of information on a new line:
Salary: [Extracted Salary Information]
Deadline: [Extracted Deadline]
Timeframe: [Extracted Timeframe]
Start Date: [Extracted Start Date]
Job Type: [Extracted Job Type]

If any piece of information cannot be clearly identified from the text, use the value "Not found" for that specific field. Avoid including surrounding sentences.
"""
    insights = {"salary": None, "deadline": None, "timeframe": None, "start_date": None, "job_type": None}

    generation_config = genai.types.GenerationConfig(
        max_output_tokens=256,
        temperature=1,
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
             logger.warning("Insights call returned no candidates without explicit blocking reason from prompt_feedback.")
             return insights

        if response.candidates[0].finish_reason == genai.types.FinishReason.SAFETY:
            logger.warning(f"Insights extraction stopped due to safety reasons. Ratings: {response.candidates[0].safety_ratings}")
            return insights
        
        gemini_response_text = response.text
        logger.debug(f"Google AI Gemma raw response for insights: {gemini_response_text}")

        for line in gemini_response_text.split('\n'):
            if ":" in line:
                key, value = line.split(":", 1)
                key_formatted = key.strip().lower().replace(" ", "_") 
                value_stripped = value.strip()

                if value_stripped.lower() == "not found":
                    value_stripped = None 
                
                if key_formatted == "salary" and value_stripped:
                    insights["salary"] = value_stripped
                elif key_formatted == "timeframe" and value_stripped:
                    insights["timeframe"] = value_stripped
                elif key_formatted == "job_type" and value_stripped: # Added job_type parsing
                    insights["job_type"] = value_stripped
                elif (key_formatted == "deadline" or key_formatted == "start_date") and value_stripped:
                    try:
                        parsed_date = dateparser.parse(value_stripped, settings={'PREFER_DATES_FROM': 'future', 'STRICT_PARSING': False})
                        if parsed_date:
                            insights[key_formatted] = parsed_date.strftime("%Y-%m-%d")
                        else:
                            insights[key_formatted] = value_stripped 
                    except Exception:
                        insights[key_formatted] = value_stripped 
        
    except Exception as e:
        logger.error(f"Error during Google AI Gemma interaction for insights: {e}", exc_info=True)
        return insights # Return partially filled or empty insights on error

    # Fallback regex for salary if Gemma didn't find it
    if not insights.get("salary"):
        salary_patterns = [
            r'[\$€£]\s?\d{1,3}(?:[,\.\s]?\d{3})*\s?[Kk]?\s?(?:(?:to|-|–)\s?[\$€£]?\s?\d{1,3}(?:[,\.\s]?\d{3})*\s?[Kk]?)?',
            r'\d{1,3}(?:[,\.\s]?\d{3})*\s?[Kk]?\s?[\$€£]\s?(?:(?:to|-|–)\s?\d{1,3}(?:[,\.\s]?\d{3})*\s?[Kk]?\s?[\$€£]?)?'
        ]
        raw_salaries = []
        for pattern in salary_patterns:
            match = re.search(pattern, description) 
            if match:
                raw_salaries.append(match.group(0))
        if raw_salaries:
            insights["salary"] = raw_salaries[0].strip()
            logger.debug(f"Found potential salary via regex fallback: {insights['salary']}")
            
    return insights


def extract_insights_from_description(desc: str, gemini_model_instance: genai.GenerativeModel) -> dict:
    # Ensure the initial dictionary includes job_type so it's always present in the return
    insights = {"salary": None, "deadline": None, "timeframe": None, "start_date": None, "job_type": None}
    if not desc or not gemini_model_instance:
        return insights 
    
    # Update the insights dictionary with results from the API call
    extracted_insights = _call_gemini_api_for_insights(desc, gemini_model_instance)
    insights.update(extracted_insights) # Merge results, preferring those from API
    
    logger.debug(f"Google AI Gemma extracted insights: {insights}")
    return insights

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


def extract_text(html: str, url: str, gemini_model_instance: genai.GenerativeModel) -> tuple[str, dict]:
    dom = BeautifulSoup(html, "lxml")
    host = urlparse(url).netloc.lower()

    if "indeed." in host:
        return _extract_indeed(dom, gemini_model_instance)
    elif "linkedin." in host:
        return _extract_linkedin(dom, gemini_model_instance)
    
    # For generic websites, immediately use the entire body text
    logger.info(f"Using generic body text extractor for {url}")
    body_text = _text_of(dom.body) if dom.body else dom.get_text(" ", strip=True)
    
    meta = {} 
    if body_text and gemini_model_instance: 
         insights = extract_insights_from_description(body_text, gemini_model_instance)
         meta.update(insights) # This will now include 'job_type' if found by the AI

    return body_text, meta


def _text_of(el: Tag | None) -> str:
    """Extracts text from a BeautifulSoup Tag, joining with spaces and stripping."""
    if not el:
        return ""
    # Consider more sophisticated text extraction if needed, e.g., handling <br> as newlines
    # For now, get_text with " " separator is a good general approach.
    return el.get_text(" ", strip=True)


def _extract_indeed(dom: BeautifulSoup, gemini_model_instance: genai.GenerativeModel) -> tuple[str, dict]:
    logger.debug("Extracting content from Indeed page.")
    title = _text_of(dom.select_one("h1.jobsearch-JobInfoHeader-title"))
    
    # Attempt to find company name more reliably
    company_el = dom.select_one('div[data-testid="job-info-header"] div[data-company-name="true"]')
    if not company_el: # Fallback selectors
        company_el = dom.select_one('div.jobsearch-CompanyInfoContainer meta[itemprop="name"]')
        company = company_el['content'].strip() if company_el and company_el.has_attr('content') else ""
    else:
        company = _text_of(company_el.find('a') if company_el.find('a') else company_el)

    if not company: # Further fallback
        company_raw_el = dom.select_one('div.jobsearch-InlineCompanyRating div:first-child, .jobsearch-DesktopStickyContainer-companyrating div:first-child')
        company = _text_of(company_raw_el).split('-')[0].strip() if company_raw_el else ""


    # Attempt to find location more reliably
    location_el = dom.select_one('div[data-testid="job-info-header"] div[data-testid="inlineHeader-companyLocation"]')
    if not location_el: # Fallback selectors
        location_el = dom.select_one('div.jobsearch-CompanyInfoContainer div[data-testid="companyLocation"]')
    location_text = _text_of(location_el)
    # Clean up location text if it contains "Location" prefix or other noise
    location = location_text.replace("Location", "").strip() if location_text else ""
    location = location.split('\n')[0].strip() # Take first line if multiple


    body_el = dom.select_one("div#jobDescriptionText")
    body = _text_of(body_el)

    meta = {"company": company.strip(), "position": title.strip(), "location": location.strip()}
    if body and gemini_model_instance: 
        insights = extract_insights_from_description(body, gemini_model_instance)
        meta.update(insights)
        
    logger.debug(f"Indeed extracted meta: {meta}")
    return body or dom.get_text(" ", strip=True), meta


def _extract_linkedin(dom: BeautifulSoup, gemini_model_instance: genai.GenerativeModel) -> tuple[str, dict]:
    logger.debug("Extracting content from LinkedIn page.")
    title    = _text_of(dom.select_one("h1.top-card-layout__title, h1.job-title, .job-details-jobs-unified-top-card__job-title, .topcard__title"))
    company  = _text_of(dom.select_one("a.topcard__org-name-link, span.topcard__flavor:first-of-type, a.job-card-container__company-name, .job-details-jobs-unified-top-card__company-name a, .topcard__flavor:first-child"))
    
    # LinkedIn location can be tricky, try a few selectors
    location_el = dom.select_one("span.topcard__flavor--bullet") # Often the most reliable
    if not location_el:
        location_el = dom.select_one("span.topcard__flavor:nth-of-type(2)") # Second flavor item
    if not location_el:
        location_el = dom.select_one(".job-details-jobs-unified-top-card__primary-description-container > div:nth-child(2) span.tvm__text") # More specific path
    location = _text_of(location_el)


    body_el  = (dom.select_one("div.description__text section.show-more-less-html") 
                or dom.select_one("div.description__text")
                or dom.select_one("div.show-more-less-html__markup")
                or dom.select_one("#job-details section.description") # More specific section
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
    logger.debug(f"LinkedIn Scraper (BeautifulSoup) extracted: Company='{company.strip()}', Position='{title.strip()}', Location='{location.strip()}'")
    logger.debug(f"LinkedIn extracted meta: {meta}")
    return body or dom.get_text(" ", strip=True), meta