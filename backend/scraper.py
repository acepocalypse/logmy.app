from __future__ import annotations

import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urlparse
import logging
import re
import dateparser
# import time # No longer strictly needed without Selenium sleep

# Import for Google AI (Gemini API / google-genai)
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold # For safety settings

# REMOVED Selenium Imports

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36" # Using a common User-Agent
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "DNT": "1", # Do Not Track header
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
    insights = {"salary": None, "deadline": None, "timeframe": None, "start_date": None, "job_type": None}
    if not desc or not gemini_model_instance:
        return insights
    extracted_insights = _call_gemini_api_for_insights(desc, gemini_model_instance)
    insights.update(extracted_insights)
    logger.debug(f"Google AI Gemma extracted insights: {insights}")
    return insights

# REVERTED fetch_page to use requests
def fetch_page(url: str) -> str:
    logger.info(f"Attempting to fetch page with requests: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=20, allow_redirects=True)
        logger.info(f"Requests fetch for {url} returned status code: {response.status_code}")
        logger.debug(f"Response Headers: {response.headers}")

        # Specifically log LinkedIn content to see what we get
        if "linkedin." in urlparse(url).netloc.lower():
             logger.warning(f"LinkedIn Response (Status: {response.status_code}). Text (first 1000 chars): {response.text[:1000]}")

        # Check if the status code indicates a block or redirect (e.g., 999 is LinkedIn's 'Forbidden')
        if response.status_code == 999:
            raise ScrapeError(f"LinkedIn returned status 999 (Blocked/Forbidden) for {url}")

        response.raise_for_status() # Raise HTTPError for other bad responses (4xx or 5xx)

        logger.info(f"Requests successfully fetched page.")
        return response.text
    except requests.exceptions.Timeout:
        logger.error(f"Timeout error fetching {url}", exc_info=True)
        raise ScrapeError(f"Timeout while trying to fetch {url}")
    except requests.exceptions.RequestException as exc:
        logger.error(f"Requests error fetching {url}: {exc}", exc_info=True)
        raise ScrapeError(f"Requests failed to fetch {url}: {exc}")


def extract_text(html: str, url: str, gemini_model_instance: genai.GenerativeModel) -> tuple[str, dict]:
    dom = BeautifulSoup(html, "lxml")
    host = urlparse(url).netloc.lower()

    # If LinkedIn returns a page, it's often a login/authwall page without JS.
    # BS4 *will* parse it, but it won't be the job description.
    # We rely on the fetch_page logs to see what's happening.
    if "linkedin." in host:
        return _extract_linkedin(dom, gemini_model_instance)
    elif "indeed." in host:
        return _extract_indeed(dom, gemini_model_instance)

    logger.info(f"Using generic body text extractor for {url}")
    body_text = _text_of(dom.body) if dom.body else dom.get_text(" ", strip=True)

    meta = {}
    if body_text and gemini_model_instance:
         insights = extract_insights_from_description(body_text, gemini_model_instance)
         meta.update(insights)

    return body_text, meta


def _text_of(el: Tag | None) -> str:
    if not el:
        return ""
    return el.get_text(" ", strip=True)


def _extract_indeed(dom: BeautifulSoup, gemini_model_instance: genai.GenerativeModel) -> tuple[str, dict]:
    logger.debug("Extracting content from Indeed page.")
    title = _text_of(dom.select_one("h1.jobsearch-JobInfoHeader-title"))
    company_el = dom.select_one('div[data-testid="job-info-header"] div[data-company-name="true"]')
    if not company_el: # Fallback selectors
        company_el = dom.select_one('div.jobsearch-CompanyInfoContainer meta[itemprop="name"]')
        company = company_el['content'].strip() if company_el and company_el.has_attr('content') else ""
    else:
        company = _text_of(company_el.find('a') if company_el.find('a') else company_el)
    if not company: # Further fallback
        company_raw_el = dom.select_one('div.jobsearch-InlineCompanyRating div:first-child, .jobsearch-DesktopStickyContainer-companyrating div:first-child')
        company = _text_of(company_raw_el).split('-')[0].strip() if company_raw_el else ""
    location_el = dom.select_one('div[data-testid="job-info-header"] div[data-testid="inlineHeader-companyLocation"]')
    if not location_el: # Fallback selectors
        location_el = dom.select_one('div.jobsearch-CompanyInfoContainer div[data-testid="companyLocation"]')
    location_text = _text_of(location_el)
    location = location_text.replace("Location", "").strip() if location_text else ""
    location = location.split('\n')[0].strip()
    body_el = dom.select_one("div#jobDescriptionText")
    body = _text_of(body_el)
    meta = {"company": company.strip(), "position": title.strip(), "location": location.strip()}
    if body and gemini_model_instance:
        insights = extract_insights_from_description(body, gemini_model_instance)
        meta.update(insights)
    logger.debug(f"Indeed extracted meta: {meta}")
    return body or dom.get_text(" ", strip=True), meta


def _extract_linkedin(dom: BeautifulSoup, gemini_model_instance: genai.GenerativeModel) -> tuple[str, dict]:
    logger.debug("Extracting content from LinkedIn page (using BS4).")
    # These selectors MIGHT work if LinkedIn serves a basic HTML, but often won't.
    title    = _text_of(dom.select_one("h1.top-card-layout__title, h1.job-title, .job-details-jobs-unified-top-card__job-title, .topcard__title, .jobs-unified-top-card__title"))
    company  = _text_of(dom.select_one("a.topcard__org-name-link, span.topcard__flavor:first-of-type, a.job-card-container__company-name, .job-details-jobs-unified-top-card__company-name a, .topcard__flavor:first-child, .jobs-unified-top-card__company-name"))
    location = _text_of(dom.select_one("span.topcard__flavor--bullet, span.topcard__flavor:nth-of-type(2), .job-details-jobs-unified-top-card__primary-description-container > div:nth-child(2) span.tvm__text, .jobs-unified-top-card__job-insight span.jobs-unified-top-card__bullet"))
    body_el  = (dom.select_one("div.description__text section.show-more-less-html")
                or dom.select_one("div.description__text")
                or dom.select_one("#job-details")
                or dom.select_one(".jobs-description__content")
                )
    body = _text_of(body_el)

    # Log what BS4 found, it might be very little.
    logger.warning(f"BS4 LinkedIn Extraction: Title='{title}', Company='{company}', Location='{location}', Body_Len={len(body)}")

    meta = {"company": company.strip(), "position": title.strip(), "location": location.strip()}
    if body and gemini_model_instance:
        insights = extract_insights_from_description(body, gemini_model_instance)
        meta.update(insights)

    logger.debug(f"LinkedIn extracted meta (BS4): {meta}")
    # Return whatever BS4 managed to find, even if it's just the body of a login page.
    return body or dom.get_text(" ", strip=True), meta