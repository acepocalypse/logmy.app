from __future__ import annotations

import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urlparse
import logging
import re
import dateparser

# Import for Google AI (Gemini API / google-genai)
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

class ScrapeError(Exception):
    """Raised when a page cannot be fetched or parsed."""

# --- _call_gemini_api_for_insights remains the same ---
def _call_gemini_api_for_insights(description: str, gemini_model_instance: genai.GenerativeModel) -> dict:
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
    generation_config = genai.types.GenerationConfig(max_output_tokens=256, temperature=1)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    try:
        response = gemini_model_instance.generate_content(
            contents=[prompt], generation_config=generation_config, safety_settings=safety_settings
        )
        if not response.candidates: return insights
        if response.candidates[0].finish_reason == genai.types.FinishReason.SAFETY: return insights
        gemini_response_text = response.text
        for line in gemini_response_text.split('\n'):
            if ":" in line:
                key, value = line.split(":", 1)
                key_formatted = key.strip().lower().replace(" ", "_")
                value_stripped = value.strip()
                if value_stripped.lower() == "not found": value_stripped = None
                if key_formatted in insights and value_stripped:
                    if (key_formatted == "deadline" or key_formatted == "start_date"):
                        try:
                            parsed_date = dateparser.parse(value_stripped, settings={'PREFER_DATES_FROM': 'future', 'STRICT_PARSING': False})
                            insights[key_formatted] = parsed_date.strftime("%Y-%m-%d") if parsed_date else value_stripped
                        except Exception: insights[key_formatted] = value_stripped
                    else: insights[key_formatted] = value_stripped
    except Exception as e: logger.error(f"Error during Google AI Gemma interaction for insights: {e}", exc_info=True)
    return insights

# --- extract_insights_from_description remains the same ---
def extract_insights_from_description(desc: str, gemini_model_instance: genai.GenerativeModel) -> dict:
    insights = {"salary": None, "deadline": None, "timeframe": None, "start_date": None, "job_type": None}
    if not desc or not gemini_model_instance: return insights
    extracted_insights = _call_gemini_api_for_insights(desc, gemini_model_instance)
    insights.update(extracted_insights)
    return insights

# --- fetch_page uses requests ---
def fetch_page(url: str) -> str:
    logger.info(f"Attempting to fetch page with requests: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=20, allow_redirects=True)
        logger.info(f"Requests fetch for {url} returned status code: {response.status_code}")
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as exc:
        logger.error(f"Requests error fetching {url}: {exc}", exc_info=True)
        raise ScrapeError(f"Requests failed to fetch {url}: {exc}")

# --- extract_text and helpers remain largely the same, using BS4 ---
def extract_text(html: str, url: str, gemini_model_instance: genai.GenerativeModel) -> tuple[str, dict]:
    dom = BeautifulSoup(html, "lxml")
    host = urlparse(url).netloc.lower()
    if "indeed." in host: return _extract_indeed(dom, gemini_model_instance)
    elif "linkedin." in host: return _extract_linkedin(dom, gemini_model_instance)
    logger.info(f"Using generic body text extractor for {url}")
    body_text = _text_of(dom.body) if dom.body else dom.get_text(" ", strip=True)
    meta = {}
    if body_text and gemini_model_instance: meta.update(extract_insights_from_description(body_text, gemini_model_instance))
    return body_text, meta

def _text_of(el: Tag | None) -> str:
    if not el: return ""
    return el.get_text(" ", strip=True)

def _extract_indeed(dom: BeautifulSoup, gemini_model_instance: genai.GenerativeModel) -> tuple[str, dict]:
    title = _text_of(dom.select_one("h1.jobsearch-JobInfoHeader-title"))
    company_el = dom.select_one('div[data-testid="job-info-header"] div[data-company-name="true"]')
    company = _text_of(company_el.find('a') if company_el.find('a') else company_el) if company_el else ""
    location = _text_of(dom.select_one('div[data-testid="job-info-header"] div[data-testid="inlineHeader-companyLocation"]')).replace("Location", "").strip()
    body = _text_of(dom.select_one("div#jobDescriptionText"))
    meta = {"company": company.strip(), "position": title.strip(), "location": location.strip()}
    if body and gemini_model_instance: meta.update(extract_insights_from_description(body, gemini_model_instance))
    return body or dom.get_text(" ", strip=True), meta

def _extract_linkedin(dom: BeautifulSoup, gemini_model_instance: genai.GenerativeModel) -> tuple[str, dict]:
    title = _text_of(dom.select_one("h1.top-card-layout__title, .topcard__title"))
    company = _text_of(dom.select_one("a.topcard__org-name-link, .topcard__flavor:first-child"))
    location = _text_of(dom.select_one("span.topcard__flavor--bullet"))
    body = _text_of(dom.select_one("div.description__text, #job-details"))
    meta = {"company": company.strip(), "position": title.strip(), "location": location.strip()}
    if body and gemini_model_instance: meta.update(extract_insights_from_description(body, gemini_model_instance))
    return body or dom.get_text(" ", strip=True), meta