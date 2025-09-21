import os
import time
import requests
import google.generativeai as genai
from typing import Generator, Dict, Any

# Configure the Gemini API with your API key
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    raise RuntimeError("Please set the GOOGLE_API_KEY environment variable.")

def get_arxiv_pdf_content(paper_id: str) -> bytes:
    """
    Downloads the content of an arXiv PDF as raw bytes.
    """
    url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve {url}: {e}")
        return b"" # Return an empty byte string on error

def scrape_arxiv_papers(paper_ids: list[str], refresh_interval: int) -> Generator[Dict[str, Any], None, None]:
    """
    A generator function that scrapes the content of arXiv papers over time.
    """
    processed_ids = set()
    while True:
        for paper_id in paper_ids:
            # Only scrape new papers or papers not yet processed
            if paper_id not in processed_ids:
                print(f"🔄 Scraping data for paper ID: {paper_id}...")
                
                # We don't need a separate HTML scraper anymore, the Pathway pipeline
                # will download the PDF and use UnstructuredParser.
                # We just need to feed the paper ID into the pipeline.
                yield {"paper_id": paper_id}
                processed_ids.add(paper_id)
        
        print(f"💤 Waiting for {refresh_interval} seconds before next scrape cycle.")
        time.sleep(refresh_interval)

def classify_with_gemini(paper_info: Dict[str, Any], subdomains: list[str]) -> str:
    """
    Uses the Gemini API to classify the paper's content.
    This function is designed to be a Pathway UDF.
    """
    if "error" in paper_info or not paper_info["abstract"]:
        return "Classification failed: Missing paper information."

    prompt = f"""
    You are an expert AI researcher. Classify the following paper into one or more of the following subdomains: {', '.join(subdomains)}.
    If no subdomain fits, choose 'Other'.

    **Paper Details:**
    Title: {paper_info['title']}
    Authors: {', '.join(paper_info['authors'])}
    Abstract: {paper_info['abstract']}

    **Instructions:**
    - Provide only the subdomain(s) as a comma-separated list.
    - Do not include any extra text, explanations, or formatting.
    """
    model = genai.GenerativeModel('gemini-pro')
    try:
        response = model.generate_content(prompt)
        classification = response.text.strip()
        return classification
    except Exception as e:
        return f"Classification failed: {e}"