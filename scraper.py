import requests
import xmltodict
import json
from datetime import datetime, timedelta

# --- Configuration ---
# Define the arXiv categories you are interested in.
# These align with your project's focus on AI, ML, and Computer Vision.
ARXIV_CATEGORIES = ["cs.AI", "cs.CV", "cs.LG", "cs.CL", "cs.NE"]
MAX_RESULTS = 20  # Max papers to fetch (e.g., last 100 papers in the categories)
OUTPUT_FILENAME = "arxiv_papers.jsonl" # Using .jsonl for easy streaming ingestion

def fetch_and_save_arxiv_papers():
    """
    Fetches the latest papers from specified arXiv categories,
    parses the XML response, and saves the data as a JSON Lines file.
    This file can be used as a static data source for a Pathway pipeline.
    """
    print("🚀 Starting arXiv paper fetch...")

    # Construct the search query for the arXiv API
    # We are searching for papers in any of our specified categories.
    search_query = "+OR+".join([f"cat:{cat}" for cat in ARXIV_CATEGORIES])

    # arXiv API endpoint
    base_url = "http://export.arxiv.org/api/query?"
    query = f"search_query={search_query}&sortBy=submittedDate&sortOrder=descending&max_results={MAX_RESULTS}"
    
    full_url = base_url + query
    print(f"Querying API: {full_url}")

    try:
        # Make the GET request to the arXiv API
        response = requests.get(full_url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return

    print("✅ Successfully fetched data from arXiv.")
    print("... Parsing XML response to JSON format.")

    # Parse the XML response using xmltodict
    data_dict = xmltodict.parse(response.content)

    papers = []
    entries = data_dict.get('feed', {}).get('entry', [])
    if not isinstance(entries, list): # Handle case where only one entry is returned
        entries = [entries]

    for entry in entries:
        try:
            # Extract authors: handles single or multiple authors
            authors_data = entry.get('author', [])
            if isinstance(authors_data, list):
                authors = [author['name'] for author in authors_data]
            else:
                authors = [authors_data.get('name', 'N/A')]

            # Extract PDF link
            links = entry.get('link', [])
            if not isinstance(links, list):
                links = [links] # Handle single link case
            
            pdf_url = ""
            for link in links:
                if link.get('@title') == 'pdf':
                    pdf_url = link.get('@href')
                    break
            
            # Construct a clean dictionary for each paper
            paper_info = {
                "id": entry.get('id', 'N/A').split('/abs/')[-1], # Clean the ID
                "title": entry.get('title', 'N/A').strip().replace('\n', ' '),
                "abstract": entry.get('summary', 'N/A').strip().replace('\n', ' '),
                "authors": authors,
                "published_date": entry.get('published', 'N/A'),
                "url": entry.get('id', 'N/A'),
                "pdf_url": pdf_url,
                "primary_category": entry.get('arxiv:primary_category', {}).get('@term', 'N/A'),
            }
            papers.append(paper_info)

        except Exception as e:
            print(f"⚠️ Warning: Could not parse an entry. Skipping. Error: {e}")

    # Save the data to a JSON Lines file
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            for paper in papers:
                f.write(json.dumps(paper) + '\n')
        print(f"✅ Successfully saved {len(papers)} papers to '{OUTPUT_FILENAME}'.")
    except IOError as e:
        print(f"❌ Failed to write to file: {e}")


if __name__ == "__main__":
    fetch_and_save_arxiv_papers()