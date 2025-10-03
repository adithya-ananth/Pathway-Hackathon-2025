import requests
import xmltodict
import json
import urllib.parse
import os

# --- Configuration ---
# --- MODIFICATION: Read keywords from a config file ---
CONFIG_FILENAME = "config.json"
MAX_RESULTS = 3
OUTPUT_FILENAME = "arxiv_papers.jsonl"

def load_keywords_from_config(filename: str) -> list:
    """Loads search keywords from a JSON configuration file."""
    if not os.path.exists(filename):
        print(f"Warning: Config file '{filename}' not found.")
        print("   Using default fallback keywords: ['computer vision', 'robotics']")
        return ["computer vision", "robotics"] # Fallback keywords
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            keywords = config_data.get("search_keywords", [])
            if not keywords:
                raise ValueError("No keywords found in config file.")
            print(f"Successfully loaded keywords from '{filename}': {keywords}")
            return keywords
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading config file '{filename}': {e}")
        print("   Using default fallback keywords: ['computer vision', 'robotics']")
        return ["computer vision", "robotics"] # Fallback keywords


def fetch_and_save_arxiv_papers():
    """
    Fetches the latest paper metadata from arXiv based on keywords from the config
    file and saves it as a JSON Lines file.
    """
    print("\n Starting arXiv paper metadata fetch...")
    
    # Load keywords from the external config file
    search_keywords = load_keywords_from_config(CONFIG_FILENAME)
    
    # Build the search query from the loaded keywords
    search_query = "+".join([f'all:"{urllib.parse.quote(kw)}"' for kw in search_keywords])
    
    # relavance
    query1 = f"search_query={search_query}&sortBy=relevance&sortOrder=descending&max_results={MAX_RESULTS}"
    full_url1 = "http://export.arxiv.org/api/query?" + query1

    # latest
    query2 = f"search_query={search_query}&sortBy=submittedDate&sortOrder=descending&max_results={MAX_RESULTS}"
    full_url2 = "http://export.arxiv.org/api/query?" + query2

    print("URL = ",full_url1)
    
    print(f"   Querying API: {full_url1[:150]}...") # Print a truncated URL
    try:
        response = requests.get(full_url1)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return

    print("Successfully fetched data from arXiv.")
    data_dict = xmltodict.parse(response.content)
    
    entries = data_dict.get('feed', {}).get('entry', [])
    if not isinstance(entries, list):
        entries = [entries] if entries else []

    print("URL = ",full_url2)
    
    print(f"   Querying API: {full_url2[:150]}...") # Print a truncated URL
    try:
        response = requests.get(full_url2)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return

    print("Successfully fetched data from arXiv.")
    data_dict = xmltodict.parse(response.content)
    
    entries2 = data_dict.get('feed', {}).get('entry', [])
    if not isinstance(entries2, list):
        entries2 = [entries2] if entries2 else []

    if not entries and not entries2:
        print("No papers found for the given keywords. The output file will be empty.")
        return

    entries = entries + entries2

    papers = []
    for entry in entries:
        # ... (rest of the parsing logic is the same as before) ...
        try:
            authors_data = entry.get('author', [])
            authors = [author['name'] for author in authors_data] if isinstance(authors_data, list) else [authors_data.get('name')]
            links = entry.get('link', [])
            pdf_url = next((link.get('@href') for link in links if link.get('@title') == 'pdf'), None)
            paper_info = {
                "doc_id": entry.get('id', 'N/A').split('/abs/')[-1],
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
            print(f"Warning: Could not parse an entry. Skipping. Error: {e}")

    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            for paper in papers:
                f.write(json.dumps(paper) + '\n')
        print(f"\nSuccessfully saved {len(papers)} paper metadata entries to '{OUTPUT_FILENAME}'.")
    except IOError as e:
        print(f"Failed to write to file: {e}")


if __name__ == "__main__":
    fetch_and_save_arxiv_papers()