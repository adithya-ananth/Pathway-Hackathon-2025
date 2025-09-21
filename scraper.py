import requests
import xmltodict
import json
import urllib.parse

# --- Configuration ---
# --- MODIFICATION: Define search keywords instead of categories ---
# Use specific, multi-word phrases for more targeted results.
SEARCH_KEYWORDS = [
    "Neural Radiance Fields",
    "Monocular Depth Estimation"
]
MAX_RESULTS = 20  # Keep this low for testing, increase for production
OUTPUT_FILENAME = "arxiv_papers.jsonl"

def fetch_and_save_arxiv_papers():
    """
    Fetches the latest paper metadata from arXiv based on keywords and saves it
    as a JSON Lines file. This file acts as the input for our main Pathway pipeline.
    """
    print("🚀 Starting arXiv paper metadata fetch based on keywords...")

    # --- MODIFICATION: Build query from keywords ---
    # The 'all:' prefix searches across all metadata fields (title, abstract, etc.).
    # Quotes are used to handle multi-word keywords as exact phrases.
    # We join them with "+OR+" to get papers matching any of the keywords.
    search_query = "+OR+".join([f'all:"{urllib.parse.quote(kw)}"' for kw in SEARCH_KEYWORDS])
    
    query = f"search_query={search_query}&sortBy=submittedDate&sortOrder=descending&max_results={MAX_RESULTS}"
    full_url = "http://export.arxiv.org/api/query?" + query
    
    print(f"Querying API: {full_url}")
    try:
        response = requests.get(full_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return

    print("✅ Successfully fetched data from arXiv.")
    data_dict = xmltodict.parse(response.content)
    
    # Handle cases where no entries or only one entry is returned
    entries = data_dict.get('feed', {}).get('entry', [])
    if not isinstance(entries, list):
        entries = [entries] if entries else []

    if not entries:
        print("⚠️ No papers found for the given keywords. The output file will be empty.")
        return

    papers = []
    for entry in entries:
        try:
            authors_data = entry.get('author', [])
            # Handle single vs. multiple authors
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
                "journal_ref": entry.get('arxiv:journal_ref', {}).get('#text'),
                "doi": entry.get('arxiv:doi', {}).get('#text'),
            }
            papers.append(paper_info)
        except Exception as e:
            print(f"⚠️ Warning: Could not parse an entry. Skipping. Error: {e}")

    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            for paper in papers:
                f.write(json.dumps(paper) + '\n')
        print(f"✅ Successfully saved {len(papers)} paper metadata entries to '{OUTPUT_FILENAME}'.")
    except IOError as e:
        print(f"❌ Failed to write to file: {e}")

if __name__ == "__main__":
    fetch_and_save_arxiv_papers()