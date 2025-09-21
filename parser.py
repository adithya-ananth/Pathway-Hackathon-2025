import os
import json
import requests
import fitz  # PyMuPDF
import time

# --- Configuration ---
SOURCE_JSONL_FILE = 'arxiv_papers.jsonl'
DOWNLOADED_PDF_DIR = 'downloaded_papers'
PAPERS_TEXT_DIR = 'papers_text'

def process_paper(paper_info, pdf_dir, text_dir):
    """
    Downloads a paper's PDF, converts it to text, and saves the text file.
    This function will skip steps if the files already exist.
    """
    doc_id = paper_info.get("doc_id")
    pdf_url = paper_info.get("pdf_url")
    title = paper_info.get("title", "No Title")

    if not doc_id or not pdf_url:
        print(f"⚠️  Skipping entry due to missing 'doc_id' or 'pdf_url': {title}")
        return

    pdf_filepath = os.path.join(pdf_dir, f"{doc_id}.pdf")
    txt_filepath = os.path.join(text_dir, f"{doc_id}.txt")
    
    # Check if the final text file already exists. If so, we can skip everything.
    if os.path.exists(txt_filepath):
        print(f"✅ Text file for {doc_id} already exists. Skipping.")
        return

    # Step 1: Download the PDF if it doesn't already exist
    if not os.path.exists(pdf_filepath):
        try:
            print(f"   -> Downloading PDF for {doc_id}...")
            response = requests.get(pdf_url, timeout=20)
            response.raise_for_status()  # Raise an exception for bad status codes
            with open(pdf_filepath, 'wb') as f:
                f.write(response.content)
            time.sleep(1)  # Be polite to the arXiv server
        except requests.exceptions.RequestException as e:
            print(f"   - ❌ ERROR downloading {doc_id}: {e}")
            return # Stop processing this paper if download fails

    # Step 2: Extract text from the PDF and save it
    try:
        print(f"   -> Extracting text from PDF for {doc_id}...")
        full_text = ""
        with fitz.open(pdf_filepath) as doc:
            for page in doc:
                full_text += page.get_text()
        
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"   -> ✅ Successfully created text file for {doc_id}")

    except Exception as e:
        print(f"   - ❌ ERROR extracting text for {doc_id}: {e}")

def main():
    """
    Main workflow: Reads paper metadata, then for each paper, downloads the PDF
    and saves its content as a text file.
    """
    print("🚀 Starting PDF to Text conversion pipeline...")
    
    # --- Setup: Create necessary directories ---
    os.makedirs(DOWNLOADED_PDF_DIR, exist_ok=True)
    os.makedirs(PAPERS_TEXT_DIR, exist_ok=True)

    if not os.path.exists(SOURCE_JSONL_FILE):
        print(f"❌ FATAL ERROR: Source file not found at '{SOURCE_JSONL_FILE}'")
        return

    # --- Load initial metadata from arxiv_papers.jsonl ---
    try:
        with open(SOURCE_JSONL_FILE, 'r', encoding='utf-8') as f_in:
            papers_to_process = [json.loads(line) for line in f_in if line.strip()]
    except Exception as e:
        print(f"❌ FATAL ERROR reading or parsing '{SOURCE_JSONL_FILE}'. Details: {e}")
        return
        
    print(f"✅ Loaded metadata for {len(papers_to_process)} papers. Starting processing...")
    print("-" * 50)

    # --- Process each paper ---
    for i, paper_meta in enumerate(papers_to_process):
        print(f"Processing paper {i+1}/{len(papers_to_process)}: {paper_meta.get('doc_id')}")
        process_paper(paper_meta, DOWNLOADED_PDF_DIR, PAPERS_TEXT_DIR)

    print("-" * 50)
    print(f"🎉 Pipeline finished! Text files are located in the '{PAPERS_TEXT_DIR}/' directory.")

if __name__ == "__main__":
    main()