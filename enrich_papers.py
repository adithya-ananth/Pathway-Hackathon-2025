import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv
import time
from datetime import datetime

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- File I/O Configuration ---
SOURCE_JSONL_FILE = 'arxiv_papers.jsonl'
PAPERS_TEXT_DIR = 'papers_text'
FINAL_OUTPUT_FILE = 'content_stream/enriched_papers.jsonl'

MAX_CHARS_PER_PAPER_FOR_API = 4000

# --- NEW: Heavily revised, two-strategy reference extraction function ---
def extract_references_with_llm(full_text_content, model_instance):
    """
    Uses the Gemini model to intelligently find and extract references
    with a hardened prompt and a final quality filter.
    """
    # Take the last 15% of the paper, where references are most likely to be.
    last_chunk_start = int(len(full_text_content) * 0.85)
    text_snippet = full_text_content[last_chunk_start:]

    prompt = f"""
    You are a specialized academic parser. Your task is to analyze the following text from the end of a research paper and extract the list of references.

    RULES:
    1.  Carefully identify the reference or bibliography section.
    2.  Extract each individual citation as a complete, separate string. Each string must be a full bibliographic reference, typically including authors, title, and year.
    3.  **CRITICAL:** Do NOT extract only citation numbers (e.g., "[23]"). Extract the full text of the reference associated with that number.
    4.  Return the result as a single, valid JSON array of strings.
    5.  If you cannot find any valid, complete references, return an empty JSON array: [].
    6.  Do not include any explanations or text outside of the JSON array.

    TEXT TO ANALYZE:
    ---
    {text_snippet}
    ---
    """
    try:
        response = model_instance.generate_content(prompt)
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        parsed_json = json.loads(cleaned_text)

        if isinstance(parsed_json, list):
            # Final quality check: filter out any leftover partials or citation-only numbers.
            # A real reference will almost always have more than 3 words.
            return [ref for ref in parsed_json if len(ref.split()) > 3]
        else:
            return []
            
    except Exception:
        # If the API call or JSON parsing fails for any reason, return an empty list.
        return []

def build_prompt(papers_data):
    """Constructs the single, comprehensive prompt for the Gemini API."""
    prompt_header = """
    You are an expert AI research scientist. Your task is to analyze the content of the research papers provided below and generate 2-3 specific, relevant sub-categories for each.

    RULES:
    1.  Return your response as a single, valid JSON array.
    2.  Each object in the array must contain the 'doc_id' and a list of strings called 'sub_categories'.
    3.  Do not include any text, explanations, or markdown formatting outside of the JSON array.
    --- START OF PAPERS DATA ---
    """
    prompt_body = ""
    for paper in papers_data:
        content_snippet = paper['content'][:MAX_CHARS_PER_PAPER_FOR_API]
        prompt_body += f"""
        --------------------
        - doc_id: "{paper['doc_id']}"
        - primary_category: "{paper['primary_category']}"
        - content: "{content_snippet}..."
        """
    prompt_footer = "\n--- END OF PAPERS DATA ---\n" \
                    "Generate the JSON response."
    return prompt_header + prompt_body + prompt_footer


def create_rag_compatible_papers(all_papers_map):
    """
    Convert papers to RAG-compatible format and save to content stream
    """
    # Create content_stream directory
    os.makedirs('./content_stream', exist_ok=True)
    
    rag_papers = []
    for doc_id, paper_meta in all_papers_map.items():
        # Convert to RAG format
        rag_paper = {
            "id": doc_id,
            "title": paper_meta.get('title', ''),
            "abstract": paper_meta.get('abstract', ''),
            "authors": paper_meta.get('authors', []),
            "published_date": paper_meta.get('published_date', ''),
            "url": paper_meta.get('url', ''),
            "pdf_url": paper_meta.get('pdf_url', paper_meta.get('url', '')),
            "primary_category": paper_meta.get('primary_category', 'unknown'),
            "secondary_categories": paper_meta.get('sub_categories', []),
            "text": paper_meta.get('abstract', ''),  # Use abstract as fallback text
            "citations": paper_meta.get('references', [])
        }
        rag_papers.append(rag_paper)
    
    # Save to timestamped file for RAG system
    timestamp = int(datetime.now().timestamp())
    timestamped_file = f'./content_stream/papers_{timestamp}.jsonl'
    
    with open(timestamped_file, 'w', encoding='utf-8') as f:
        for paper in rag_papers:
            f.write(json.dumps(paper) + '\n')
    
    print(f"✅ Added {len(rag_papers)} papers to RAG content stream: {timestamped_file}")
    return rag_papers

def main():
    """
    Main workflow: Reads source data, enriches it with references and AI-generated
    sub-categories, and saves a final, complete dataset in JSONL format.
    """
    print("🚀 Starting the paper enrichment pipeline...")

    if not os.path.exists(SOURCE_JSONL_FILE):
        print(f"❌ FATAL ERROR: Source file not found at '{SOURCE_JSONL_FILE}'")
        return

    all_papers_map = {}
    try:
        with open(SOURCE_JSONL_FILE, 'r', encoding='utf-8') as f_in:
            for i, line in enumerate(f_in):
                if not line.strip(): continue
                paper_data = json.loads(line)
                all_papers_map[paper_data['doc_id']] = paper_data
    except Exception as e:
        print(f"❌ FATAL ERROR reading or parsing '{SOURCE_JSONL_FILE}' on line {i+1}. Details: {e}")
        return

    if not all_papers_map:
        print("⚠️ Warning: No papers were loaded.")
        return
    print(f"✅ Loaded metadata for {len(all_papers_map)} papers from '{SOURCE_JSONL_FILE}'.")

    papers_to_process_for_api = []
    print("📖 Reading text files and extracting references (using robust v3 parser)...")
    
    # First, set default values for all papers
    for doc_id in all_papers_map:
        all_papers_map[doc_id]['references'] = []
        all_papers_map[doc_id]['sub_categories'] = []
    
    # Try to find text files with various naming patterns
    text_files_found = 0
    for doc_id, paper_meta in all_papers_map.items():
        content = None
        file_found = False
        
        # Try different possible file paths
        possible_paths = [
            os.path.join(PAPERS_TEXT_DIR, f"{doc_id}.txt"),
            os.path.join(PAPERS_TEXT_DIR, f"{doc_id}_text.txt"),
            os.path.join('.', f"{doc_id}.txt"),
            os.path.join('text_files', f"{doc_id}.txt")
        ]
        
        for file_path in possible_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    file_found = True
                    text_files_found += 1
                    print(f"   ✅ Found text file for {doc_id}")
                    break
            except FileNotFoundError:
                continue
        
        if file_found and content:
            # Extract references from full text
            all_papers_map[doc_id]['references'] = extract_references_with_llm(content, model)
            papers_to_process_for_api.append({
                "doc_id": doc_id,
                "primary_category": paper_meta.get('primary_category', 'N/A'),
                "content": content
            })
        else:
            # Use abstract as fallback content
            abstract = paper_meta.get('abstract', '')
            if abstract:
                papers_to_process_for_api.append({
                    "doc_id": doc_id,
                    "primary_category": paper_meta.get('primary_category', 'N/A'),
                    "content": abstract
                })
            print(f"   - Warning: Text file not found for {doc_id}. Using abstract as fallback.")

    print(f"📊 Processing status: {text_files_found} full text files found, {len(papers_to_process_for_api)} papers total")

    # Always process papers (with full text or abstracts)
    if papers_to_process_for_api:
        final_prompt = build_prompt(papers_to_process_for_api)
        print(f"🤖 Sending request for {len(papers_to_process_for_api)} papers to Gemini API...")
        try:
            response = model.generate_content(final_prompt)
            cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            result_data = json.loads(cleaned_response_text)
            print("✅ Received and parsed AI-generated sub-categories.")

            results_map = {result['doc_id']: result['sub_categories'] for result in result_data}
            for doc_id, sub_categories in results_map.items():
                if doc_id in all_papers_map:
                    all_papers_map[doc_id]['sub_categories'] = sub_categories

        except Exception as e:
            print(f"⚠️ API call failed, continuing with basic metadata: {e}")
            # Continue without AI-generated sub-categories

    # Always create the final dataset
    final_dataset = list(all_papers_map.values())
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(FINAL_OUTPUT_FILE), exist_ok=True)
    
    with open(FINAL_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for paper_record in final_dataset:
            f.write(json.dumps(paper_record) + '\n')

    print(f"✅ Created enriched dataset at '{FINAL_OUTPUT_FILE}' with {len(final_dataset)} papers")

    # Create RAG-compatible papers and add to content stream
    create_rag_compatible_papers(all_papers_map)
    
    print(f"\n📄🎉 Success! Papers processed and added to RAG content stream.")
    print(f"   - {text_files_found} papers with full text processed")
    print(f"   - {len(final_dataset)} total papers available for RAG system")


if __name__ == "__main__":
    main()