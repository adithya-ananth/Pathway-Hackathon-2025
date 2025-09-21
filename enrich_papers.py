import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv
import time

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
# --- MODIFIED: Output file now has a .jsonl extension ---
FINAL_OUTPUT_FILE = 'complete_papers_data.jsonl'

MAX_CHARS_PER_PAPER_FOR_API = 4000

def extract_references(full_text_content):
    """
    Extracts the list of references from the full text of a paper using a more
    robust, multi-step parsing strategy.
    """
    match = re.search(r'^\s*(references|bibliography)\s*$', full_text_content, re.IGNORECASE | re.MULTILINE)

    if not match:
        return []

    references_section = full_text_content[match.end():].strip()
    references = re.split(r'\n\s*\n', references_section)

    cleaned_references = []
    for ref in references:
        cleaned_ref = re.sub(r'\s*\n\s*', ' ', ref).strip()
        if len(cleaned_ref) > 25:
            cleaned_references.append(cleaned_ref)

    return cleaned_references


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

def main():
    """
    Main workflow: Reads source data, enriches it with references and AI-generated
    sub-categories, and saves a final, complete dataset in JSONL format.
    """
    print("🚀 Starting the paper enrichment pipeline...")

    # 1. Read the source metadata from the JSONL file.
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

    # 2. Prepare data for API call and extract references from text files.
    papers_to_process_for_api = []
    print("📖 Reading text files and extracting references...")
    for doc_id, paper_meta in all_papers_map.items():
        file_path = os.path.join(PAPERS_TEXT_DIR, f"{doc_id}.txt")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            all_papers_map[doc_id]['references'] = extract_references(content)
            papers_to_process_for_api.append({
                "doc_id": doc_id,
                "primary_category": paper_meta.get('primary_category', 'N/A'),
                "content": content
            })
        except FileNotFoundError:
            print(f"   - Warning: Text file not found for {doc_id}. Skipping.")
            all_papers_map[doc_id]['references'] = []

    # 3. Call the Gemini API to get sub-categories
    if not papers_to_process_for_api:
        print("❌ No text files were found to process. Cannot call API.")
        return

    final_prompt = build_prompt(papers_to_process_for_api)
    print(f"🤖 Sending request for {len(papers_to_process_for_api)} papers to Gemini API...")
    try:
        response = model.generate_content(final_prompt)
        cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        result_data = json.loads(cleaned_response_text)
        print("✅ Received and parsed AI-generated sub-categories.")

        # 4. Merge the AI results back into our main data map
        results_map = {result['doc_id']: result['sub_categories'] for result in result_data}
        for doc_id, sub_categories in results_map.items():
            if doc_id in all_papers_map:
                all_papers_map[doc_id]['sub_categories'] = sub_categories

        # 5. --- MODIFIED: Save the final, complete dataset as a JSONL file ---
        final_dataset = list(all_papers_map.values())
        with open(FINAL_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for paper_record in final_dataset:
                # Convert each paper's dictionary to a JSON string and write it as a new line
                f.write(json.dumps(paper_record) + '\n')

        print(f"\n📄🎉 Success! Final, comprehensive dataset created at '{FINAL_OUTPUT_FILE}'.")

    except Exception as e:
        print(f"\n❌ An error occurred during the API call or final processing: {e}")
        if 'response' in locals():
            with open('error_response.txt', 'w') as f:
                f.write(response.text)
            print("   - Raw API response saved to 'error_response.txt' for debugging.")

if __name__ == "__main__":
    main()