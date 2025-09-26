import os
import json
import google.generativeai as genai
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

# --- Configure Gemini API ---
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY_EXTRACT"])
except KeyError:
    raise RuntimeError("Please set the GEMINI_API_KEY environment variable.")

# import google.generativeai as genai
# genai.configure(api_key="GEMINI_API_KEY")
# model = genai.GenerativeModel("gemini-2.0-flash-lite")
# resp = model.generate_content("Say hello")
# print(resp.text)

def extract_keywords_from_prompt(user_prompt: str) -> List[str]:
    """
    Uses the Gemini API to extract keywords from a user prompt.
    Returns a list of keyword strings.
    """
    model = genai.GenerativeModel("gemini-2.0-flash-lite") # Using 1.5-flash for potentially better extraction
    prompt_template = f"""
    You are an expert at identifying specific, technical keywords and topics from research paper queries.
    Given a user's prompt, extract the main topics, research areas, and keywords.
    Provide the keywords as a single, comma-separated string. Do not add any other text or formatting.

    User Prompt: "latest advancements in computer vision, especially on adversarial attacks on vision transformers"
    Keywords: Computer Vision, Adversarial Attacks, Vision Transformers

    User Prompt: "Recent advances in deep reinforcement learning for robotics"
    Keywords: Deep Reinforcement Learning, Robotics

    User Prompt: "I need papers on ORB-SLAM3"
    Keywords: ORB-SLAM3, SLAM, Visual Odometry

    User Prompt: "{user_prompt}"
    Keywords:
    """

    try:
        response = model.generate_content(prompt_template)
        # Clean up the response to get a clean, comma-separated string
        keywords_str = response.text.strip().replace("Keywords:", "").strip()
        
        # Split the string into a list of keywords
        keywords_list = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
        return keywords_list
    except Exception as e:
        print(f"Keyword extraction failed: {e}")
        return []

def run_keyword_extraction(input_prompt: str) -> None:
    """
    Main function to run keyword extraction and create config/query files.
    """
    config_filename = "config.json"
    query_stream_dir = "query_stream"
    query_filename = "input_query.jsonl"

    print(f"Generating keywords from prompt: \"{input_prompt}\"")
    extracted_keywords = extract_keywords_from_prompt(input_prompt)

    print("Extracted keywrods: ", extracted_keywords)

    # If keyword extraction fails, create fallback keywords from the prompt
    if not extracted_keywords:
        print("API extraction failed, creating fallback keywords from prompt...")
        # Simple keyword extraction as fallback
        words = input_prompt.lower().split()
        # Remove common words and keep meaningful terms
        stop_words = {'tell', 'me', 'about', 'on', 'in', 'the', 'of', 'and', 'or', 'for', 'with', 'to', 'a', 'an'}
        extracted_keywords = [word.title() for word in words if word not in stop_words and len(word) > 2]
        if not extracted_keywords:
            extracted_keywords = [input_prompt.title()]  # Use whole prompt as keyword if nothing else works

    if extracted_keywords:
        # Create a dictionary to store the keywords in a structured way
        config_data = {"search_keywords": extracted_keywords}
        
        # Write the keywords to the config file
        with open(config_filename, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4)
        
        print(f"Successfully extracted keywords and saved them to '{config_filename}'.")
        print(f"   Keywords: {extracted_keywords}")
        
        # Create query_stream directory if it doesn't exist
        os.makedirs(query_stream_dir, exist_ok=True)
        
        # Create query for RAG system
        query_data = {
            "query": input_prompt,
            "top_k": 5,
            "keywords": extracted_keywords
        }
        
        query_filepath = os.path.join(query_stream_dir, query_filename)
        with open(query_filepath, 'w', encoding='utf-8') as f:
            json.dump(query_data, f)
            f.write('\n')  # Add newline for JSONL format
        
        print(f"Created query file '{query_filepath}' for RAG system.")
        print(f"   Query: \"{input_prompt}\"")
        print(f"   Top K: 5")
        print(f"   Keywords: {extracted_keywords}")
    else:
        print("Could not extract any keywords. The config file was not created.")