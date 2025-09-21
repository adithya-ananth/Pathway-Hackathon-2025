import os
import json
import google.generativeai as genai
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

# --- Configure Gemini API ---
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    raise RuntimeError("Please set the GEMINI_API_KEY environment variable.")

def extract_keywords_from_prompt(user_prompt: str) -> List[str]:
    """
    Uses the Gemini API to extract keywords from a user prompt.
    Returns a list of keyword strings.
    """
    model = genai.GenerativeModel('gemini-1.5-flash') # Using 1.5-flash for potentially better extraction
    prompt_template = f"""
    You are an expert at identifying specific, technical keywords and topics from research paper queries.
    Given a user's prompt, extract the main topics, research areas, and keywords.
    Provide the keywords as a single, comma-separated string. Do not add any other text or formatting.

    User Prompt: "latest advancements in computer vision, especially on adversarial attacks on vision transformers"
    Keywords: Computer Vision, Adversarial Attacks, Vision Transformers

    User Prompt: "Recent advances in deep reinforcement learning for robotics"
    Keywords: Deep Reinforcement Learning, Robotics

    User Prompt: "I need papers on ORB-SLAM3"
    Keywords: ORB-SLAM3, SLAM, Visual Odometry, Computer Vision

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

if __name__ == "__main__":
    # You can change the input prompt here
    input_prompt = "I need papers on ORB-SLAM3 and real-time object detection with transformers."
    config_filename = "config.json"

    print(f"▶️  Generating keywords from prompt: \"{input_prompt}\"")
    extracted_keywords = extract_keywords_from_prompt(input_prompt)

    if extracted_keywords:
        # Create a dictionary to store the keywords in a structured way
        config_data = {"search_keywords": extracted_keywords}
        
        # Write the keywords to the config file
        with open(config_filename, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4)
        
        print(f"✅  Successfully extracted keywords and saved them to '{config_filename}'.")
        print(f"   Keywords: {extracted_keywords}")
    else:
        print("❌  Could not extract any keywords. The config file was not created.")