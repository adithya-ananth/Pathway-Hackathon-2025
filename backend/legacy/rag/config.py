import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY_SUMMARIZE")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY_SUMMARIZE not found. Please set it in your .env file.")

genai.configure(api_key=API_KEY)

LAST_COMPREHENSIVE_ANSWER: str | None = None
LAST_TOP5_DOCS: list[dict] | None = None
