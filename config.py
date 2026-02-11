from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

# Check API
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY không tồn tại trong file .env!")

MODEL_ID = "gemini-2.5-flash" 

client = genai.Client(api_key=API_KEY)

TEMPERATURE = 0.7
PROMPT_DIR = "prompts"