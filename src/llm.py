# src/llm.py
import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key
try:
    client = genai.Client()
except Exception as e:
    # It's better to log the error than to print it
    print(f"Error initializing the Gemini API client: {e}")

# The model name is now a parameter to generate_content
MODEL_NAME = "gemini-1.5-flash"

def analyze_document(document_text: str, user_prompt: str, lang_code: str) -> str:
    """
    Analyzes a given document text based on a user-provided prompt.
    The response is instructed to be in the language specified by lang_code.
    """
    # The prompt now includes a clear instruction for the model to respond in the selected language.
    prompt = f"The user has requested the response in the language with code '{lang_code}'. Please provide the analysis in this language. Prompt: {user_prompt}\n\nDocument Text:\n{document_text}"

    try:
        # Use the client.models.generate_content method
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text
    except Exception as e:
        # Return a user-friendly error message
        return f"Error calling LLM: {e}"