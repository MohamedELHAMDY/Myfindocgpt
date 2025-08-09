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
    # Use the os.getenv function to get the API key from the environment variable
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    # This will now correctly show an error if the API key is not found
    print(f"Error initializing the Gemini API client: {e}")
    client = None # Set client to None if initialization fails

# The model name is now a parameter to generate_content
MODEL_NAME = "gemini-1.5-flash"

def analyze_document(document_text: str, user_prompt: str) -> str:
    """
    Analyzes a given document text based on a user-provided prompt.
    This function replaces the specific analysis functions for 10-K sections.
    """
    # Check if the client was successfully initialized
    if client is None:
        return "API Client not initialized. Please ensure your GOOGLE_API_KEY is set correctly in the .env file."

    # The prompt now combines the user's question with the document text.
    # This makes the analysis generic and adaptable to any user query.
    prompt = f"{user_prompt}\n\nDocument Text:\n{document_text}"

    try:
        # Use the client.models.generate_content method
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error calling LLM: {e}"

