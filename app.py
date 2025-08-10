import streamlit as st
import os
import io
from google_genai import genai
from PyMuPDF import fitz
from dotenv import load_dotenv

# --- 1. Set up API Key and LLM Client ---
# Load environment variables from .env file for local development.
# This is handled automatically by Streamlit in the cloud via secrets.toml.
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Use st.secrets for Streamlit Cloud deployment.
# This is the recommended and secure way to handle secrets.
if not GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except KeyError:
        st.error("API Key not found. Please set it in your Streamlit secrets or .env file.")
        st.stop() # Stop the app if the API key is not available

# Configure the Gemini API client
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    client = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    st.error(f"Error initializing the Gemini API client: {e}")
    st.stop() # Stop the app if the client can't be initialized


# --- 2. Define the core logic functions ---
def extract_text_from_pdf(pdf_file) -> str:
    """
    Extracts text from a PDF file uploaded via Streamlit.
    It uses PyMuPDF (fitz) for efficient text extraction.
    """
    text = ""
    try:
        # Re-open the file stream to handle multiple reads
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        st.stop()
    return text

@st.cache_data
def analyze_document(document_text: str, user_prompt: str) -> str:
    """
    Analyzes a given document text based on a user-provided prompt using the Gemini API.
    The @st.cache_data decorator caches the results, preventing redundant API calls
    and speeding up the application.
    """
    prompt = f"{user_prompt}\n\nDocument Text:\n{document_text}"
    try:
        response = client.generate_content(prompt)
        return response.text
    except Exception as e:
        # Return a more informative error message to the user
        return f"Error calling LLM: {e}. Please try again or with a different prompt."


# --- 3. Build the Streamlit App UI ---
def main():
    """
    Main function to build the Streamlit application interface.
    """
    st.set_page_config(page_title="FinDocGPT", layout="wide")
    st.title("FinDocGPT: AI Financial Document Analyzer")
    st.markdown("An AI-powered financial analyst that can parse, analyze, and summarize 10-K filings and other financial documents.")

    uploaded_file = st.file_uploader("Upload a 10-K PDF file", type=["pdf"])

    if uploaded_file:
        with st.spinner("Extracting text from PDF..."):
            # The file is passed to the extraction function
            document_text = extract_text_from_pdf(uploaded_file)
        
        st.success("Text extraction complete!")

        # Pre-defined prompts for user convenience
        analysis_options = {
            "Summarize Risk Factors": "Summarize the key risk factors and future outlook mentioned in the document.",
            "Extract Key Financial Metrics": "Identify and list the key financial metrics (e.g., revenue, net income, cash flow, debt) from the document.",
            "Analyze Management Discussion": "Provide a high-level summary of the management's discussion and analysis (MD&A) section.",
            "Custom Query": "Write your own prompt below."
        }
        
        analysis_type = st.selectbox("Select an analysis type:", list(analysis_options.keys()))

        user_prompt = ""
        if analysis_type == "Custom Query":
            user_prompt = st.text_area("Enter your custom analysis prompt:", height=150)
        else:
            user_prompt = analysis_options[analysis_type]

        if st.button("Analyze Document", use_container_width=True):
            if user_prompt:
                with st.spinner("Analyzing document with Gemini..."):
                    result = analyze_document(document_text, user_prompt)
                    st.subheader("Analysis Result:")
                    st.write(result)
            else:
                st.warning("Please select a predefined analysis or enter a custom prompt.")

if __name__ == "__main__":
    main()
