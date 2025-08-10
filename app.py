# app.py
import streamlit as st
import os
import json
import google.genai as genai
from src.parser import parse_uploaded_file

# ------------------------------
#   Language & Configuration Setup
# ------------------------------
LOCALE_DIR = "locales"

LANGUAGES = {
    "en": "English",
    "es": "Español",
    "ar": "العربية",
    "fr": "Français",
    "pt": "Português",
    "ru": "Русский",
    "zh": "中文",
    "ja": "日本語",
    "hi": "हिन्दी"
}

def load_strings(lang_code):
    """Load translation strings from locales folder."""
    file_path = os.path.join(LOCALE_DIR, f"{lang_code}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# ------------------------------
#   Configure Page
# ------------------------------
st.set_page_config(page_title="FinDocGPT", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        text-align: center;
        letter-spacing: -1px;
    }
    h3 {
        color: #34495e;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 600;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 10px;
    }
    .stButton>button {
        color: white;
        background-color: #3498db;
        border-radius: 12px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
    }
    .stSpinner > div > div {
        border-top-color: #3498db !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
#   Configure Gemini API
# ------------------------------
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
MODEL_NAME = "gemini-1.5-flash"

def analyze_document(document_text: str, user_prompt: str, lang_code: str) -> str:
    """Send the document and prompt to Gemini API."""
    prompt = (
        f"The user has requested the response in the language with code '{lang_code}'.\n"
        f"Please provide the analysis in this language.\n\n"
        f"Prompt: {user_prompt}\n\n"
        f"Document Text:\n{document_text}"
    )
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling LLM: {e}"

# ------------------------------
#   UI Layout
# ------------------------------
# Sidebar language selection
st.sidebar.title("Language")
selected_lang_name = st.sidebar.selectbox("Choose a language:", list(LANGUAGES.values()))
selected_lang_code = [key for key, value in LANGUAGES.items() if value == selected_lang_name][0]
strings = load_strings(selected_lang_code)

# Header
st.title(strings.get("app_title", "FinDocGPT"))
st.markdown(f"### {strings.get('subtitle', 'AI-powered financial analyst.')}")
st.markdown("---")

# ------------------------------
#   Document Upload Section
# ------------------------------
st.subheader(strings.get("section_1_header", "1. Upload a Document or Paste Text"))

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        strings.get("upload_label", "Upload a PDF, TXT, or HTML file:"),
        type=strings.get("upload_formats", ["pdf", "txt", "html"]),
        key="file_uploader"
    )

with col2:
    pasted_text = st.text_area(
        strings.get("or_text", "Or, paste the document text here:"),
        height=300,
        key="text_area"
    )

document_text = ""
if uploaded_file:
    document_text = parse_uploaded_file(uploaded_file)
elif pasted_text:
    document_text = pasted_text

st.markdown("---")

# ------------------------------
#   Analysis Prompt Section
# ------------------------------
st.subheader(strings.get("section_2_header", "2. Ask the AI a question about the document"))
user_prompt = st.text_area(
    strings.get("prompt_label", "Enter your analysis prompt here:"),
    placeholder=strings.get("prompt_placeholder", "e.g., 'Summarize the main risks in 3 bullet points.'"),
    height=100,
    key="prompt_area"
)

# ------------------------------
#   Analysis Button
# ------------------------------
if st.button(strings.get("analyze_button", "Analyze Document")):
    if not document_text:
        st.error(strings.get("error_file_upload", "Please upload a file or paste text to analyze."))
    elif not user_prompt:
        st.error(strings.get("error_prompt", "Please enter a prompt for the analysis."))
    else:
        with st.spinner(strings.get("loading_message", "Analyzing your document with AI...")):
            analysis_result = analyze_document(document_text, user_prompt, selected_lang_code)
            st.subheader(strings.get("result_header", "✨ Analysis Result"))
            st.write(analysis_result)

st.markdown("---")

# Disclaimer
st.markdown(strings.get(
    "disclaimer",
    "⚠️ **Disclaimer:** This tool is for demonstration purposes only and should not be used for making financial decisions."
))
