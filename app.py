# The following is the updated requirements.txt file with the missing dependency.
# This file should be placed in the same directory as app.py.

# requirements.txt
streamlit
google-generativeai
python-dotenv
PyMuPDF
streamlit_option_menu

# ---

# app.py - Fully Optimized Version
import streamlit as st
import os
import json
import google.generativeai as genai
from src.parser import parse_uploaded_file # Assumes this file exists for parsing documents
import pandas as pd
import time
import plotly.express as px
from streamlit_option_menu import option_menu # Using a library for a better sidebar menu

# ======================================================
#   Application Configuration
# ======================================================

# Directory for localization files
LOCALE_DIR = "locales"

# Supported languages
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

# Cache translation strings to avoid re-loading on every rerun
@st.cache_data
def load_strings(lang_code):
    """Loads translation strings from a JSON file in the locales folder."""
    file_path = os.path.join(LOCALE_DIR, f"{lang_code}.json")
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            st.error(f"Language file not found: {file_path}")
    except (IOError, json.JSONDecodeError) as e:
        st.error(f"Error loading language file for '{lang_code}': {e}")
    return {}

# Set Streamlit page configuration for a wide, clean layout
st.set_page_config(page_title="FinDocGPT", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a modern and polished look
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
<style>
    :root {
        --primary-color: #3498db;
        --secondary-color: #2c3e50;
        --background-color: #ecf0f1;
        --container-bg-color: #ffffff;
        --button-hover-color: #2980b9;
        --text-color: #34495e;
    }
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding: 2rem;
    }
    h1 {
        color: var(--primary-color);
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        text-align: center;
        letter-spacing: -1px;
    }
    h3 {
        color: var(--secondary-color);
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        border-bottom: 2px solid var(--background-color);
        padding-bottom: 10px;
        margin-top: 1.5rem;
    }
    .stButton>button {
        color: white;
        background-color: var(--primary-color);
        border-radius: 12px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: var(--button-hover-color);
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .stDownloadButton>button {
        background-color: #2ecc71;
    }
    .stDownloadButton>button:hover {
        background-color: #27ae60;
    }
    .stSpinner > div > div {
        border-top-color: var(--primary-color) !important;
    }
    .result-container {
        border: 1px solid var(--background-color);
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        background-color: var(--container-bg-color);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .quick-prompt-button {
        background-color: var(--background-color);
        color: var(--text-color);
        border: 1px solid #dcdcdc;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 14px;
        margin-right: 5px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .quick-prompt-button:hover {
        background-color: #d8e1e7;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 8px;
        border: 1px solid #ccc;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================
#   Gemini API Interaction
# ======================================================

# Configure Gemini API
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
MODEL_NAME = "gemini-1.5-flash"

def api_call_with_backoff(func, *args, **kwargs):
    """Wrapper to handle API calls with exponential backoff and retries."""
    retries = 5
    delay = 1
    progress_bar = st.empty()
    status_text = st.empty()
    
    for i in range(retries):
        try:
            status_text.info(f"Attempt {i + 1}/{retries}: Analyzing document with AI...")
            progress_bar.progress((i + 1) / retries)
            return func(*args, **kwargs)
        except Exception as e:
            if i < retries - 1:
                status_text.warning(f"API call failed. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                status_text.error(f"Final API call failed after {retries} attempts: {e}")
                raise
    progress_bar.empty()
    status_text.empty()

def analyze_document(document_text, user_prompt, selected_lang_code):
    """
    Sends a combined prompt to the Gemini API and returns the analysis result.
    Handles both structured (JSON) and unstructured text responses.
    """
    # Check if the prompt suggests a table or data extraction
    if any(keyword in user_prompt.lower() for keyword in ["table", "figures", "extract data", "json"]):
        full_prompt = (
            f"You are a financial analyst. The user has provided a document and wants to extract specific data into a table. "
            f"Please identify the relevant financial figures mentioned in the document and organize them into a JSON object "
            f"that represents a table. The object should have keys like 'metric' and 'value'. "
            f"Document: {document_text}\n"
            f"User's Question: {user_prompt}\n"
            f"Please respond ONLY with the JSON object."
        )
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            response = api_call_with_backoff(
                model.generate_content,
                full_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            json_str = response.candidates[0].content.parts[0].text
            return json.loads(json_str)
        except (genai.types.generation_types.BlockedPromptException, json.JSONDecodeError) as e:
            st.error(f"Error processing structured output: {e}. Falling back to a text response.")
            return "Could not generate a structured table. Please try a different prompt or format."
        except Exception as e:
            return f"An error occurred during API processing: {e}. Please check your API key and try again."
    else:
        # Standard text-based response
        full_prompt = (
            f"You are a financial analyst. The user has provided a document and a question. "
            f"Keep your response concise and to the point. The user's language preference is '{selected_lang_code}'.\n"
            f"Document: {document_text}\n"
            f"User's Question: {user_prompt}"
        )
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            response = api_call_with_backoff(model.generate_content, full_prompt)
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                return "No content was generated by the model. Please try a different prompt."
        except genai.types.generation_types.BlockedPromptException:
            return "I am sorry, but your prompt was flagged for safety. Please try rephrasing."
        except Exception as e:
            return f"An error occurred during API processing: {e}. Please check your API key and try again."

@st.cache_data
def generate_dynamic_prompts(document_text):
    """Generates three dynamic prompts based on the document's content."""
    if not document_text:
        return []
    
    prompt = (
        f"Based on the following financial document, generate three very concise and specific questions "
        f"a financial analyst might ask. Respond with a JSON array of three strings.\n\n"
        f"Document: {document_text[:1000]}..."
    )
    model = genai.GenerativeModel(MODEL_NAME)
    try:
        response = api_call_with_backoff(
            model.generate_content,
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        json_str = response.candidates[0].content.parts[0].text
        return json.loads(json_str)
    except Exception as e:
        st.error(f"Could not generate dynamic prompts. Defaulting to static prompts.")
        return ["Summarize the main points.", "Identify key risks and opportunities.", "Extract all financial figures in a table."]

@st.cache_data
def generate_document_summary(document_text):
    """Generates a structured summary of the document with key sections."""
    if not document_text:
        return []

    prompt = (
        f"Break down the following financial document into 3-5 key sections. "
        f"For each section, provide a concise title and a short summary. "
        f"Respond with a JSON array of objects, where each object has keys 'title' and 'summary'.\n\n"
        f"Document: {document_text[:2000]}..."
    )
    model = genai.GenerativeModel(MODEL_NAME)
    try:
        response = api_call_with_backoff(
            model.generate_content,
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        json_str = response.candidates[0].content.parts[0].text
        return json.loads(json_str)
    except Exception as e:
        st.error(f"Could not generate a document summary.")
        return []


# ======================================================
#   UI Layout and Logic
# ======================================================

# Initialize session state for consistent UI behavior
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "is_document_loaded" not in st.session_state:
    st.session_state.is_document_loaded = False
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = ""
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = ""
if "dynamic_prompts" not in st.session_state:
    st.session_state.dynamic_prompts = []
if "document_summary" not in st.session_state:
    st.session_state.document_summary = []
if "chart_type" not in st.session_state:
    st.session_state.chart_type = "Bar Chart"

# Callback function to set the prompt text from a button click
def set_prompt(prompt_text):
    st.session_state.user_prompt = prompt_text

# Sidebar for language selection and navigation
with st.sidebar:
    st.title("Settings")
    selected_lang_name = st.selectbox("Choose a language:", list(LANGUAGES.values()))
    selected_lang_code = [key for key, value in LANGUAGES.items() if value == selected_lang_name][0]
    strings = load_strings(selected_lang_code)

    st.markdown("---")
    st.header("About")
    st.info(strings.get(
        "disclaimer",
        "⚠️ **Disclaimer:** This tool is for demonstration purposes only and should not be used for making financial decisions."
    ))

# Header and Main Title
st.title(strings.get("app_title", "FinDocGPT"))
st.markdown(f"### {strings.get('subtitle', 'AI-powered financial analyst.')}")
st.markdown("---")

# ------------------------------
#   Document Upload Section
# ------------------------------
st.subheader(strings.get("section_1_header", "1. Upload a Document or Paste Text"))
upload_col, text_col = st.columns([1, 1])

with upload_col:
    uploaded_file = st.file_uploader(
        strings.get("upload_label", "Upload a PDF, TXT, HTML, DOCX, or XLSX file:"),
        type=["pdf", "txt", "html", "docx", "xlsx"],
        key="file_uploader",
        disabled=st.session_state.is_document_loaded
    )

with text_col:
    pasted_text = st.text_area(
        strings.get("or_text", "Or, paste the document text here:"),
        height=200,
        key="text_area",
        disabled=st.session_state.is_document_loaded
    )

# Logic to handle document loading
if uploaded_file and not st.session_state.is_document_loaded:
    with st.spinner("Parsing document..."):
        try:
            st.session_state.document_text = parse_uploaded_file(uploaded_file)
            st.session_state.is_document_loaded = True
            st.success("Document loaded successfully!")
        except Exception as e:
            st.error(f"Failed to parse the uploaded file: {e}")
elif pasted_text and not st.session_state.is_document_loaded:
    st.session_state.document_text = pasted_text
    st.session_state.is_document_loaded = True
    st.success("Text pasted successfully!")

# Reset button to clear all session state
if st.button("Reset Session", help="Clear the current document and analysis."):
    st.session_state.clear()
    st.rerun()

st.markdown("---")

# ------------------------------
#   Analysis Prompt Section
# ------------------------------
st.subheader(strings.get("section_2_header", "2. Ask the AI a question about the document"))

# Dynamic prompts are generated only after a document is loaded
if st.session_state.is_document_loaded:
    st.markdown("##### Dynamic Prompts: <i class='fas fa-lightbulb'></i>")
    st.markdown("<p style='font-size: 14px; color: #666;'>These prompts were generated by the AI based on the document's content.</p>", unsafe_allow_html=True)
    st.session_state.dynamic_prompts = generate_dynamic_prompts(st.session_state.document_text)

    prompt_cols = st.columns(3)
    for i, prompt_text in enumerate(st.session_state.dynamic_prompts):
        with prompt_cols[i]:
            st.button(prompt_text, key=f"quick_prompt_{i}", on_click=set_prompt, args=(prompt_text,))
else:
    st.markdown("##### Quick Prompts: <i class='fas fa-question-circle'></i>")
    st.markdown("<p style='font-size: 14px; color: #666;'>Upload a document to see more specific prompts!</p>", unsafe_allow_html=True)
    prompt_cols = st.columns(3)
    static_prompts = [
        strings.get("prompt_summarize", "Summarize the main points."),
        strings.get("prompt_risks", "Identify key risks and opportunities."),
        strings.get("prompt_figures", "Extract all financial figures in a table.")
    ]
    for i, prompt_text in enumerate(static_prompts):
        with prompt_cols[i]:
            st.button(prompt_text, key=f"quick_prompt_{i}", on_click=set_prompt, args=(prompt_text,))

user_prompt = st.text_area(
    strings.get("prompt_label", "Enter your analysis prompt here:"),
    value=st.session_state.user_prompt,
    placeholder=strings.get("prompt_placeholder", "e.g., 'Summarize the main risks in 3 bullet points.'"),
    height=100,
    key="user_prompt_area"
)

# Analysis Button
if st.button(strings.get("analyze_button", "Analyze Document")):
    if not st.session_state.is_document_loaded:
        st.error(strings.get("error_file_upload", "Please upload a file or paste text to analyze."))
    elif not user_prompt:
        st.error(strings.get("error_prompt", "Please enter a prompt for the analysis."))
    else:
        # Progress bar and status messages for a better user experience
        with st.empty():
            with st.spinner(strings.get("loading_message", "Analyzing your document with AI...")):
                st.session_state.analysis_result = analyze_document(st.session_state.document_text, user_prompt, selected_lang_code)

# ------------------------------
#   Analysis Result Section
# ------------------------------
if st.session_state.analysis_result:
    st.subheader(strings.get("result_header", "✨ Analysis Result"))
    result_container = st.container()

    if isinstance(st.session_state.analysis_result, list) and all(isinstance(item, dict) for item in st.session_state.analysis_result):
        with result_container:
            try:
                df = pd.DataFrame(st.session_state.analysis_result)
                st.dataframe(df, use_container_width=True)

                if len(df.columns) >= 2:
                    st.markdown("##### Data Visualization <i class='fas fa-chart-bar'></i>", unsafe_allow_html=True)
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
                    
                    if numeric_cols and non_numeric_cols:
                        chart_type = st.selectbox(
                            "Select a Chart Type:",
                            ("Bar Chart", "Line Chart", "Scatter Plot"),
                            key="chart_type"
                        )
                        x_col = st.selectbox(
                            "Select X-axis:",
                            options=non_numeric_cols,
                            key="x_col"
                        )
                        y_col = st.selectbox(
                            "Select Y-axis:",
                            options=numeric_cols,
                            key="y_col"
                        )

                        if chart_type == "Bar Chart":
                            fig = px.bar(df, x=x_col, y=y_col, color=y_col, color_continuous_scale=px.colors.sequential.Viridis, labels={col: col.replace('_', ' ').title() for col in df.columns})
                        elif chart_type == "Line Chart":
                            fig = px.line(df, x=x_col, y=y_col, labels={col: col.replace('_', ' ').title() for col in df.columns})
                        elif chart_type == "Scatter Plot":
                            fig = px.scatter(df
