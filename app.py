# app.py - Fully Optimized Version
import streamlit as st
import os
import json
import google.generativeai as genai
from src.parser import parse_uploaded_file # Assumes this file exists for parsing documents
import pandas as pd
import time
import plotly.express as px

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
<style>
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding: 2rem;
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
    .result-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        background-color: #f8f8f8;
    }
    .quick-prompt-button {
        background-color: #f0f4f8;
        color: #34495e;
        border: 1px solid #dcdcdc;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 14px;
        margin-right: 5px;
        cursor: pointer;
    }
    .quick-prompt-button:hover {
        background-color: #e9eef2;
    }
    .stDownloadButton > button {
        background-color: #2ecc71;
    }
    .stDownloadButton > button:hover {
        background-color: #27ae60;
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
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if i < retries - 1:
                st.warning(f"API call failed. Retrying in {delay}s... (Attempt {i + 1}/{retries})")
                time.sleep(delay)
                delay *= 2
            else:
                st.error(f"Final API call failed after {retries} attempts: {e}")
                raise

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
        key="file_uploader"
    )

with text_col:
    pasted_text = st.text_area(
        strings.get("or_text", "Or, paste the document text here:"),
        height=200,
        key="text_area"
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
if st.button("Reset Session"):
    st.session_state.clear()
    st.rerun()

st.markdown("---")

# ------------------------------
#   Analysis Prompt Section
# ------------------------------
st.subheader(strings.get("section_2_header", "2. Ask the AI a question about the document"))

# Dynamic prompts are generated only after a document is loaded
if st.session_state.is_document_loaded:
    st.markdown("##### Dynamic Prompts:")
    st.session_state.dynamic_prompts = generate_dynamic_prompts(st.session_state.document_text)

    prompt_cols = st.columns(3)
    for i, prompt_text in enumerate(st.session_state.dynamic_prompts):
        prompt_cols[i].button(prompt_text, key=f"quick_prompt_{i}", on_click=set_prompt, args=(prompt_text,))
else:
    st.markdown("##### Quick Prompts (Upload a document to see more!):")
    prompt_cols = st.columns(3)
    static_prompts = [
        strings.get("prompt_summarize", "Summarize the main points."),
        strings.get("prompt_risks", "Identify key risks and opportunities."),
        strings.get("prompt_figures", "Extract all financial figures in a table.")
    ]
    for i, prompt_text in enumerate(static_prompts):
        prompt_cols[i].button(prompt_text, key=f"quick_prompt_{i}", on_click=set_prompt, args=(prompt_text,))

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
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()

                    if numeric_cols and non_numeric_cols:
                        x_col = non_numeric_cols[0]
                        y_col = numeric_cols[0]
                        st.markdown("##### Data Visualization")
                        fig = px.bar(
                            df, x=x_col, y=y_col,
                            title=f"Distribution of {y_col} by {x_col}",
                            color=y_col,
                            color_continuous_scale=px.colors.sequential.Viridis,
                            labels={col: col.replace('_', ' ').title() for col in df.columns},
                            hover_data=df.columns
                        )
                        fig.update_layout(title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Helvetica", size=14, color="#333"), xaxis={'categoryorder':'total descending'}, hovermode="x unified", showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download as CSV", data=csv, file_name='analysis_results.csv', mime='text/csv')
            except Exception as e:
                st.error(f"Could not display the table correctly: {e}")
                st.markdown(f'<div class="result-container">{st.session_state.analysis_result}</div>', unsafe_allow_html=True)
    else:
        with result_container:
            st.markdown(f'<div class="result-container">{st.session_state.analysis_result}</div>', unsafe_allow_html=True)
            st.download_button(label="Download as TXT", data=st.session_state.analysis_result.encode('utf-8'), file_name='analysis_results.txt', mime='text/plain')

st.markdown("---")

# ------------------------------
#   Document Navigation and Content Section
# ------------------------------
if st.session_state.is_document_loaded:
    st.subheader("Document Content and Navigation")
    with st.expander("View Document and Summary"):
        st.session_state.document_summary = generate_document_summary(st.session_state.document_text)

        if st.session_state.document_summary:
            st.markdown("##### Key Sections:")
            summary_cols = st.columns(len(st.session_state.document_summary))
            for i, section in enumerate(st.session_state.document_summary):
                with summary_cols[i]:
                    st.markdown(f"**{section['title']}**")
                    st.write(f"_{section['summary']}_")
        else:
            st.info("No summary available. You can view the full document text below.")

        st.text_area("Full Document Text", value=st.session_state.document_text, height=500, disabled=True)
