# app.py - Optimized and Enhanced Version with Reordered UI
import streamlit as st
import os
import json
import google.generativeai as genai
from src.parser import parse_uploaded_file # Assuming this file exists and works as intended
import pandas as pd
import time
import plotly.express as px

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

# Use Streamlit's cache to load strings only once per language selection
@st.cache_data
def load_strings(lang_code):
    """Load translation strings from locales folder."""
    file_path = os.path.join(LOCALE_DIR, f"{lang_code}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            st.error(f"Error loading language file: {e}")
            return {}
    return {}

# ------------------------------
#   Configure Page
# ------------------------------
st.set_page_config(page_title="FinDocGPT", layout="wide")

# Custom CSS for a more polished look
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
</style>
""", unsafe_allow_html=True)

# ------------------------------
#   Configure Gemini API
# ------------------------------
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
MODEL_NAME = "gemini-1.5-flash"

def analyze_document(document_text, user_prompt, selected_lang_code):
    """
    Sends a combined prompt to the Gemini API and returns the analysis result.
    If the prompt suggests tabular data, it attempts to return a structured JSON.
    """
    # Check if the prompt suggests a table or data extraction
    if "table" in user_prompt.lower() or "figures" in user_prompt.lower() or "extract data" in user_prompt.lower():
        full_prompt = (
            f"You are a financial analyst. The user has provided a document and wants to extract specific data into a table. "
            f"Please identify the relevant financial figures mentioned in the document and organize them into a JSON object "
            f"that represents a table. The object should have keys like 'metric' and 'value'.\n\n"
            f"Document: {document_text}\n\n"
            f"User's Question: {user_prompt}\n\n"
            f"Please respond ONLY with the JSON object."
        )
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(
                full_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            json_str = response.candidates[0].content.parts[0].text
            return json.loads(json_str)
        except (genai.types.generation_types.BlockedPromptException, json.JSONDecodeError) as e:
            st.error(f"Error processing structured output: {e}. Falling back to text response.")
            return "Could not generate a structured table. Please try a different prompt or format."
        except Exception as e:
            return f"An error occurred: {e}. Please check your API key and try again."
    else:
        # Standard text-based response
        full_prompt = (
            f"You are a financial analyst. The user has provided a document and a question. "
            f"Keep your response concise and to the point. The user's language preference is '{selected_lang_code}'.\n\n"
            f"Document: {document_text}\n\n"
            f"User's Question: {user_prompt}"
        )
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(full_prompt)
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                return "No content was generated by the model. Please try a different prompt."
        except genai.types.generation_types.BlockedPromptException:
            return "I am sorry, but your prompt was flagged for safety. Please try rephrasing."
        except Exception as e:
            return f"An error occurred: {e}. Please check your API key and try again."

# Use caching to generate dynamic prompts based on the document
@st.cache_data
def generate_dynamic_prompts(document_text):
    """Generates three dynamic prompts based on the document's content."""
    if not document_text:
        return []

    retries = 3
    delay = 1
    for i in range(retries):
        try:
            prompt = (
                f"Based on the following financial document, generate three very concise and specific questions "
                f"a financial analyst might ask. Respond with a JSON array of three strings.\n\n"
                f"Document: {document_text[:1000]}..." # Use a truncated version for efficiency
            )
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            json_str = response.candidates[0].content.parts[0].text
            return json.loads(json_str)
        except Exception as e:
            time.sleep(delay)
            delay *= 2
            st.warning(f"Attempt {i+1} failed to generate dynamic prompts. Retrying...")
    return ["Summarize the main points.", "Identify key risks and opportunities.", "Extract all financial figures in a table."]

@st.cache_data
def generate_document_summary(document_text):
    """Generates a structured summary of the document with key sections."""
    if not document_text:
        return []

    retries = 3
    delay = 1
    for i in range(retries):
        try:
            prompt = (
                f"Break down the following financial document into 3-5 key sections. "
                f"For each section, provide a concise title and a short summary. "
                f"Respond with a JSON array of objects, where each object has keys 'title', 'summary', and 'content_snippet'. "
                f"The 'content_snippet' should be a small piece of the original text from that section.\n\n"
                f"Document: {document_text[:2000]}..." # Use a truncated version for efficiency
            )
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            json_str = response.candidates[0].content.parts[0].text
            return json.loads(json_str)
        except Exception as e:
            time.sleep(delay)
            delay *= 2
            st.warning(f"Attempt {i+1} failed to generate document summary. Retrying...")
    return []

# Callback function to set the prompt text
def set_prompt(prompt_text):
    st.session_state.user_prompt = prompt_text

# ------------------------------
#   UI Layout and Session State
# ------------------------------
# Initialize session state for document content and analysis result
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
        strings.get("upload_label", "Upload a PDF, TXT, HTML, DOCX, or XLSX file:"),
        type=strings.get("upload_formats", ["pdf", "txt", "html", "docx", "xlsx"]),
        key="file_uploader"
    )

with col2:
    pasted_text = st.text_area(
        strings.get("or_text", "Or, paste the document text here:"),
        height=300,
        key="text_area"
    )

# Logic to load document text into session state
if uploaded_file and not st.session_state.is_document_loaded:
    with st.spinner("Parsing document..."):
        # The parser.py file would need to be updated to handle docx and xlsx files
        # e.g., using libraries like python-docx and openpyxl.
        st.session_state.document_text = parse_uploaded_file(uploaded_file)
        st.session_state.is_document_loaded = True
        st.success("Document loaded successfully! You can now ask a question.")
elif pasted_text and not st.session_state.is_document_loaded:
    st.session_state.document_text = pasted_text
    st.session_state.is_document_loaded = True
    st.success("Text pasted successfully! You can now ask a question.")

# Reset button to clear all session state
if st.button("Reset Session"):
    st.session_state.document_text = ""
    st.session_state.is_document_loaded = False
    st.session_state.analysis_result = ""
    st.session_state.user_prompt = ""
    st.session_state.dynamic_prompts = []
    st.session_state.document_summary = []
    st.rerun()

st.markdown("---")

# ------------------------------
#   Analysis Prompt Section
# ------------------------------
st.subheader(strings.get("section_2_header", "2. Ask the AI a question about the document"))

# Dynamically generate prompts if a document is loaded
if st.session_state.is_document_loaded:
    st.markdown("##### Dynamic Prompts:")
    # Generate and cache dynamic prompts
    st.session_state.dynamic_prompts = generate_dynamic_prompts(st.session_state.document_text)

    prompt_cols = st.columns(3)
    for i, prompt_text in enumerate(st.session_state.dynamic_prompts):
        if prompt_cols[i].button(prompt_text, key=f"quick_prompt_{i}", on_click=set_prompt, args=(prompt_text,)):
            pass
else:
    # Fallback to static prompts if no document is loaded
    st.markdown("##### Quick Prompts:")
    prompt_cols = st.columns(3)
    static_prompts = [
        strings.get("prompt_summarize", "Summarize the main points."),
        strings.get("prompt_risks", "Identify key risks and opportunities."),
        strings.get("prompt_figures", "Extract all financial figures in a table.")
    ]
    for i, prompt_text in enumerate(static_prompts):
        if prompt_cols[i].button(prompt_text, key=f"quick_prompt_{i}", on_click=set_prompt, args=(prompt_text,)):
            pass


user_prompt = st.text_area(
    strings.get("prompt_label", "Enter your analysis prompt here:"),
    value=st.session_state.user_prompt,
    placeholder=strings.get("prompt_placeholder", "e.g., 'Summarize the main risks in 3 bullet points.'"),
    height=100,
    key="user_prompt_area"
)

# Analysis Button
if st.button(strings.get("analyze_button", "Analyze Document")):
    if not st.session_state.document_text:
        st.error(strings.get("error_file_upload", "Please upload a file or paste text to analyze."))
    elif not user_prompt:
        st.error(strings.get("error_prompt", "Please enter a prompt for the analysis."))
    else:
        with st.spinner(strings.get("loading_message", "Analyzing your document with AI...")):
            analysis_result = analyze_document(st.session_state.document_text, user_prompt, selected_lang_code)
            st.session_state.analysis_result = analysis_result

# Display analysis result with structured output handling
if st.session_state.analysis_result:
    st.subheader(strings.get("result_header", "✨ Analysis Result"))
    # Check if the result is a structured JSON table
    if isinstance(st.session_state.analysis_result, list) and all(isinstance(item, dict) for item in st.session_state.analysis_result):
        # Convert JSON to a DataFrame for a clean table display
        try:
            df = pd.DataFrame(st.session_state.analysis_result)
            st.dataframe(df, use_container_width=True)

            # Check if the data is suitable for plotting and display a chart
            if len(df.columns) >= 2:
                # Attempt to find a numerical column and a non-numerical one
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()

                if numeric_cols and non_numeric_cols:
                    x_col = non_numeric_cols[0]
                    y_col = numeric_cols[0]

                    st.markdown("##### Data Visualization")
                    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                    st.plotly_chart(fig, use_container_width=True)

            # Add a download button for the table
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name='analysis_results.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"Could not display the table correctly: {e}")
            st.markdown(f'<div class="result-container">{st.session_state.analysis_result}</div>', unsafe_allow_html=True)
    else:
        # Fallback for standard text output
        st.markdown(f'<div class="result-container">{st.session_state.analysis_result}</div>', unsafe_allow_html=True)
        st.download_button(
            label="Download as TXT",
            data=st.session_state.analysis_result,
            file_name='analysis_results.txt',
            mime='text/plain',
        )

st.markdown("---")

# ------------------------------
#   Document Navigation and Content Section
# ------------------------------
if st.session_state.is_document_loaded:
    st.subheader("Document Content and Navigation")
    with st.expander("View Document and Summary"):
        # Generate and cache document summary
        st.session_state.document_summary = generate_document_summary(st.session_state.document_text)
        
        # Display clickable summaries
        if st.session_state.document_summary:
            st.markdown("##### Key Sections:")
            for section in st.session_state.document_summary:
                if st.button(f"**{section['title']}**", key=f"section_btn_{section['title']}"):
                    st.session_state.document_text = section['content_snippet']
                    st.info(f"Showing content snippet for: **{section['title']}**")
        
        # Display the full document content in a read-only area
        st.text_area("Full Document Text", value=st.session_state.document_text, height=500, disabled=True)

st.markdown("---")

# Disclaimer
st.markdown(strings.get(
    "disclaimer",
    "⚠️ **Disclaimer:** This tool is for demonstration purposes only and should not be used for making financial decisions."
))
