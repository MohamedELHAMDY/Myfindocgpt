# app.py - Fully Optimized Version with Document Comparison
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
    /* Style for the file uploaders when a file is present */
    .stFileUploader.has-files label > div {
        background-color: #d4edda !important;
        border-color: #c3e6cb !important;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================
#   Gemini API Interaction
# ======================================================

# Configure Gemini API
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    MODEL_NAME = "gemini-1.5-flash"
else:
    st.error("Please set the GOOGLE_API_KEY in your Streamlit secrets.")
    st.stop()


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
    if not document_text:
        return "No document content provided for analysis."

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

def compare_documents(doc1_text, doc2_text, user_prompt, selected_lang_code):
    """
    Sends a prompt to compare two documents to the Gemini API.
    """
    if not doc1_text or not doc2_text:
        return "Please upload both documents for comparison."

    full_prompt = (
        f"You are a financial analyst. The user has provided two financial documents and a question. "
        f"Your task is to compare these two documents based on the question. "
        f"Keep your response concise and to the point. The user's language preference is '{selected_lang_code}'.\n"
        f"Document 1: {doc1_text}\n"
        f"Document 2: {doc2_text}\n"
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
if "document_text_1" not in st.session_state: st.session_state.document_text_1 = ""
if "is_document_loaded_1" not in st.session_state: st.session_state.is_document_loaded_1 = False
if "document_text_2" not in st.session_state: st.session_state.document_text_2 = ""
if "is_document_loaded_2" not in st.session_state: st.session_state.is_document_loaded_2 = False
if "analysis_result" not in st.session_state: st.session_state.analysis_result = ""
if "user_prompt" not in st.session_state: st.session_state.user_prompt = ""
if "dynamic_prompts" not in st.session_state: st.session_state.dynamic_prompts = []
if "document_summary_1" not in st.session_state: st.session_state.document_summary_1 = []
if "document_summary_2" not in st.session_state: st.session_state.document_summary_2 = []
if "chart_type" not in st.session_state: st.session_state.chart_type = "Bar Chart"
if "comparison_mode" not in st.session_state: st.session_state.comparison_mode = False

# Callback function to set the prompt text from a button click
def set_prompt(prompt_text):
    st.session_state.user_prompt = prompt_text

# Sidebar for language selection
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

# Mode selection toggle
comparison_mode = st.toggle("Compare Two Documents", value=st.session_state.comparison_mode)
st.session_state.comparison_mode = comparison_mode
st.markdown("---")

# ------------------------------
#   Document Upload Section
# ------------------------------
if comparison_mode:
    st.subheader("1. Upload Two Documents for Comparison")
    upload_col1, upload_col2 = st.columns(2)
    with upload_col1:
        st.markdown("##### Document 1")
        uploaded_file_1 = st.file_uploader(
            "Upload a PDF, TXT, HTML, DOCX, or XLSX file:",
            type=["pdf", "txt", "html", "docx", "xlsx"],
            key="file_uploader_1"
        )
        pasted_text_1 = st.text_area("Or, paste text for Document 1 here:", height=200, key="text_area_1")
    
    with upload_col2:
        st.markdown("##### Document 2")
        uploaded_file_2 = st.file_uploader(
            "Upload a PDF, TXT, HTML, DOCX, or XLSX file:",
            type=["pdf", "txt", "html", "docx", "xlsx"],
            key="file_uploader_2"
        )
        pasted_text_2 = st.text_area("Or, paste text for Document 2 here:", height=200, key="text_area_2")

    # Logic to handle document loading for comparison mode
    if uploaded_file_1 and not st.session_state.is_document_loaded_1:
        with st.spinner("Parsing Document 1..."):
            try:
                st.session_state.document_text_1 = parse_uploaded_file(uploaded_file_1)
                st.session_state.is_document_loaded_1 = True
            except Exception as e:
                st.error(f"Failed to parse Document 1: {e}")
    elif pasted_text_1 and not st.session_state.is_document_loaded_1:
        st.session_state.document_text_1 = pasted_text_1
        st.session_state.is_document_loaded_1 = True

    if uploaded_file_2 and not st.session_state.is_document_loaded_2:
        with st.spinner("Parsing Document 2..."):
            try:
                st.session_state.document_text_2 = parse_uploaded_file(uploaded_file_2)
                st.session_state.is_document_loaded_2 = True
            except Exception as e:
                st.error(f"Failed to parse Document 2: {e}")
    elif pasted_text_2 and not st.session_state.is_document_loaded_2:
        st.session_state.document_text_2 = pasted_text_2
        st.session_state.is_document_loaded_2 = True

    if st.session_state.is_document_loaded_1 and st.session_state.is_document_loaded_2:
        st.success("Both documents loaded successfully!")

else: # Single document mode
    st.subheader("1. Upload a Document or Paste Text")
    upload_col, text_col = st.columns([1, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload a PDF, TXT, HTML, DOCX, or XLSX file:",
            type=["pdf", "txt", "html", "docx", "xlsx"],
            key="file_uploader"
        )
    with text_col:
        pasted_text = st.text_area(
            "Or, paste the document text here:",
            height=200,
            key="text_area"
        )

    # Logic to handle document loading for single mode
    if uploaded_file and not st.session_state.is_document_loaded_1:
        with st.spinner("Parsing document..."):
            try:
                st.session_state.document_text_1 = parse_uploaded_file(uploaded_file)
                st.session_state.is_document_loaded_1 = True
                st.success("Document loaded successfully!")
            except Exception as e:
                st.error(f"Failed to parse the uploaded file: {e}")
    elif pasted_text and not st.session_state.is_document_loaded_1:
        st.session_state.document_text_1 = pasted_text
        st.session_state.is_document_loaded_1 = True
        st.success("Text pasted successfully!")

# Reset button to clear all session state
if st.button("Reset Session", help="Clear the current documents and analysis."):
    st.session_state.clear()
    st.rerun()

st.markdown("---")

# ------------------------------
#   Analysis Prompt Section
# ------------------------------
st.subheader(strings.get("section_2_header", "2. Ask the AI a question about the document"))

# Dynamic prompts are generated only after a document is loaded
if not comparison_mode and st.session_state.is_document_loaded_1:
    st.markdown("##### Dynamic Prompts: <i class='fas fa-lightbulb'></i>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 14px; color: #666;'>These prompts were generated by the AI based on the document's content.</p>", unsafe_allow_html=True)
    st.session_state.dynamic_prompts = generate_dynamic_prompts(st.session_state.document_text_1)
    prompt_cols = st.columns(3)
    for i, prompt_text in enumerate(st.session_state.dynamic_prompts):
        with prompt_cols[i]:
            st.button(prompt_text, key=f"quick_prompt_{i}", on_click=set_prompt, args=(prompt_text,))
else:
    st.markdown("##### Quick Prompts: <i class='fas fa-question-circle'></i>", unsafe_allow_html=True)
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
if comparison_mode:
    if st.button("Compare Documents"):
        if not st.session_state.is_document_loaded_1 or not st.session_state.is_document_loaded_2:
            st.error("Please upload two files or paste text to compare.")
        elif not user_prompt:
            st.error("Please enter a prompt for the comparison.")
        else:
            with st.empty():
                with st.spinner("Comparing your documents with AI..."):
                    st.session_state.analysis_result = compare_documents(
                        st.session_state.document_text_1,
                        st.session_state.document_text_2,
                        user_prompt,
                        selected_lang_code
                    )
else: # Single document mode
    if st.button(strings.get("analyze_button", "Analyze Document")):
        if not st.session_state.is_document_loaded_1:
            st.error(strings.get("error_file_upload", "Please upload a file or paste text to analyze."))
        elif not user_prompt:
            st.error(strings.get("error_prompt", "Please enter a prompt for the analysis."))
        else:
            with st.empty():
                with st.spinner(strings.get("loading_message", "Analyzing your document with AI...")):
                    st.session_state.analysis_result = analyze_document(
                        st.session_state.document_text_1,
                        user_prompt,
                        selected_lang_code
                    )

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
                            fig = px.scatter(df, x=x_col, y=y_col, color=y_col, labels={col: col.replace('_', ' ').title() for col in df.columns})
                        
                        fig.update_layout(title_text=f"{chart_type} of {y_col} by {x_col}", title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Poppins", size=14, color="#333"), xaxis={'categoryorder':'total descending'}, hovermode="x unified", showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download as CSV", data=csv, file_name='analysis_results.csv', mime='text/csv')
            except Exception as e:
                st.error(f"Could not display the table or chart correctly: {e}")
                st.markdown(f'<div class="result-container">{st.session_state.analysis_result}</div>', unsafe_allow_html=True)
    else:
        with result_container:
            st.markdown(f'<div class="result-container">{st.session_state.analysis_result}</div>', unsafe_allow_html=True)
            st.download_button(label="Download as TXT", data=st.session_state.analysis_result.encode('utf-8'), file_name='analysis_results.txt', mime='text/plain')

st.markdown("---")

# ------------------------------
#   Document Navigation and Content Section
# ------------------------------
if st.session_state.is_document_loaded_1 and not comparison_mode:
    st.subheader("Document Content and Navigation")
    with st.expander("View Document and Summary"):
        st.session_state.document_summary_1 = generate_document_summary(st.session_state.document_text_1)

        if st.session_state.document_summary_1:
            st.markdown("##### Key Sections: <i class='fas fa-clipboard-list'></i>", unsafe_allow_html=True)
            summary_cols = st.columns(len(st.session_state.document_summary_1))
            for i, section in enumerate(st.session_state.document_summary_1):
                with summary_cols[i]:
                    st.markdown(f"**{section['title']}**")
                    st.write(f"_{section['summary']}_")
        else:
            st.info("No summary available. You can view the full document text below.")

        st.text_area("Full Document Text", value=st.session_state.document_text_1, height=500, disabled=True)

if st.session_state.is_document_loaded_1 and st.session_state.is_document_loaded_2 and comparison_mode:
    st.subheader("Documents for Comparison")
    doc_view_col1, doc_view_col2 = st.columns(2)
    with doc_view_col1:
        with st.expander("View Document 1 Summary"):
            st.session_state.document_summary_1 = generate_document_summary(st.session_state.document_text_1)
            st.text_area("Full Document 1 Text", value=st.session_state.document_text_1, height=400, disabled=True)
    with doc_view_col2:
        with st.expander("View Document 2 Summary"):
            st.session_state.document_summary_2 = generate_document_summary(st.session_state.document_text_2)
            st.text_area("Full Document 2 Text", value=st.session_state.document_text_2, height=400, disabled=True)
# ======================================================
#   Privacy Policy Section
# ======================================================

def display_privacy_policy():
    """Displays a detailed privacy policy in an expander."""
    st.markdown("---")
    st.markdown("### Privacy Policy")
    with st.expander("Read our Privacy Policy"):
        st.markdown("""
        **1. Introduction**
        FinDocGPT is committed to protecting your privacy. This policy explains how we handle your information when you use our application.

        **2. Information We Collect**
        We do not store or permanently save any documents, text, or prompts you upload or enter into the application. All data is processed in real-time and is discarded immediately after the analysis is complete. We do not collect any personal information, such as your name, email address, or location.

        **3. How We Use Your Information**
        The text and prompts you provide are sent to the Google Gemini API for the sole purpose of generating an analysis. The content is used to perform the requested task and is not stored or used for any other purpose.

        **4. Third-Party Services**
        Our application uses the Google Gemini API. Your data is subject to Google's privacy policy when it is transmitted to their services. We are not responsible for the privacy practices of third-party services.

        **5. Data Security**
        We implement reasonable security measures to protect your information during transmission. However, no method of transmission over the internet or method of electronic storage is 100% secure.

        **6. Changes to This Privacy Policy**
        We may update our Privacy Policy from time to time. We will notify you of any changes by posting the new policy on this page.

        **7. Contact Us**
        If you have any questions about this Privacy Policy, please contact us.
        """)

# Add this line to the end of the script to display the privacy policy
display_privacy_policy()
