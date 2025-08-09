# app.py
import streamlit as st
import os
from src.llm import analyze_document
from src.parser import parse_uploaded_file

# Set up the Streamlit page configuration
st.set_page_config(page_title="FinDocGPT", layout="wide")
st.title("📊 FinDocGPT – AI Document Analyst")

st.markdown("---")

# User input: Upload file or paste text
st.subheader("1. Upload or paste your document")
uploaded_file = st.file_uploader("Upload a financial report (TXT, HTML, PDF)", type=["txt", "html", "pdf"])
pasted_text = st.text_area("Or, paste the document text here:", height=300)

document_text = ""
if uploaded_file:
    # Use the parser to get the text from the uploaded file
    document_text = parse_uploaded_file(uploaded_file)
elif pasted_text:
    document_text = pasted_text

# User input: The analysis prompt
st.subheader("2. Ask the AI a question about the document")
user_prompt = st.text_area(
    "Enter your analysis prompt here (e.g., 'Summarize the main risks in 3 bullet points.', 'What is the total revenue for the latest fiscal year?', 'Analyze the sentiment of this report.')"
)

# Analysis button
if st.button("Analyze"):
    if not document_text:
        st.error("Please upload a file or paste text to analyze.")
    elif not user_prompt:
        st.error("Please enter a prompt for the analysis.")
    else:
        with st.spinner("Analyzing your document with AI..."):
            # Call the new generic analysis function
            analysis_result = analyze_document(document_text, user_prompt)
            st.subheader("✨ Analysis Result")
            st.write(analysis_result)

st.markdown("---")
st.markdown("⚠️ **Disclaimer:** This tool is for demonstration purposes only and should not be used for making financial decisions.")
st.markdown("This application uses the Gemini API.")
