# src/parser.py
import io
import fitz # PyMuPDF
import os

def parse_uploaded_file(uploaded_file) -> str:
    """
    Parses an uploaded file (PDF, TXT, or HTML) and extracts its text content.
    This function is now a generic text extractor, no longer splitting by 10-K sections.
    """
    string_data = ""
    # Check the file type
    if uploaded_file.name.endswith(".pdf"):
        try:
            # Read the file bytes
            pdf_bytes = uploaded_file.getvalue()
            # Open the PDF with PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            # Extract text from each page
            for page in doc:
                text += page.get_text()
            # Close the document
            doc.close()
            string_data = text
        except Exception as e:
            print(f"Error parsing PDF file: {e}")
            return "Could not parse PDF."
    # Handle text-based files (TXT, HTML)
    else:
        string_data = io.StringIO(uploaded_file.getvalue().decode("utf-8")).read()

    # Truncate the document text to avoid exceeding the model's token limit
    # The current limit for gemini-1.5-flash is very high, but this is a good practice.
    # We will keep the first 50,000 characters.
    max_chars = 50000
    if len(string_data) > max_chars:
        return string_data[:max_chars]
        
    return string_data
