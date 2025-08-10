MyFinDocGPT: AI-Powered Financial Document Analysis
FinDocGPT is a powerful, web-based application built with Streamlit and the Google Gemini API. It is designed to assist financial analysts and users with a need to quickly understand, summarize, and compare financial documents. By leveraging a large language model, the tool can answer specific questions, extract key figures into tables, and even visualize data.

🌟 Key Features
Multi-Document Analysis: Easily upload and analyze a single document or compare two documents side-by-side.

Multi-Format Support: The application can parse various file types, including PDF, TXT, HTML, DOCX, and XLSX.

AI-Powered Insights: Ask the AI questions about your documents and get concise, intelligent answers.

Structured Data Extraction: The AI can extract key financial figures and present them in a clean, interactive table.

Interactive Data Visualization: When the AI returns tabular data, the app can automatically generate charts (bar, line, scatter plots) to help you visualize the data.

Dynamic Prompt Suggestions: Based on the content of your document, the AI will suggest relevant questions to ask, helping you get started.

Multi-Language Support: The user interface is available in multiple languages, with easy selection from the sidebar.

Robust Error Handling: Includes a built-in exponential backoff mechanism for API calls to ensure stability and reliability.

🚀 Prerequisites
To run this application, you will need:

Python 3.7+

A Google API Key for the Gemini API. You can get one from the Google AI Studio.

💻 Installation
Follow these steps to get the application up and running on your local machine.

Clone the repository:

git clone https://github.com/your-username/findocgpt.git
cd findocgpt

Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required packages:

pip install -r requirements.txt

Configure your Google API Key:
Create a new file named .streamlit/secrets.toml in the root directory of your project.
Add your API key to this file:

GOOGLE_API_KEY = "YOUR_API_KEY_HERE"

Run the application:

streamlit run app.py

The application will open in your web browser.

📝 Usage Guide
Single Document Analysis
Upload: Use the "Upload a file" button or paste text directly into the text area.

Prompt: Enter your question in the text area labeled "Enter your analysis prompt here."

Analyze: Click the "Analyze Document" button. The AI's response will appear below. If the response is a table, it will be displayed as an interactive dataframe with a chart.

Document Comparison
Toggle: Click the "Compare Two Documents" toggle at the top of the page.

Upload: Upload or paste text for both "Document 1" and "Document 2" in their respective sections.

Prompt: Enter a comparative question, such as "What are the key differences between these two reports?"

Compare: Click the "Compare Documents" button to get the AI's analysis.

⚠️ Disclaimer
This tool is for demonstration purposes only and should not be used for making financial decisions. The analysis provided is a result of a large language model and may not always be accurate. Always consult with a qualified professional for financial advice.
