FinDocGPT
An AI-powered financial analyst that parses, analyzes, and summarizes financial filings like 10-K reports to help users make informed decisions.

Features
Upload a PDF document (e.g., a 10-K filing).

Extract text from the document.

Use predefined prompts to analyze the document with the Gemini API (e.g., summarizing risk factors, extracting key metrics).

Write custom prompts for any kind of analysis.

Local Development
Clone the repository:

git clone https://github.com/your-username/FinDocGPT.git
cd FinDocGPT

Set up a virtual environment:

python -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Configure API Key:

Create a .env file in the root of the project.

Add your Google API key to the file: GOOGLE_API_KEY="your_api_key_here"

Run the application:

streamlit run app.py

Deployment on Streamlit Cloud
Push your code to a GitHub repository.

Go to Streamlit Cloud and click "New app".

Select your repository and branch.

Add your Google API key as a secret. In the "Advanced settings," add GOOGLE_API_KEY with your API key as the value. The app.py code is already configured to read this secret.

Deploy!
