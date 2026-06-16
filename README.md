# AI-Powered Scam Message and Suspicious Web Content Detection Using NLP

This is a simple Streamlit prototype for detecting whether a text message or web page content looks like scam/spam or legitimate content.

## Features

- Python + Streamlit web app
- 3 NLP models:
  - Multinomial Naive Bayes
  - Support Vector Machine (SVM)
  - Logistic Regression
- TF-IDF feature extraction
- Basic text preprocessing:
  - lowercase
  - remove punctuation
  - remove special characters
- Message analysis page:
  - paste a message
  - choose a model
  - click **Analyze**
  - view prediction and confidence
- URL analysis page:
  - enter a URL
  - extract page text with `requests` and `BeautifulSoup`
  - run scam/spam detection on the extracted text
  - show a friendly error if extraction fails
- Model comparison section:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

## Project Files

- [`app.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\app.py): main Streamlit app
- [`requirements.txt`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\requirements.txt): Python dependencies
- [`data/raw/sms_spam_demo.csv`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\data\raw\sms_spam_demo.csv): small demo dataset

## How To Run

1. Open this folder in VS Code.
2. Open the terminal in VS Code.
3. Install the required packages:

```powershell
pip install -r requirements.txt
```

4. Run the Streamlit app:

```powershell
streamlit run app.py
```

## Notes

- The dataset included here is a small demo dataset so the prototype can work quickly.
- You can later replace it with a larger public SMS spam dataset without changing the app structure.
- The code is written simply and includes comments so it is easier to explain during your presentation.
