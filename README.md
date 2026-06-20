# AI-Powered Scam Message and Suspicious Web Content Detection Using NLP

This project is a Streamlit-based NLP prototype for classifying text as `Scam/Spam` or `Legitimate`.
It supports pasted messages and extracted webpage content, and compares multiple classic NLP models.

## Current Project Status

This project has already been refactored into a modular structure.

Important:
- `app.py` is now only a small Streamlit entry point
- the main logic is inside the `src/` folder
- older files such as `src/preprocessing.py`, `src/train.py`, `src/evaluate.py`, and `src/predict.py` are no longer part of the current structure
- if those old filenames are still open in VS Code tabs, close them and reopen the current files from the Explorer

## Features

- Python + Streamlit web app
- 3 NLP models:
  - Multinomial Naive Bayes
  - Support Vector Machine (SVM)
  - Logistic Regression
- TF-IDF feature extraction
- Basic preprocessing:
  - lowercase
  - remove punctuation
  - remove special characters
  - remove extra spaces
- Message analysis page:
  - paste a message
  - choose a model
  - click `Analyze`
  - view prediction and confidence
- URL analysis page:
  - enter a URL
  - extract visible page text with `requests` and `BeautifulSoup`
  - classify extracted text
  - show a friendly error if extraction fails
- Model comparison page:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Dataset preview page

## Latest Structure

- [`app.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\app.py): small app entry point
- [`requirements.txt`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\requirements.txt): Python dependencies
- [`data/raw/sms_spam_demo.csv`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\data\raw\sms_spam_demo.csv): demo dataset
- [`src/constants.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\constants.py): shared paths, app title, model names
- [`src/text_processing.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\text_processing.py): text cleaning logic
- [`src/data_loader.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\data_loader.py): dataset loading
- [`src/modeling.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\modeling.py): model training, evaluation, prediction
- [`src/web_tools.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\web_tools.py): webpage extraction helpers
- [`src/ui.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\ui.py): Streamlit UI layout and styling

## How To Run

Open this project folder in VS Code, then run the app from the terminal.

Recommended command on your current machine:

```powershell
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe -m streamlit run app.py
```

If you need to install the dependencies first:

```powershell
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe -m pip install -r requirements.txt
```

If your `python` command already points to the correct installation, this also works:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Notes

- The included dataset is a small demo dataset so the prototype can run quickly.
- You can later replace it with a larger public dataset without changing the overall app structure.
- The SVM setup now uses a calibrated linear classifier so the app can still display confidence scores cleanly.
- The UI has been redesigned and the logic has been separated into modules to make the code easier to explain in your assignment and presentation.
