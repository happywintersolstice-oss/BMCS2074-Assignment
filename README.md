# AI-Powered Scam Message and Suspicious Web Content Detection Using NLP

This project is a Python + Streamlit NLP assignment app that classifies SMS text, email text, and webpage text as `Scam/Spam` or `Legitimate`.

It currently supports:
- pasted message analysis
- webpage text analysis from a URL
- model comparison
- dataset preview

## Current Scope

The app uses:
- TF-IDF for feature extraction
- Naive Bayes
- SVM
- Logistic Regression

The current preprocessing flow:
- lowercase text
- remove punctuation
- remove special characters
- remove extra spaces

## App Features

- `Message Analysis`
  - paste suspicious SMS or email text
  - choose a model
  - click `Analyze`
  - view prediction and confidence

- `URL Analysis`
  - enter a URL
  - extract readable webpage text with `requests` and `BeautifulSoup`
  - classify the extracted text
  - show a friendly error if extraction fails

- `Model Comparison`
  - Accuracy
  - Precision
  - Recall
  - F1 Score

- `Dataset Preview`
  - view the current dataset rows used by the app

## Project Structure

- [`app.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\app.py)
  Streamlit entry point

- [`data/raw/sms_spam_demo.csv`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\data\raw\sms_spam_demo.csv)
  Current dataset file

- [`src/constants.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\constants.py)
  Shared paths, app title, and model names

- [`src/text_processing.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\text_processing.py)
  Text cleaning logic

- [`src/data_loader.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\data_loader.py)
  Dataset loading

- [`src/modeling.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\modeling.py)
  Shared training coordinator and prediction flow

- [`src/web_tools.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\web_tools.py)
  Webpage text extraction helpers

- [`src/ui.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\ui.py)
  Streamlit layout, styling, and page rendering

## Model Ownership

Each model now has its own file so each group member can clearly own one model:

- [`src/models/naive_bayes_model.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\models\naive_bayes_model.py)
  Naive Bayes pipeline and training

- [`src/models/svm_model.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\models\svm_model.py)
  SVM pipeline and training

- [`src/models/logistic_regression_model.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\models\logistic_regression_model.py)
  Logistic Regression pipeline and training

## How To Run

Install dependencies:

```powershell
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe -m pip install -r requirements.txt
```

Run the app:

```powershell
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe -m streamlit run app.py
```

If your `python` command already points to the correct installation, this also works:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Notes

- The current dataset is still small and should be replaced or expanded later for the final assignment.
- The current dataset is still small and SMS-heavy, so it should be replaced or expanded later with stronger SMS-and-email coverage for the final assignment.
- The UI has already been refactored into a cleaner single-screen tab layout.
- SVM uses a calibrated linear classifier so the app can still show confidence scores.
- Old files such as `src/preprocessing.py`, `src/train.py`, `src/evaluate.py`, and `src/predict.py` are no longer part of the current structure.

## Extra Ideas For Future Enhancement

- `Risk factor breakdown`
  - show suspicious keyword groups such as urgency, reward, payment, account, or verification language

- `Scam risk level`
  - show a simple Low / Medium / High risk indicator in addition to the raw prediction

- `Highlight suspicious words`
  - visually point out the words or terms that most influenced the prediction

- `Compare all models on the same message`
  - show how Naive Bayes, SVM, and Logistic Regression each classify the same input

- `Batch analysis mode`
  - allow users to paste multiple messages and classify them all in one run

- `URL warning heuristics`
  - inspect the URL structure itself for suspicious patterns before or alongside webpage text analysis

- `Explanation by model`
  - let each model explain its result in a slightly different way based on how that model works

- `Session history`
  - keep a temporary list of recently analyzed messages, predictions, confidence scores, and selected models

- `Evaluation charts`
  - add confusion matrices or simple metric charts for model comparison

- `Scam category tagging`
  - classify likely scam type such as prize scam, account verification scam, loan scam, delivery scam, or general promotional spam

- `OCR image scanning`
  - allow the user to upload an image
  - extract text from the image using OCR
  - send the extracted text into the same NLP classification pipeline
  - return prediction, confidence, and explanation
  - useful for scam posters, screenshot scams, fake reward images, and fake banking alerts
