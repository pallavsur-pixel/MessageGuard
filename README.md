# MessageGuard — SMS Spam Detection

A Machine Learning Lab mini-project that compares four classification algorithms and provides an interactive Streamlit UI.

## Features

- Dataset cleaning and inspection
- Stratified train/test split
- TF-IDF text feature extraction inside model pipelines
- Four models: Multinomial Naive Bayes, Logistic Regression, Linear SVM, Random Forest
- Accuracy, spam precision, spam recall, spam F1, macro F1, ROC-AUC
- Confusion matrix, ROC curves, classification report, model comparison charts
- Interactive SMS prediction page
- Upload your own labelled CSV and download cleaned data / metrics
- Responsive Streamlit layout

## 1. Requirements

- Python 3.10 or newer recommended
- Internet connection for the default UCI dataset download (unless you place the dataset locally)

## 2. Run locally on Windows

Open PowerShell in this project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this once for the current terminal or use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.\.venv\Scripts\python.exe -m streamlit run app.py`.

Streamlit will print a local URL, usually `http://localhost:8501`.

## 3. Dataset & CSV Formats

### A. Default Dataset
The default dataset is the UCI SMS Spam Collection:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The app downloads it automatically on first run (or uses `data/SMSSpamCollection` if placed locally).

### B. Custom Labeled CSV (Training & Evaluation)
Upload a CSV or TXT file with labeled messages. Supported columns (case-insensitive):
- **Message column:** `message`, `text`, `sms`, `v2`, `content`, `sms_text`, `body`
- **Label column:** `label`, `category`, `class`, `v1`, `target`, `tag`
- **Accepted labels:** `ham` / `spam` or `0` / `1`
- **Encodings supported:** UTF-8, Latin-1, CP1252, ISO-8859-1 (handles Kaggle `spam.csv` automatically).

Sample Labeled CSV:
```csv
label,message
ham,Are we still having the team meeting at 4 PM?
spam,WINNER! You have won a £1,000 cash prize. Call 0800123456 to claim now.
```

### C. Unlabeled CSV (Bulk SMS Prediction)
On the **Bulk Prediction** page, upload any CSV or TXT file with an SMS message column (`message`, `text`, `sms`, `v2`, etc.). Labels are **not** required. The system will classify every message in bulk and let you download the enriched CSV.

Sample Bulk Unlabeled CSV:
```csv
message
Urgent! Your mobile number has won a £2,000 cash prize. Call now.
Hey are you coming to class today?
```

## 4. Models and fair comparison

Each model is trained and tested on the same stratified split (fixed random state 42). TF-IDF vectorization is included in each scikit-learn Pipeline, so the vocabulary is learned from training data only. The app ranks models by spam F1-score, with recall and precision as tie-breakers.

Do not write expected scores into your report. Run the app, record the results produced on your run, and explain the trade-offs.

## 5. Deployment using Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `README.md`, and `.gitignore`.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose the repository, branch, and `app.py` as the main file.
4. Deploy and wait for the build to finish.
5. Open the public URL on both a desktop and a mobile browser.

Do not upload private messages or personal data. The app uses a public dataset by default and may download that dataset at startup.

## 6. Suggested repository structure

```text
sms-spam-detection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── (optional local SMSSpamCollection file)
```

## 7. Suggested 10-page report structure

1. Title page and abstract
2. Introduction and problem statement
3. Objectives and existing-system limitations
4. Dataset description and data cleaning
5. Exploratory data analysis and class distribution
6. TF-IDF, train/test split, and methodology
7. Four models and their working principles
8. Performance comparison table, confusion matrix, ROC curves
9. UI screenshots, deployment, GitHub structure
10. Conclusion, limitations, future scope, references

## 8. Important viva points

- TF-IDF converts text into numerical features.
- Stratification preserves approximate class proportions.
- Accuracy alone can be misleading with imbalanced data.
- Precision measures the correctness of spam predictions.
- Recall measures how much actual spam is detected.
- F1-score is the harmonic mean of precision and recall.
- ROC-AUC measures ranking quality across thresholds.
- Data leakage is avoided by fitting TF-IDF only on training data.
- A model that wins on one split is not guaranteed to generalize to all future messages.

## Dataset citation

Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering: New Collection and Results. Proceedings of the 2011 ACM Symposium on Document Engineering. Dataset hosted by the UCI Machine Learning Repository.
