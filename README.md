# MessageGuard 🛡️ — SMS Spam Detection & Model Comparison

A comprehensive Machine Learning laboratory system that benchmarks four classification algorithms for SMS spam detection, featuring an interactive Streamlit UI, multi-dataset file upload, leakage-free TF-IDF feature pipelines, in-depth diagnostic analytics, and batch CSV inference.

---

## 🌟 Key Features

1. **Multi-Dataset Upload & Merging**:
   - Simultaneous upload of multiple `.csv`, `.tsv`, and `.txt` dataset files.
   - Automatic column detection supporting standard (`label`/`message`), Kaggle-style (`v1`/`v2` with comma-split text reconstruction), and custom headers.
   - Fault-tolerant processing: invalid or corrupted files are isolated in the manifest with descriptive error messages while valid files continue to merge seamlessly.
   - Rigorous data cleaning: whitespace stripping, label normalization (`ham`/`spam`, `0`/`1`), duplicate record removal, and conflicting-label dropping.
   - Transparent manifest table displaying individual file names, rows parsed, detected columns, and error diagnostics.

2. **Executive Overview Dashboard**:
   - Live system status indicators: Active Dataset, Clean Usable Messages, Ham/Spam prevalence, and Model Training status.
   - Dynamic benchmark snapshot table automatically populated upon model training.
   - Prominent single-click training action directly on the Overview page.
   - 5-Phase end-to-end pipeline architecture visual layout.

3. **Leakage-Free Model Training & Fair Comparison**:
   - Four distinct classification algorithms:
     1. **Multinomial Naive Bayes** (`alpha=1.0`)
     2. **Logistic Regression** (`C=1.0`, `max_iter=1000`)
     3. **Linear Support Vector Machine (LinearSVC)** (`C=1.0`)
     4. **Random Forest** (`n_estimators=100`, `random_state=42`)
   - Strict data leakage prevention: TF-IDF vectorizer vocabularies and IDF weights are learned exclusively from the training split inside each Scikit-learn Pipeline.
   - Configurable stratified test/train split slider (10% to 40%).
   - Rigorous model selection rule: Primary rank by **Spam F1-Score**, with **Spam Recall** and **Spam Precision** as explicit tie-breakers.

4. **In-Depth Performance Analytics**:
   - Interactive, annotated Confusion Matrix for each evaluated algorithm.
   - Multi-model ROC Curve comparison with independent continuous decision curves and AUC scores.
   - Complete Scikit-learn Classification Report dataframe (Precision, Recall, F1-Score, Support).

5. **Individual SMS Spam Detector**:
   - Model selection dropdown with live test F1 score.
   - Instant message classification with pre-configured sample triggers (Lottery Scam, Phishing URL, Authentic Ham).
   - High-contrast, accessible verdict boxes that **do not rely on color alone** (prominent icons, uppercase verdict badges, and descriptive context).
   - Calibrated probability progress bars or SVM decision boundary scores.

6. **Bulk SMS Prediction & Export**:
   - Batch classification on unlabeled CSV, TSV, or TXT files.
   - Automatic message column detection with manual confirmation.
   - Batch summary metrics, class distribution donut chart, and preview dataframe.
   - One-click CSV export with preserved input row order and attached predictions.

---

## 🚀 Quickstart & Local Installation

### Prerequisites
- Python 3.10 or newer
- Windows PowerShell, macOS Terminal, or Linux Shell

### Setup Steps
```powershell
# 1. Clone repository
git clone https://github.com/pallavsur-pixel/MessageGuard.git
cd MessageGuard

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Linux/macOS: source .venv/bin/activate

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch Streamlit Application
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📂 Supported Dataset Formats

### A. Default UCI Dataset
- Default benchmark: **UCI SMS Spam Collection** (5,572 raw records).
- Stored locally in `data/SMSSpamCollection` for offline execution.

### B. Custom Labeled Datasets (Single or Multiple Upload)
Supported extensions: `.csv`, `.tsv`, `.txt`.
- **Accepted Message Columns (case-insensitive):** `message`, `text`, `sms`, `v2`, `content`, `sms_text`, `body`
- **Accepted Label Columns (case-insensitive):** `label`, `category`, `class`, `v1`, `target`, `tag`
- **Accepted Label Values:** `ham` / `spam` or `0` / `1`
- **Encodings:** UTF-8, Latin-1, CP1252, ISO-8859-1 (automatically resolves Kaggle `spam.csv` encoding issues).

Sample Labeled CSV:
```csv
label,message
ham,Are we still having the project discussion at 4 PM in the library?
spam,WINNER! You have won a £1,000 cash prize. Call 0800123456 to claim now.
```

### C. Unlabeled Datasets (Bulk Prediction)
- Requires only a column containing text messages.
- Any additional columns are preserved in the final downloadable CSV.

---

## 🧪 Automated Test Suite

MessageGuard includes 18 unit and integration tests covering data ingestion, column detection, deduplication, model benchmarking, ROC curves, multi-file uploads, and error resilience.

To run the full master test suite:
```powershell
python test_master_suite.py
```

To run regression tests:
```powershell
python test_runner.py
```

---

## 🎓 Academic Viva & Lab Demonstration Notes

- **Why Spam F1-Score over Overall Accuracy?** SMS datasets exhibit severe class imbalance (~87% Ham vs ~13% Spam). A trivial baseline predicting always *Ham* achieves 87% accuracy while completely failing to stop spam. Spam F1-Score balances precision and recall.
- **Precision vs Recall Trade-off:**
  - *Spam Recall* ($\frac{TP}{TP+FN}$): Measures the fraction of actual spam caught. High recall prevents dangerous phishing links from reaching users.
  - *Spam Precision* ($\frac{TP}{TP+FP}$): Measures the fraction of flagged messages that were genuinely spam. High precision ensures critical personal messages and 2FA OTPs are never blocked.
- **Data Leakage Guarantee:** The TF-IDF vectorizer vocabulary, IDF weights, and classifiers are fit exclusively on the training fold during `train_test_split`.
- **Fault-Tolerant Multi-Dataset Merging:** Processing multiple files independently standardizes heterogeneous schemas before concatenation and global deduplication, ensuring that malformed files do not abort valid data ingestion.
- **Core ML Points:**
  - TF-IDF converts raw text into numerical features based on term frequency and document specificity.
  - Stratification preserves approximate class proportions across train and test partitions.
  - ROC-AUC measures ranking quality across continuous decision thresholds.
  - A model that wins on one split is not guaranteed to generalize to all future messages.

## Dataset citation

Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering: New Collection and Results. Proceedings of the 2011 ACM Symposium on Document Engineering. Dataset hosted by the UCI Machine Learning Repository.
