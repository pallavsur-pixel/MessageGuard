import io
import os
import hashlib
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# ---------------- Streamlit App Configuration ----------------
st.set_page_config(
    page_title="MessageGuard | SMS Spam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
LOCAL_DATA = Path("data/SMSSpamCollection")

LABEL_CANDIDATES = [
    "label", "v1", "category", "class", "target", "type",
    "tag", "classification", "is_spam", "spam_label", "status"
]

TEXT_CANDIDATES = [
    "message", "text", "sms", "v2", "content", "message_text",
    "sms_text", "body", "email", "mail", "clean_text", "sentence", "description"
]


# ==============================================================================
# Helper Functions: Label Normalization & Data Loading
# ==============================================================================
def normalize_label(value):
    """
    Normalize supported labels to 'spam' or 'ham'.
    Does NOT silently map arbitrary labels; unsupported labels return None.
    """
    if pd.isna(value):
        return None
    val_str = str(value).strip().lower()
    if val_str in {"spam", "1", "1.0", "yes", "true", "s", "positive"}:
        return "spam"
    if val_str in {"ham", "not spam", "not_spam", "legitimate", "legit", "0", "0.0", "no", "false", "h", "negative"}:
        return "ham"
    return None


def load_uploaded_dataset(uploaded_file):
    """
    Robustly read an uploaded CSV/text file with multi-encoding fallback
    (UTF-8, Latin-1, CP1252, ISO-8859-1, etc.) and safe delimiter handling.
    """
    if hasattr(uploaded_file, "getvalue"):
        raw_bytes = uploaded_file.getvalue()
    elif hasattr(uploaded_file, "read"):
        raw_bytes = uploaded_file.read()
    elif isinstance(uploaded_file, (bytes, bytearray)):
        raw_bytes = bytes(uploaded_file)
    else:
        with open(uploaded_file, "rb") as f:
            raw_bytes = f.read()

    if not raw_bytes or len(raw_bytes.strip()) == 0:
        raise ValueError("The uploaded file is empty. Please upload a file containing dataset rows.")

    encodings_to_try = ["utf-8", "latin-1", "cp1252", "iso-8859-1", "utf-8-sig"]

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, on_bad_lines="skip")
            # If all columns collapsed into 1, check if tab, semicolon, or pipe separated
            if len(df.columns) == 1:
                first_chunk = raw_bytes[:2048]
                for sep_cand in ["\t", ";", "|"]:
                    if sep_cand.encode("latin-1") in first_chunk:
                        try:
                            df_cand = pd.read_csv(
                                io.BytesIO(raw_bytes), sep=sep_cand,
                                encoding=enc, on_bad_lines="skip"
                            )
                            if len(df_cand.columns) > 1:
                                df = df_cand
                                break
                        except Exception:
                            pass
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            continue

    # Final fallback: UTF-8 with replacement characters so it never crashes
    return pd.read_csv(
        io.BytesIO(raw_bytes), encoding="utf-8", encoding_errors="replace",
        on_bad_lines="skip"
    )


@st.cache_data(show_spinner=False)
def load_default_dataset():
    """Load default dataset from local disk or fetch from UCI ML repository."""
    if LOCAL_DATA.exists():
        raw_bytes = LOCAL_DATA.read_bytes()
    else:
        request = urllib.request.Request(
            UCI_URL, headers={"User-Agent": "Mozilla/5.0"}
        )
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                archive_bytes = response.read()
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
                member = next(
                    (n for n in zf.namelist() if n.endswith("SMSSpamCollection")),
                    None
                )
                if member is None:
                    raise ValueError("The downloaded UCI archive did not contain SMSSpamCollection.")
                raw_bytes = zf.read(member)
        except Exception as exc:
            raise ConnectionError(
                f"Could not download the default UCI dataset ({exc}). "
                "Please check your internet connection or upload a labeled CSV file."
            )

    # The official UCI dataset is tab-separated and has no header.
    for enc in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
        try:
            return pd.read_csv(
                io.BytesIO(raw_bytes), sep="\t", header=None,
                names=["label", "message"], encoding=enc, on_bad_lines="skip"
            )
        except UnicodeDecodeError:
            continue
    return pd.read_csv(
        io.BytesIO(raw_bytes), sep="\t", header=None,
        names=["label", "message"], encoding="utf-8", encoding_errors="replace", on_bad_lines="skip"
    )


# ==============================================================================
# Column Detection & Cleaning Pipeline
# ==============================================================================
def detect_columns(df):
    """
    Detect candidate message and label columns with case-insensitivity and trimming.
    Returns (detected_label_col, detected_text_col, is_ambiguous).
    """
    df_cols = list(df.columns)
    lower_map = {str(c).strip().lower(): c for c in df_cols}

    label_col = None
    for cand in LABEL_CANDIDATES:
        if cand in lower_map:
            label_col = lower_map[cand]
            break

    text_col = None
    for cand in TEXT_CANDIDATES:
        if cand in lower_map and lower_map[cand] != label_col:
            text_col = lower_map[cand]
            break

    # Two-column fallback heuristic if candidate match didn't find both
    is_ambiguous = False
    if label_col is None or text_col is None:
        if len(df_cols) >= 2:
            col0, col1 = df_cols[0], df_cols[1]
            s0 = df[col0].dropna().astype(str).head(100).map(normalize_label)
            s1 = df[col1].dropna().astype(str).head(100).map(normalize_label)
            if len(s0) and s0.notna().mean() >= 0.6:
                label_col, text_col = col0, col1
            elif len(s1) and s1.notna().mean() >= 0.6:
                label_col, text_col = col1, col0
            else:
                is_ambiguous = True
        else:
            is_ambiguous = True

    return label_col, text_col, is_ambiguous


def clean_dataframe(df, label_col=None, text_col=None):
    """
    Standardize labels and text, validate columns, remove empty/invalid rows,
    handle conflicting labels, and remove duplicate records.
    """
    if df is None or len(df) == 0:
        raise ValueError("The dataset is empty. Please provide a dataset with records.")

    df_working = df.copy()

    # Reconstruct Kaggle spam.csv where commas split text across Unnamed: 2, 3, 4
    lower_cols = [str(c).strip().lower() for c in df_working.columns]
    if "v1" in lower_cols and "v2" in lower_cols:
        v1_real = df_working.columns[lower_cols.index("v1")]
        v2_real = df_working.columns[lower_cols.index("v2")]
        unnamed_cols = [c for c in df_working.columns if str(c).strip().lower().startswith("unnamed")]
        if unnamed_cols:
            df_working[v2_real] = df_working[[v2_real] + unnamed_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()

    # Column identification
    if label_col is None or text_col is None:
        det_label, det_text, is_ambiguous = detect_columns(df_working)
        label_col = label_col or det_label
        text_col = text_col or det_text

    if label_col is None or text_col is None:
        found = list(df_working.columns)
        raise ValueError(
            f"Could not identify both the label and message columns. Found columns: {found}. "
            "Please ensure your CSV has a label column (e.g., 'label', 'category', 'v1') "
            "and a message column (e.g., 'message', 'text', 'sms', 'v2')."
        )

    if label_col not in df_working.columns or text_col not in df_working.columns:
        raise ValueError(f"Selected columns ('{label_col}', '{text_col}') do not exist in the dataset.")

    original_rows = len(df_working)

    # 1. Normalize labels
    mapped_labels = df_working[label_col].map(normalize_label)
    clean_messages = df_working[text_col].fillna("").astype(str).str.strip()

    raw_labels = df_working[label_col].dropna()
    invalid_labels_count = int(raw_labels.map(normalize_label).isna().sum())
    empty_messages_count = int((clean_messages.str.len() == 0).sum())

    sub_df = pd.DataFrame({
        "label": mapped_labels,
        "message": clean_messages
    })

    # Filter out invalid labels and blank messages
    sub_df = sub_df.dropna(subset=["label"])
    sub_df = sub_df[sub_df["message"].str.len() > 0]

    # 2. Handle identical messages with conflicting labels
    # If the exact same message is labeled as both 'ham' and 'spam', supervision is contradictory.
    label_counts_per_msg = sub_df.groupby("message")["label"].nunique()
    conflicting_messages = label_counts_per_msg[label_counts_per_msg > 1].index.tolist()
    conflicting_count = len(conflicting_messages)

    if conflicting_count > 0:
        sub_df = sub_df[~sub_df["message"].isin(conflicting_messages)]

    # 3. Deduplication of identical (label, message) pairs
    rows_before_dedup = len(sub_df)
    clean_out = sub_df.drop_duplicates(subset=["label", "message"]).reset_index(drop=True)
    duplicates_removed = rows_before_dedup - len(clean_out)
    final_rows = len(clean_out)

    # Validation checks
    if final_rows < 20:
        raise ValueError(
            f"Only {final_rows} valid messages remained after cleaning (minimum required: 20). "
            "Please provide a dataset with at least 20 valid ham/spam messages."
        )

    unique_classes = clean_out["label"].unique().tolist()
    if len(unique_classes) < 2:
        raise ValueError(
            f"The cleaned dataset must contain both 'ham' and 'spam' classes to train models. "
            f"Found only: {unique_classes}."
        )

    ham_count = int((clean_out["label"] == "ham").sum())
    spam_count = int((clean_out["label"] == "spam").sum())
    spam_ratio = spam_count / final_rows if final_rows > 0 else 0.0

    # Diagnostic warnings
    warnings = []
    total_removed = original_rows - final_rows
    if original_rows > 0 and (total_removed / original_rows) > 0.25:
        pct = (total_removed / original_rows) * 100
        warnings.append(
            f"{total_removed:,} rows ({pct:.1f}%) were removed during cleaning due to invalid labels, "
            "empty text, conflicting labels, or duplicates."
        )
    if conflicting_count > 0:
        warnings.append(
            f"{conflicting_count} messages had conflicting labels (marked as both ham and spam) "
            "and were excluded to prevent corrupting model supervision."
        )
    if spam_ratio < 0.05 or spam_ratio > 0.95:
        warnings.append(
            f"Severe class imbalance detected: only {spam_ratio:.1%} of messages are spam. "
            "Models utilize class-weight balancing and stratified splitting to optimize detection."
        )

    stats = {
        "original_rows": original_rows,
        "removed_invalid_labels": invalid_labels_count,
        "removed_empty_messages": empty_messages_count,
        "conflicting_messages_count": conflicting_count,
        "removed_duplicates": duplicates_removed,
        "total_removed": total_removed,
        "final_rows": final_rows,
        "ham_count": ham_count,
        "spam_count": spam_count,
        "spam_ratio": spam_ratio,
        "label_col_used": label_col,
        "text_col_used": text_col,
        "warnings": warnings,
    }

    return clean_out, stats


def load_and_combine_datasets(files):
    """
    Process multiple uploaded dataset files (CSV, TXT, TSV),
    detect columns per file (including Kaggle v1/v2 handling),
    standardize to ['label', 'message'], combine valid files,
    and apply comprehensive cleaning, deduplication, and conflict resolution.

    If one file fails, processing continues with the remaining valid files.
    Returns: (cleaned_df, cleaning_stats, file_reports, raw_combined_df)
    """
    if not files:
        raise ValueError("No files provided for loading.")

    valid_sub_dfs = []
    file_reports = []

    for idx, f in enumerate(files):
        fname = getattr(f, "name", os.path.basename(f) if isinstance(f, str) else f"file_{idx+1}")
        try:
            df_raw = load_uploaded_dataset(f)
            if df_raw is None or len(df_raw) == 0:
                raise ValueError("The file contains no data rows or is empty.")

            # Reconstruct Kaggle spam.csv where commas split text across Unnamed: 2, 3, 4
            df_working = df_raw.copy()
            lower_cols = [str(c).strip().lower() for c in df_working.columns]
            if "v1" in lower_cols and "v2" in lower_cols:
                v1_real = df_working.columns[lower_cols.index("v1")]
                v2_real = df_working.columns[lower_cols.index("v2")]
                unnamed_cols = [c for c in df_working.columns if str(c).strip().lower().startswith("unnamed")]
                if unnamed_cols:
                    df_working[v2_real] = df_working[[v2_real] + unnamed_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()

            det_lbl, det_txt, is_ambig = detect_columns(df_working)
            if det_lbl is None or det_txt is None:
                found_cols = list(df_working.columns)
                raise ValueError(
                    f"Could not identify both label and message columns. Found: {found_cols}. "
                    "Ensure columns like 'label'/'v1' and 'message'/'v2' exist."
                )

            sub_df = pd.DataFrame({
                "label": df_working[det_lbl],
                "message": df_working[det_txt]
            })
            valid_sub_dfs.append(sub_df)
            file_reports.append({
                "filename": fname,
                "status": "Valid",
                "raw_rows": len(df_working),
                "label_col": det_lbl,
                "text_col": det_txt,
                "error": None
            })
        except Exception as exc:
            file_reports.append({
                "filename": fname,
                "status": "Skipped / Invalid",
                "raw_rows": 0,
                "label_col": None,
                "text_col": None,
                "error": str(exc)
            })

    if not valid_sub_dfs:
        failed_msgs = [f"• {r['filename']}: {r['error']}" for r in file_reports if r['error']]
        raise ValueError(
            "None of the uploaded files could be parsed as valid labeled datasets:\n" +
            "\n".join(failed_msgs)
        )

    combined_raw = pd.concat(valid_sub_dfs, ignore_index=True)
    cleaned_df, stats = clean_dataframe(combined_raw, label_col="label", text_col="message")

    # Enrich stats with multi-file metrics
    stats["total_files_uploaded"] = len(files)
    stats["valid_files_count"] = len(valid_sub_dfs)
    stats["failed_files_count"] = len(files) - len(valid_sub_dfs)
    stats["file_reports"] = file_reports

    return cleaned_df, stats, file_reports, combined_raw


# ==============================================================================
# Model Pipelines & Evaluation Logic
# ==============================================================================
def build_models(dataset_size=5000):
    """
    Construct Scikit-learn pipelines with TF-IDF vectorization and four classifiers:
    1. Multinomial Naive Bayes
    2. Logistic Regression
    3. Linear Support Vector Machine
    4. Random Forest
    """
    min_df_val = 1 if dataset_size < 300 else 2
    return {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                      ngram_range=(1, 2), min_df=min_df_val, max_features=50000,
                                      sublinear_tf=True)),
            ("classifier", MultinomialNB(alpha=0.5)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                      ngram_range=(1, 2), min_df=min_df_val, max_features=50000,
                                      sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced",
                                              random_state=42)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                      ngram_range=(1, 2), min_df=min_df_val, max_features=50000,
                                      sublinear_tf=True)),
            ("classifier", LinearSVC(class_weight="balanced", random_state=42, dual="auto")),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                      ngram_range=(1, 2), min_df=min_df_val, max_features=30000,
                                      sublinear_tf=True)),
            ("classifier", RandomForestClassifier(
                n_estimators=180, max_depth=None, min_samples_leaf=1,
                class_weight="balanced_subsample", n_jobs=-1, random_state=42
            )),
        ]),
    }


def get_scores(model, X):
    """Extract continuous prediction scores for spam (probabilities or decision distances)."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, list(model.classes_).index("spam")]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        probs = raw if list(model.classes_).index("spam") == 1 else -raw
    else:
        probs = None
    return probs


def train_and_evaluate_models(cleaned_df, test_size=0.20):
    """
    Execute stratified train/test split, fit TF-IDF only on X_train,
    train all 4 classifiers, and calculate test-set metrics.
    """
    X = cleaned_df["message"].astype(str)
    y = cleaned_df["label"].astype(str)

    # Stratified split when class counts permit
    min_class_samples = y.value_counts().min()
    stratify_target = y if min_class_samples >= 2 else None
    stratify_warning = None
    if min_class_samples < 2:
        stratify_warning = "Minority class has fewer than 2 examples; train/test split conducted without stratification."

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_target
    )

    if y_train.nunique() < 2:
        raise ValueError(
            f"The training split contains only class '{y_train.iloc[0]}'. "
            "Binary classification models require examples of both 'ham' and 'spam' in the training set."
        )

    models = build_models(dataset_size=len(cleaned_df))
    fitted = {}
    rows = []
    predictions = {}
    score_values = {}
    model_statuses = {}

    for name, pipeline in models.items():
        try:
            model = clone(pipeline)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            scores = get_scores(model, X_test)

            metrics = {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision (spam)": precision_score(y_test, pred, pos_label="spam", zero_division=0),
                "Recall (spam)": recall_score(y_test, pred, pos_label="spam", zero_division=0),
                "F1-score (spam)": f1_score(y_test, pred, pos_label="spam", zero_division=0),
                "F1-score (macro)": f1_score(y_test, pred, average="macro", zero_division=0),
            }

            if scores is not None and len(np.unique(y_test)) == 2:
                try:
                    metrics["ROC-AUC"] = roc_auc_score((y_test == "spam").astype(int), scores)
                except Exception:
                    metrics["ROC-AUC"] = np.nan
            else:
                metrics["ROC-AUC"] = np.nan

            rows.append(metrics)
            fitted[name] = model
            predictions[name] = pred
            score_values[name] = scores
            model_statuses[name] = "Success"
        except Exception as exc:
            model_statuses[name] = f"Failed: {exc}"

    if not rows:
        raise RuntimeError("All models failed to train.")

    # Selection rule: 1. Highest Spam F1 -> 2. Highest Spam Recall -> 3. Highest Spam Precision
    results = pd.DataFrame(rows).sort_values(
        ["F1-score (spam)", "Recall (spam)", "Precision (spam)"],
        ascending=False
    ).reset_index(drop=True)

    best_name = results.iloc[0]["Model"]
    return {
        "results": results,
        "models": fitted,
        "predictions": predictions,
        "scores": score_values,
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
        "best_name": best_name,
        "stratify_warning": stratify_warning,
        "model_statuses": model_statuses,
        "test_size": test_size,
    }


def predict_bulk(model_pipeline, df, text_col):
    """
    Run bulk prediction on an unlabeled DataFrame.
    Preserves all original rows, columns, and order.
    """
    out_df = df.copy()
    raw_msgs = out_df[text_col].fillna("").astype(str)

    # Distinguish non-blank messages from blank lines
    valid_mask = raw_msgs.str.strip().str.len() > 0
    valid_msgs = raw_msgs[valid_mask]

    predicted_labels = pd.Series(index=out_df.index, dtype=object)
    predicted_probs = pd.Series(index=out_df.index, dtype=object)

    if len(valid_msgs) > 0:
        preds = model_pipeline.predict(valid_msgs)
        predicted_labels.loc[valid_mask] = preds

        if hasattr(model_pipeline, "predict_proba"):
            spam_idx = list(model_pipeline.classes_).index("spam")
            probs = model_pipeline.predict_proba(valid_msgs)[:, spam_idx]
            predicted_probs.loc[valid_mask] = [f"{p:.1%}" for p in probs]
        elif hasattr(model_pipeline, "decision_function"):
            scores = model_pipeline.decision_function(valid_msgs)
            predicted_probs.loc[valid_mask] = [f"{s:+.3f}" for s in scores]

    # Handle blank/skipped messages
    predicted_labels.loc[~valid_mask] = "SKIPPED (blank message)"
    predicted_probs.loc[~valid_mask] = "N/A"

    out_df["predicted_label"] = predicted_labels
    score_col_name = "spam_probability" if hasattr(model_pipeline, "predict_proba") else "decision_score"
    out_df[score_col_name] = predicted_probs

    total_rows = len(out_df)
    valid_count = int(valid_mask.sum())
    skipped_count = total_rows - valid_count
    spam_count = int((out_df["predicted_label"] == "spam").sum())
    ham_count = int((out_df["predicted_label"] == "ham").sum())

    stats = {
        "total_rows": total_rows,
        "valid_count": valid_count,
        "skipped_count": skipped_count,
        "spam_count": spam_count,
        "ham_count": ham_count,
        "score_col_name": score_col_name,
    }
    return out_df, stats


# ==============================================================================
# Custom CSS
# ==============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Hero Header */
    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #1e3a8a 100%);
        color: #ffffff;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.18);
    }
    .hero-title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 0.35rem;
    }
    .hero h1 {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .hero-tag {
        background: rgba(13, 148, 136, 0.25);
        border: 1px solid #14b8a6;
        color: #5eead4;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .hero p {
        font-size: 0.96rem;
        color: #cbd5e1;
        margin: 0;
        line-height: 1.5;
    }

    /* Theme-Adaptive Status Bar */
    .status-bar {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 10px;
        padding: 0.7rem 1.1rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.6rem;
    }
    .status-bar-left {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .status-bar-right {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.6rem;
    }

    /* Badges & Pills */
    .badge-success {
        background: rgba(34, 197, 94, 0.16);
        color: #16a34a;
        border: 1px solid rgba(34, 197, 94, 0.35);
        font-weight: 600;
        padding: 0.22rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    .badge-warning {
        background: rgba(234, 179, 8, 0.16);
        color: #ca8a04;
        border: 1px solid rgba(234, 179, 8, 0.35);
        font-weight: 600;
        padding: 0.22rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    .badge-top-model {
        background: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.35);
        font-weight: 600;
        padding: 0.22rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.82rem;
    }

    /* Prediction Result Cards (Spam vs Ham) */
    .verdict-box {
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .verdict-box-spam {
        background: rgba(220, 38, 38, 0.08);
        border: 2px solid #dc2626;
    }
    .verdict-box-ham {
        background: rgba(22, 163, 74, 0.08);
        border: 2px solid #16a34a;
    }
    .verdict-badge-spam {
        display: inline-block;
        background: #dc2626;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        letter-spacing: 0.04em;
    }
    .verdict-badge-ham {
        display: inline-block;
        background: #16a34a;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        letter-spacing: 0.04em;
    }
    .verdict-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0.6rem 0 0.3rem 0;
    }
    .verdict-desc {
        font-size: 0.92rem;
        opacity: 0.9;
        margin: 0;
        line-height: 1.45;
    }

    /* Workflow Cards */
    .workflow-card {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        height: 100%;
    }
    .workflow-card h4 {
        margin: 0 0 0.4rem 0;
        font-size: 0.98rem;
        font-weight: 600;
    }
    .workflow-card p {
        font-size: 0.85rem;
        opacity: 0.85;
        margin: 0;
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# Session State Initialization
# ==============================================================================
if "data_source" not in st.session_state:
    st.session_state.data_source = "default"  # 'default' or 'upload'
if "active_dataset_name" not in st.session_state:
    st.session_state.active_dataset_name = "UCI SMS Spam Collection (Default)"
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "cleaning_stats" not in st.session_state:
    st.session_state.cleaning_stats = None
if "file_reports" not in st.session_state:
    st.session_state.file_reports = None
if "dataset_fingerprint" not in st.session_state:
    st.session_state.dataset_fingerprint = None
if "trained_fingerprint" not in st.session_state:
    st.session_state.trained_fingerprint = None
if "model_data" not in st.session_state:
    st.session_state.model_data = None
if "models_stale" not in st.session_state:
    st.session_state.models_stale = True
if "test_size" not in st.session_state:
    st.session_state.test_size = 0.20
if "upload_error" not in st.session_state:
    st.session_state.upload_error = None
if "bulk_results" not in st.session_state:
    st.session_state.bulk_results = None
if "bulk_stats" not in st.session_state:
    st.session_state.bulk_stats = None


# ==============================================================================
# Sidebar: Navigation & Controls
# ==============================================================================
with st.sidebar:
    st.markdown("## 🛡️ MessageGuard")
    st.caption("SMS Spam Detection & Model Comparison")

    page = st.radio(
        "Navigation",
        [
            "1. Overview",
            "2. Dataset & Data Cleaning",
            "3. Model Training & Comparison",
            "4. Performance Analytics",
            "5. SMS Spam Detector",
            "6. Bulk SMS Prediction"
        ],
        label_visibility="visible"
    )

    st.divider()
    st.markdown("### ⚙️ Quick Actions")

    # Global train button in sidebar
    sidebar_train_clicked = st.button("🚀 Train / Evaluate Models", use_container_width=True, type="primary")

    st.divider()
    st.markdown("**Evaluated Algorithms**")
    st.caption("1. Multinomial Naive Bayes\n2. Logistic Regression\n3. Linear Support Vector Machine\n4. Random Forest")
    st.divider()
    st.caption("Machine Learning Mini-Project. Designed for transparency and reproducible model evaluation.")


# ==============================================================================
# Hero Header
# ==============================================================================
st.markdown(
    """
    <div class="hero">
      <div class="hero-title-row">
        <h1>MessageGuard 🛡️</h1>
        <span class="hero-tag">ML Laboratory Edition</span>
      </div>
      <p>Intelligent SMS Spam Detection with transparent ML model comparison, multi-dataset ingestion, in-depth performance analytics, and batch prediction.</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# Auto-Initialize Default Dataset on First Load
# ==============================================================================
if st.session_state.raw_df is None and st.session_state.data_source == "default":
    try:
        raw_default = load_default_dataset()
        cleaned_default, stats_default = clean_dataframe(raw_default)
        st.session_state.raw_df = raw_default
        st.session_state.cleaned_df = cleaned_default
        st.session_state.cleaning_stats = stats_default
        st.session_state.file_reports = None
        st.session_state.active_dataset_name = "UCI SMS Spam Collection (Default)"
        st.session_state.dataset_fingerprint = f"default_{len(cleaned_default)}"
        st.session_state.models_stale = (st.session_state.trained_fingerprint != st.session_state.dataset_fingerprint)
    except Exception as exc:
        st.session_state.upload_error = str(exc)


# ==============================================================================
# Global Model Training Handler (Triggered by Button)
# ==============================================================================
def trigger_model_training():
    if st.session_state.cleaned_df is None:
        st.error("No valid dataset available to train. Please load or upload a dataset first.")
        return

    with st.spinner(f"Training and evaluating 4 models on {len(st.session_state.cleaned_df):,} messages..."):
        try:
            m_data = train_and_evaluate_models(
                st.session_state.cleaned_df,
                test_size=float(st.session_state.test_size)
            )
            st.session_state.model_data = m_data
            st.session_state.trained_fingerprint = st.session_state.dataset_fingerprint
            st.session_state.models_stale = False
            st.success("✅ Model training and evaluation completed successfully!")
        except Exception as exc:
            st.error(f"❌ Training failed: {exc}")


# Check if sidebar button was clicked
if sidebar_train_clicked:
    trigger_model_training()


# ==============================================================================
# Active Status Bar
# ==============================================================================
has_models = st.session_state.model_data is not None and not st.session_state.models_stale
status_badge = (
    '<span class="badge-success">● Models Trained & Ready</span>'
    if has_models else
    '<span class="badge-warning">● Models Require Training</span>'
)
best_model_name = st.session_state.model_data["best_name"] if has_models else "Pending Training"
clean_count_str = f"{len(st.session_state.cleaned_df):,}" if st.session_state.cleaned_df is not None else "0"

st.markdown(
    f"""
    <div class="status-bar">
      <div class="status-bar-left">
        <span style="font-weight: 600;">Active Dataset:</span>
        <span style="font-weight: 700; color: #0d9488;">{st.session_state.active_dataset_name}</span>
        <span style="opacity: 0.75; font-size: 0.88rem;">({clean_count_str} clean messages)</span>
      </div>
      <div class="status-bar-right">
        {status_badge}
        <span class="badge-top-model">Top Model: {best_model_name}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# SECTION 1: OVERVIEW
# ==============================================================================
if page == "1. Overview":
    st.subheader("1. Overview & Machine Learning Architecture")

    st.markdown("""
    **MessageGuard** is an end-to-end machine learning laboratory system designed to protect mobile communications
    by accurately detecting and classifying SMS messages as **Spam** or **Ham (Legitimate)**.
    Built for academic rigor and model transparency, it provides multi-file dataset ingestion, strict deduplication,
    leakage-free TF-IDF feature extraction, side-by-side evaluation of four classification algorithms, and real-time inference.
    """)

    col_def1, col_def2 = st.columns(2)
    with col_def1:
        st.info(
            "🚨 **What is Spam? (Positive Class)**\n\n"
            "Unsolicited, deceptive, or fraudulent mobile text messages. Common variants include phishing links, "
            "fake prize/lottery alerts, urgent banking scams, spoofed delivery notices, and aggressive marketing."
        )
    with col_def2:
        st.success(
            "✅ **What is Ham? (Negative Class)**\n\n"
            "Authentic, legitimate communications sent between individuals or verified transactional services. "
            "Examples include personal chat messages, two-factor authentication (2FA) OTPs, and appointment confirmations."
        )

    st.markdown("#### 📊 System State & Key Indicators")
    ov1, ov2, ov3, ov4 = st.columns(4)

    if st.session_state.cleaned_df is not None and st.session_state.cleaning_stats is not None:
        stats = st.session_state.cleaning_stats
        total_clean = stats["final_rows"]
        spam_pct = (stats["spam_count"] / total_clean * 100) if total_clean > 0 else 0
        ov1.metric("Clean Messages", f"{total_clean:,}")
        ov2.metric("Spam Prevalence", f"{stats['spam_count']:,} ({spam_pct:.1f}%)", f"Ham: {stats['ham_count']:,}")
    else:
        ov1.metric("Clean Messages", "0")
        ov2.metric("Spam Prevalence", "0 (0%)", "No data loaded")

    num_trained = len(st.session_state.model_data["models"]) if has_models else 0
    ov3.metric("Trained Models", f"{num_trained} of 4", "Ready" if has_models else "Untrained / Stale")

    if has_models:
        top_f1 = st.session_state.model_data["results"].iloc[0]["F1-score (spam)"]
        ov4.metric("Top Algorithm", f"{best_model_name}", f"Spam F1: {top_f1:.4f}")
    else:
        ov4.metric("Top Algorithm", "Pending Training", "Needs execution")

    # Dynamic Training Callout & Benchmark Snapshot
    if has_models:
        st.success(
            f"🏆 **Benchmark Available:** All 4 machine learning pipelines have been trained and evaluated on the active dataset. "
            f"**{best_model_name}** achieved the top score with a Spam F1 of **{top_f1:.4f}**."
        )
        st.markdown("##### 📈 Evaluated Model Comparison Snapshot")
        snapshot_df = st.session_state.model_data["results"].copy()
        for col in snapshot_df.columns:
            if col != "Model":
                snapshot_df[col] = snapshot_df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        st.dataframe(snapshot_df, use_container_width=True, hide_index=True)

        col_retrain, _ = st.columns([1.5, 3])
        with col_retrain:
            if st.button("🔄 Retrain / Re-Evaluate Models", key="overview_retrain_btn", use_container_width=True):
                trigger_model_training()
                st.rerun()
    else:
        st.warning(
            "⚠️ **Models are not yet trained on the active dataset configuration.** "
            "Trigger model training now to benchmark all 4 algorithms on identical held-out test data."
        )
        col_ov_train, _ = st.columns([1.5, 3])
        with col_ov_train:
            if st.button("🚀 Train & Compare Models Now", type="primary", use_container_width=True, key="overview_train_action"):
                trigger_model_training()
                st.rerun()

    st.divider()
    st.markdown("#### ⚙️ End-to-End Pipeline Architecture")

    wf1, wf2, wf3, wf4, wf5 = st.columns(5)
    with wf1:
        st.markdown("""
        <div class="workflow-card">
          <h4>1. Ingestion</h4>
          <p>Multi-file CSV/TSV/TXT loading, Kaggle v1/v2 schema handling, automatic column detection.</p>
        </div>
        """, unsafe_allow_html=True)
    with wf2:
        st.markdown("""
        <div class="workflow-card">
          <h4>2. Preprocessing</h4>
          <p>Label standardization (ham/spam), whitespace cleanup, deduplication, conflict resolution.</p>
        </div>
        """, unsafe_allow_html=True)
    with wf3:
        st.markdown("""
        <div class="workflow-card">
          <h4>3. Feature Eng.</h4>
          <p>TF-IDF unigram & bigram vectorization with sublinear scaling, fitted on training fold only.</p>
        </div>
        """, unsafe_allow_html=True)
    with wf4:
        st.markdown("""
        <div class="workflow-card">
          <h4>4. ML Benchmark</h4>
          <p>4 distinct algorithms: Naive Bayes, Logistic Reg., Linear SVM, and Random Forest.</p>
        </div>
        """, unsafe_allow_html=True)
    with wf5:
        st.markdown("""
        <div class="workflow-card">
          <h4>5. Inference</h4>
          <p>Single-message spam analyzer with probability bar and bulk batch CSV prediction & export.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    with st.expander("ℹ️ Academic Evaluation Methodology & Metric Selection Policy"):
        st.markdown("""
        - **Why Spam F1-Score over Overall Accuracy?** SMS datasets exhibit severe class imbalance (~87% Ham vs ~13% Spam).
          A trivial baseline predicting always *Ham* achieves 87% accuracy while completely failing to stop spam.
          The primary metric used is **Spam F1-Score**, the harmonic mean of Spam Precision and Spam Recall.
        - **Model Selection Rule:**
          1. **Highest Spam F1-Score** (Primary rank)
          2. **Highest Spam Recall** (Tie-breaker 1 — prioritize catching unsolicited fraud)
          3. **Highest Spam Precision** (Tie-breaker 2 — minimize false alarms on authentic chats)
        - **Data Leakage Guarantee:** The TF-IDF vectorizer vocabulary, IDF weights, and classifiers are fit **exclusively** on the training fold during `train_test_split`.
        """)


# ==============================================================================
# SECTION 2: DATASET & DATA CLEANING
# ==============================================================================
elif page == "2. Dataset & Data Cleaning":
    st.subheader("2. Dataset Loading & Data Cleaning Pipeline")
    st.write("Acquire, inspect, validate, and preprocess labeled SMS datasets with multi-file upload support.")

    col_src, col_opts = st.columns([1.5, 1])
    with col_src:
        selected_source = st.radio(
            "Select Training Data Source",
            ["UCI SMS Spam Collection (Default)", "Upload Labeled Dataset(s)"],
            index=0 if st.session_state.data_source == "default" else 1,
            key="source_radio_selector"
        )

    # 1. Handle Option A: Default UCI Dataset
    if selected_source == "UCI SMS Spam Collection (Default)":
        if st.session_state.data_source != "default" or st.session_state.raw_df is None:
            try:
                with st.spinner("Loading UCI SMS Spam Collection..."):
                    raw_uci = load_default_dataset()
                    cleaned_uci, stats_uci = clean_dataframe(raw_uci)
                    st.session_state.data_source = "default"
                    st.session_state.active_dataset_name = "UCI SMS Spam Collection (Default)"
                    st.session_state.raw_df = raw_uci
                    st.session_state.cleaned_df = cleaned_uci
                    st.session_state.cleaning_stats = stats_uci
                    st.session_state.file_reports = None
                    new_fp = f"default_{len(cleaned_uci)}"
                    if st.session_state.trained_fingerprint != new_fp:
                        st.session_state.models_stale = True
                    st.session_state.dataset_fingerprint = new_fp
                    st.session_state.upload_error = None
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading default dataset: {e}")

    # 2. Handle Option B: Upload Labeled Datasets (Multiple files supported)
    else:
        st.session_state.data_source = "upload"
        uploaded_files = st.file_uploader(
            "Upload Labeled SMS Datasets (CSV, TXT, TSV)",
            type=["csv", "txt", "tsv"],
            accept_multiple_files=True,
            key="labeled_datasets_uploader",
            help="Upload one or multiple files simultaneously. MessageGuard standardizes columns, combines valid files, and applies strict deduplication."
        )

        if uploaded_files:
            # Generate composite fingerprint based on filenames, sizes, and content hashes
            fp_parts = []
            for f in sorted(uploaded_files, key=lambda x: x.name):
                f_bytes = f.getvalue()
                f_hash = hashlib.md5(f_bytes).hexdigest()[:8]
                fp_parts.append(f"{f.name}_{len(f_bytes)}_{f_hash}")
            upload_fp = "multi_" + hashlib.md5("_".join(fp_parts).encode()).hexdigest()[:12]

            if st.session_state.dataset_fingerprint != upload_fp:
                try:
                    with st.spinner(f"Processing and combining {len(uploaded_files)} dataset file(s)..."):
                        cleaned_up, stats_up, file_reps, raw_combined = load_and_combine_datasets(uploaded_files)

                    st.session_state.raw_df = raw_combined
                    num_valid = stats_up["valid_files_count"]
                    total_f = len(uploaded_files)
                    if total_f == 1 and num_valid == 1:
                        st.session_state.active_dataset_name = f"Uploaded File: '{uploaded_files[0].name}'"
                    else:
                        st.session_state.active_dataset_name = f"Combined Dataset ({num_valid} valid of {total_f} files)"

                    st.session_state.dataset_fingerprint = upload_fp
                    st.session_state.cleaned_df = cleaned_up
                    st.session_state.cleaning_stats = stats_up
                    st.session_state.file_reports = file_reps
                    st.session_state.models_stale = True
                    st.session_state.upload_error = None
                    st.rerun()
                except Exception as exc:
                    st.session_state.upload_error = str(exc)
                    st.session_state.cleaned_df = None
                    st.session_state.cleaning_stats = None
                    st.session_state.file_reports = None
                    st.session_state.models_stale = True
        elif not uploaded_files and st.session_state.cleaned_df is None:
            st.info("👆 Please select one or more labeled dataset files (.csv, .tsv, .txt) above to begin.")

        if st.session_state.upload_error:
            st.error(f"❌ Dataset Processing Error:\n{st.session_state.upload_error}")
            st.info(
                "💡 **Expected schema:** Each dataset file should include a label column (e.g. `label`, `category`, `class`, `v1`) "
                "with `ham`/`spam` (or `0`/`1`) and an SMS message text column (e.g. `message`, `text`, `sms`, `v2`)."
            )

    # 3. Uploaded Files Manifest Table (Display when files were processed)
    if st.session_state.data_source == "upload" and st.session_state.file_reports:
        st.markdown("#### 📁 Uploaded Files Processing Manifest")
        manifest_records = []
        for rep in st.session_state.file_reports:
            is_valid = (rep["status"] == "Valid")
            manifest_records.append({
                "File Name": rep["filename"],
                "Status": "✅ Valid" if is_valid else "⚠️ Skipped / Error",
                "Rows Parsed": f"{rep['raw_rows']:,}" if is_valid else "0",
                "Label Column": rep["label_col"] or "Not found",
                "Message Column": rep["text_col"] or "Not found",
                "Diagnostic Details": "Parsed and merged successfully" if is_valid else f"Error: {rep['error']}"
            })
        manifest_df = pd.DataFrame(manifest_records)
        st.dataframe(manifest_df, use_container_width=True, hide_index=True)

        failed_files = [r for r in st.session_state.file_reports if r["status"] != "Valid"]
        valid_files = [r for r in st.session_state.file_reports if r["status"] == "Valid"]
        if failed_files and valid_files:
            st.warning(
                f"⚠️ **Partial Ingestion:** {len(failed_files)} file(s) failed validation and were skipped. "
                f"The remaining {len(valid_files)} valid file(s) were successfully combined into the active dataset."
            )
        elif valid_files and not failed_files and len(valid_files) > 1:
            st.success(f"✅ Successfully combined all {len(valid_files)} files into one unified dataset.")

    # 4. Column Selection Override (for single custom datasets or manual tuning)
    if st.session_state.raw_df is not None:
        raw_cols = list(st.session_state.raw_df.columns)
        if len(raw_cols) > 2 or ("label" not in raw_cols or "message" not in raw_cols):
            det_lbl, det_txt, is_ambig = detect_columns(st.session_state.raw_df)
            with st.expander("🛠️ Column Mapping Settings (Override Auto-Detection)", expanded=is_ambig):
                st.caption("Change column mappings if auto-detection picked unexpected columns:")
                c_sel1, c_sel2 = st.columns(2)
                default_lbl_idx = raw_cols.index(det_lbl) if det_lbl in raw_cols else 0
                default_txt_idx = raw_cols.index(det_txt) if det_txt in raw_cols else (1 if len(raw_cols) > 1 else 0)

                chosen_lbl = c_sel1.selectbox("Label Column", raw_cols, index=default_lbl_idx)
                chosen_txt = c_sel2.selectbox("Message Text Column", raw_cols, index=default_txt_idx)

                if st.button("Apply Column Mapping"):
                    try:
                        cleaned_override, stats_override = clean_dataframe(
                            st.session_state.raw_df, label_col=chosen_lbl, text_col=chosen_txt
                        )
                        st.session_state.cleaned_df = cleaned_override
                        st.session_state.cleaning_stats = stats_override
                        st.session_state.models_stale = True
                        st.success("Column mappings applied successfully!")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed with selected columns: {exc}")

    # 5. Display Cleaning Results & Visualizations
    if st.session_state.cleaned_df is not None and st.session_state.cleaning_stats is not None:
        stats = st.session_state.cleaning_stats
        st.divider()
        st.markdown(f"#### 🧹 Preprocessing Summary — {st.session_state.active_dataset_name}")

        cl1, cl2, cl3, cl4 = st.columns(4)
        cl1.metric("Original Rows Loaded", f"{stats['original_rows']:,}")
        cl2.metric("Invalid / Blank Filtered", f"{stats['removed_invalid_labels'] + stats['removed_empty_messages']:,}")
        cl3.metric("Duplicates & Conflicts Dropped", f"{stats['conflicting_messages_count'] + stats['removed_duplicates']:,}")
        cl4.metric("Clean Usable Messages", f"{stats['final_rows']:,}")

        if stats.get("warnings"):
            for w in stats["warnings"]:
                st.warning(f"⚠️ {w}")

        col_chart, col_preview = st.columns([1, 1.4])
        with col_chart:
            st.markdown("#### Class Balance")
            counts_df = pd.DataFrame({
                "Class": ["ham", "spam"],
                "Count": [stats["ham_count"], stats["spam_count"]],
                "Percentage": [
                    f"{stats['ham_count'] / stats['final_rows'] * 100:.1f}%",
                    f"{stats['spam_count'] / stats['final_rows'] * 100:.1f}%"
                ]
            })
            fig_bar = px.bar(
                counts_df, x="Class", y="Count", color="Class",
                color_discrete_map={"ham": "#2563eb", "spam": "#dc2626"},
                text="Percentage"
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_preview:
            st.markdown("#### Cleaned Dataset Preview")
            sample_n = min(15, len(st.session_state.cleaned_df))
            st.dataframe(st.session_state.cleaned_df.head(sample_n), use_container_width=True, hide_index=True)

            cleaned_csv_bytes = st.session_state.cleaned_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Cleaned Dataset (CSV)",
                cleaned_csv_bytes,
                "cleaned_sms_dataset.csv",
                "text/csv",
                type="secondary"
            )


# ==============================================================================
# SECTION 3: MODEL TRAINING & COMPARISON
# ==============================================================================
elif page == "3. Model Training & Comparison":
    st.subheader("3. Model Training & Algorithm Comparison")
    st.write(
        "Train and evaluate four Scikit-learn pipelines with TF-IDF text features on identical held-out test data."
    )

    if st.session_state.cleaned_df is None:
        st.warning("⚠️ No clean dataset is available. Please navigate to **2. Dataset & Data Cleaning** first.")
        st.stop()

    # Split configuration
    col_cfg1, col_cfg2 = st.columns([1.5, 1])
    with col_cfg1:
        new_test_size = st.select_slider(
            "Test Set Fraction",
            options=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
            value=st.session_state.test_size,
            format_func=lambda x: f"{int(x*100)}% Test / {int((1-x)*100)}% Train"
        )
        if new_test_size != st.session_state.test_size:
            st.session_state.test_size = new_test_size
            st.session_state.models_stale = True

    with col_cfg2:
        st.info(
            "🔒 **Data Leakage Protection:**\n\n"
            "TF-IDF vectorizer vocabularies and IDF weights are learned exclusively from the training set inside each pipeline."
        )

    # Explicit Train Button
    st.write("")
    btn_col1, btn_col2 = st.columns([1.5, 3])
    with btn_col1:
        train_now = st.button("🚀 Train / Evaluate Models Now", type="primary", use_container_width=True)

    if train_now:
        trigger_model_training()

    if st.session_state.models_stale or st.session_state.model_data is None:
        st.warning(
            "⚠️ Current models are not trained on the active dataset configuration. "
            "Click **🚀 Train / Evaluate Models Now** above to train the models."
        )
        st.stop()

    # Results Table & Details
    m_data = st.session_state.model_data
    results_df = m_data["results"]
    best_model_str = m_data["best_name"]
    top_metric_row = results_df.iloc[0]

    st.divider()
    st.success(
        f"🏆 **Top Performer by Selection Rule:** **{best_model_str}** "
        f"(Spam F1: **{top_metric_row['F1-score (spam)']:.4f}**, "
        f"Recall: **{top_metric_row['Recall (spam)']:.4f}**, "
        f"Precision: **{top_metric_row['Precision (spam)']:.4f}**, "
        f"Accuracy: **{top_metric_row['Accuracy']:.4f}**)\n\n"
        "*Selection rule: 1. Highest Spam F1-Score -> 2. Highest Spam Recall (tie-breaker) -> 3. Highest Spam Precision.*"
    )

    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("Top Model Spam F1", f"{top_metric_row['F1-score (spam)']:.4f}", f"Rank 1 ({best_model_str})")
    tc2.metric("Spam Recall", f"{top_metric_row['Recall (spam)']:.4f}", "Fraud capture rate")
    tc3.metric("Spam Precision", f"{top_metric_row['Precision (spam)']:.4f}", "Legit msg safety")
    tc4.metric("Overall Accuracy", f"{top_metric_row['Accuracy']:.4f}", "Test split")

    with st.expander("📚 Metric Definitions & Evaluation Criteria Glossary"):
        st.markdown("""
        | Metric | Definition | Critical Significance in SMS Spam Detection |
        | :--- | :--- | :--- |
        | **Spam F1-Score** | $2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$ | **Primary evaluation benchmark**. Balances false positives and false negatives under severe class imbalance (~87% Ham / 13% Spam). |
        | **Spam Recall** | $\\frac{\\text{True Spam Caught}}{\\text{All Actual Spam}}$ | Fraction of unsolicited fraudulent messages intercepted. High recall ensures malicious phishing links do not reach users. |
        | **Spam Precision** | $\\frac{\\text{True Spam Caught}}{\\text{All Predicted Spam}}$ | Fraction of flagged messages that were truly spam. High precision ensures critical personal messages and 2FA OTPs are never blocked. |
        | **Overall Accuracy** | $\\frac{\\text{Correct Predictions}}{\\text{Total Predictions}}$ | Overall accuracy across both classes. Less informative than F1 due to majority class dominance, but provided for baseline reference. |
        | **ROC-AUC** | Area under ROC Curve | Evaluates class discriminability across all continuous decision thresholds, independent of default classification cutoffs. |
        """)

    st.markdown("#### Measured Performance Metrics (Held-Out Test Set)")
    display_results = results_df.copy()
    for col in display_results.columns:
        if col != "Model":
            display_results[col] = display_results[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    st.dataframe(display_results, use_container_width=True, hide_index=True)

    # Comparative Bar Chart
    melted_df = results_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision (spam)", "Recall (spam)", "F1-score (spam)", "F1-score (macro)"],
        var_name="Metric", value_name="Score"
    )
    fig_comp = px.bar(
        melted_df, x="Model", y="Score", color="Metric", barmode="group",
        range_y=[0, 1.05], title="Performance Metrics Across Models"
    )
    fig_comp.update_layout(margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig_comp, use_container_width=True)

    csv_metrics = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Model Metrics (CSV)",
        csv_metrics,
        "model_comparison_metrics.csv",
        "text/csv"
    )


# ==============================================================================
# SECTION 4: PERFORMANCE ANALYTICS
# ==============================================================================
elif page == "4. Performance Analytics":
    st.subheader("4. In-Depth Performance Analytics")

    if st.session_state.model_data is None or st.session_state.models_stale:
        st.warning("⚠️ Models have not been trained on the active dataset. Please navigate to **3. Model Training & Comparison** and train models.")
        st.stop()

    m_data = st.session_state.model_data
    results_df = m_data["results"]
    X_test = m_data["X_test"]
    y_test = m_data["y_test"]
    best_m = m_data["best_name"]

    col_diag1, col_diag2 = st.columns(2)

    # Confusion Matrix
    with col_diag1:
        st.markdown("#### Confusion Matrix")
        chosen_cm_m = st.selectbox(
            "Inspect Confusion Matrix for Model:",
            list(m_data["models"].keys()),
            index=list(m_data["models"].keys()).index(best_m),
            key="analytics_cm_select"
        )
        cm = confusion_matrix(y_test, m_data["predictions"][chosen_cm_m], labels=["ham", "spam"])
        cm_df = pd.DataFrame(cm, index=["Actual Ham", "Actual Spam"], columns=["Predicted Ham", "Predicted Spam"])
        fig_cm = px.imshow(
            cm_df, text_auto=True, color_continuous_scale="Blues", aspect="auto",
            title=f"Confusion Matrix — {chosen_cm_m}"
        )
        fig_cm.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    # ROC Curves
    with col_diag2:
        st.markdown("#### ROC Curves")
        if len(np.unique(y_test)) < 2:
            st.info("ℹ️ The test split contains only a single class. ROC curves and ROC-AUC require both ham and spam examples in the test split.")
        else:
            roc_fig = go.Figure()
            line_styles = {
                "Multinomial Naive Bayes": dict(dash="solid", width=2.5),
                "Logistic Regression": dict(dash="dash", width=2.5),
                "Linear SVM": dict(dash="dot", width=2.5),
                "Random Forest": dict(dash="dashdot", width=2.5),
            }
            for name, scores in m_data["scores"].items():
                if scores is not None:
                    try:
                        fpr, tpr, _ = roc_curve((y_test == "spam").astype(int), scores)
                        auc_series = results_df.loc[results_df["Model"] == name, "ROC-AUC"]
                        auc_score = auc_series.iloc[0] if not auc_series.empty else np.nan
                        auc_str = f"{auc_score:.4f}" if pd.notna(auc_score) else "N/A"
                        style = line_styles.get(name, dict(dash="solid", width=2))
                        roc_fig.add_trace(go.Scatter(
                            x=fpr, y=tpr, mode="lines",
                            name=f"{name} (AUC={auc_str})",
                            line=style
                        ))
                    except Exception:
                        continue
            roc_fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Random Chance", line=dict(dash="dash", color="gray")
            ))
            roc_fig.update_layout(
                title="ROC Curve Comparison",
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(roc_fig, use_container_width=True)

    # Classification Report
    st.markdown(f"#### Classification Report — {chosen_cm_m}")
    report_dict = classification_report(
        y_test, m_data["predictions"][chosen_cm_m],
        labels=["ham", "spam"], output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report_dict).transpose().round(4), use_container_width=True)


# ==============================================================================
# SECTION 5: SMS SPAM DETECTOR (INDIVIDUAL)
# ==============================================================================
elif page == "5. SMS Spam Detector":
    st.subheader("5. Individual SMS Spam Detector")
    st.write("Test individual text messages with the trained machine learning pipelines.")

    if st.session_state.model_data is None or st.session_state.models_stale:
        st.warning("⚠️ No trained models available. Please train models in **3. Model Training & Comparison** first.")
        st.stop()

    m_data = st.session_state.model_data
    best_m = m_data["best_name"]

    col_sm1, col_sm2 = st.columns([2, 1])
    with col_sm1:
        selected_model_name = st.selectbox(
            "Select Prediction Model",
            list(m_data["models"].keys()),
            index=list(m_data["models"].keys()).index(best_m),
            key="individual_model_selector"
        )
    with col_sm2:
        model_f1 = m_data["results"].loc[m_data["results"]["Model"] == selected_model_name, "F1-score (spam)"].iloc[0]
        st.metric(f"{selected_model_name} Test F1", f"{model_f1:.4f}")

    message_text = st.text_area(
        "Enter SMS Message Text:",
        placeholder="Type or paste an SMS message here to classify...",
        height=130,
        key="individual_text_area"
    )

    b1, b2, b3, b4 = st.columns([1.3, 1, 1, 1])
    with b1:
        predict_single = st.button("🔍 Classify Message", type="primary", use_container_width=True)
    with b2:
        if st.button("Sample Lottery Spam", use_container_width=True):
            st.session_state["inserted_sample"] = "URGENT! You have won a £2,000 cash prize or a holiday voucher. Claim now: call 09061701461 to claim your reward code."
            st.rerun()
    with b3:
        if st.button("Sample Phishing URL", use_container_width=True):
            st.session_state["inserted_sample"] = "Your bank account has been temporarily locked due to unusual logins. Verify your identity immediately at http://secure-verify-auth.com"
            st.rerun()
    with b4:
        if st.button("Sample Normal Ham", use_container_width=True):
            st.session_state["inserted_sample"] = "Hey Alex, are we still meeting in the computer lab after class at 3:30 PM?"
            st.rerun()

    if "inserted_sample" in st.session_state and not message_text:
        message_text = st.session_state.pop("inserted_sample")
        predict_single = True

    if predict_single:
        if not message_text.strip():
            st.warning("Please enter an SMS message before predicting.")
        else:
            pipeline = m_data["models"][selected_model_name]
            clean_msg = message_text.strip()
            prediction = pipeline.predict([clean_msg])[0]

            if prediction == "spam":
                st.markdown(
                    """
                    <div class="verdict-box verdict-box-spam">
                      <span class="verdict-badge-spam">🚨 VERDICT: SPAM DETECTED</span>
                      <div class="verdict-title" style="color: #dc2626;">Fraudulent, Unsolicited, or Phishing SMS</div>
                      <p class="verdict-desc">This message exhibits high-risk spam indicators such as urgency triggers, promotional prize phrasing, or unverified external links.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="verdict-box verdict-box-ham">
                      <span class="verdict-badge-ham">✅ VERDICT: HAM (LEGITIMATE)</span>
                      <div class="verdict-title" style="color: #16a34a;">Authentic Personal or Transactional Communication</div>
                      <p class="verdict-desc">This message conforms to benign conversation or transactional communications (personal chats, verified 2FA OTPs, service updates) with no spam indicators detected.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Meaningful probability vs decision score
            if hasattr(pipeline, "predict_proba"):
                spam_idx = list(pipeline.classes_).index("spam")
                probs = pipeline.predict_proba([clean_msg])[0]
                spam_prob = float(probs[spam_idx])
                ham_prob = 1.0 - spam_prob
                col_pb1, col_pb2 = st.columns(2)
                col_pb1.metric("Spam Confidence", f"{spam_prob:.1%}")
                col_pb2.metric("Ham Confidence", f"{ham_prob:.1%}")
                st.progress(spam_prob, text=f"Calibrated Spam Probability: {spam_prob:.1%}")
            elif hasattr(pipeline, "decision_function"):
                dec_score = float(pipeline.decision_function([clean_msg])[0])
                st.info(f"📊 **Linear SVM Decision Score:** `{dec_score:+.3f}` (Positive values denote Spam class; negative values denote Ham class).")


# ==============================================================================
# SECTION 6: BULK SMS PREDICTION
# ==============================================================================
elif page == "6. Bulk SMS Prediction":
    st.subheader("6. Bulk SMS Prediction (Unlabeled CSV)")
    st.write(
        "Upload an unlabeled CSV, TSV, or TXT file containing new SMS messages without ground-truth labels. "
        "The model will classify all messages in batch, preserve original rows, and export predictions."
    )

    if st.session_state.model_data is None or st.session_state.models_stale:
        st.warning("⚠️ No trained models available. Please train models in **3. Model Training & Comparison** first.")
        st.stop()

    m_data = st.session_state.model_data
    best_m = m_data["best_name"]

    col_bm1, col_bm2 = st.columns([2, 1])
    with col_bm1:
        bulk_chosen_model = st.selectbox(
            "Select Model for Bulk Prediction",
            list(m_data["models"].keys()),
            index=list(m_data["models"].keys()).index(best_m),
            key="bulk_model_select_box"
        )
    with col_bm2:
        st.caption(f"Using **{bulk_chosen_model}** trained on {len(st.session_state.cleaned_df):,} messages.")

    bulk_file = st.file_uploader(
        "Upload Unlabeled SMS Dataset (CSV, TXT, TSV)",
        type=["csv", "txt", "tsv"],
        key="bulk_file_uploader_input"
    )

    if bulk_file is not None:
        try:
            with st.spinner("Loading bulk file..."):
                raw_bulk_df = load_uploaded_dataset(bulk_file)

            if len(raw_bulk_df) == 0:
                st.warning("Uploaded file contains no rows.")
            else:
                bulk_cols = list(raw_bulk_df.columns)
                # Detect message column
                _, auto_text_col, _ = detect_columns(raw_bulk_df)
                detected_text = auto_text_col if auto_text_col in bulk_cols else bulk_cols[0]

                st.write("")
                col_mc1, col_mc2 = st.columns([1.5, 2])
                with col_mc1:
                    chosen_text_col = st.selectbox(
                        "Confirm SMS Message Column:",
                        bulk_cols,
                        index=bulk_cols.index(detected_text)
                    )

                with st.spinner(f"Predicting {len(raw_bulk_df):,} messages using {bulk_chosen_model}..."):
                    model_pipe = m_data["models"][bulk_chosen_model]
                    pred_df, b_stats = predict_bulk(model_pipe, raw_bulk_df, chosen_text_col)
                    st.session_state.bulk_results = pred_df
                    st.session_state.bulk_stats = b_stats

                st.success(f"✅ Predicted {b_stats['valid_count']:,} messages successfully!")

                # Summary Metrics
                bk1, bk2, bk3, bk4 = st.columns(4)
                bk1.metric("Total Input Rows", f"{b_stats['total_rows']:,}")
                spam_pct = (b_stats['spam_count'] / b_stats['valid_count'] * 100) if b_stats['valid_count'] > 0 else 0
                bk2.metric("Predicted Spam", f"{b_stats['spam_count']:,}", f"{spam_pct:.1f}%")
                bk3.metric("Predicted Ham", f"{b_stats['ham_count']:,}", f"{100 - spam_pct:.1f}%")
                bk4.metric("Blank / Skipped", f"{b_stats['skipped_count']:,}")

                col_bchart, col_bprev = st.columns([1, 1.4])
                with col_bchart:
                    counts_p = pd.DataFrame({
                        "Label": ["ham", "spam"],
                        "Count": [b_stats["ham_count"], b_stats["spam_count"]]
                    })
                    fig_p = px.pie(
                        counts_p, names="Label", values="Count", hole=0.5,
                        color="Label", color_discrete_map={"ham": "#2563eb", "spam": "#dc2626"},
                        title="Prediction Class Distribution"
                    )
                    fig_p.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_p, use_container_width=True)

                with col_bprev:
                    st.markdown("#### Prediction Output Preview")
                    st.dataframe(pred_df.head(20), use_container_width=True, hide_index=True)

                # Working Download Button
                csv_pred_bytes = pred_df.to_csv(index=False).encode("utf-8")
                safe_model_str = bulk_chosen_model.lower().replace(" ", "_")
                st.download_button(
                    "📥 Download Complete Predictions (CSV)",
                    csv_pred_bytes,
                    f"bulk_predictions_{safe_model_str}.csv",
                    "text/csv",
                    type="primary"
                )

        except Exception as exc:
            st.error(f"❌ Error during bulk prediction: {exc}")
    else:
        st.info("Upload an unlabeled CSV file above to run bulk SMS spam prediction.")
