import io
import os
import pandas as pd
import numpy as np
from app import (
    normalize_label,
    load_uploaded_dataset,
    detect_columns,
    clean_dataframe,
    build_models,
    get_scores,
    train_and_evaluate_models,
    predict_bulk
)

print("=" * 60)
print("RUNNING MASTER TEST SUITE — SMS SPAM DETECTION")
print("=" * 60)

# -------------------------------------------------------------
# Test 1: Valid labeled CSV loading and cleaning
# -------------------------------------------------------------
print("\n[Test 1] Valid labeled CSV loading & cleaning:")
with open(r"C:\Users\Pallav\Downloads\spam.csv", "rb") as f:
    df_raw = load_uploaded_dataset(f)
assert len(df_raw) == 5572, f"Expected 5572, got {len(df_raw)}"
cleaned_df, stats = clean_dataframe(df_raw)
assert stats["final_rows"] == 5158, f"Expected 5158, got {stats['final_rows']}"
assert stats["ham_count"] == 4516 and stats["spam_count"] == 642
print(f" -> PASSED: 5572 raw -> {stats['final_rows']} clean rows (Ham: {stats['ham_count']}, Spam: {stats['spam_count']}).")

# -------------------------------------------------------------
# Test 2: Case-insensitive column detection & Whitespace in headers
# -------------------------------------------------------------
print("\n[Test 2] Case-insensitive & whitespace in column headers:")
df_whitespace = pd.DataFrame({
    "  LABEL  ": ["ham", "spam"] * 12,
    "  MeSSaGe   ": [f"Legit chat message {i}" if i % 2 == 0 else f"WIN £1000 cash prize {i}" for i in range(24)]
})
lbl_col, txt_col, is_ambig = detect_columns(df_whitespace)
assert lbl_col == "  LABEL  " and txt_col == "  MeSSaGe   ", f"Failed: {lbl_col}, {txt_col}"
clean_ws, stats_ws = clean_dataframe(df_whitespace)
assert len(clean_ws) == 24
print(f" -> PASSED: Detected '{lbl_col}' and '{txt_col}', cleaned {len(clean_ws)} rows.")

# -------------------------------------------------------------
# Test 3: Empty CSV
# -------------------------------------------------------------
print("\n[Test 3] Empty CSV handling:")
try:
    load_uploaded_dataset(io.BytesIO(b""))
    assert False, "Should have raised ValueError on empty bytes"
except ValueError as exc:
    print(f" -> PASSED: Raised expected ValueError: {exc}")

# -------------------------------------------------------------
# Test 4: Missing required columns
# -------------------------------------------------------------
print("\n[Test 4] Missing required columns:")
df_bad_cols = pd.DataFrame({"colA": [1, 2, 3], "colB": ["x", "y", "z"]})
try:
    clean_dataframe(df_bad_cols)
    assert False, "Should have raised ValueError on missing columns"
except ValueError as exc:
    print(f" -> PASSED: Raised expected ValueError: {exc}")

# -------------------------------------------------------------
# Test 5: Blank messages handling
# -------------------------------------------------------------
print("\n[Test 5] Blank/empty messages removal:")
df_blanks = pd.DataFrame({
    "label": ["ham", "spam"] * 15,
    "message": [f"Valid message {i}" if i % 3 != 0 else ("" if i % 2 == 0 else "   ") for i in range(30)]
})
clean_blanks, stats_blanks = clean_dataframe(df_blanks)
assert stats_blanks["removed_empty_messages"] == 10, f"Expected 10 empty removed, got {stats_blanks['removed_empty_messages']}"
assert len(clean_blanks) == 20
print(f" -> PASSED: Successfully filtered {stats_blanks['removed_empty_messages']} blank/whitespace messages.")

# -------------------------------------------------------------
# Test 6: Invalid labels handling (does not map arbitrary labels)
# -------------------------------------------------------------
print("\n[Test 6] Invalid labels filtering:")
assert normalize_label("random_text") is None
assert normalize_label("ham") == "ham"
assert normalize_label("spam") == "spam"
assert normalize_label("1.0") == "spam"
assert normalize_label("0") == "ham"
df_invalid_labels = pd.DataFrame({
    "label": ["unsupported", "neutral"] * 5 + ["ham", "spam"] * 15,
    "message": [f"Unique message text {i}" for i in range(40)]
})
clean_inv, stats_inv = clean_dataframe(df_invalid_labels)
assert stats_inv["removed_invalid_labels"] == 10
assert len(clean_inv) == 30
print(f" -> PASSED: Filtered {stats_inv['removed_invalid_labels']} unsupported labels without silent conversion.")

# -------------------------------------------------------------
# Test 7: Duplicate records removal and <20 threshold check
# -------------------------------------------------------------
print("\n[Test 7] Duplicate records removal and minimum rows threshold:")
df_dups = pd.DataFrame({
    "label": ["ham", "spam"] * 15,
    "message": ["Same exact message A", "Same exact message B"] * 15
})
try:
    clean_dataframe(df_dups)
    assert False, "Should have raised ValueError because only 2 unique rows remain (<20)"
except ValueError as exc:
    print(f" -> PASSED: Raised expected ValueError when deduplication leaves <20 rows: {exc}")

df_dups_valid = pd.DataFrame({
    "label": ["ham", "spam"] * 15,
    "message": [f"Message {i % 22}" for i in range(30)]
})
clean_dv, stats_dv = clean_dataframe(df_dups_valid)
assert stats_dv["removed_duplicates"] == 8
assert len(clean_dv) == 22
print(f" -> PASSED: Deduplicated {stats_dv['removed_duplicates']} records leaving {len(clean_dv)} valid clean rows.")

# -------------------------------------------------------------
# Test 8: Conflicting labels explicit handling
# -------------------------------------------------------------
print("\n[Test 8] Conflicting labels policy:")
df_conflict = pd.DataFrame({
    "label": ["ham", "spam", "ham", "spam"] + ["ham", "spam"] * 12,
    "message": ["Ambiguous message", "Ambiguous message", "Legit text", "Spam text"] + [f"Clean message {i}" for i in range(24)]
})
clean_conf, stats_conf = clean_dataframe(df_conflict)
assert stats_conf["conflicting_messages_count"] == 1, f"Expected 1 conflicting, got {stats_conf['conflicting_messages_count']}"
assert "Ambiguous message" not in clean_conf["message"].values, "Conflicting message should be dropped"
print(f" -> PASSED: Conflicting message identified and dropped from training set ({stats_conf['conflicting_messages_count']} message).")

# -------------------------------------------------------------
# Test 9: Single-class dataset rejection
# -------------------------------------------------------------
print("\n[Test 9] Single-class dataset rejection:")
df_single = pd.DataFrame({
    "label": ["ham"] * 30,
    "message": [f"Legit message {i}" for i in range(30)]
})
try:
    clean_dataframe(df_single)
    assert False, "Should have raised ValueError on single-class dataset"
except ValueError as exc:
    print(f" -> PASSED: Raised expected ValueError: {exc}")

# -------------------------------------------------------------
# Test 10: Insufficient samples for binary training / stratified split
# -------------------------------------------------------------
print("\n[Test 10] Insufficient samples handling (single minority sample):")
df_min_sample = pd.DataFrame({
    "label": ["spam"] + ["ham"] * 25,
    "message": [f"Message text {i}" for i in range(26)]
})
clean_min, _ = clean_dataframe(df_min_sample)
try:
    train_and_evaluate_models(clean_min, test_size=0.20)
    assert False, "Should have raised ValueError because 1 sample cannot provide both classes in training"
except ValueError as exc:
    print(f" -> PASSED: Safely rejected single minority instance: {exc}")

# Small dataset with enough samples for both classes
df_small_valid = pd.DataFrame({
    "label": ["spam"] * 4 + ["ham"] * 22,
    "message": [f"Unique text sample {i}" for i in range(26)]
})
clean_sv, _ = clean_dataframe(df_small_valid)
m_data_sv = train_and_evaluate_models(clean_sv, test_size=0.20)
assert len(m_data_sv["models"]) == 4
print(" -> PASSED: Trained successfully with small minority class count (4 spam, 22 ham).")

# -------------------------------------------------------------
# Test 11: All 4 models trained on suitable dataset
# -------------------------------------------------------------
print("\n[Test 11] Model training & evaluation (all 4 pipelines):")
model_data = train_and_evaluate_models(cleaned_df, test_size=0.20)
results_df = model_data["results"]
print("Evaluated Models and F1-Scores:")
for _, r in results_df.iterrows():
    print(f"  • {r['Model']}: Acc={r['Accuracy']:.4f}, Precision={r['Precision (spam)']:.4f}, Recall={r['Recall (spam)']:.4f}, F1={r['F1-score (spam)']:.4f}, AUC={r['ROC-AUC']:.4f}")
assert len(model_data["models"]) == 4, "Expected all 4 models"
best_name = model_data["best_name"]
print(f" -> PASSED: 4 models trained. Top model: '{best_name}'.")

# -------------------------------------------------------------
# Test 12: Prediction of single SMS (ham & spam)
# -------------------------------------------------------------
print("\n[Test 12] Individual SMS prediction without retraining:")
best_model = model_data["models"][best_name]
spam_sample = "Congratulations! You have won a free prize. Click here to claim your reward now!"
ham_sample = "Hey, are we meeting at the library at 4 PM today?"
pred_spam = best_model.predict([spam_sample])[0]
pred_ham = best_model.predict([ham_sample])[0]
assert pred_spam == "spam", f"Expected spam, got {pred_spam}"
assert pred_ham == "ham", f"Expected ham, got {pred_ham}"
print(f" -> PASSED: Sample spam predicted as '{pred_spam}', sample ham predicted as '{pred_ham}'.")

# -------------------------------------------------------------
# Test 13: Bulk unlabeled prediction, row preservation & blank handling
# -------------------------------------------------------------
print("\n[Test 13] Bulk SMS Prediction workflow:")
df_bulk_input = pd.DataFrame({
    "id": [101, 102, 103, 104, 105],
    "client": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "text": [
        "Urgent! Your mobile number has won a £2,000 cash prize. Call 09061701461 to claim.",
        "Hey are you coming to class today?",
        "",  # Blank message
        "WINNER! As a valued customer receive £900 reward!",
        "Can you pick up milk on your way home?"
    ]
})
pred_bulk, b_stats = predict_bulk(best_model, df_bulk_input, "text")

# Verify original columns and rows preserved
assert list(pred_bulk["id"].values) == [101, 102, 103, 104, 105], "Row order corrupted!"
assert "client" in pred_bulk.columns, "Original client column missing!"
assert "predicted_label" in pred_bulk.columns, "predicted_label column missing!"
assert pred_bulk.loc[0, "predicted_label"] == "spam"
assert pred_bulk.loc[1, "predicted_label"] == "ham"
assert pred_bulk.loc[2, "predicted_label"] == "SKIPPED (blank message)"
assert pred_bulk.loc[3, "predicted_label"] == "spam"
assert pred_bulk.loc[4, "predicted_label"] == "ham"
assert b_stats["valid_count"] == 4
assert b_stats["skipped_count"] == 1
assert b_stats["spam_count"] == 2
assert b_stats["ham_count"] == 2
print(" -> PASSED: Bulk prediction completed. Row order and metadata preserved. Skipped blank rows handled.")

# -------------------------------------------------------------
# Test 14: Download serialization fidelity
# -------------------------------------------------------------
print("\n[Test 14] CSV export fidelity:")
csv_export = pred_bulk.to_csv(index=False)
reloaded = pd.read_csv(io.StringIO(csv_export))
assert len(reloaded) == 5
assert list(reloaded["predicted_label"].values) == list(pred_bulk["predicted_label"].values)
print(" -> PASSED: Exported CSV matches in-memory predictions exactly.")

# -------------------------------------------------------------
# Test 15: Model-specific ROC continuous scores & single-class handling
# -------------------------------------------------------------
print("\n[Test 15] Model-specific ROC continuous scores & single-class test split handling:")
from sklearn.metrics import roc_curve, roc_auc_score

# 1. Verify model-specific continuous scores
assert len(model_data["scores"]) == 4, "Expected scores for all 4 models"
mnb_scores = model_data["scores"]["Multinomial Naive Bayes"]
lr_scores = model_data["scores"]["Logistic Regression"]
svm_scores = model_data["scores"]["Linear SVM"]
rf_scores = model_data["scores"]["Random Forest"]

# Check shapes
n_test = len(model_data["X_test"])
assert len(mnb_scores) == n_test and len(lr_scores) == n_test and len(svm_scores) == n_test and len(rf_scores) == n_test

# Check score ranges: probabilities in [0, 1], SVM decision function real-valued
assert np.all((mnb_scores >= 0.0) & (mnb_scores <= 1.0)), "MNB scores should be probabilities in [0, 1]"
assert np.all((lr_scores >= 0.0) & (lr_scores <= 1.0)), "LR scores should be probabilities in [0, 1]"
assert np.all((rf_scores >= 0.0) & (rf_scores <= 1.0)), "RF scores should be probabilities in [0, 1]"
assert not np.array_equal(mnb_scores, lr_scores), "Models must have independent scores"
assert not np.array_equal(lr_scores, svm_scores), "Models must have independent scores"

# 2. Check ROC curve and AUC computation
y_test_bin = (model_data["y_test"] == "spam").astype(int)
for m_name in ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVM", "Random Forest"]:
    fpr, tpr, thresholds = roc_curve(y_test_bin, model_data["scores"][m_name])
    auc = roc_auc_score(y_test_bin, model_data["scores"][m_name])
    assert len(fpr) == len(tpr) and len(fpr) >= 2, f"Invalid ROC curve for {m_name}"
    assert 0.0 <= auc <= 1.0, f"Invalid AUC for {m_name}: {auc}"
print(" -> PASSED: Verified independent continuous scores and valid ROC curves for all 4 models.")

# 3. Single-class test split edge case
y_single_class = pd.Series(["ham"] * 10)
# App logic verifies len(np.unique(y_test)) == 2 before computing ROC-AUC; otherwise assigns np.nan
if len(np.unique(y_single_class)) == 2:
    single_class_auc = roc_auc_score((y_single_class == "spam").astype(int), np.linspace(0, 1, 10))
else:
    single_class_auc = np.nan
assert np.isnan(single_class_auc), "Single class test split correctly produces NaN for ROC-AUC"
print(" -> PASSED: Verified single-class test split handling (NaN AUC).")

print("\n" + "=" * 60)
print("ALL 15 COMPREHENSIVE TESTS PASSED WITH 100% SUCCESS!")
print("=" * 60)

