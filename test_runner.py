import io
import os
import pandas as pd
from app import (
    normalize_label,
    load_uploaded_dataset,
    clean_dataframe,
    train_and_evaluate_models,
    build_models,
    get_scores
)

print('--- Test 1: Testing spam.csv (User download) ---')
with open(r'C:\Users\Pallav\Downloads\spam.csv', 'rb') as f:
    raw_df = load_uploaded_dataset(f)
assert len(raw_df) == 5572, f'Expected 5572 rows, got {len(raw_df)}'
cleaned_df, stats = clean_dataframe(raw_df)
print(f'Test 1 Cleaned: {stats["original_rows"]} -> {stats["final_rows"]} rows. Ham: {stats["ham_count"]}, Spam: {stats["spam_count"]}')
model_data = train_and_evaluate_models(cleaned_df, test_size=0.20)
f1_val = model_data["results"].iloc[0]["F1-score (spam)"]
print(f'Test 1 Top Model: {model_data["best_name"]} with F1: {f1_val:.4f}')

print('\n--- Test 2: Testing Bulk Unlabeled CSV ---')
with open(r'test_datasets\bulk_unlabeled.csv', 'rb') as f:
    bulk_df = load_uploaded_dataset(f)
assert len(bulk_df) == 6, f'Expected 6 rows, got {len(bulk_df)}'
best_model = model_data['models'][model_data['best_name']]
preds = best_model.predict(bulk_df['message'])
print(f'Test 2 Bulk Predictions: {preds.tolist()}')
assert len(preds) == 6

print('\n--- Test 3: Testing Empty CSV ---')
try:
    with open(r'test_datasets\empty.csv', 'rb') as f:
        load_uploaded_dataset(f)
    print('Test 3 FAILED: expected error')
except ValueError as e:
    print(f'Test 3 PASSED with expected ValueError: {e}')

print('\n--- Test 4: Testing Missing Columns ---')
try:
    with open(r'test_datasets\missing_cols.csv', 'rb') as f:
        df_missing = load_uploaded_dataset(f)
    clean_dataframe(df_missing)
    print('Test 4 FAILED: expected error')
except ValueError as e:
    print(f'Test 4 PASSED with expected ValueError: {e}')

print('\n--- Test 5: Testing Invalid Labels ---')
try:
    with open(r'test_datasets\invalid_labels.csv', 'rb') as f:
        df_invalid = load_uploaded_dataset(f)
    clean_dataframe(df_invalid)
    print('Test 5 FAILED: expected error')
except ValueError as e:
    print(f'Test 5 PASSED with expected ValueError: {e}')

print('\n--- Test 6: Testing Small Dataset (<20 rows) ---')
try:
    with open(r'test_datasets\small_dataset.csv', 'rb') as f:
        df_small = load_uploaded_dataset(f)
    clean_dataframe(df_small)
    print('Test 6 FAILED: expected error')
except ValueError as e:
    print(f'Test 6 PASSED with expected ValueError: {e}')

print('\n--- Test 7: Testing Single Class Dataset ---')
try:
    with open(r'test_datasets\single_class.csv', 'rb') as f:
        df_single = load_uploaded_dataset(f)
    clean_dataframe(df_single)
    print('Test 7 FAILED: expected error')
except ValueError as e:
    print(f'Test 7 PASSED with expected ValueError: {e}')

print('\n--- Test 8: Testing Custom Labeled CSV ---')
with open(r'test_datasets\valid_custom_labeled.csv', 'rb') as f:
    df_custom = load_uploaded_dataset(f)
cleaned_custom, stats_custom = clean_dataframe(df_custom)
print(f'Test 8 Custom Cleaned: {stats_custom["original_rows"]} -> {stats_custom["final_rows"]} rows.')
model_data_custom = train_and_evaluate_models(cleaned_custom, test_size=0.20)
f1_custom = model_data_custom["results"].iloc[0]["F1-score (spam)"]
print(f'Test 8 Custom Top Model: {model_data_custom["best_name"]} with F1: {f1_custom:.4f}')

print('\nALL 8 TESTS COMPLETED SUCCESSFULLY!')
