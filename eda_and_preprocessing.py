"""
Exploratory Data Analysis (EDA) and Preprocessing Pipeline
for Fake Job Posting Detection.
"""

import os
import re
import json
import pandas as pd
import numpy as np

DATASET_PATH = os.path.join(os.path.dirname(__file__), "fake_job_postings.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    """
    Clean unstructured text by removing HTML tags, non-alphabet characters,
    normalizing whitespace, and converting to lowercase.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # 1. Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    # 3. Keep only letters and spaces (stripping digits & punctuation)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # 4. Collapse multiple spaces & lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def raw_text_combine(row):
    """
    Combine raw text fields before cleaning.
    """
    fields = [
        str(row.get('title', '')),
        str(row.get('company_profile', '')),
        str(row.get('description', '')),
        str(row.get('requirements', '')),
        str(row.get('benefits', ''))
    ]
    return " ".join([f for f in fields if f and f.lower() != 'nan'])


def extract_features(df):
    """
    Performs full data preprocessing, text cleaning, feature engineering,
    and returns processed DataFrame alongside metadata statistics.
    """
    print("Performing Exploratory Data Analysis & Preprocessing...")
    
    df = df.copy()
    
    # Fill missing values for text fields
    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits', 'location', 'department']
    for col in text_cols:
        df[col] = df[col].fillna('')

    # 1. Combine raw text and cleaned text
    df['raw_full_text'] = df.apply(raw_text_combine, axis=1)
    df['clean_full_text'] = df['raw_full_text'].apply(clean_text)

    # 2. Structural & Length Features
    df['char_count'] = df['raw_full_text'].apply(len)
    df['word_count'] = df['raw_full_text'].apply(lambda x: len(x.split()))
    df['uppercase_count'] = df['raw_full_text'].apply(lambda x: sum(1 for c in x if c.isupper()))
    df['uppercase_ratio'] = df['uppercase_count'] / (df['char_count'] + 1)
    
    # Missing field flags
    df['has_company_profile'] = (df['company_profile'].str.strip() != '').astype(int)
    df['has_requirements'] = (df['requirements'].str.strip() != '').astype(int)
    df['has_benefits'] = (df['benefits'].str.strip() != '').astype(int)
    df['has_salary'] = (df['salary_range'].fillna('').str.strip() != '').astype(int)
    
    # Calculate total missing text fields count out of key 5 text attributes
    df['missing_text_fields_count'] = 5 - (
        (df['title'].str.strip() != '').astype(int) +
        df['has_company_profile'] +
        (df['description'].str.strip() != '').astype(int) +
        df['has_requirements'] +
        df['has_benefits']
    )

    # 3. Perform EDA Summary Statistics
    total_postings = len(df)
    fraud_postings = int(df['fraudulent'].sum())
    real_postings = total_postings - fraud_postings

    logo_fraud_rate = float(df[df['has_company_logo'] == 0]['fraudulent'].mean())
    logo_real_rate = float(df[df['has_company_logo'] == 1]['fraudulent'].mean())

    questions_fraud_rate = float(df[df['has_questions'] == 0]['fraudulent'].mean())
    questions_real_rate = float(df[df['has_questions'] == 1]['fraudulent'].mean())

    no_profile_fraud_rate = float(df[df['has_company_profile'] == 0]['fraudulent'].mean())
    profile_fraud_rate = float(df[df['has_company_profile'] == 1]['fraudulent'].mean())

    eda_stats = {
        "total_postings": total_postings,
        "real_postings": real_postings,
        "fraudulent_postings": fraud_postings,
        "fraud_percentage": round(fraud_postings / total_postings * 100, 2),
        "fraud_rate_no_company_logo": round(logo_fraud_rate * 100, 2),
        "fraud_rate_with_company_logo": round(logo_real_rate * 100, 2),
        "fraud_rate_no_screening_questions": round(questions_fraud_rate * 100, 2),
        "fraud_rate_with_screening_questions": round(questions_real_rate * 100, 2),
        "fraud_rate_no_company_profile": round(no_profile_fraud_rate * 100, 2),
        "fraud_rate_with_company_profile": round(profile_fraud_rate * 100, 2),
        "avg_word_count_real": round(float(df[df['fraudulent'] == 0]['word_count'].mean()), 1),
        "avg_word_count_fraud": round(float(df[df['fraudulent'] == 1]['word_count'].mean()), 1)
    }

    # Save EDA summary JSON
    eda_json_path = os.path.join(OUTPUT_DIR, "eda_summary.json")
    with open(eda_json_path, "w") as f:
        json.dump(eda_stats, f, indent=4)

    print(f"EDA Summary saved to {eda_json_path}")
    print(json.dumps(eda_stats, indent=2))

    return df, eda_stats


if __name__ == "__main__":
    if os.path.exists(DATASET_PATH):
        df_raw = pd.read_csv(DATASET_PATH)
        df_processed, stats = extract_features(df_raw)
        processed_path = os.path.join(OUTPUT_DIR, "processed_postings.csv")
        df_processed.to_csv(processed_path, index=False)
        print(f"Processed dataset saved to {processed_path}")
    else:
        print("Dataset not found. Please run download_data.py first.")
