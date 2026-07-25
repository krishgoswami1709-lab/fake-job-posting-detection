"""
Model Training, Experimentation, and Comparative Analysis Pipeline.
Evaluates 6 classification algorithms across multiple feature representations
and preprocessing strategies for Fake Job Posting Detection.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from scipy.sparse import hstack, csr_matrix

# Classifiers
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

PROCESSED_CSV = os.path.join(OUTPUT_DIR, "processed_postings.csv")


def load_processed_data():
    if not os.path.exists(PROCESSED_CSV):
        raise FileNotFoundError(f"Processed file not found at {PROCESSED_CSV}. Run eda_and_preprocessing.py first.")
    df = pd.read_csv(PROCESSED_CSV)
    df['clean_full_text'] = df['clean_full_text'].fillna('')
    df['raw_full_text'] = df['raw_full_text'].fillna('')
    return df


def build_feature_matrices(df, text_col='clean_full_text', vectorizer_type='tfidf', max_features=10000):
    """
    Build combined feature matrix (Text vectorizer + Non-negative Scaled Metadata)
    """
    # 1. Vectorize text
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
            stop_words='english'
        )
    else:
        vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            stop_words='english'
        )

    X_text = vectorizer.fit_transform(df[text_col])

    # 2. Extract Numerical / Flag features with non-negative MinMaxScaler
    num_cols = [
        'telecommuting', 'has_company_logo', 'has_questions',
        'has_company_profile', 'has_requirements', 'has_benefits', 'has_salary',
        'word_count', 'uppercase_ratio', 'missing_text_fields_count'
    ]
    
    scaler = MinMaxScaler()
    X_num = scaler.fit_transform(df[num_cols].fillna(0))

    # 3. Categorical One-Hot Features
    cat_cols = ['employment_type', 'required_experience', 'required_education']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat = encoder.fit_transform(df[cat_cols].fillna('Unspecified'))

    # Combine into sparse matrix
    X_combined = hstack([X_text, csr_matrix(X_num), csr_matrix(X_cat)]).tocsr()

    feature_names = (
        list(vectorizer.get_feature_names_out()) +
        num_cols +
        list(encoder.get_feature_names_out(cat_cols))
    )

    return X_combined, df['fraudulent'].values, vectorizer, scaler, encoder, feature_names


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Train model, evaluate metrics on test set, return metric dictionary.
    """
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)
    else:
        y_prob = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "model_name": model_name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "confusion_matrix": cm
    }, model


def run_experimentation_and_comparisons():
    print("\n========================================================")
    print("STARTING FAKE JOB POSTING MODEL TRAINING & EXPERIMENTATION")
    print("========================================================\n")
    
    df = load_processed_data()

    # Split dataset into train & test (80/20 stratified split)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.20,
        random_state=42,
        stratify=df['fraudulent']
    )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # 1. Main Feature Matrix Construction (TF-IDF + Metadata + Clean Text)
    X_combined, y_all, main_vec, main_scaler, main_encoder, feature_names = build_feature_matrices(
        df, text_col='clean_full_text', vectorizer_type='tfidf', max_features=10000
    )

    X_train = X_combined[train_idx]
    X_test = X_combined[test_idx]
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    print(f"Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")
    print(f"Feature Space Dimensionality: {X_train.shape[1]}")

    # Define Candidate Classifiers
    classifiers = {
        "Logistic Regression (Balanced)": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Complement Naive Bayes": ComplementNB(alpha=0.5),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=20, random_state=42, n_jobs=-1),
        "Extra Trees Classifier": ExtraTreesClassifier(n_estimators=100, class_weight='balanced', max_depth=25, random_state=42, n_jobs=-1),
        "Linear SVM (SGDClassifier)": SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42, max_iter=1000),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }

    model_results = []
    trained_models = {}

    best_f1 = -1.0
    best_model_name = None
    best_model_obj = None

    for name, clf in classifiers.items():
        print(f"Training {name}...")
        res, fitted_clf = evaluate_model(clf, X_train, X_test, y_train, y_test, model_name=name)
        model_results.append(res)
        trained_models[name] = fitted_clf
        
        print(f" -> Accuracy: {res['accuracy']:.4f} | F1: {res['f1_score']:.4f} | Recall: {res['recall']:.4f} | PR-AUC: {res['pr_auc']:.4f}")
        
        if res['f1_score'] > best_f1:
            best_f1 = res['f1_score']
            best_model_name = name
            best_model_obj = fitted_clf

    print(f"\n>>> Best Performing Model: {best_model_name} with F1-Score of {best_f1:.4f} <<<\n")

    # 2. Experimentation: Impact of Preprocessing & Feature Representations
    print("--- Running Preprocessing & Feature Impact Experiments ---")
    experiments = {}

    # Exp A: TF-IDF vs CountVectorizer (on Logistic Regression)
    X_cnt, _, _, _, _, _ = build_feature_matrices(df, text_col='clean_full_text', vectorizer_type='count', max_features=10000)
    cnt_lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    res_cnt, _ = evaluate_model(cnt_lr, X_cnt[train_idx], X_cnt[test_idx], y_train, y_test, model_name="Logistic Regression (CountVectorizer)")
    experiments["Vectorization Comparison"] = {
        "TF-IDF Vectorizer": model_results[0],
        "Count Vectorizer": res_cnt
    }

    # Exp B: Clean Text vs Raw Text (on Logistic Regression)
    X_raw, _, _, _, _, _ = build_feature_matrices(df, text_col='raw_full_text', vectorizer_type='tfidf', max_features=10000)
    raw_lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    res_raw, _ = evaluate_model(raw_lr, X_raw[train_idx], X_raw[test_idx], y_train, y_test, model_name="Logistic Regression (Raw Text)")
    experiments["Text Cleaning Impact"] = {
        "Cleaned Text": model_results[0],
        "Raw Uncleaned Text": res_raw
    }

    # Exp C: Text-Only vs Text + Metadata
    vec_text_only = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True, stop_words='english')
    X_text_only_train = vec_text_only.fit_transform(df_train['clean_full_text'])
    X_text_only_test = vec_text_only.transform(df_test['clean_full_text'])
    text_only_lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    res_text_only, _ = evaluate_model(text_only_lr, X_text_only_train, X_text_only_test, y_train, y_test, model_name="Text Only (No Metadata)")
    experiments["Metadata Feature Impact"] = {
        "Text + Structured Metadata": model_results[0],
        "Text Only": res_text_only
    }

    # 3. Interpretability & Feature Weight Analysis (Using Logistic Regression Model)
    lr_model = trained_models["Logistic Regression (Balanced)"]
    coefs = lr_model.coef_[0]

    top_fake_indices = np.argsort(coefs)[-20:][::-1]
    top_real_indices = np.argsort(coefs)[:20]

    top_fake_terms = [{"term": feature_names[i], "weight": round(float(coefs[i]), 4)} for i in top_fake_indices]
    top_real_terms = [{"term": feature_names[i], "weight": round(float(coefs[i]), 4)} for i in top_real_indices]

    # Save summary report artifact
    evaluation_summary = {
        "model_comparisons": model_results,
        "best_model_name": best_model_name,
        "best_f1_score": best_f1,
        "experiments": experiments,
        "top_fake_indicators": top_fake_terms,
        "top_legitimate_indicators": top_real_terms
    }

    eval_json_path = os.path.join(OUTPUT_DIR, "model_evaluation_report.json")
    with open(eval_json_path, "w") as f:
        json.dump(evaluation_summary, f, indent=4)

    # Save Production Artifacts
    joblib.dump(best_model_obj, os.path.join(MODELS_DIR, "best_job_fraud_model.joblib"))
    joblib.dump(main_vec, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(main_scaler, os.path.join(MODELS_DIR, "numerical_scaler.joblib"))
    joblib.dump(main_encoder, os.path.join(MODELS_DIR, "categorical_encoder.joblib"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.joblib"))

    print(f"\nEvaluation summary saved to: {eval_json_path}")
    print(f"Production model artifacts saved to: {MODELS_DIR}\n")

    return evaluation_summary


if __name__ == "__main__":
    run_experimentation_and_comparisons()
