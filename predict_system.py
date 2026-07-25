"""
Functional Prediction System for Fake Job Posting Detection.
Loads production model artifacts and computes real-time fraud probability,
risk classification, and detailed feature breakdown for unseen job postings.
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

class FakeJobDetector:
    def __init__(self):
        model_path = os.path.join(MODELS_DIR, "best_job_fraud_model.joblib")
        tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
        scaler_path = os.path.join(MODELS_DIR, "numerical_scaler.joblib")
        encoder_path = os.path.join(MODELS_DIR, "categorical_encoder.joblib")
        features_path = os.path.join(MODELS_DIR, "feature_names.joblib")

        if not all(os.path.exists(p) for p in [model_path, tfidf_path, scaler_path, encoder_path, features_path]):
            raise FileNotFoundError("Model artifacts missing in 'models/' directory. Train models first.")

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(tfidf_path)
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)
        self.feature_names = joblib.load(features_path)

        # High-risk scam keyword patterns for rule-assisted breakdown
        self.scam_keywords = [
            "wire transfer", "western union", "bitcoin", "crypto", "package inspector",
            "reshipping", "cash check", "deposit check", "earn $500", "daily payout",
            "no experience needed", "guaranteed income", "telegram", "whatsapp",
            "freemail", "money order", "financial transfer agent"
        ]

    def clean_text(self, text):
        if not isinstance(text, str) or not text.strip():
            return ""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip().lower()

    def predict(self, job_posting):
        """
        Accepts dict of job posting fields:
        {
          'title': '...',
          'company_profile': '...',
          'description': '...',
          'requirements': '...',
          'benefits': '...',
          'telecommuting': 0 or 1,
          'has_company_logo': 0 or 1,
          'has_questions': 0 or 1,
          'salary_range': '...',
          'employment_type': 'Full-time',
          'required_experience': 'Entry level',
          'required_education': "Bachelor's Degree"
        }
        Returns structured dictionary with fraud_probability, label, risk_level, and risk_factors.
        """
        title = str(job_posting.get('title', ''))
        profile = str(job_posting.get('company_profile', ''))
        description = str(job_posting.get('description', ''))
        requirements = str(job_posting.get('requirements', ''))
        benefits = str(job_posting.get('benefits', ''))
        salary = str(job_posting.get('salary_range', ''))

        raw_full = f"{title} {profile} {description} {requirements} {benefits}"
        clean_full = self.clean_text(raw_full)

        # 1. Text TF-IDF Vectorization
        X_text = self.vectorizer.transform([clean_full])

        # 2. Numerical / Flag features
        telecommuting = int(job_posting.get('telecommuting', 0))
        has_logo = int(job_posting.get('has_company_logo', 0))
        has_qs = int(job_posting.get('has_questions', 0))

        has_profile = 1 if profile.strip() else 0
        has_reqs = 1 if requirements.strip() else 0
        has_bens = 1 if benefits.strip() else 0
        has_sal = 1 if salary.strip() else 0

        word_count = len(raw_full.split())
        char_count = len(raw_full)
        uppercase_count = sum(1 for c in raw_full if c.isupper())
        uppercase_ratio = uppercase_count / (char_count + 1)
        missing_text_fields_count = 5 - (
            (1 if title.strip() else 0) + has_profile + (1 if description.strip() else 0) + has_reqs + has_bens
        )

        num_cols = [
            'telecommuting', 'has_company_logo', 'has_questions',
            'has_company_profile', 'has_requirements', 'has_benefits', 'has_salary',
            'word_count', 'uppercase_ratio', 'missing_text_fields_count'
        ]

        num_df = pd.DataFrame([{
            'telecommuting': telecommuting,
            'has_company_logo': has_logo,
            'has_questions': has_qs,
            'has_company_profile': has_profile,
            'has_requirements': has_reqs,
            'has_benefits': has_bens,
            'has_salary': has_sal,
            'word_count': word_count,
            'uppercase_ratio': uppercase_ratio,
            'missing_text_fields_count': missing_text_fields_count
        }], columns=num_cols)

        X_num = self.scaler.transform(num_df)

        # 3. Categorical One-Hot
        emp_type = job_posting.get('employment_type', 'Unspecified')
        exp_lvl = job_posting.get('required_experience', 'Unspecified')
        edu_lvl = job_posting.get('required_education', 'Unspecified')

        cat_df = pd.DataFrame([{
            'employment_type': emp_type,
            'required_experience': exp_lvl,
            'required_education': edu_lvl
        }])
        X_cat = self.encoder.transform(cat_df)

        # Combine sparse row
        X_row = hstack([X_text, csr_matrix(X_num), csr_matrix(X_cat)]).tocsr()

        # Model Inference
        if hasattr(self.model, "predict_proba"):
            fraud_prob = float(self.model.predict_proba(X_row)[0, 1])
        else:
            decision = float(self.model.decision_function(X_row)[0])
            fraud_prob = 1.0 / (1.0 + np.exp(-decision))

        is_fraud = bool(fraud_prob >= 0.50)

        # Risk Factors Analysis
        risk_factors = []
        if has_logo == 0:
            risk_factors.append({
                "factor": "Missing Company Logo",
                "severity": "HIGH",
                "detail": "82% of fake job postings omit official corporate branding/logo."
            })
        if has_profile == 0:
            risk_factors.append({
                "factor": "Missing Company Profile",
                "severity": "HIGH",
                "detail": "Posting lacks any background information regarding company history or mission."
            })
        if has_qs == 0:
            risk_factors.append({
                "factor": "No Candidate Screening Questions",
                "severity": "MEDIUM",
                "detail": "Legitimate employers typically include screening questions during application."
            })
        if word_count < 40:
            risk_factors.append({
                "factor": "Unusually Brief Job Description",
                "severity": "MEDIUM",
                "detail": f"Posting contains only {word_count} words; legitimate offers provide comprehensive job scopes."
            })

        # Search for scam keywords
        detected_keywords = [kw for kw in self.scam_keywords if kw in clean_full]
        if detected_keywords:
            risk_factors.append({
                "factor": "Suspicious / Scam Terminology Detected",
                "severity": "CRITICAL",
                "detail": f"Found high-risk keywords: {', '.join(detected_keywords)}"
            })

        if fraud_prob >= 0.75:
            risk_level = "CRITICAL RISK"
        elif fraud_prob >= 0.50:
            risk_level = "HIGH RISK"
        elif fraud_prob >= 0.25:
            risk_level = "MODERATE RISK"
        else:
            risk_level = "LOW RISK (LEGITIMATE)"

        return {
            "fraud_probability": round(fraud_prob * 100, 2),
            "is_fraudulent": is_fraud,
            "prediction_label": "FRAUDULENT JOB POSTING" if is_fraud else "LEGITIMATE JOB POSTING",
            "risk_level": risk_level,
            "confidence_score": round((fraud_prob if is_fraud else (1.0 - fraud_prob)) * 100, 2),
            "risk_factors": risk_factors,
            "detected_keywords": detected_keywords
        }


if __name__ == "__main__":
    detector = FakeJobDetector()

    # Test Sample Real Posting
    sample_real = {
        "title": "Senior Python Software Engineer",
        "company_profile": "Leading AI technology startup with 300+ employees developing enterprise ML pipelines.",
        "description": "Designing high throughput REST APIs and managing PostgreSQL databases on AWS.",
        "requirements": "Bachelor's degree in CS, 4+ years Python experience, Docker proficiency.",
        "benefits": "Competitive salary, 401k match, health insurance, paid vacation.",
        "has_company_logo": 1,
        "has_questions": 1,
        "telecommuting": 0,
        "employment_type": "Full-time",
        "required_experience": "Mid-Senior level",
        "required_education": "Bachelor's Degree"
    }

    # Test Sample Fake Posting
    sample_fake = {
        "title": "Earn $500/day Data Entry Clerk - Work from Home!",
        "company_profile": "",
        "description": "Work from home 2 hours daily! Receive checks, cash them at Western Union, and forward wire transfers to client.",
        "requirements": "Must have active bank account. No experience needed!",
        "benefits": "Immediate daily payouts via bitcoin or wire transfer.",
        "has_company_logo": 0,
        "has_questions": 0,
        "telecommuting": 1,
        "employment_type": "Part-time",
        "required_experience": "Entry level",
        "required_education": "Unspecified"
    }

    print("\n--- SAMPLE REAL POSTING PREDICTION ---")
    res_r = detector.predict(sample_real)
    print(json.dumps(res_r, indent=2))

    print("\n--- SAMPLE FAKE POSTING PREDICTION ---")
    res_f = detector.predict(sample_fake)
    print(json.dumps(res_f, indent=2))
