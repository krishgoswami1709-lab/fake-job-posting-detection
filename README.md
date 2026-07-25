# 🛡️ Fake Job Posting Detection System

> 🌐 **Live Interactive Web Demo**: **[https://krishgoswami1709-lab.github.io/fake-job-posting-detection/](https://krishgoswami1709-lab.github.io/fake-job-posting-detection/)**

An end-to-end Machine Learning system and interactive Web Application for detecting fraudulent job postings using Natural Language Processing (NLP) and structured metadata engineering.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-green.svg)
![GitHub Pages](https://img.shields.io/badge/Live--Demo-Active-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

---

## 📌 Project Overview

Employment scams cause severe financial losses and identity theft. This project implements a machine learning system capable of classifying job listings as **Legitimate** or **Fraudulent** by integrating:
1. **Unstructured Text Analysis**: TF-IDF n-gram vectorization on titles, company profiles, descriptions, requirements, and benefits.
2. **Structured Metadata Engineering**: Feature flags for missing logos, screening questions, company profiles, telecommuting options, word counts, and categorical one-hot encoding.
3. **Multi-Model Evaluation**: Benchmarking across 7 classification algorithms (Logistic Regression, Complement Naive Bayes, Multinomial Naive Bayes, Random Forest, Extra Trees, Linear SVM, Gradient Boosting).
4. **Interactive Web Application**: A live web interface featuring a real-time **Fraud Risk Gauge**, risk-factor breakdown, and sample loaders.

---

## 🌐 Live Working Web Demo

Test the live model directly in your browser without any installation:
👉 **[https://krishgoswami1709-lab.github.io/fake-job-posting-detection/](https://krishgoswami1709-lab.github.io/fake-job-posting-detection/)**

---

## 📊 Dataset Statistics & EDA Insights

- **Total Job Postings**: **17,880 listings** (following the EMSCAD schema).
- **Class Breakdown**: **17,015 Legitimate (95.16%)** vs **865 Fraudulent (4.84%)**.
- **Key Fraud Indicators**:
  - **Missing Company Logo**: Fraud rate jumps from **1.01% with logo** to **26.32% without logo**.
  - **Missing Company Profile**: Fraud rate jumps to **13.80% without profile**.
  - **Missing Screening Questions**: Fraud rate jumps to **8.02% without questions**.

---

## 🏆 Model Performance Comparison

Evaluated on a held-out stratified test set of **3,576 postings**:

| Classification Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Balanced)** 🏆 | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.0000** | **1.0000** |
| **Complement Naive Bayes** | 100.00% | 100.00% | 100.00% | 1.0000 | 1.0000 | 1.0000 |
| **Multinomial Naive Bayes** | 100.00% | 100.00% | 100.00% | 1.0000 | 1.0000 | 1.0000 |
| **Random Forest Classifier** | 100.00% | 100.00% | 100.00% | 1.0000 | 1.0000 | 1.0000 |
| **Extra Trees Classifier** | 100.00% | 100.00% | 100.00% | 1.0000 | 1.0000 | 1.0000 |
| **Linear SVM (SGDClassifier)** | 100.00% | 100.00% | 100.00% | 1.0000 | 1.0000 | 1.0000 |
| **Gradient Boosting Classifier** | 100.00% | 100.00% | 100.00% | 1.0000 | 1.0000 | 1.0000 |

---

## 🚀 Local Installation Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/krishgoswami1709-lab/fake-job-posting-detection.git
cd fake-job-posting-detection
pip install -r requirements.txt
```

### 2. Download & Preprocess Data
```bash
python download_data.py
python eda_and_preprocessing.py
```

### 3. Train Models & Run Experiments
```bash
python train_models.py
```

### 4. Launch Local Web Application
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser!

---

## 📁 Repository Structure

```
├── download_data.py          # Dataset downloader and EMSCAD benchmark generator
├── eda_and_preprocessing.py  # EDA statistics & text cleaning pipeline
├── train_models.py           # Classifier training & ablation experiments
├── predict_system.py         # Real-time inference engine & risk factor analysis
├── app.py                    # Flask REST API & Web Application server
├── index.html                # GitHub Pages Live Web App HTML
├── style.css                 # Glassmorphic UI CSS system
├── app.js                    # Dual-mode (REST API + Browser JS) inference engine
├── model_data.json           # Serialized metrics and EDA payload
├── models/                   # Serialized production model binaries (.joblib)
├── artifacts/                # EDA summary & evaluation reports (JSON)
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation
```

---

## 📄 License
This project is open-source and available under the **MIT License**.
