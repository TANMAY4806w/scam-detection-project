# 🛡️ ScamGuard AI — Fake Investment Scam & Ponzi Scheme Detection

> **Final Year Capstone Project** | AI-Based detection of financial scams and Ponzi schemes on social media using NLP and Machine Learning.

---

## 📌 Project Overview

ScamGuard AI is a full-stack AI system that classifies social media messages as either:

- ✅ **Legitimate** — Regular financial discussion
- 🚨 **Scam / Ponzi Scheme** — Fraudulent investment promotion

It uses a **dual-engine ML pipeline** combining **TF-IDF + XGBoost** (keyword-based) and **FinBERT + Logistic Regression** (semantic-based) to achieve robust, real-world scam detection.

---

## 🎯 Key Features

| Feature | Description |
|--------|-------------|
| 🤖 **Dual Engine AI** | TF-IDF XGBoost + FinBERT LR, averaged for higher accuracy |
| 📊 **Model Accuracy** | 99.3% on test set, 95.9% Scam Recall |
| 🔎 **Explainable AI** | LIME-based word-level indicators of scam prediction |
| ⚡ **FastAPI Backend** | REST API serving predictions in real-time |
| 🌐 **Web Interface** | Premium dark-mode frontend with probability visualization |
| 💻 **GPU Support** | XGBoost uses CUDA for accelerated training |

---

## 🗂️ Project Structure

```
scam-detection-project/
│
├── datasets/
│   ├── original/
│   │   └── merged_scam_dataset.csv     ← Raw merged dataset (21,731 records)
│   ├── processed/
│   │   ├── cleaned_scam_dataset.csv    ← Preprocessed text
│   │   ├── X_tfidf.pkl                 ← TF-IDF feature matrix
│   │   ├── X_finbert.pkl               ← FinBERT embeddings (768-dim)
│   │   └── y_labels.pkl                ← Target labels
│   └── README.md                       ← Dataset documentation
│
├── model_files/
│   ├── tfidf_vectorizer.pkl            ← Saved TF-IDF Vectorizer
│   ├── scam_model.pkl                  ← TF-IDF Best Model (XGBoost)
│   ├── scam_model_tfidf.pkl            ← TF-IDF model backup
│   ├── scam_model_finbert.pkl          ← FinBERT model backup
│   └── primary_scam_model_lr_finbert.pkl ← Best model: LR on FinBERT
│
├── notebooks/
│   ├── 01_data_download.py             ← Download datasets from HuggingFace
│   ├── 02_data_preprocessing.py        ← Text cleaning pipeline
│   ├── 03_feature_extraction.py        ← TF-IDF extraction + FinBERT embeddings
│   ├── 04_model_training.py            ← Train LR, NB, XGBoost, Voting Classifier
│   ├── 05_model_evaluation.py          ← Confusion matrix & evaluation
│   ├── 06_real_life_test.py            ← LIME explainability + live testing
│   └── 07_custom_evaluation.py         ← 20-example custom benchmark
│
├── backend/
│   ├── main.py                         ← FastAPI server (dual-engine API)
│   └── test_api.py                     ← API test script
│
├── frontend/
│   ├── index.html                      ← Web interface
│   ├── style.css                       ← Dark-mode premium UI
│   └── app.js                          ← Frontend logic & API integration
│
└── README.md                           ← This file
```

---

## 📦 Datasets Used

| Dataset | Source | Label | Count |
|---------|--------|-------|-------|
| `redasers/difraud` (SMS/Phishing) | HuggingFace | 0 & 1 | 21,846 |
| `txnguyen292/adversarial-scam-dataset` | Kaggle | 0 & 1 | 1,200 |
| **Merged Dataset (Deduplicated)** | Combined | 0 + 1 | **21,731** |

---

## 🧠 Models & Performance

| Model | Features | Accuracy | Scam Recall |
|-------|----------|----------|-------------|
| Naive Bayes | TF-IDF | 96.2% | 85.8% |
| XGBoost (GPU) | TF-IDF | 94.0% | 91.0% |
| Logistic Regression | FinBERT | 93.5% | 94.3% |
| **Ensemble (50/50)** ⭐ | **TF-IDF + FinBERT** | **94.5%** | **94.0%** |

> **Why is Recall the key metric?** A missed scam (False Negative) can cause a user to trust a fraudulent scheme and lose money. We optimize for recall to ensure maximum scam coverage.

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install pandas datasets scikit-learn xgboost transformers torch fastapi uvicorn joblib lime
```

### 2. Download & Prepare Datasets
```bash
cd notebooks
python 01_data_download.py
python 02_data_preprocessing.py
python 03_feature_extraction.py
python 04_model_training.py
```

### 3. Start the Backend API
```bash
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 4. Start the Frontend
```bash
cd frontend
python -m http.server 5500
```

### 5. Open in Browser
```
http://127.0.0.1:5500
```

---

## 🔌 API Reference

### `POST /predict`

**Request:**
```json
{
  "text": "Guaranteed 200% profit daily! Join our mining pool NOW!"
}
```

**Response:**
```json
{
  "prediction": "Scam",
  "probability": 0.87,
  "risk_level": "High"
}
```

**Swagger UI:** http://127.0.0.1:8000/docs

---

## ⚙️ Technologies Used

| Layer | Technology |
|-------|-----------|
| Data | HuggingFace Datasets |
| NLP | FinBERT (`ProsusAI/finbert`), TF-IDF |
| ML Models | Scikit-learn, XGBoost (CUDA GPU) |
| Explainability | LIME |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Serialization | Joblib |

---

## 🎓 Academic Subject Mapping

| Subject | Application |
|---------|------------|
| **Advanced AI (AAI)** | FinBERT Transformers, Ensemble Learning, Explainable AI |
| **AI for Financial & Banking (AIFBA)** | Ponzi scheme detection, investment fraud identification |
| **Social Media Analytics (SMA)** | Text analytics on financial social media posts |

---

## 👤 Author

**Tanmay** | Final Year Capstone Project | 2026
