from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import sys
import os

# Import shared clean_text utility
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks"))
from utils import clean_text

# Initialize FastAPI app
app = FastAPI(
    title="ScamGuard AI — Scam & Phishing Detection API",
    description="API for detecting scams, phishing, and smishing using a dual-engine ML pipeline (TF-IDF + FinBERT).",
    version="2.0.0"
)

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
device = None
tokenizer = None
finbert_model = None
lr_model = None         # FinBERT-based LR model
tfidf_vectorizer = None
xgb_model = None        # TF-IDF-based model

# Input/Output Schemas
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_level: str

@app.on_event("startup")
def load_models():
    """Load ML models and tokenizers on API startup"""
    global device, tokenizer, finbert_model, lr_model, tfidf_vectorizer, xgb_model
    try:
        print("Loading models into memory...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load FinBERT for contextual embeddings
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        finbert_model = AutoModel.from_pretrained("ProsusAI/finbert")
        finbert_model.to(device)
        finbert_model.eval()
        
        # Load FinBERT-based Logistic Regression
        lr_model = joblib.load("../model_files/primary_scam_model_lr_finbert.pkl")
        
        # Load TF-IDF vectorizer + model
        tfidf_vectorizer = joblib.load("../model_files/tfidf_vectorizer.pkl")
        xgb_model = joblib.load("../model_files/scam_model.pkl")
        
        print("All models loaded successfully! Dual-engine pipeline is ready.")
    except Exception as e:
        print(f"Failed to load models: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the ScamGuard AI API. Go to /docs for Swagger UI."}

@app.post("/predict", response_model=PredictionResponse)
def predict_scam(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    try:
        # Step 1: Text Preprocessing
        cleaned = clean_text(request.text)
        
        # --- ENGINE 1: TF-IDF + Model (keyword-based) ---
        tfidf_features = tfidf_vectorizer.transform([cleaned])
        prob_scam_tfidf = float(xgb_model.predict_proba(tfidf_features)[0][1])
        
        # --- ENGINE 2: FinBERT + LR (semantic-based) ---
        inputs = tokenizer(cleaned, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = finbert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        prob_scam_finbert = float(lr_model.predict_proba(cls_embedding)[0][1])
        
        # --- ENSEMBLE: Weighted average (50/50) ---
        prob_scam = round((0.5 * prob_scam_tfidf) + (0.5 * prob_scam_finbert), 4)
        
        # Format Output
        prediction_label = "Scam" if prob_scam > 0.5 else "Legitimate"
        
        if prob_scam > 0.70:
            risk_level = "High"
        elif prob_scam > 0.40:
            risk_level = "Medium"
        else:
            risk_level = "Low"
            
        return PredictionResponse(
            prediction=prediction_label,
            probability=prob_scam,
            risk_level=risk_level
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
