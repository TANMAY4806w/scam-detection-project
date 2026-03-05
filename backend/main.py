from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Initialize FastAPI app
app = FastAPI(
    title="Fake Investment Scam Detection API",
    description="API for detecting Ponzi schemes and investment scams using a dual-engine ML pipeline (TF-IDF + FinBERT).",
    version="1.0.0"
)

# Enable CORS for the frontend to communicate with the backend
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
lr_model = None     # FinBERT-based LR model
tfidf_vectorizer = None
xgb_model = None    # TF-IDF-based XGBoost model

# Input Data Schema
class PredictionRequest(BaseModel):
    text: str

# Output Data Schema
class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_level: str

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.on_event("startup")
def load_models():
    """Load the machine learning models and tokenizers on API startup"""
    global device, tokenizer, finbert_model, lr_model, tfidf_vectorizer, xgb_model
    try:
        print("Loading models into memory...")
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load HuggingFace FinBERT Model (for contextual embeddings)
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        finbert_model = AutoModel.from_pretrained("ProsusAI/finbert")
        finbert_model.to(device)
        finbert_model.eval()
        
        # Load FinBERT-based Logistic Regression model
        lr_model = joblib.load("../model_files/primary_scam_model_lr_finbert.pkl")
        
        # Load TF-IDF vectorizer + XGBoost model (for keyword-based detection)
        tfidf_vectorizer = joblib.load("../model_files/tfidf_vectorizer.pkl")
        xgb_model = joblib.load("../model_files/scam_model.pkl")
        
        print("All models loaded successfully! Dual-engine pipeline is ready.")
    except Exception as e:
        print(f"Failed to load models: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fake Investment Scam Detection API. Go to /docs for Swagger UI."}

@app.post("/predict", response_model=PredictionResponse)
def predict_scam(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    try:
        # Step 1: Text Preprocessing
        cleaned = clean_text(request.text)
        
        # --- ENGINE 1: TF-IDF + XGBoost (keyword-based) ---
        tfidf_features = tfidf_vectorizer.transform([cleaned])
        prob_scam_tfidf = float(xgb_model.predict_proba(tfidf_features)[0][1])
        
        # --- ENGINE 2: FinBERT + Logistic Regression (semantic-based) ---
        inputs = tokenizer(cleaned, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = finbert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        prob_scam_finbert = float(lr_model.predict_proba(cls_embedding)[0][1])
        
        # --- ENSEMBLE: Take weighted average (TF-IDF: 50%, FinBERT: 50%) ---
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
