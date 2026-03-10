# %% [markdown]
# # Step 4: Model Training
# Trains multiple ML models on both TF-IDF and FinBERT features:
# - Logistic Regression (with class_weight='balanced')
# - Naive Bayes (TF-IDF only — can't handle negative values)
# - XGBoost (with scale_pos_weight for class imbalance)
# Saves the best model from each feature set.

# %%
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

# %%
def train_and_evaluate(X, y, feature_type=""):
    """Train multiple models and return results with trained models"""
    print(f"\n{'='*60}")
    print(f"Training models using {feature_type} features...")
    print(f"{'='*60}")
    
    # Stratified Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Calculate class imbalance ratio for XGBoost
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

    print(f"Training Data — Legitimate: {num_neg}, Scam: {num_pos}")
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight, random_state=42,
            eval_metric='logloss', tree_method='hist'
        )
    }
    
    # Naive Bayes only works with non-negative features (TF-IDF)
    if feature_type == "TF-IDF":
        models["Naive Bayes"] = MultinomialNB()

    # %%
    trained_models = {}
    best_model_name = None
    best_recall = -1
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"  Accuracy: {acc:.4f} | Recall (Scam): {rec:.4f} | F1: {f1:.4f}")
        
        trained_models[name] = model
        
        # Track best model by recall (catching scams is priority)
        if rec > best_recall:
            best_recall = rec
            best_model_name = name

    print(f"\nBest {feature_type} model by Recall: {best_model_name} ({best_recall:.4f})")
    
    # Print detailed classification report for best model
    y_pred_best = trained_models[best_model_name].predict(X_test)
    print(f"\nDetailed Classification Report ({best_model_name}):")
    print(classification_report(y_test, y_pred_best, target_names=['Legitimate', 'Scam']))

    return trained_models, best_model_name

# %%
def main():
    # Load features
    try:
        X_tfidf = joblib.load("../datasets/processed/X_tfidf.pkl")
        X_finbert = joblib.load("../datasets/processed/X_finbert.pkl")
        y = joblib.load("../datasets/processed/y_labels.pkl")
    except Exception as e:
        print(f"Error loading features: {e}")
        print("Run 03_feature_extraction.py first.")
        return
        
    os.makedirs("../model_files", exist_ok=True)
    
    # %%
    # Train on TF-IDF features
    models_tfidf, best_tfidf_name = train_and_evaluate(X_tfidf, y, "TF-IDF")
    
    # Save best TF-IDF model
    best_tfidf_model = models_tfidf[best_tfidf_name]
    joblib.dump(best_tfidf_model, "../model_files/scam_model_tfidf.pkl")
    joblib.dump(best_tfidf_model, "../model_files/scam_model.pkl")
    print(f"\n[Saved {best_tfidf_name} as scam_model_tfidf.pkl and scam_model.pkl]")

    # %%
    # Train on FinBERT features
    models_finbert, best_finbert_name = train_and_evaluate(X_finbert, y, "FinBERT")
    
    # Save best FinBERT model
    best_finbert_model = models_finbert[best_finbert_name]
    joblib.dump(best_finbert_model, "../model_files/scam_model_finbert.pkl")
    print(f"\n[Saved {best_finbert_name} as scam_model_finbert.pkl]")
    
    # Also save Logistic Regression (FinBERT) specifically for the dual-engine API
    if "Logistic Regression" in models_finbert:
        lr_finbert = models_finbert["Logistic Regression"]
        joblib.dump(lr_finbert, "../model_files/primary_scam_model_lr_finbert.pkl")
        print("[Saved LR (FinBERT) as primary_scam_model_lr_finbert.pkl for API]")

# %%
if __name__ == "__main__":
    main()
