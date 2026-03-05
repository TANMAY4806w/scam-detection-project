import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_evaluate(X, y, feature_type=""):
    print(f"\n{'='*50}")
    print(f"Training models using {feature_type} features...")
    print(f"{'='*50}")
    
    # Stratified Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Calculate scale_pos_weight for XGBoost to handle class imbalance
    # scale_pos_weight = sum(negative instances) / sum(positive instances)
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

    print(f"Training Data - Legitimate: {num_neg}, Scam: {num_pos}")
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")

    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        # Note: MultinomialNB doesn't directly take 'class_weight' in fit cleanly in sklearn without priors modifications,
        # but works fine on TF-IDF. However, MultinomialNB cannot handle negative values, so we cannot use it for FinBERT.
        "XGBoost": xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss', tree_method='hist', device='cuda')
    }
    
    if feature_type == "TF-IDF":
        models["Naive Bayes"] = MultinomialNB()

    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        print(f"-> Accuracy: {acc:.4f} | Recall (Scam): {rec:.4f}")
        
        trained_models[name] = model

    print("\nTraining Ensemble Voting Classifier...")
    # For Voting Classifier, we will re-use the untrained forms to fit
    estimators = [(name, model) for name, model in trained_models.items()]
    voting_clf = VotingClassifier(estimators=estimators, voting='soft')
    
    # Naive Bayes might not output expected proba if not calibrated, but soft voting works okay on TFIDF
    try:
        voting_clf.fit(X_train, y_train)
        y_pred = voting_clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        print(f"-> Ensemble Accuracy: {acc:.4f} | Ensemble Recall (Scam): {rec:.4f}")
        trained_models["Voting Classifier"] = voting_clf
    except Exception as e:
        print(f"-> Warning: Could not train Voting Classifier: {e}")

    return trained_models

def main():
    # 1. Load Data
    try:
        X_tfidf = joblib.load("../datasets/X_tfidf.pkl")
        X_finbert = joblib.load("../datasets/X_finbert.pkl")
        y = joblib.load("../datasets/y_labels.pkl")
    except Exception as e:
        print("Error loading features. Ensure Step 4 completed successfully.")
        return
        
    os.makedirs("../model_files", exist_ok=True)
    
    # 2. Train on TF-IDF
    models_tfidf = train_and_evaluate(X_tfidf, y, "TF-IDF")
    
    # Save best TF-IDF model (let's save the Voting Classifier or XGBoost)
    if "Voting Classifier" in models_tfidf:
        best_tfidf_model = models_tfidf["Voting Classifier"]
    else:
        best_tfidf_model = models_tfidf["XGBoost"]
        
    joblib.dump(best_tfidf_model, "../model_files/scam_model_tfidf.pkl")
    print("\n[Saved TF-IDF Best Model to model_files]")

    # 3. Train on FinBERT
    # Note: FinBERT embeddings can be negative, standard MultinomialNB won't work.
    models_finbert = train_and_evaluate(X_finbert, y, "FinBERT")
    
    best_finbert_model = models_finbert.get("Voting Classifier", models_finbert["XGBoost"])
    joblib.dump(best_finbert_model, "../model_files/scam_model_finbert.pkl")
    print("\n[Saved FinBERT Best Model to model_files]")

    # Will choose one of these models as `scam_model.pkl` for the final API backend
    joblib.dump(best_tfidf_model, "../model_files/scam_model.pkl")
    print("\n[Also saved TF-IDF as the default scam_model.pkl for API Backend]")

if __name__ == "__main__":
    main()
