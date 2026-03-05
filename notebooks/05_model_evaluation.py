import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
import json

def generate_confusion_matrix():
    print("Loading test data (FinBERT features)...")
    try:
        X_finbert = joblib.load("../datasets/X_finbert.pkl")
        y = joblib.load("../datasets/y_labels.pkl")
    except Exception as e:
        print(f"Error loading features: {e}")
        return

    # We need to re-split exactly as we did in training to get the test set
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X_finbert, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Loading Logistic Regression (FinBERT) model...")
    # NOTE: The previous script output showed Logistic Regression (FinBERT) had 95.9% recall.
    # However, we only explicitly saved `best_finbert_model` (Voting Classifer or XGBoost based on dict extraction).
    # Since we didn't save LR, let's retrain LR quickly here just to get the exact confusion matrix,
    # OR if we just want the predictions, we can use the saved `scam_model_finbert.pkl` which is XGBoost.
    
    # Actually, we chose Logistic Regression conceptually, let's quickly fit it to get the confusion matrix.
    # This takes 2 seconds on FinBERT embeddings.
    from sklearn.linear_model import LogisticRegression
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    _, _, y_train, _ = train_test_split(X_finbert, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Fitting Logistic Regression to get Confusion Matrix...")
    lr_model.fit(X_finbert, y) # Quick train on all for simplicity, or train/test split. We'll use split.
    lr_model.fit(X_finbert[:len(y_train)], y_train) # Approximate split manually without redefining X_train to save memory
    
    # Let's do it cleanly
    X_train, X_test, y_train, y_test = train_test_split(X_finbert, y, test_size=0.2, random_state=42, stratify=y)
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nScam Class (1):")
    print(f"True Positives (Caught Scams): {cm[1, 1]}")
    print(f"False Negatives (Missed Scams): {cm[1, 0]}")
    print("\nLegitimate Class (0):")
    print(f"True Negatives: {cm[0, 0]}")
    print(f"False Positives: {cm[0, 1]}")
    
    # Re-Saving this Logistic Regression model as our absolute best primary model
    joblib.dump(lr_model, "../model_files/primary_scam_model_lr_finbert.pkl")
    print("\nSaved Logistic Regression (FinBERT) as `primary_scam_model_lr_finbert.pkl`")

if __name__ == "__main__":
    generate_confusion_matrix()
