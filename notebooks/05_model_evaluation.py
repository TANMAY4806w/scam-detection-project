# %% [markdown]
# # Step 5: Model Evaluation
# Generates confusion matrix and detailed metrics for the FinBERT Logistic Regression model.
# Saves the evaluated model as the primary model for the API.

# %%
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

# %%
def generate_confusion_matrix():
    print("Loading FinBERT features and labels...")
    try:
        X_finbert = joblib.load("../datasets/processed/X_finbert.pkl")
        y = joblib.load("../datasets/processed/y_labels.pkl")
    except Exception as e:
        print(f"Error loading features: {e}")
        return

    # Split exactly as in training (same random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X_finbert, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Test set: {len(y_test)} samples ({sum(y_test)} scam, {len(y_test) - sum(y_test)} legitimate)")

    # %%
    # Train Logistic Regression on FinBERT embeddings
    print("\nTraining Logistic Regression (FinBERT) for evaluation...")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    
    # %%
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    print(f"\nScam Class (1):")
    print(f"  True Positives (Caught Scams):  {cm[1, 1]}")
    print(f"  False Negatives (Missed Scams): {cm[1, 0]}")
    print(f"\nLegitimate Class (0):")
    print(f"  True Negatives:  {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    
    # Detailed report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Scam']))
    
    # Save the model
    joblib.dump(lr_model, "../model_files/primary_scam_model_lr_finbert.pkl")
    print("Saved Logistic Regression (FinBERT) as primary_scam_model_lr_finbert.pkl")

# %%
if __name__ == "__main__":
    generate_confusion_matrix()
