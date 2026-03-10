# %% [markdown]
# # Step 6: Real-Life Test with LIME Explainability
# Tests the model on custom messages and uses LIME to explain predictions.

# %%
import joblib
import numpy as np
import os
import sys
from lime.lime_text import LimeTextExplainer

# Import shared clean_text
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import clean_text

# %%
def test_real_life_input(custom_message):
    """Test a single message and show LIME explanation"""
    print(f"\n{'-'*50}")
    print("TESTING REAL-LIFE INPUT")
    print(f"{'-'*50}")
    print(f"Message: '{custom_message}'\n")

    cleaned_msgs = [clean_text(custom_message)]

    try:
        # Load TF-IDF model (LIME works best with text → TF-IDF → prediction pipeline)
        vectorizer = joblib.load("../model_files/tfidf_vectorizer.pkl")
        xgboost_model = joblib.load("../model_files/scam_model.pkl")
        
        # Predict
        tfidf_features = vectorizer.transform(cleaned_msgs)
        predict_proba = xgboost_model.predict_proba(tfidf_features)[0]
        prediction = "Scam" if predict_proba[1] > 0.5 else "Legitimate"

        print("--- Prediction ---")
        print(f"Prediction : {prediction}")
        print(f"Probability: {predict_proba[1]*100:.2f}% Scam")
        
        if predict_proba[1] > 0.70:
            print("Risk Level : HIGH")
        elif predict_proba[1] > 0.40:
            print("Risk Level : MEDIUM")
        else:
            print("Risk Level : LOW")

        # %%
        # LIME Explanation
        print("\n--- Explainable AI (LIME) Indicators ---")
        
        def predictor_fn(texts):
            cleaned = [clean_text(t) for t in texts]
            vecs = vectorizer.transform(cleaned)
            return xgboost_model.predict_proba(vecs)
        
        explainer = LimeTextExplainer(class_names=['Legitimate', 'Scam'])
        exp = explainer.explain_instance(custom_message, predictor_fn, num_features=5)
        
        print("Top words contributing to the prediction:")
        for word, weight in exp.as_list():
            if weight > 0:
                print(f"  -> '{word}' (Pushes towards Scam: +{weight:.4f})")
            else:
                print(f"  -> '{word}' (Pushes towards Legitimate: {weight:.4f})")
                
    except Exception as e:
        print(f"Error: {e}")

# %%
if __name__ == "__main__":
    # Test with a phishing/scam message
    test_1 = "URGENT: Your bank account has been compromised! Click this link immediately to verify your identity and secure your funds: www.secure-banking-verify.com"
    test_real_life_input(test_1)
    
    # Test with a legitimate financial message
    test_2 = "Apple Inc. announced its quarterly earnings today. Revenue increased by 5%, beating expectations. The stock market responded positively."
    test_real_life_input(test_2)
    
    # Test with a crypto scam
    test_3 = "Hey bro! I just found this amazing new trading bot that consistently doubles your crypto every week. DM for the link, limited spots available!"
    test_real_life_input(test_3)
