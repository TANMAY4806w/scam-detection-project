import joblib
import re
import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def test_real_life_input(custom_message):
    print(f"\n{'-'*50}")
    print("TESTING REAL-LIFE INPUT")
    print(f"{'-'*50}")
    print(f"Message: '{custom_message}'\n")

    cleaned_msgs = [clean_text(custom_message)]

    try:
        # Load TF-IDF components for Explainability
        # Explainability (LIME/SHAP) is much easier to visualize on text directly using TF-IDF
        # because FinBERT processes dense vectors that don't map back to single words cleanly for visualization.
        vectorizer = joblib.load("../model_files/tfidf_vectorizer.pkl")
        xgboost_model = joblib.load("../model_files/scam_model.pkl") # this is the TF-IDF Best Model saved in Step 5
        
        # We transform the text
        tfidf_features = vectorizer.transform(cleaned_msgs)
        
        predict_proba_xgboost = xgboost_model.predict_proba(tfidf_features)[0]
        prediction_xgboost = "Scam" if predict_proba_xgboost[1] > 0.5 else "Legitimate"

        print("--- prediction ---")
        print(f"Prediction : {prediction_xgboost}")
        print(f"Probability: {predict_proba_xgboost[1]*100:.2f}% Scam")
        
        if predict_proba_xgboost[1] > 0.70:
            print("Risk Level : HIGH")
        elif predict_proba_xgboost[1] > 0.40:
            print("Risk Level : MEDIUM")
        else:
            print("Risk Level : LOW")

        print("\n--- Explainable AI (LIME) Indicators ---")
        
        # We need a function that takes raw text and outputs probabilities for LIME
        def predictor_fn(texts):
            # clean texts
            cleaned = [clean_text(t) for t in texts]
            # vectorize
            vecs = vectorizer.transform(cleaned)
            # predict
            return xgboost_model.predict_proba(vecs)
        
        explainer = LimeTextExplainer(class_names=['Legitimate', 'Scam'])
        exp = explainer.explain_instance(custom_message, predictor_fn, num_features=5)
        
        # Get top contributing words
        print("Top words contributing to the 'Scam' prediction:")
        for word, weight in exp.as_list():
            if weight > 0:
                print(f" -> '{word}' (Weight: +{weight:.4f})")
            else:
                print(f" -> '{word}' (Decreases Scam Prob: {weight:.4f})")
                
    except Exception as e:
        print(f"Error executing real-life test: {e}")

if __name__ == "__main__":
    test_1 = "Hey man! I just found a guaranteed way to double your money in 24 hours using smart crypto mining returns. Join my private telegram group now, limited investment opportunity!"
    test_real_life_input(test_1)
    
    test_2 = "Apple Inc. announced its quarterly earnings today. Revenue increased by 5%, beating expectations. The stock market responded positively, with the S&P 500 up slightly."
    test_real_life_input(test_2)
