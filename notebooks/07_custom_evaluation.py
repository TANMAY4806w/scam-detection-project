# %% [markdown]
# # Step 7: Custom Evaluation Benchmark
# Evaluates all TF-IDF models on a hand-crafted benchmark of 30 messages
# covering phishing, smishing, financial fraud, and legitimate content.

# %%
import joblib
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import clean_text

# %%
def test_models_on_custom_data():
    print("Loading TF-IDF Vectorizer and features...")
    try:
        vectorizer = joblib.load("../model_files/tfidf_vectorizer.pkl")
        X_tfidf = joblib.load("../datasets/processed/X_tfidf.pkl")
        y = joblib.load("../datasets/processed/y_labels.pkl")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # %%
    # Retrain models for comparison (takes ~1 second on TF-IDF)
    print("Training models for benchmark comparison...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    import xgboost as xgb
    
    scale_pos_weight = np.sum(y == 0) / np.sum(y == 1) if np.sum(y == 1) > 0 else 1.0

    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "XGBoost": xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss', tree_method='hist')
    }
    
    for name, model in models.items():
        model.fit(X_tfidf, y)

    # %%
    # 30 Custom Examples: diverse scam types + legitimate messages + edge cases
    custom_dataset = [
        # --- PHISHING / SMISHING (Label 1) ---
        ("URGENT: Your bank account has been locked. Click here to verify your identity immediately: www.secure-bank-login.com", 1),
        ("We detected suspicious activity on your PayPal. Confirm your account now or it will be suspended: paypal-verify.net", 1),
        ("Your package could not be delivered. Update your address and pay $1.99 shipping fee here: usps-redeliver.com", 1),
        ("IRS ALERT: You owe back taxes of $4,500. Pay immediately to avoid arrest. Call 1-800-555-0199", 1),
        ("Netflix: Your payment failed. Update your billing info within 24 hours or lose access: netflix-billing-update.com", 1),
        
        # --- FINANCIAL / CRYPTO SCAMS (Label 1) ---
        ("Guaranteed returns of 200% daily in crypto! Join our VIP mining pool now, limited spots available!", 1),
        ("Hey bro, I found this amazing new trading bot that doubles your money every week. DM for the invite link", 1),
        ("Elon Musk is giving back to the community! Send ETH to the address below and get 2x back instantly!", 1),
        ("Invest just $50 today and I will manage your forex trades. Profits guaranteed $5000 in 3 days.", 1),
        ("Secret matrix system just launched! Getting in early means you get paid from everyone who joins later. $10 to start.", 1),
        
        # --- RECRUITMENT / ADVANCE FEE SCAMS (Label 1) ---
        ("Congratulations! You've been selected for a work-from-home job paying $500/day. No experience needed. Send $50 registration fee.", 1),
        ("You won the WhatsApp Financial Lottery of $1,000,000! Pay the $500 transfer fee to receive your prize.", 1),
        ("Hi darling, I'm a successful trader. I made $50k last month from my phone. I can teach you for a small signup fee.", 1),
        ("ATTENTION: Your Social Security number has been suspended. Press 1 to speak with an officer immediately.", 1),
        ("Double your ethereum overnight with our new staking DApp. Secure, audited, and mathematically guaranteed returns.", 1),
        
        # --- LEGITIMATE FINANCIAL MESSAGES (Label 0) ---
        ("Federal Reserve announces a new interest rate hike of 0.25% in an effort to curb inflation.", 0),
        ("Apple Inc. reported strong quarterly earnings beating analyst estimates. Revenue grew 7% year over year.", 0),
        ("Goldman Sachs reports that oil prices may stabilize next quarter due to expected supply chain resolutions.", 0),
        ("According to my technical analysis, BTC might test support at the 200 EMA before any potential bounce.", 0),
        ("Just dollar cost averaging into my Vanguard S&P 500 index fund this month. Staying disciplined.", 0),
        
        # --- LEGITIMATE EVERYDAY MESSAGES (Label 0) ---
        ("Hey, are we still meeting for coffee at 3pm today?", 0),
        ("Your Amazon order #112-3456 has shipped and will arrive by Thursday.", 0),
        ("Reminder: Your dentist appointment is scheduled for March 15 at 10:00 AM.", 0),
        ("Can anyone recommend a good book on value investing? I've already read The Intelligent Investor.", 0),
        ("The European Central Bank plans to inject liquidity into the banking sector to prevent a recession.", 0),
        
        # --- EDGE CASES (Label 0) ---
        ("I need to rebalance my 401k portfolio since US equities have grown to 85% of my allocation.", 0),
        ("Congratulations on your promotion! Well deserved. Let's celebrate this weekend.", 0),
        ("Breaking: Major cybersecurity breach reported at a Fortune 500 company. Investigation ongoing.", 0),
        ("Your monthly bank statement is ready. Log in to your account at chase.com to view it.", 0),
        ("Warning: Severe weather alert for your area. Stay indoors and avoid travel.", 0),
    ]

    # %%
    print(f"\nEvaluating {len(custom_dataset)} Custom Real-Life Messages...")
    
    texts = [item[0] for item in custom_dataset]
    true_labels = [item[1] for item in custom_dataset]
    
    cleaned_texts = [clean_text(t) for t in texts]
    X_custom_tfidf = vectorizer.transform(cleaned_texts)

    results = []
    
    for name, model in models.items():
        preds = model.predict(X_custom_tfidf)
        acc = accuracy_score(true_labels, preds)
        
        correct_scams = sum(1 for yt, yp in zip(true_labels, preds) if yt == 1 and yp == 1)
        total_scams = sum(true_labels)
        scam_recall = correct_scams / total_scams if total_scams > 0 else 0
        
        correct_legit = sum(1 for yt, yp in zip(true_labels, preds) if yt == 0 and yp == 0)
        total_legit = len(true_labels) - total_scams
        legit_precision = correct_legit / total_legit if total_legit > 0 else 0
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Scam Recall": scam_recall,
            "Correct Scams": correct_scams,
            "Total Scams": total_scams,
            "Legit Correct": correct_legit,
            "Total Legit": total_legit
        })
        
    # %%
    # Print Results
    print("\n" + "=" * 70)
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Scams Caught':<15} | {'Legit Correct':<15}")
    print("=" * 70)
    for res in results:
        print(f"{res['Model']:<25} | {res['Accuracy']*100:>6.1f} %   | {res['Correct Scams']}/{res['Total Scams']} ({res['Scam Recall']*100:.1f}%)      | {res['Legit Correct']}/{res['Total Legit']}")
    print("=" * 70)
    
    # Show line-by-line predictions for best model
    best_model = models["Logistic Regression"]
    lr_preds = best_model.predict(X_custom_tfidf)
    
    print("\nLine-by-line predictions (Logistic Regression):")
    for i, (text, true_label) in enumerate(custom_dataset):
        pred_label = lr_preds[i]
        true_str = "SCAM " if true_label == 1 else "LEGIT"
        pred_str = "SCAM " if pred_label == 1 else "LEGIT"
        
        match = "OK" if true_label == pred_label else "XX"
        print(f"[{match}] True: {true_str} | Pred: {pred_str} | {text[:60]}...")

# %%
if __name__ == "__main__":
    test_models_on_custom_data()
