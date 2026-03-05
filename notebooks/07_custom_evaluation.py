import joblib
import re
import pandas as pd
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def test_models_on_custom_data():
    print("Loading TF-IDF Vectorizer and Models from Step 5...")
    try:
        vectorizer = joblib.load("../model_files/tfidf_vectorizer.pkl")
        
        # In Step 5, we overrode `scam_model_tfidf.pkl` and `scam_model.pkl` to just hold the "best" TF-IDF model.
        # But we trained them all in a dictionary inside the script. We need to grab them from where they might be saved
        # Oh wait, we didn't save the dictionary of ALL models. Let's just quickly retrain them on the loaded features
        # because retraining TF-IDF models takes ~1 second.
        
        X_tfidf = joblib.load("../datasets/X_tfidf.pkl")
        y = joblib.load("../datasets/y_labels.pkl")
    except Exception as e:
        print(f"Error loading required files: {e}")
        return

    print("Quickly retraining the 4 models on TF-IDF space to compare exactly on this task...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    import xgboost as xgb
    from sklearn.ensemble import VotingClassifier
    import numpy as np
    
    scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)

    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "XGBoost": xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss', tree_method='hist')
    }
    
    for name, model in models.items():
        if name != "XGBoost":
            model.fit(X_tfidf, y)
        else:
            model.fit(X_tfidf, y)
            
    voting_clf = VotingClassifier(estimators=[(n, m) for n, m in models.items()], voting='soft')
    voting_clf.fit(X_tfidf, y)
    models["Voting Classifier"] = voting_clf

    # 20 Custom Examples: 12 Scams (60%), 8 Legitimate (40%)
    custom_dataset = [
        # --- 12 SCAMS (Label 1) ---
        ("GUARANTEED returns of 100% daily in crypto! Click here to join my VIP mining pool now limits apply!!", 1),
        ("You have been selected to win 5 BTC. Please send 0.1 BTC to this wallet to verify your identity and claim.", 1),
        ("Hey bro, I found this amazing new trading bot that consistently doubles your money every week. DM for link", 1),
        ("URGENT: Your Binance account is blocked. Verify your funds at www.binance-secure-auth.com immediately to unlock.", 1),
        ("Invest just $50 today and I will manage your forex trades. Profits guaranteed $5000 in 3 days. Risk free.", 1),
        ("Elon Musk is giving back to the community! Send ETH to the address below and get 2x back instantly! Limited time promotion.", 1),
        ("This new altcoin is about to pump 1000x tomorrow! Join our pump and dump signal group right now before it moons.", 1),
        ("Hello sir, I am an expert account manager. I trade for you and we share 50/50 profits. Just fund your mt4 account and give me the password.", 1),
        ("Double your ethereum overnight with our new staking DApp protocol. Secure, audited, and mathematically guaranteed.", 1),
        ("Hi darling, I made $50k last month just trading binary options from my phone. I can teach you for a small signup fee.", 1),
        ("Congratulations! Your number won the WhatsApp Financial Lottery of $1,000,000. Pay the $500 transfer fee to receive it.", 1),
        ("Secret Ponzi matrix system just launched! Getting in early means you get paid from everyone who joins later. Secure your slot for $10.", 1),
        
        # --- 8 LEGITIMATE (Label 0) ---
        ("Federal Reserve announces a new interest rate hike of 0.25% in an effort to curb inflation. Markets closed relatively flat.", 0),
        ("I just read the Wall Street Journal article about the upcoming Apple earnings report. They expect strong iPhone sales.", 0),
        ("According to my technical analysis on the daily chart, BTC might test support at the 200 EMA before any potential bounce.", 0),
        ("Just dollar cost averaging into my Vanguard S&P 500 index fund this month like usual. Staying disciplined with long term investing.", 0),
        ("Can anyone recommend a good book on value investing? I've already read The Intelligent Investor by Benjamin Graham.", 0),
        ("Goldman Sachs reports that oil prices may stabilize next quarter due to expected supply chain resolutions.", 0),
        ("I need to rebalance my 401k portfolio since US equities have grown to be 85% of my allocation.", 0),
        ("The European Central Bank plans to inject liquidity into the banking sector to prevent a recession.", 0)
    ]

    print("\nEvaluating the 20 Custom Real-Life Inputs...")
    
    # Preprocess custom dataset
    texts = [item[0] for item in custom_dataset]
    true_labels = [item[1] for item in custom_dataset]
    
    cleaned_texts = [clean_text(t) for t in texts]
    X_custom_tfidf = vectorizer.transform(cleaned_texts)

    results = []
    
    for name, model in models.items():
        preds = model.predict(X_custom_tfidf)
        acc = accuracy_score(true_labels, preds)
        
        # Calculate correct scams manually
        correct_scams = sum(1 for yt, yp in zip(true_labels, preds) if yt == 1 and yp == 1)
        total_scams = sum(true_labels)
        scam_recall = correct_scams / total_scams
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Scam Recall": scam_recall,
            "Correct Scams": correct_scams,
            "Total Scams": total_scams
        })
        
    # Print Results nicely
    print("\n" + "="*60)
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Scams Caught':<15}")
    print("="*60)
    for res in results:
        print(f"{res['Model']:<25} | {res['Accuracy']*100:>6.1f} %   | {res['Correct Scams']}/{res['Total Scams']} ({(res['Scam Recall']*100):.1f}%)")
    print("="*60)
    
    # Show predictions line by line for the best model (Logistic Regression)
    best_model = models["Logistic Regression"]
    lr_preds = best_model.predict(X_custom_tfidf)
    
    print("\nLine by line predictions (Logistic Regression):")
    for i, (text, true_label) in enumerate(custom_dataset):
        pred_label = lr_preds[i]
        true_str = "SCAM" if true_label == 1 else "LEGIT"
        pred_str = "SCAM" if pred_label == 1 else "LEGIT"
        
        match = "✅" if true_label == pred_label else "❌"
        print(f"[{match}] True: {true_str} | Pred: {pred_str} | Text: {text[:50]}...")


if __name__ == "__main__":
    test_models_on_custom_data()
