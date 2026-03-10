# %% [markdown]
# # Step 3: Feature Extraction
# Extracts two types of features from the cleaned text:
# 1. **TF-IDF** — keyword-based sparse features (fast, interpretable)
# 2. **FinBERT** — contextual dense embeddings from a finance-pretrained BERT model

# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import torch
from transformers import AutoTokenizer, AutoModel

# %%
def extract_tfidf(df):
    """Extract TF-IDF features from cleaned text"""
    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(df['cleaned_text'])
    
    print(f"TF-IDF shape: {X_tfidf.shape}")
    
    # Save the vectorizer for API usage
    os.makedirs("../model_files", exist_ok=True)
    joblib.dump(vectorizer, "../model_files/tfidf_vectorizer.pkl")
    print("Saved TF-IDF Vectorizer to ../model_files/tfidf_vectorizer.pkl")
    
    return X_tfidf

# %%
def extract_finbert(df):
    """Extract FinBERT [CLS] token embeddings as dense features"""
    print("Extracting FinBERT embeddings... (this may take a few minutes)")
    
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    embeddings = []
    batch_size = 128
    texts = df['cleaned_text'].tolist()
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            
            # Use [CLS] token embedding as sentence representation
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
            
            if (i // batch_size) % 10 == 0:
                print(f"Processed batch {i // batch_size + 1} / {(len(texts) + batch_size - 1) // batch_size}")
                
    X_finbert = np.vstack(embeddings)
    print(f"FinBERT embeddings shape: {X_finbert.shape}")
    
    return X_finbert

# %%
def main():
    input_path = "../datasets/processed/cleaned_scam_dataset.csv"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run 02_data_preprocessing.py first.")
        return
        
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['cleaned_text'])
    df['cleaned_text'] = df['cleaned_text'].astype(str)
    
    print(f"Loaded {len(df)} cleaned messages")
    
    # 1. TF-IDF
    X_tfidf = extract_tfidf(df)
    os.makedirs("../datasets/processed", exist_ok=True)
    joblib.dump(X_tfidf, "../datasets/processed/X_tfidf.pkl")
    
    # 2. FinBERT
    X_finbert = extract_finbert(df)
    joblib.dump(X_finbert, "../datasets/processed/X_finbert.pkl")
    
    # Save labels
    y = df['label'].values
    joblib.dump(y, "../datasets/processed/y_labels.pkl")
    
    print("\nFeature Extraction Complete!")
    print(f"Saved X_tfidf ({X_tfidf.shape}), X_finbert ({X_finbert.shape}), and y_labels ({len(y)}) to disk.")

# %%
if __name__ == "__main__":
    main()
