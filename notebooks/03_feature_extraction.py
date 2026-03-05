import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import torch
from transformers import AutoTokenizer, AutoModel

def extract_tfidf(df):
    print("Extracting TF-IDF features...")
    # Configuration: max_features approx 10000, remove english stopwords
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    
    # Fit and transform the cleaned text
    X_tfidf = vectorizer.fit_transform(df['cleaned_text'])
    
    print(f"TF-IDF shape: {X_tfidf.shape}")
    
    # Save the vectorizer for future API usage
    os.makedirs("../model_files", exist_ok=True)
    joblib.dump(vectorizer, "../model_files/tfidf_vectorizer.pkl")
    print("Saved TF-IDF Vectorizer to ../model_files/tfidf_vectorizer.pkl")
    
    return X_tfidf

def extract_finbert(df):
    print("Extracting FinBERT embeddings... (this may take a few minutes)")
    # Load HuggingFace FinBERT Model
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    embeddings = []
    
    # Process in batches to manage memory
    batch_size = 128
    texts = df['cleaned_text'].tolist()
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            # Tokenize batch
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get model outputs
            outputs = model(**inputs)
            
            # Use the [CLS] token embedding (first token) as sentence representation
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
            
            if (i // batch_size) % 10 == 0:
                print(f"Processed batch {i // batch_size} / {len(texts) // batch_size}")
                
    X_finbert = np.vstack(embeddings)
    print(f"FinBERT embeddings shape: {X_finbert.shape}")
    
    return X_finbert


def main():
    input_path = "../datasets/cleaned_scam_dataset.csv"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run preprocessing first.")
        return
        
    df = pd.read_csv(input_path)
    # Ensure cleaned_text is string (drop any str(nan))
    df = df.dropna(subset=['cleaned_text'])
    df['cleaned_text'] = df['cleaned_text'].astype(str)
    
    # 1. TF-IDF
    X_tfidf = extract_tfidf(df)
    joblib.dump(X_tfidf, "../datasets/X_tfidf.pkl")
    
    # 2. FinBERT
    X_finbert = extract_finbert(df)
    joblib.dump(X_finbert, "../datasets/X_finbert.pkl")
    
    # Save labels array
    y = df['label'].values
    joblib.dump(y, "../datasets/y_labels.pkl")
    
    print("Feature Extraction Complete! Saved X_tfidf, X_finbert, and y_labels to disk.")


if __name__ == "__main__":
    main()
