import pandas as pd
import re
import os

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase conversion
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Remove punctuation and special characters (keeping only words and spaces)
    text = re.sub(r'[^\w\s]', '', text)
    
    # 4. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_data():
    input_path = "../datasets/merged_scam_dataset.csv"
    output_path = "../datasets/cleaned_scam_dataset.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
        
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    print("Applying text cleaning (lowercase, remove URLs, punctuation, special chars, extra spaces)...")
    # Create the cleaned text column
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Drop rows where cleaned_text might have become completely empty
    initial_len = len(df)
    df = df[df['cleaned_text'].str.len() > 0]
    dropped = initial_len - len(df)
    print(f"Dropped {dropped} rows with empty cleaned text.")
    
    print("Saving cleaned dataset...")
    df.to_csv(output_path, index=False)
    
    print(f"Saved to {output_path}. Total records: {len(df)}")
    
    # Display a few samples
    print("\nSamples of original vs cleaned text:")
    for i in range(min(5, len(df))):
        print(f"--- Sample {i+1} ---")
        print(f"Original: {df.iloc[i]['text']}")
        print(f"Cleaned : {df.iloc[i]['cleaned_text']}")
        print(f"Label   : {'Scam' if df.iloc[i]['label'] == 1 else 'Legitimate'}")

if __name__ == "__main__":
    preprocess_data()
