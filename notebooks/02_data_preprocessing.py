# %% [markdown]
# # Step 2: Data Preprocessing
# Cleans the merged dataset: lowercase, remove URLs, remove punctuation, remove extra spaces.

# %%
import pandas as pd
import os
import sys

# Add parent directory for shared utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import clean_text

# %%
def preprocess_data():
    input_path = "../datasets/original/merged_scam_dataset.csv"
    output_path = "../datasets/processed/cleaned_scam_dataset.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run 01_data_download.py first.")
        return
        
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} messages")
    
    # %%
    print("Applying text cleaning (lowercase, remove URLs, punctuation, special chars, extra spaces)...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Drop rows where cleaned_text is empty
    initial_len = len(df)
    df = df[df['cleaned_text'].str.len() > 0]
    dropped = initial_len - len(df)
    print(f"Dropped {dropped} rows with empty cleaned text.")
    
    # %%
    print("Saving cleaned dataset...")
    df.to_csv(output_path, index=False)
    
    print(f"Saved to {output_path}. Total records: {len(df)}")
    print(f"\nClass Distribution:")
    print(df['label'].value_counts())
    
    # Display a few samples
    print("\nSamples of original vs cleaned text:")
    for i in range(min(5, len(df))):
        print(f"--- Sample {i+1} ---")
        print(f"Original: {df.iloc[i]['text'][:100]}")
        print(f"Cleaned : {df.iloc[i]['cleaned_text'][:100]}")
        print(f"Label   : {'Scam' if df.iloc[i]['label'] == 1 else 'Legitimate'}")

# %%
if __name__ == "__main__":
    preprocess_data()
