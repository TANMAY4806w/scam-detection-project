import pandas as pd
from datasets import load_dataset
import os

def download_and_merge_datasets():
    print("Downloading Legitimate Financial Data...")
    # Load twitter financial news sentiment dataset from huggingface
    finance_dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    df_finance = pd.DataFrame(finance_dataset)
    # The finance dataset has 'text' and 'label' columns. All of them are legitimate text, so map to 0.
    df_finance['label'] = 0
    df_finance = df_finance[['text', 'label']]
    print(f"Loaded {len(df_finance)} legitimate financial messages.")

    print("Downloading Spam/Scam Data...")
    # Load SMS spam collection dataset from huggingface
    spam_dataset = load_dataset("sms_spam", split="train")
    df_spam_all = pd.DataFrame(spam_dataset)
    # Label 1 is spam. We only want spam to represent scams/ponzi schemas/phishing.
    df_spam = df_spam_all[df_spam_all['label'] == 1].copy()
    # Ensure label is 1 for Scam
    df_spam['label'] = 1
    # Rename 'sms' column to 'text' if it exists
    if 'sms' in df_spam.columns:
        df_spam = df_spam.rename(columns={'sms': 'text'})
    df_spam = df_spam[['text', 'label']]
    print(f"Loaded {len(df_spam)} scam/spam messages.")

    print("Merging datasets...")
    # Merge and shuffle
    df_merged = pd.concat([df_finance, df_spam], ignore_index=True)
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to disk
    os.makedirs("../datasets", exist_ok=True)
    out_path = "../datasets/merged_scam_dataset.csv"
    df_merged.to_csv(out_path, index=False)
    print(f"Saved merged dataset to {out_path} with {len(df_merged)} total messages.")
    
    print("Class Distribution:")
    print(df_merged['label'].value_counts())

if __name__ == "__main__":
    download_and_merge_datasets()
