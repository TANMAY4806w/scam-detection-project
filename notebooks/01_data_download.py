# %% [markdown]
# # Step 1: Data Download
# Downloads and merges scam/fraud datasets:
# 1. **redasers/difraud** (HuggingFace) — SMS fraud + Phishing emails (JSONL files)
# 2. **Kaggle Adversarial Scam Dataset** — real-world financial fraud, recruitment scams

# %%
import pandas as pd
import os
import json

# %%
def download_difraud_subset(subset_name):
    """Download a subset (sms, phishing, etc.) from redasers/difraud via raw JSONL files"""
    from huggingface_hub import hf_hub_download
    
    all_rows = []
    for split in ["train", "validation", "test"]:
        filepath = f"{subset_name}/{split}.jsonl"
        print(f"  Downloading {filepath}...")
        try:
            local_path = hf_hub_download(
                repo_id="redasers/difraud",
                filename=filepath,
                repo_type="dataset"
            )
            with open(local_path, 'r', encoding='utf-8') as f:
                for line in f:
                    row = json.loads(line.strip())
                    all_rows.append(row)
        except Exception as e:
            print(f"  Warning: Could not download {filepath}: {e}")
    
    df = pd.DataFrame(all_rows)
    return df

# %%
def download_huggingface_difraud():
    """Download SMS fraud + Phishing subsets from redasers/difraud"""
    print("=" * 50)
    print("Downloading DIFrauD Dataset (SMS + Phishing)...")
    print("=" * 50)
    
    # Download SMS fraud subset
    print("\n[1/2] SMS Fraud subset:")
    df_sms = download_difraud_subset("sms")
    print(f"  Loaded {len(df_sms)} SMS messages")
    
    # Download Phishing subset
    print("\n[2/2] Phishing subset:")
    df_phishing = download_difraud_subset("phishing")
    print(f"  Loaded {len(df_phishing)} phishing/email messages")
    
    # Merge SMS + Phishing
    df_hf = pd.concat([df_sms, df_phishing], ignore_index=True)
    
    # Ensure 'text' and 'label' columns exist
    # DIFrauD uses: text (string), label (0=non-deceptive, 1=deceptive)
    df_hf = df_hf[['text', 'label']]
    
    print(f"\nCombined DIFrauD (SMS + Phishing):")
    print(f"  Total: {len(df_hf)}")
    print(f"  Scam/Fraud (1): {len(df_hf[df_hf['label'] == 1])}")
    print(f"  Legitimate (0): {len(df_hf[df_hf['label'] == 0])}")
    
    return df_hf

# %%
def download_kaggle_adversarial_scam():
    """Download Adversarial Scam Dataset from Kaggle"""
    print("\n" + "=" * 50)
    print("Downloading Kaggle Adversarial Scam Dataset...")
    print("=" * 50)
    
    kaggle_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "key")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if not os.path.exists(kaggle_json):
        print("WARNING: kaggle.json not found in key/ folder.")
        return None
    
    # Set up Kaggle credentials
    with open(kaggle_json) as f:
        creds = json.load(f)
    os.environ['KAGGLE_USERNAME'] = creds['username']
    os.environ['KAGGLE_KEY'] = creds['key']
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        download_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "kaggle_temp")
        os.makedirs(download_dir, exist_ok=True)
        
        # Download the adversarial scam dataset
        api.dataset_download_files("txnguyen292/adversarial-scam-dataset", path=download_dir, unzip=True)
        
        # Find CSV files
        csv_files = [f for f in os.listdir(download_dir) if f.endswith('.csv')]
        if not csv_files:
            print("WARNING: No CSV found in Kaggle download.")
            return None
        
        df = pd.read_csv(os.path.join(download_dir, csv_files[0]))
        print(f"  Kaggle columns: {list(df.columns)}")
        print(f"  Loaded {len(df)} messages")
        
        # Remap labels: scam (0, 1) → 1, non-scam (-1) → 0
        if 'label' in df.columns:
            df['label'] = df['label'].apply(lambda x: 0 if x == -1 else 1)
        elif 'Label' in df.columns:
            df = df.rename(columns={'Label': 'label'})
            df['label'] = df['label'].apply(lambda x: 0 if x == -1 else 1)
        
        # Normalize text column name
        text_col = None
        for col in ['text', 'Text', 'message', 'Message', 'content', 'Content']:
            if col in df.columns:
                text_col = col
                break
        
        if text_col and text_col != 'text':
            df = df.rename(columns={text_col: 'text'})
        
        df = df[['text', 'label']]
        
        print(f"  After remapping — Scam: {len(df[df['label'] == 1])}, Legit: {len(df[df['label'] == 0])}")
        
        # Cleanup temp folder
        import shutil
        shutil.rmtree(download_dir, ignore_errors=True)
        
        return df
        
    except Exception as e:
        print(f"WARNING: Kaggle download failed: {e}")
        print("Continuing with HuggingFace dataset only...")
        return None

# %%
def download_and_merge_datasets():
    """Main function: download, merge, and save all datasets"""
    
    # Source 1: DIFrauD (SMS + Phishing)
    df_hf = download_huggingface_difraud()
    
    # Source 2: Kaggle adversarial scam
    df_kaggle = download_kaggle_adversarial_scam()
    
    # Merge all available datasets
    frames = [df_hf]
    if df_kaggle is not None:
        frames.append(df_kaggle)
    
    print("\n" + "=" * 50)
    print("Merging datasets...")
    print("=" * 50)
    
    df_merged = pd.concat(frames, ignore_index=True)
    
    # Drop duplicates
    initial_len = len(df_merged)
    df_merged = df_merged.drop_duplicates(subset=['text'], keep='first')
    print(f"Dropped {initial_len - len(df_merged)} duplicate messages")
    
    # Shuffle
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    os.makedirs("../datasets/original", exist_ok=True)
    out_path = "../datasets/original/merged_scam_dataset.csv"
    df_merged.to_csv(out_path, index=False)
    
    print(f"\nSaved merged dataset to {out_path}")
    print(f"Total messages: {len(df_merged)}")
    print(f"\nClass Distribution:")
    print(df_merged['label'].value_counts())
    scam_ratio = len(df_merged[df_merged['label'] == 1]) / len(df_merged) * 100
    print(f"\nScam ratio: {scam_ratio:.1f}%")

# %%
if __name__ == "__main__":
    download_and_merge_datasets()
