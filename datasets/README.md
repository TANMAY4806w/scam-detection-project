# Scam Detection Dataset

## Overview
This dataset contains a mixture of legitimate financial discussions and fraudulent scam messages. It was compiled from reliable open-source datasets to train the AI Fake Investment Scam & Ponzi Scheme Detection model.

## Data Sources

### 1. Legitimate Class (`label = 0`)
- **Dataset Source:** Hugging Face
- **Name:** `zeroshot/twitter-financial-news-sentiment`
- **Description:** An annotated corpus of English-language finance-related tweets to represent standard financial discussions on social media.
- **Record Count:** 9,543 entries.

### 2. Scam/Spam Class (`label = 1`)
- **Dataset Source:** Hugging Face (Originally SMS Spam Collection)
- **Name:** `sms_spam`
- **Description:** A classic dataset filtered specifically for the `spam` label, representing fraudulent messages, phishing, fake lottery, and "double your money" messages.
- **Record Count:** 747 entries.

## Final Merged Dataset
- **File Name:** `merged_scam_dataset.csv`
- **Total Records:** 10,290
- **Format:** CSV
- **Columns:**
  - `text`: The string content of the message.
  - `label`: Binary classification label (0 = Legitimate, 1 = Scam).
- **Class Distribution:**
  - `0`: 9,543 (92.74%)
  - `1`: 747 (7.26%)
- **Data Balancing Strategy Needed?** Yes, the data is highly imbalanced towards the legitimate class. Techniques like class weighting or SMOTE will be applied during modeling.

## Generation Process
The dataset was auto-downloaded, filtered, mapped to binary labels, concatenated, and shuffled using the `notebooks/01_data_download.py` script.
