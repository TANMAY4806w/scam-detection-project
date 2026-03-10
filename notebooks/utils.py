# %%
# Shared utility functions for the ScamGuard AI pipeline
# This module is imported by all notebooks and the backend API.

import re

# %%
def clean_text(text):
    """
    Clean and normalize text for ML processing.
    Steps: lowercase → remove URLs → remove punctuation → remove extra spaces
    """
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
