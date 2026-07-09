# clean_corpus.py
import pandas as pd
import os
import re
import config

def normalize_text(text):
    """
    Applies comprehensive NLP text cleaning.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove HTML tags (e.g., <br>, <div>)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 3. Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # 4. Remove email addresses (reduces noise and protects PII)
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # 5. Remove special characters and punctuation (keep only letters, numbers, and spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 6. Replace all whitespace (newlines, tabs, multiple spaces) with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def execute_cleaning_pipeline():
    print("Beginning standalone corpus cleaning pipeline...")
    
    if not os.path.exists(config.RAW_DATA_PATH):
        print(f"Error: Could not find raw file '{config.RAW_DATA_PATH}'. Place it in the directory.")
        return

    # Load raw data
    df = pd.read_csv(config.RAW_DATA_PATH)
    print(f"Raw dataset loaded: {len(df):,} records found.")

    # 1. Filter for English rows
    if 'language' in df.columns:
        df = df[df['language'] == 'en']
        print(f"Filtered to English only. Rows remaining: {len(df):,}")

    # 2. Drop rows missing essential classification items
    df = df.dropna(subset=['body', 'queue']).copy()

    # 3. Drop ALL useless tracking columns (Added 'language' here since we already filtered by it)
    columns_to_drop = ['language', 'version', 'tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6', 'tag_7', 'tag_8']
    df = df.drop(columns=columns_to_drop, errors='ignore').reset_index(drop=True)
    
    # 4. Create unified operational columns
    df['clean_body'] = df['subject'].fillna('') + " " + df['body'].fillna('')
    df['clean_answer'] = df['answer'].fillna('no solution available')
    
    df['type'] = df['type'].fillna('request')
    df['priority'] = df['priority'].fillna('medium')

    # 5. Apply Deep Text Cleaning to BOTH body and answer
    print("Applying deep NLP text normalization to bodies and answers (this may take a moment)...")
    df['clean_body'] = df['clean_body'].apply(normalize_text)
    
    # FIXED: This line is now active and will strip out all special chars from the answers
    df['clean_answer'] = df['clean_answer'].apply(normalize_text) 

    # 6. Drop any rows that became completely empty after cleaning
    df = df[df['clean_body'].str.len() > 0]

    # 7. Save clean corpus file
    df.to_csv(config.CLEANED_DATA_PATH, index=False)
    print(f"Cleaned corpus file saved to: '{config.CLEANED_DATA_PATH}' ({len(df):,} rows)")
    print("Milestone 1 Data Preprocessing Complete!")

if __name__ == "__main__":
    execute_cleaning_pipeline()