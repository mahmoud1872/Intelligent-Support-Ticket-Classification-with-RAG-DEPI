# data_processing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import config

def load_clean_corpus():
    """Loads the pre-cleaned dataset file safely."""
    try:
        return pd.read_csv(config.CLEANED_DATA_PATH)
    except FileNotFoundError:
        print(f" Error: Clean file '{config.CLEANED_DATA_PATH}' missing. Run clean_corpus.py first!")
        raise

def encode_labels(df):
    """Applies LabelEncoding to the categorical classification targets."""
    le_priority = LabelEncoder()
    le_type = LabelEncoder()
    le_queue = LabelEncoder()

    df['priority_label'] = le_priority.fit_transform(df['priority'])
    df['type_label'] = le_type.fit_transform(df['type'])
    df['queue_label'] = le_queue.fit_transform(df['queue'])

    encoders = {
        'priority': le_priority,
        'type': le_type,
        'queue': le_queue
    }
    return df, encoders