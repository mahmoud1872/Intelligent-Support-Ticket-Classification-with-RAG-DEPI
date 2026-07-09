# config.py
import os

# API Keys & Third-Party Configurations
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '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')
USE_OLLAMA_FALLBACK = False
OLLAMA_MODEL = 'llama3.2'
USE_GROQ = bool(GROQ_API_KEY)

# Model Definitions
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
TFIDF_MAX_FEATURES = 20000

# File Paths
RAW_DATA_PATH = "raw_data.csv"
CLEANED_DATA_PATH = "cleaned_tickets.csv"
EMBEDDINGS_PATH = "embeddings.npy"

# Exported Artifacts
TFIDF_PICKLE_PATH = "tfidf_vectorizer.pkl"
CLASSIFIER_PICKLE_PATH = "best_classifier.pkl"
ENCODERS_PICKLE_PATH = "label_encoders.pkl"
FAISS_INDEX_PATH = "faiss_index.bin"