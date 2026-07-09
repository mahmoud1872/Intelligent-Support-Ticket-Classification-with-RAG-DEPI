# train.py
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import warnings

import config
import clean_corpus
import data_processing
import models

warnings.filterwarnings('ignore')

def main():
    print("Initiating Master Training Pipeline...")
    
    # Step 1: Execute standalone data cleaning
    clean_corpus.execute_cleaning_pipeline()
    
    # Step 2: Load clean features and apply label encoding layers
    df = data_processing.load_clean_corpus()
    df, encoders = data_processing.encode_labels(df)
    
    # Step 3: Feature Extraction & Classifier Training
    X, tfidf = models.extract_tfidf_features(df)
    best_clf, summary_metrics = models.train_classifiers(X, df)
    
    print("\n --- TRAINED CLASSIFIER SUMMARY BASELINES ---")
    for row in summary_metrics:
        print(f"Model: {row['Model']} | Overall F1: {row['Overall_F1']:.4f} | Queue Acc: {row['Queue_Acc']:.4f}")
        
    # Step 4: Vector Search Embeddings Indexing
    embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    faiss_index, embeddings = models.build_faiss_index(df, embedder)
    
    # Step 5: Save Binary Artifacts for Deployment (Milestone 3 FastAPI)
    print("\n💾 Saving structural production assets...")
    np.save(config.EMBEDDINGS_PATH, embeddings)
    
    with open(config.TFIDF_PICKLE_PATH, 'wb') as f: pickle.dump(tfidf, f)
    with open(config.CLASSIFIER_PICKLE_PATH, 'wb') as f: pickle.dump(best_clf, f)
    with open(config.ENCODERS_PICKLE_PATH, 'wb') as f: pickle.dump(encoders, f)
    faiss.write_index(faiss_index, config.FAISS_INDEX_PATH)
    
    print("Complete Training Phase Finished successfully!")

if __name__ == "__main__":
    main()