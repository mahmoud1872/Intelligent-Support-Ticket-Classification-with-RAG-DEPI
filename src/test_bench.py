# test_bench.py
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import config
import rag_pipeline

def run_test_suite():
    print(" Loading saved production assets...")
    with open(config.TFIDF_PICKLE_PATH, 'rb') as f: tfidf = pickle.load(f)
    with open(config.CLASSIFIER_PICKLE_PATH, 'rb') as f: best_model = pickle.load(f)
    with open(config.ENCODERS_PICKLE_PATH, 'rb') as f: encoders = pickle.load(f)
    
    index = faiss.read_index(config.FAISS_INDEX_PATH)
    embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    
    df_saved = pd.read_csv(config.CLEANED_DATA_PATH)
    kb_data = {
        'bodies': df_saved['clean_body'].fillna('').tolist(),
        'answers': df_saved['clean_answer'].fillna('').tolist(),
        'queues': df_saved['queue'].fillna('').tolist(),
        'types': df_saved['type'].fillna('').tolist()
    }
    print(" System assets loaded offline successfully!\n" + "="*60)

    def process_ticket(text):
        x_feat = tfidf.transform([text])
        predictions = best_model.predict(x_feat)[0]
        
        pred_p = encoders['priority'].inverse_transform([predictions[0]])[0]
        pred_t = encoders['type'].inverse_transform([predictions[1]])[0]
        pred_q = encoders['queue'].inverse_transform([predictions[2]])[0]
        
        rag_out = rag_pipeline.generate_rag_response(text, index, embedder, kb_data, top_k=3)
        
        print(f"\n📥 INCOMING TICKET: \"{text[:100]}...\"")
        print(f"🚨 Predicted Priority : {pred_p.upper()}")
        print(f"📁 Predicted Issue Type: {pred_t}")
        print(f"🎯 Assigned Department : {pred_q}")
        print(f"🤖 Output ({rag_out['backend']}): {rag_out['response']}\n" + "="*60)

    # Execution Validation Samples
    process_ticket("Centralized account management portal is completely offline. Bypassing my login credential prompts.")
    process_ticket("Can you provide details regarding integration patterns for popular smart home ecosystems like Alexa?")

if __name__ == "__main__":
    run_test_suite()