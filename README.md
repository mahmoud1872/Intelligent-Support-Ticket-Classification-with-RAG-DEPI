# Intelligent-Support-Ticket-Classification-with-RAG-DEPI

## Project Overview
This project builds an AI-powered support system that:
- Classifies support tickets (priority, type, queue)
- Retrieves similar past tickets using vector search
- Generates intelligent responses using RAG (Retrieval-Augmented Generation)

It simulates real-world ML Engineering systems combining NLP, embeddings, and LLMs.

---

## Pipeline Architecture

Ticket Text  
→ TF-IDF / ML Classifier  
→ SBERT Embeddings  
→ FAISS Vector Search  
→ Top Similar Tickets  
→ Groq LLM (Llama 3.3 70B)  
→ Generated Response  

---

## Features
- Multi-label classification (Priority / Type / Queue)
- Semantic search using SBERT embeddings
- FAISS vector database for retrieval
- RAG-based response generation
- Traditional vs RAG comparison

---

## Models Used
- TF-IDF (bigrams)
- Random Forest / SVM / Logistic Regression
- Sentence-BERT (all-MiniLM-L6-v2)
- FAISS
- Groq LLM API (Llama 3.3 70B)

---

## How to Run
1. Open the notebook in Google Colab
2. Install dependencies
3. Run all cells step-by-step

---

## Evaluation Metrics
- Accuracy (Classification)
- F1 Score
- Precision@K (Retrieval)
- BLEU Score (Response quality)

---

## Tech Stack
Python, Scikit-learn, SBERT, FAISS, Groq API, NLTK
