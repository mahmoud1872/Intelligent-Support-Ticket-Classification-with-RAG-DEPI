# Intelligent-Support-Ticket-Classification-with-RAG-DEPI

## Project Overview
This project builds an intelligent customer support system that:
- Classifies support tickets (Priority, Type, Queue)
- Retrieves similar historical tickets using semantic search (FAISS + SBERT)
- Generates automated responses using a Retrieval-Augmented Generation (RAG) pipeline with Groq Llama 3.3 70B

---

## Pipeline Architecture

Ticket → TF-IDF / SBERT → Classification + Retrieval → FAISS → LLM (Groq / Ollama) → Response

---

## Tech Stack
- Python
- Scikit-learn
- Sentence Transformers (SBERT)
- FAISS
- Groq API (Llama 3.3 70B)
- NLTK
- Pandas / NumPy

---

## Key Features
- Multi-output ticket classification
- Semantic search using embeddings
- RAG-based response generation
- Traditional vs Transformer comparison
- BLEU + Accuracy evaluation

---

## How to Run
```bash
pip install -r requirements.txt
python src/main.py
