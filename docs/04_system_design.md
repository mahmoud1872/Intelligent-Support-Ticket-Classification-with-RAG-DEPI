# System Analysis & Design

---

## Problem Statement
Manual support ticket handling is slow and inconsistent. We need an automated intelligent system.

---

## System Architecture

Ticket Input  
→ Preprocessing  
→ Classification Model  
→ Embedding (SBERT)  
→ FAISS Retrieval  
→ LLM Generation (Groq)  
→ Final Response  

---

## Data Flow

1. User submits ticket
2. System cleans text
3. Classifier predicts labels
4. FAISS retrieves similar tickets
5. LLM generates response
6. Output returned

---

## Components
- Preprocessing Module
- ML Classifiers
- Embedding Model (SBERT)
- Vector DB (FAISS)
- LLM Generator (Groq API)

---

## Testing
- Accuracy evaluation
- F1 score
- BLEU score
- Retrieval precision@K

---

## Deployment (Conceptual)
- Python backend
- FAISS index storage
- API integration with Groq
