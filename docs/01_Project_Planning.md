# Project Planning & Management

## Project Proposal
This project aims to build an intelligent support ticket system that combines:
- NLP classification
- Vector search
- LLM-based response generation (RAG)

---

## Objectives
- Automate support ticket classification
- Improve response time using AI
- Provide context-aware responses using retrieval + generation

---

## Project Plan

| Phase | Task |
|------|------|
| Phase 1 | Data preprocessing & cleaning |
| Phase 2 | Model training (TF-IDF, ML models) |
| Phase 3 | Embedding + FAISS retrieval |
| Phase 4 | RAG integration |
| Phase 5 | Evaluation |

---

## Roles
- ML Engineer: Model training
- NLP Engineer: Embeddings + RAG
- Developer: Pipeline integration

---

## Risk Assessment
- API rate limits (Groq)
- Model accuracy issues
- Data imbalance

### Mitigation
- Fallback to retrieval-only mode
- Class weighting
- Data preprocessing

---

## KPIs
- Classification Accuracy
- F1 Score
- Retrieval Precision@K
- BLEU Score
