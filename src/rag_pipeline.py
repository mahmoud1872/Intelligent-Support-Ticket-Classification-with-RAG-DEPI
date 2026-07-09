# rag_pipeline.py
from groq import Groq
import config

def find_similar(query_text, index, embedder, kb_data, top_k=3):
    qv = embedder.encode([query_text], normalize_embeddings=True)
    scores, idxs = index.search(qv.astype('float32'), top_k)
    
    results = []
    for r in range(top_k):
        match_idx = idxs[0][r]
        results.append({
            'rank': r+1,
            'score': round(float(scores[0][r]), 4),
            'ticket': kb_data['bodies'][match_idx],
            'answer': kb_data['answers'][match_idx],
            'queue': kb_data['queues'][match_idx],
            'type': kb_data['types'][match_idx]
        })
    return results

def generate_rag_response(query_text, index, embedder, kb_data, top_k=3):
    similar = find_similar(query_text, index, embedder, kb_data, top_k=top_k)
    context = "".join([f"\nContext Example {r['rank']}:\nIssue: {r['ticket'][:150]}\nAnswer: {r['answer'][:150]}\n" for r in similar])
    
    prompt = f"You are an expert customer support agent. Review this context:\n{context}\n\nNew Ticket: {query_text}\n\nWrite a 3-sentence actionable response."
    
    if config.USE_GROQ:
        try:
            client = Groq(api_key=config.GROQ_API_KEY)
            chat = client.chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=200, 
                temperature=0.3
            )
            return {'backend': 'groq', 'response': chat.choices[0].message.content, 'retrieved': similar}
        except Exception:
            pass
            
    return {'backend': 'retrieval-only', 'response': similar[0]['answer'], 'retrieved': similar}