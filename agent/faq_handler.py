"""
FAQ Retrieval Handler using semantic search.
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import os

BASE_DIR = Path(__file__).parent
FAQS_PATH = BASE_DIR / "faqs.json"

_model = None
_faqs = []
_faq_embeddings = None

def _load_faqs():
    global _faqs
    if not FAQS_PATH.exists():
        return []
    try:
        with open(FAQS_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading FAQs: {e}")
        return []

def _get_model():
    global _model, _faqs, _faq_embeddings
    if _model is None:
        print("Loading SentenceTransformer model for FAQs...", flush=True)
        # Using a small and fast embedding model
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        _faqs = _load_faqs()
        if _faqs:
            questions = [faq["question"] for faq in _faqs]
            _faq_embeddings = _model.encode(questions, convert_to_tensor=True)
        else:
            _faq_embeddings = None
    return _model

def get_answer(user_question: str) -> str:
    """Find the best matching FAQ answer using semantic search."""
    model = _get_model()
    if not _faqs or _faq_embeddings is None:
        return "I'm sorry, but I don't have any FAQs loaded right now."
    
    # Encode the user's question
    query_embedding = model.encode(user_question, convert_to_tensor=True)
    
    # Compute cosine similarities
    cosine_scores = util.cos_sim(query_embedding, _faq_embeddings)[0]
    
    # Find the best match
    best_match_idx = int(cosine_scores.argmax())
    best_score = float(cosine_scores[best_match_idx])
    
    # Threshold for a good match (tune as needed)
    if best_score > 0.35:
        return _faqs[best_match_idx]["answer"]
    else:
        return "I'm not sure about that. Try asking another question or let me know if you need help with anything else."
