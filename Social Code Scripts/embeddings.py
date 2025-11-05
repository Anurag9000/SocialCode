"""
embeddings.py
- Provides `embed_texts(texts, model_name)` using sentence-transformers if available.
- Falls back to TF-IDF if sentence-transformers is not installed (keeps API the same).
"""
from typing import List, Tuple
import numpy as np

def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Tuple[object, np.ndarray, str]:
    """
    Returns (model_or_vectorizer, embeddings ndarray, backend)
    backend in {"sentence-transformers", "tfidf"}
    """
    texts = [t if isinstance(t, str) else "" for t in texts]

    # Try sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embs = np.asarray(model.encode(texts, convert_to_numpy=True, normalize_embeddings=True))
        return model, embs, "sentence-transformers"
    except Exception:
        pass

    # Fallback: TF-IDF (shared vectorizer must be used consistently)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words='english')
    X = vec.fit_transform([t.lower() for t in texts])
    X = normalize(X)  # L2 normalize
    return vec, X, "tfidf"

def embed_with(model_or_vec, texts: List[str], backend: str) -> "np.ndarray":
    texts = [t if isinstance(t, str) else "" for t in texts]
    if backend == "sentence-transformers":
        # model is a SentenceTransformer
        embs = model_or_vec.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(embs)
    else:
        # scikit TF-IDF vectorizer
        from sklearn.preprocessing import normalize
        X = model_or_vec.transform([t.lower() for t in texts])
        X = normalize(X)
        return X

def cosine_sim(a, b):
    """Cosine similarity for dense or sparse matrices."""
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(a, b)
