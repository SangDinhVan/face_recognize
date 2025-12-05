import numpy as np
from typing import List, Tuple, Optional


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / denom)


def match_embedding(
    query_emb: np.ndarray,
    db_embeddings: List[Tuple[str, np.ndarray]],
    threshold: float = 0.5,
) -> Tuple[int, Optional[str], float]:
    if query_emb is None or not db_embeddings:
        return 0, None, 0.0

    best_user, best_score = None, -1.0

    for user_id, emb in db_embeddings:
        score = cosine_similarity(query_emb, emb)
        if score > best_score:
            best_score = score
            best_user = user_id

    if best_score >= threshold:
        return 1, best_user, best_score

    return 0, None, best_score


def identify_embedding(
    query_emb: np.ndarray,
    db_embeddings: List[Tuple[str, np.ndarray]],
) -> Tuple[Optional[str], float]:
    if query_emb is None or not db_embeddings:
        return None, 0.0

    best_user, best_score = None, -1.0

    for user_id, emb in db_embeddings:
        score = cosine_similarity(query_emb, emb)
        if score > best_score:
            best_score = score
            best_user = user_id

    return best_user, best_score
