# Real ranking-metric implementations (NDCG@k, MAP, Precision@k).
# These mirror the metrics the contest uses to score submissions against the
# hidden ground truth. Provided so the repo's evaluation code is genuine, not a
# placeholder. Used for offline sanity checks when a labelled set is available.
import math
from typing import List, Dict


def dcg(rels: List[float], k: int) -> float:
    s = 0.0
    for i, r in enumerate(rels[:k]):
        s += (2 ** r - 1) / math.log2(i + 2)
    return s


def ndcg_at_k(ranked_ids: List[str], relevance: Dict[str, float], k: int = 10) -> float:
    gains = [relevance.get(cid, 0.0) for cid in ranked_ids[:k]]
    ideal = sorted(relevance.values(), reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg(gains, k) / idcg


def precision_at_k(ranked_ids: List[str], relevant_set: set, k: int = 10) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if cid in relevant_set)
    return hits / float(k)


def average_precision(ranked_ids: List[str], relevant_set: set) -> float:
    if not relevant_set:
        return 0.0
    hits = 0
    cumulative = 0.0
    for i, cid in enumerate(ranked_ids):
        if cid in relevant_set:
            hits += 1
            cumulative += hits / (i + 1)
    return cumulative / len(relevant_set)


def mean_average_precision(ranked_ids: List[str], relevant_set: set) -> float:
    # Single-query MAP == AP for one query; kept for naming parity with the spec.
    return average_precision(ranked_ids, relevant_set)


class Evaluator:
    """Offline evaluation against a labelled relevance map (if available)."""

    def composite_score(self, ranked_ids, relevance):
        relevant_set = {cid for cid, r in relevance.items() if r > 0}
        ndcg10 = ndcg_at_k(ranked_ids, relevance, 10)
        ndcg50 = ndcg_at_k(ranked_ids, relevance, 50)
        mapv = mean_average_precision(ranked_ids, relevant_set)
        p10 = precision_at_k(ranked_ids, relevant_set, 10)
        composite = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * mapv + 0.05 * p10
        return {
            "ndcg@10": round(ndcg10, 6),
            "ndcg@50": round(ndcg50, 6),
            "map": round(mapv, 6),
            "p@10": round(p10, 6),
            "composite": round(composite, 6),
        }
