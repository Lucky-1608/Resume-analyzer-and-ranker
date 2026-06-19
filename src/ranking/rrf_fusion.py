# Reciprocal Rank Fusion (Cormack, Clarke & Buttcher, SIGIR 2009).
# RRF(d) = sum_r weight_r / (k + rank_r(d)), k=60.
# Operates on RANKS only, so signals on incomparable scales (BM25 unbounded,
# cosine in [0,1], rule score) combine without normalization or tuning.
# Deterministic: ties in any signal are broken by candidate_id ascending before
# ranks are assigned, so the fused result is identical across runs.

RRF_K = 60


def _rank_map(score_dict):
    """cid -> 1-based rank. Higher score = better (rank 1). Tie-break: id asc."""
    ordered = sorted(score_dict.items(), key=lambda x: (-x[1], str(x[0])))
    return {cid: i + 1 for i, (cid, _) in enumerate(ordered)}


def reciprocal_rank_fusion(signal_scores, weights=None, k=RRF_K):
    """
    signal_scores: dict name -> {cid: score}
    weights:       dict name -> float (default 1.0 each)
    Returns: dict cid -> fused RRF score (higher = better).
    """
    weights = weights or {}
    rank_maps = {name: _rank_map(sd) for name, sd in signal_scores.items()}
    all_ids = set()
    for sd in signal_scores.values():
        all_ids.update(sd.keys())
    fused = {}
    for cid in all_ids:
        s = 0.0
        for name in sorted(signal_scores.keys()):  # sorted => deterministic sum
            w = weights.get(name, 1.0)
            r = rank_maps[name].get(cid)
            if r is not None:
                s += w / (k + r)
        fused[cid] = round(s, 9)
    return fused
