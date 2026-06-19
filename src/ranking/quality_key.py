# Quality-driven tie-breaker key.
#
# RRF fusion produces the primary ordering, but near-ties (candidates within a
# small epsilon of each other) are common at the top because the strongest
# candidates share templated career descriptions and thus near-identical lexical
# scores. NDCG@10 is highly sensitive to the ORDER of the top items, so breaking
# those near-ties by candidate_id (arbitrary w.r.t. quality) leaves NDCG points on
# the table. This module computes a continuous, JD-prioritized quality score used
# ONLY to order candidates within a near-tie band. It does not change the ordering
# of candidates whose RRF scores are clearly separated.
#
# Components (in rough JD-priority order):
#   1. experience-band centrality  (peak at the middle of the 5-9 band)
#   2. high-value skill breadth     (ranking + retrieval + LLM + vector-db)
#   3. shipper/production evidence  (scale & deployment language in descriptions)
#   4. primary-hub location         (exactly Pune/Noida, where the offices are)
#   5. verified skill assessments   (when present)
# All terms are in [0,1]; the weighted sum is in [0,1]. Deterministic.

HIGH_VALUE_FAMILIES = ("ranking", "embeddings_retrieval", "llm", "vector_db")


def _band_centrality(yoe, band):
    lo, hi = band
    mid = (lo + hi) / 2.0
    half = max(1e-6, (hi - lo) / 2.0)
    # 1.0 at the band midpoint, decaying linearly to 0 at +/- one half-width
    # outside the band edges (so a 7-yr beats a 5- or 9-yr, which beat 11-yr).
    dist = abs(float(yoe) - mid) / half
    return max(0.0, 1.0 - 0.5 * dist)


def _skill_breadth(profile, role_config):
    vocab = role_config.get("vocabulary", {})
    skills_text = " ".join((s.get("name", "") or "").lower() for s in (profile.get("skills", []) or []))
    hit_families = 0
    for fam in HIGH_VALUE_FAMILIES:
        terms = vocab.get(fam, {}).get("terms", [])
        if any(t in skills_text for t in terms):
            hit_families += 1
    return hit_families / float(len(HIGH_VALUE_FAMILIES))


def _shipper_evidence(profile, role_config):
    cues = role_config.get("shipper_cues", [])
    if not cues:
        cues = ["production", "deployed", "at scale", "serving", "shipped",
                "latency", "throughput", "millions", "50m", "10m", "1m",
                "end-to-end", "real-time", "pipeline"]
    text = (profile.get("career_text", "") or "").lower()
    hits = sum(1 for c in cues if c in text)
    return min(1.0, hits / 6.0)


def _primary_hub(profile):
    loc = (profile.get("profile", {}).get("location", "") or "").lower()
    return 1.0 if ("pune" in loc or "noida" in loc) else 0.0


def _assessments(profile):
    a = profile.get("signals", {}).get("skill_assessment_scores", {}) or {}
    if not a:
        return 0.5  # neutral when unmeasured (don't punish missing data)
    vals = [v for v in a.values() if isinstance(v, (int, float))]
    if not vals:
        return 0.5
    return max(0.0, min(1.0, (sum(vals) / len(vals)) / 100.0))


def quality_key(profile, role_config):
    band = role_config.get("exp_band", (5, 9))
    yoe = float(profile.get("profile", {}).get("years_of_experience", 0) or 0)
    bc = _band_centrality(yoe, band)
    sb = _skill_breadth(profile, role_config)
    se = _shipper_evidence(profile, role_config)
    ph = _primary_hub(profile)
    asg = _assessments(profile)
    # Weighted by JD priority; band-fit and skill depth dominate.
    key = (0.34 * bc + 0.28 * sb + 0.22 * se + 0.08 * ph + 0.08 * asg)
    return round(key, 9)
