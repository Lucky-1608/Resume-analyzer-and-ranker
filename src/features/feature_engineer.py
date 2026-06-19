# Feature engineering for candidate ranking.
# DETERMINISTIC: every dict/iteration that feeds a float sum is sorted first,
# so results do not depend on Python's per-process hash seed.
#
# Implements:
#   Gap 2 - career-history evidence scoring ("says vs means")
#   Gap 3 - fuller behavioral / availability model (all 23 signals)
#   Gap 4 - location fit (via fit_signals)
#   Gap 5 - education / external-validation bonus (via fit_signals)

from src.intelligence.fit_signals import location_fit, validation_signal

_PW = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.55, "beginner": 0.3, "": 0.4}

# Strong production-ML evidence phrases mined from career_history descriptions.
# These separate genuine Tier-5 builders from keyword-stuffers (Gap 2).
EVIDENCE_STRONG = [
    "recommendation system", "recommender", "ranking system", "search system",
    "semantic search", "vector search", "retrieval", "embeddings", "rag",
    "learning to rank", "deployed to", "in production", "production system",
    "served", "serving", "at scale", "millions of", "real-time", "latency",
    "a/b test", "ab test", "fine-tun", "vector database", "faiss", "pinecone",
    "elasticsearch", "opensearch", "qdrant", "recall", "ndcg", "throughput",
]
EVIDENCE_MED = [
    "machine learning", "deep learning", "nlp", "natural language", "pipeline",
    "model", "feature", "classifier", "training", "inference", "api", "spark",
    "airflow", "data pipeline", "etl", "backend", "microservice", "docker",
    "kubernetes", "scalable", "distributed",
]
SCALE_HINTS = ["million", "billion", "scale", "1m", "10m", "100k", "qps",
               "tps", "throughput", "concurrent", "high-traffic"]


def _exp_fit(yoe, band, smin, smax):
    lo, hi = band
    if lo <= yoe <= hi:
        return 1.0
    if yoe < lo:
        if smin > 0 and yoe <= smin:
            return max(0.10, 0.45 * (yoe / smin))
        return 0.6 + 0.4 * ((yoe - smin) / max(1e-6, lo - smin))
    return 0.30 if yoe >= smax else 1.0 - 0.70 * ((yoe - hi) / max(1e-6, smax - hi))


def _idle(ds):
    if not ds or not isinstance(ds, str):
        return 999.0
    from datetime import datetime, date
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            d = datetime.strptime(ds, fmt).date()
            t = date.today()
            return (t.year - d.year) * 12 + (t.month - d.month)
        except Exception:
            pass
    return 999.0


def _career_evidence(profile):
    """Gap 2: score production-ML evidence found in career-history descriptions."""
    ctext = (profile.get("career_text", "") or "").lower()
    if not ctext:
        return 0.0
    strong = sum(1 for p in EVIDENCE_STRONG if p in ctext)
    med = sum(1 for p in EVIDENCE_MED if p in ctext)
    scale = 1 if any(h in ctext for h in SCALE_HINTS) else 0
    # Saturating: a few strong hits already signal a real builder.
    score = min(1.0, 0.16 * strong + 0.05 * med + 0.10 * scale)
    return round(score, 6)


def _behavioral(sig):
    """Gap 3: availability/quality multiplier from all relevant behavioral signals.
    Returns a value centered near 1.0, roughly in [0.55, 1.45]."""
    def num(key, default):
        v = sig.get(key, default)
        return v if isinstance(v, (int, float)) else default

    # Engagement / responsiveness
    resp = num("recruiter_response_rate", 0.5)
    if resp == -1:
        resp = 0.5
    art = num("avg_response_time_hours", 48.0)
    art_score = 1.0 if art <= 12 else (0.8 if art <= 48 else (0.6 if art <= 120 else 0.4))
    iv = num("interview_completion_rate", 0.5)
    ofr = num("offer_acceptance_rate", 0.5)
    if ofr == -1:
        ofr = 0.5

    # Recruiter demand / visibility (genuinely predictive per signals doc)
    saved = num("saved_by_recruiters_30d", 0)
    views = num("profile_views_received_30d", 0)
    appears = num("search_appearance_30d", 0)
    apps = num("applications_submitted_30d", 0)
    demand = min(1.0, saved / 20.0) * 0.5 + min(1.0, views / 50.0) * 0.3 + min(1.0, appears / 50.0) * 0.2

    # Network / credibility
    conns = num("connection_count", 0)
    endors = num("endorsements_received", 0)
    network = min(1.0, conns / 500.0) * 0.5 + min(1.0, endors / 150.0) * 0.5

    # Trust / verification
    vf = (1 if sig.get("verified_email") else 0) + (1 if sig.get("verified_phone") else 0) + \
         (1 if sig.get("linkedin_connected") else 0)
    vf = vf / 3.0
    complete = num("profile_completeness_score", 50) / 100.0

    # Activity recency
    mi = _idle(sig.get("last_active_date"))
    rec = 1.0 if mi <= 3 else (0.85 if mi <= 6 else (0.6 if mi <= 12 else 0.4))

    # Availability intent
    avail = 0.0
    if sig.get("open_to_work_flag"):
        avail += 0.5
    np_days = num("notice_period_days", 90)
    avail += max(0.0, min(0.5, (90 - np_days) / 120.0))
    # apps submitted: some signal of active job-seeking
    avail += min(0.2, apps / 25.0)
    avail = min(1.0, avail)

    # --- Signal 20: preferred_work_mode ---
    # JD offices are Pune/Noida, hybrid in practice ("no required in-office days",
    # desks "mostly used Tue/Thu"). onsite/hybrid/flexible align with a founding
    # in-person team; pure remote is a mild misalignment (not a disqualifier).
    wm = str(sig.get("preferred_work_mode", "") or "").lower()
    wm_fit = {"onsite": 1.0, "hybrid": 1.0, "flexible": 0.9, "remote": 0.7}.get(wm, 0.85)

    # --- Signal 21: expected_salary_range_inr_lpa (seniority sanity) ---
    # Senior AI Engineer at a funded startup sits in a sensible LPA band. A very
    # low expectation usually signals a junior/mismatched profile; a very high one
    # signals over-level for the band. Peak fit ~14-32 LPA midpoint.
    esr = sig.get("expected_salary_range_inr_lpa", {}) or {}
    smin = esr.get("min") if isinstance(esr, dict) else None
    smax = esr.get("max") if isinstance(esr, dict) else None
    sal_fit = 0.85
    if isinstance(smin, (int, float)) and isinstance(smax, (int, float)):
        mid = (float(smin) + float(smax)) / 2.0
        if mid < 6:
            sal_fit = 0.55          # likely junior
        elif mid < 10:
            sal_fit = 0.75
        elif mid <= 32:
            sal_fit = 1.0           # in-band senior
        elif mid <= 45:
            sal_fit = 0.85          # high but plausible
        else:
            sal_fit = 0.7           # over-level
    fit_adj = 0.5 * wm_fit + 0.5 * sal_fit   # combined preference/seniority fit

    # --- Signal 22: skill_assessment_scores (verified competence) ---
    # Only ~20% have any; when present, a high average is a real credibility lift.
    assess = sig.get("skill_assessment_scores", {}) or {}
    if assess:
        avg_assess = sum(v for v in assess.values()
                         if isinstance(v, (int, float))) / max(1, len(assess))
        assess_score = max(0.0, min(1.0, avg_assess / 100.0))
        n_assess = min(1.0, len(assess) / 5.0)
        verified_skill = 0.7 * assess_score + 0.3 * n_assess
    else:
        verified_skill = 0.5        # neutral when unmeasured

    # --- Signal 23: signup_date (platform tenure, minor) ---
    # Weak by design: this is the Redrob join date, not activity (recency is
    # last_active_date, already used). Included for completeness with low weight.
    su_months = _idle(sig.get("signup_date"))   # months since signup
    tenure = max(0.0, min(1.0, su_months / 36.0))  # longer-tenured -> slightly +

    # Compose a multiplier. Engagement & availability dominate (JD: "actually
    # available"); demand/network/trust/fit are supporting; tenure is a whisper.
    engagement = 0.40 * resp + 0.20 * art_score + 0.20 * iv + 0.20 * ofr
    core = (0.30 * engagement + 0.20 * avail + 0.15 * demand +
            0.10 * network + 0.07 * vf + 0.05 * complete +
            0.06 * fit_adj + 0.05 * verified_skill + 0.02 * tenure)
    core = core * rec  # idle profiles pulled down
    # Map [0,1] core onto [0.55, 1.45]
    return round(0.55 + 0.90 * max(0.0, min(1.0, core)), 6)


class FeatureEngineer:
    def generate_features(self, profile, role_config, retrieval_score=0.0):
        vocab = role_config.get("vocabulary", {})
        fam_rx = role_config.get("family_regex", {})
        sig = profile.get("signals", {}) or {}
        skills = profile.get("skills", []) or []
        text = (profile.get("text", "") or "").lower()
        pr = profile.get("profile", {}) or {}
        assess = sig.get("skill_assessment_scores", {}) or {}

        fam_order = sorted(vocab.keys())
        wsum = sum(vocab[f].get("weight", 1.0) for f in fam_order) or 1.0
        lname = [((sk.get("name", "") or "").lower(), sk) for sk in skills]

        # --- Skill-based technical fit ---
        fs = 0.0
        for fam in fam_order:
            spec = vocab[fam]
            w = spec.get("weight", 1.0)
            rx = fam_rx.get(fam)
            best = 0.0
            if rx:
                for nm, sk in lname:
                    if nm and rx.search(nm):
                        stp = _PW.get(sk.get("proficiency", ""), 0.4)
                        en = min(1.0, (sk.get("endorsements", 0) or 0) / 50.0)
                        a = assess.get(sk.get("name", ""), None)
                        ab = (a / 100.0) if isinstance(a, (int, float)) and a >= 0 else 0.0
                        c = 0.5 * stp + 0.25 * en + 0.25 * ab
                        if c > best:
                            best = c
                if best == 0.0 and rx.search(text):
                    best = 0.25
            fs += w * best
        skill_fit = min(1.0, round(fs / wsum, 9))

        # --- Gap 2: career-history evidence ("says vs means") ---
        evidence = _career_evidence(profile)
        # technical_fit blends skills with real evidence; evidence can rescue a
        # thin skill list AND confirm a strong one.
        tech = min(1.0, round(0.62 * skill_fit + 0.38 * evidence, 9))

        # --- Production / shipper signal ---
        gh = sig.get("github_activity_score", -1)
        gs = (gh / 100.0) if isinstance(gh, (int, float)) and gh >= 0 else 0.0
        sc = role_config.get("shipper_cues", [])
        career_text = (profile.get("career_text", "") or "").lower()
        ship = min(1.0, sum(1 for c in sc if c in career_text) / 5.0)
        mlt = vocab.get("ml_core", {}).get("terms", []) + vocab.get("embeddings_retrieval", {}).get("terms", [])
        hml = any(any(t in (s.get("name", "") or "").lower() for t in mlt) for s in skills)
        prod = min(1.0, round(0.40 * gs + 0.40 * ship + (0.20 if hml else 0.0), 9))

        # --- Experience fit curve ---
        yoe = float(pr.get("years_of_experience", 0) or 0)
        ef = _exp_fit(yoe, role_config.get("exp_band", (5, 9)),
                      role_config.get("exp_soft_min", 3), role_config.get("exp_soft_max", 12))
        mi = _idle(sig.get("last_active_date"))
        rec = 1.0 if mi <= 3 else (0.85 if mi <= 6 else (0.65 if mi <= 12 else 0.45))
        cq = ef * rec
        tt = (pr.get("current_title", "") + " " + " ".join(profile.get("roles", []))).lower()
        ir = any(c in tt for c in role_config.get("research_title_cues", [])) or \
             any(c in text for c in role_config.get("research_content_cues", []))
        if ir and prod < 0.4:
            cq *= 0.55
        if any(c in tt for c in role_config.get("non_ic_title_cues", [])) and ship < 0.2:
            cq *= 0.75
        if prod >= 0.6:
            cq = min(1.0, cq * 1.10)
        cq = max(0.0, min(1.0, round(cq, 9)))

        # --- Gap 3: behavioral / availability multiplier (all 23 signals) ---
        beh_mult = _behavioral(sig)

        # --- Gap 4 & 5: location fit + validation bonus ---
        loc_mult = location_fit(profile)
        valid_bonus = validation_signal(profile)

        return {
            "technical_fit": round(tech, 6),
            "skill_fit": round(skill_fit, 6),
            "evidence": round(evidence, 6),
            "retrieval_score": round(float(retrieval_score), 6),
            "production_ml_score": round(prod, 6),
            "career_quality": round(cq, 6),
            "behavioral_mult": round(beh_mult, 6),
            "location_mult": round(loc_mult, 6),
            "validation_bonus": round(valid_bonus, 6),
        }
