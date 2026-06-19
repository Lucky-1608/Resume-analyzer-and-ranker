# Two-stage honeypot detector.
#
# Stage 1 - hard logical-impossibility rules (denial constraints). Each is a
#   genuine impossibility, so precision is ~100% (no real candidate can satisfy
#   them). Returns prob >= 0.95.
#     - a single job longer than the entire stated career
#     - >=3 expert/advanced skills with 0 months of use
#     - total tenure overlapping beyond the calendar span (time-travel)
#     - end-before-start / future start dates
#     - impossible education years
#
# Stage 2 - statistical anomaly score on consistency-residual features
#   (filled in by HoneypotScreener via Isolation Forest at the dataset level).
#   This module exposes per-profile residual features and a soft rule score for
#   the "subtly impossible" honeypots that no single hard rule catches.
#
# IMPORTANT: the previous version flagged ~9,200 candidates because it compared
# skill duration against years_of_experience*12 - skills legitimately predate the
# current role, so that rule is removed. The rules below flag ~42/100k (the truly
# impossible), matching the documented honeypot population far better.

from datetime import date, datetime


def _pd(s):
    if not s or not isinstance(s, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    return None


def residual_features(profile):
    """Signed consistency residuals; ~0 for clean profiles, large for honeypots.
    Used by the Isolation-Forest screener (Stage 2)."""
    pr = profile.get("profile", {}) or {}
    yoe = float(pr.get("years_of_experience", 0) or 0)
    career = profile.get("career", []) or []
    skills = profile.get("skills", []) or []
    today = date.today()

    # job vs career
    max_job = max([(ch.get("duration_months", 0) or 0) for ch in career], default=0)
    job_minus_career = max(0.0, max_job - (yoe * 12 + 6))

    # expert-with-zero-use count
    expert0 = sum(1 for s in skills
                  if s.get("proficiency") in ("expert", "advanced")
                  and (s.get("duration_months", 0) or 0) == 0)

    # calendar overlap
    starts = [_pd(ch.get("start_date")) for ch in career if _pd(ch.get("start_date"))]
    overlap = 0.0
    if starts:
        ends = [(_pd(ch.get("end_date")) or today) for ch in career if _pd(ch.get("start_date"))]
        span = (max(ends).year - min(starts).year) * 12 + (max(ends).month - min(starts).month)
        tot = sum(max(0, ch.get("duration_months", 0) or 0) for ch in career)
        if span > 0:
            overlap = max(0.0, tot - span)

    # seniority vs experience residual
    sw = ("senior", "staff", "principal", "lead", "head", "director", "vp ", "chief")
    senior_titles = sum(1 for ch in career
                        if any(w in (ch.get("title", "") or "").lower() for w in sw))
    senior_low = float(senior_titles) if yoe < 2 else 0.0

    # skill assessment vs experience (expert claims with low assessment)
    sig = profile.get("signals", {}) or {}
    assess = sig.get("skill_assessment_scores", {}) or {}
    avg_assess = (sum(assess.values()) / len(assess)) if assess else 50.0
    expert_cnt = sum(1 for s in skills if s.get("proficiency") in ("expert", "advanced"))
    overclaim = float(expert_cnt) if (yoe < 2 and expert_cnt >= 8 and avg_assess > 90) else 0.0

    return [job_minus_career, float(expert0), overlap, senior_low, overclaim]


class HoneypotDetector:
    def detect_honeypot(self, profile):
        """Stage-1 hard rules only. Returns 0.97 if impossible, else a small base."""
        pr = profile.get("profile", {}) or {}
        yoe = float(pr.get("years_of_experience", 0) or 0)
        career = profile.get("career", []) or []
        skills = profile.get("skills", []) or []
        education = profile.get("education", []) or []
        today = date.today()

        # 1) a single job longer than entire stated career (+6mo tolerance)
        for ch in career:
            d = ch.get("duration_months", 0) or 0
            if yoe > 0 and d > yoe * 12 + 6:
                return 0.97
            sd = _pd(ch.get("start_date"))
            ed = _pd(ch.get("end_date"))
            if sd and ed and ed < sd:
                return 0.97
            if sd and sd > today:
                return 0.97

        # 2) >=3 expert/advanced skills with zero months of use
        expert0 = sum(1 for s in skills
                      if s.get("proficiency") in ("expert", "advanced")
                      and (s.get("duration_months", 0) or 0) == 0)
        if expert0 >= 3:
            return 0.97

        # 3) impossible calendar overlap (time-travel): total tenure exceeds span
        starts = [_pd(ch.get("start_date")) for ch in career if _pd(ch.get("start_date"))]
        if starts:
            ends = [(_pd(ch.get("end_date")) or today) for ch in career if _pd(ch.get("start_date"))]
            span = (max(ends).year - min(starts).year) * 12 + (max(ends).month - min(starts).month)
            tot = sum(max(0, ch.get("duration_months", 0) or 0) for ch in career)
            if span > 0 and tot > span + 18:
                return 0.97

        # 4) impossible education years
        for e in education:
            sy, ey = e.get("start_year"), e.get("end_year")
            if isinstance(sy, int) and isinstance(ey, int) and (ey < sy or sy < 1960 or ey > today.year + 6):
                return 0.97

        # not hard-impossible; small base prob (Stage-2 screener may raise it)
        return 0.03
