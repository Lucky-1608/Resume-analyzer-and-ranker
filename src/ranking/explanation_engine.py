# JD-aware, grounded, unique reasoning generator (Gap 6).
# - Cites only the candidate's REAL skills/roles (no hallucination).
# - References the actual fit logic in JD terms (product vs services, location,
#   availability, evidence, over/under-seniority, flags).
# - Varied sentence structure; never inserts the candidate's name.

_DQ_PHRASES = {
    "services_only_career": "career has been mostly at IT-services firms (JD prefers product-company experience)",
    "services_industry_only": "primarily services-industry background",
    "off_domain_role": "current role is outside the AI/ML domain",
    "stopped_coding": "recent roles look lead/architecture rather than hands-on coding",
    "cv_speech_without_nlp": "vision/speech focus with limited NLP/IR exposure",
    "research_without_production": "research-leaning with limited production deployment",
    "job_hopper": "frequent short stints",
}


def _r1(x):
    try:
        return f"{float(x):.1f}"
    except Exception:
        return str(x)


class ExplanationEngine:
    def generate_explanation(self, profile, features, honeypot_prob=0.0, dq_tags=None):
        dq_tags = dq_tags or []
        pr = profile.get("profile", {}) or {}
        title = pr.get("current_title") or (profile.get("roles") or ["Professional"])[0]
        yoe = pr.get("years_of_experience", 0) or 0
        sig = profile.get("signals", {}) or {}
        loc = pr.get("location", "") or ""
        country = pr.get("country", "") or ""

        skills = sorted(profile.get("skills", []) or [],
                        key=lambda s: (s.get("endorsements", 0), s.get("duration_months", 0)),
                        reverse=True)
        cited = [s.get("name", "") for s in skills if s.get("name")][:3]
        st = ", ".join(cited) if cited else "a general skill set"

        tech = features.get("technical_fit", 0.0)
        evidence = features.get("evidence", 0.0)
        prod = features.get("production_ml_score", 0.0)
        cq = features.get("career_quality", 0.0)

        # Opener varies by strength AND by a deterministic per-candidate index so
        # the top list doesn't read as one repeated template (Stage-4 robustness).
        # Determinism: index derived from candidate_id via a stable checksum, not
        # Python's hash(), so it is identical across runs / hash seeds.
        cid = str(profile.get("id", ""))
        vidx = sum((i + 1) * ord(ch) for i, ch in enumerate(cid)) % 3
        if tech >= 0.6:
            templates = [
                f"Strong fit: {title} with {_r1(yoe)} yrs and directly relevant {st}",
                f"{title} ({_r1(yoe)} yrs) is a strong match, with directly relevant {st}",
                f"Strong candidate — {_r1(yoe)}-yr {title} whose core skills ({st}) map directly to the role",
            ]
        elif tech >= 0.38:
            templates = [
                f"Solid match: {title} ({_r1(yoe)} yrs) with relevant {st}",
                f"{title} with {_r1(yoe)} yrs brings relevant {st}",
                f"Reasonable fit — {_r1(yoe)}-yr {title} with relevant {st}",
            ]
        else:
            templates = [
                f"Adjacent profile: {title} ({_r1(yoe)} yrs); partial overlap via {st}",
                f"{title} ({_r1(yoe)} yrs) overlaps partially through {st}",
                f"Partial match — {_r1(yoe)}-yr {title}; some relevant exposure via {st}",
            ]
        bits = [templates[vidx]]

        # Evidence from career history (the "says vs means" signal).
        if evidence >= 0.5:
            ev_phr = ["career history shows hands-on retrieval/ranking/recommendation work",
                      "described work covers hands-on retrieval/ranking/recommendation",
                      "track record includes hands-on ranking/retrieval delivery"]
            bits.append(ev_phr[vidx])
        elif evidence >= 0.3:
            ev_phr = ["career history shows applied ML/production work",
                      "described work reflects applied ML in production",
                      "background includes applied ML delivery"]
            bits.append(ev_phr[vidx])

        # Production / shipping.
        gh = sig.get("github_activity_score", -1)
        if prod >= 0.6:
            bits.append("clear production/shipping signal")
        elif isinstance(gh, (int, float)) and 0 <= gh < 20 and prod < 0.4:
            bits.append("limited recent production signal")

        # Seniority band commentary.
        if cq < 0.45 and yoe and float(yoe) > 12:
            bits.append("seniority above the role's 5-9 yr target band")
        elif cq < 0.45 and yoe and float(yoe) < 3:
            bits.append("earlier-career for the role band")

        # Location (JD-named preference).
        ll = loc.lower()
        if any(h in ll for h in ["pune", "noida"]):
            bits.append(f"based in {loc.split(',')[0]} (matches office location)")
        elif any(h in ll for h in ["hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "bangalore", "bengaluru"]):
            bits.append(f"{loc.split(',')[0]}-based")
        elif country and country.lower() != "india":
            if sig.get("willing_to_relocate"):
                bits.append(f"based outside India ({country}) but open to relocating")
            else:
                bits.append(f"based outside India ({country}); no visa sponsorship offered")

        # Availability / engagement.
        resp = float(sig.get("recruiter_response_rate", 0.5) or 0.5)
        if sig.get("open_to_work_flag") and resp >= 0.7:
            bits.append("actively open and responsive")
        elif resp < 0.2:
            bits.append("low recruiter responsiveness")
        np_days = sig.get("notice_period_days", 90) or 90
        if isinstance(np_days, (int, float)) and np_days <= 30:
            bits.append("short notice period")

        # Disqualifier context (one, the most salient).
        for t in dq_tags:
            if t in _DQ_PHRASES:
                bits.append(_DQ_PHRASES[t])
                break

        # Honeypot flag.
        if honeypot_prob >= 0.5:
            bits.append("profile has timeline inconsistencies (flagged)")

        s = "; ".join(bits) + "."
        return s[0].upper() + s[1:]
