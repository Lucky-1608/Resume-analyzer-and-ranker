import json


class CandidateParser:
    def __init__(self, candidates_file=None, raw_candidates=None):
        self.candidates_file = candidates_file
        self.raw_candidates = raw_candidates
        self._profiles = []

    def _iter_raw(self):
        if self.raw_candidates:
            for c in self.raw_candidates:
                yield c
            return
        with open(self.candidates_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except Exception:
                        pass

    @staticmethod
    def _structure(c):
        p = c.get("profile", {}) or {}
        sig = c.get("redrob_signals", {}) or {}

        skills = [{
            "name": s.get("name", "") or "",
            "proficiency": (s.get("proficiency", "") or "").lower(),
            "endorsements": s.get("endorsements", 0) or 0,
            "duration_months": s.get("duration_months", 0) or 0,
        } for s in (c.get("skills", []) or []) if isinstance(s, dict)]

        career = [{
            "company": ch.get("company", "") or "",
            "title": ch.get("title", "") or "",
            "start_date": ch.get("start_date"),
            "end_date": ch.get("end_date"),
            "duration_months": ch.get("duration_months", 0) or 0,
            "is_current": bool(ch.get("is_current")),
            "industry": ch.get("industry", "") or "",
            "company_size": ch.get("company_size", "") or "",
            "description": ch.get("description", "") or "",
        } for ch in (c.get("career_history", []) or []) if isinstance(ch, dict)]

        education = [{
            "institution": e.get("institution", "") or "",
            "degree": e.get("degree", "") or "",
            "field_of_study": e.get("field_of_study", "") or "",
            "start_year": e.get("start_year"),
            "end_year": e.get("end_year"),
            "tier": e.get("tier", "unknown") or "unknown",
        } for e in (c.get("education", []) or []) if isinstance(e, dict)]

        certifications = [{
            "name": cert.get("name", "") or "",
            "issuer": cert.get("issuer", "") or "",
            "year": cert.get("year"),
        } for cert in (c.get("certifications", []) or []) if isinstance(cert, dict)]

        roles = [ch["title"] for ch in career if ch.get("title")]

        blob = " ".join([
            p.get("headline", "") or "",
            p.get("summary", "") or "",
            " ".join((s["name"] or "") for s in skills),
            " ".join(roles),
            " ".join((ch["description"] or "") for ch in career),
        ]).strip()

        # Career-history description text only (for "says vs means" evidence mining)
        career_text = " ".join((ch["description"] or "") for ch in career).strip()

        return {
            "id": c.get("candidate_id"),
            "profile": {
                "years_of_experience": p.get("years_of_experience", 0) or 0,
                "current_title": p.get("current_title", "") or "",
                "current_company": p.get("current_company", "") or "",
                "current_company_size": p.get("current_company_size", "") or "",
                "current_industry": p.get("current_industry", "") or "",
                "summary": p.get("summary", "") or "",
                "headline": p.get("headline", "") or "",
                "location": p.get("location", "") or "",
                "country": p.get("country", "") or "",
            },
            "skills": skills,
            "career": career,
            "roles": roles,
            "education": education,
            "certifications": certifications,
            "signals": sig,
            "text": blob,
            "career_text": career_text,
        }

    def load_candidates(self):
        self._profiles = [self._structure(c) for c in self._iter_raw()]
        print(f"Loaded {len(self._profiles)} candidates.")
        return self._profiles

    def extract_profiles(self):
        if not self._profiles:
            self.load_candidates()
        return self._profiles
