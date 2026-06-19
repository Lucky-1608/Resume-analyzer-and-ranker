class EvidenceGraph:
    def compute_evidence_score(self, profile):
        skills=profile.get("skills",[]) or []
        if not skills: return 0.3
        desc=" ".join((ch.get("description","") or "") for ch in profile.get("career",[]) or []).lower()
        cap=max(sum(max(0,ch.get("duration_months",0) or 0) for ch in profile.get("career",[]) or []),float(profile.get("profile",{}).get("years_of_experience",0) or 0)*12,1.0)
        top=sorted(skills,key=lambda s:(s.get("endorsements",0),s.get("duration_months",0)),reverse=True)[:8]
        sup=0.0
        for s in top:
            n=(s.get("name","") or "").lower()
            if not n: continue
            bt=n in desc; tp=(s.get("duration_months",0) or 0)<=cap+6
            sup+=(1.0 if bt and tp else 0.5 if bt or tp else 0.0)
        return round(min(1.0,0.25+0.75*(sup/len(top))),6)
