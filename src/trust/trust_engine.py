import math
from src.trust.evidence_graph import EvidenceGraph

class TrustEngine:
    def __init__(self): self._ev=EvidenceGraph()

    def calculate_trust_score(self, profile):
        sig=profile.get("signals",{}) or {}; skills=profile.get("skills",[]) or []
        ev=self._ev.compute_evidence_score(profile)
        end=min(1.0,math.log1p(sum(s.get("endorsements",0) or 0 for s in skills))/math.log1p(200))
        assess=sig.get("skill_assessment_scores",{}) or {}
        backed=0.0; chk=0
        for s in skills:
            if s.get("proficiency","") in ("advanced","expert"):
                chk+=1; a=assess.get(s.get("name",""),None)
                backed+=(1.0 if isinstance(a,(int,float)) and a>=70 else 0.5 if isinstance(a,(int,float)) and a>=50 else 0.0)
        cl=backed/chk if chk else 0.6
        vf=sum([1 if sig.get("verified_email") else 0,1 if sig.get("verified_phone") else 0,1 if sig.get("linkedin_connected") else 0])/3.0
        cp=float(sig.get("profile_completeness_score",50) or 50)/100.0
        return round(max(0.0,min(1.0,0.35*ev+0.20*end+0.20*cl+0.10*vf+0.15*cp)),6)
