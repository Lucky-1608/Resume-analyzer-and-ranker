# Proxy evaluation harness.
#
# There is no public leaderboard and the true ground truth is hidden, so we build
# an INDEPENDENT relevance labeling directly from the JD's stated criteria and
# score our submission against it with the same metrics the contest uses
# (NDCG@10, NDCG@50, MAP, P@10) plus the documented composite.
#
# This is NOT the organizers' ground truth. It is a defensible, transparent proxy:
# if our ranking scores high here, it is strong evidence we are aligned with what
# the JD describes as an ideal hire. The labeling logic below is deliberately
# INDEPENDENT of the ranking model's scoring (it uses raw profile facts and simple
# rules), so a high score is not circular.
#
# Relevance grades (0..3), assigned from raw facts only:
#   3 = ideal: on-domain senior title, in 5-9 band, AI/ML/IR shipper evidence,
#       India, not disqualified, not a honeypot
#   2 = strong: on-domain, near band (4-11), some AI/ML evidence, India
#   1 = marginal: on-domain-ish OR adjacent, weak evidence
#   0 = irrelevant / off-domain / honeypot / disqualified
#
# Metrics computed on the submitted top-100 order.

import csv, json, math, sys, os, io, contextlib, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime

ON_DOMAIN = ["machine learning","ml engineer","ai engineer","ai specialist",
             "data scientist","applied scientist","research engineer","nlp engineer",
             "search engineer","recommendation","analytics engineer","ml scientist"]
AIML_EVIDENCE = ["rag","ranking","retrieval","recommendation","embedding","semantic search",
                 "llm","fine-tun","learning to rank","vector","information retrieval",
                 "transformer","neural","deep learning","nlp"]
SHIP = ["production","deployed","at scale","serving","shipped","latency","throughput",
        "millions","50m","10m","real-time","end-to-end","pipeline"]
OFF_DOMAIN = ["marketing","hr manager","human resources","sales ","accountant",
              "civil engineer","mechanical engineer","content writer","graphic designer",
              "customer support","operations manager","business analyst","recruiter",
              "supply chain","financial analyst","project manager"]
CONSULTING = {"tcs","tata consultancy","infosys","wipro","accenture","cognizant",
              "capgemini","hcl","tech mahindra","mphasis","mindtree","ltimindtree",
              "deloitte","ibm","dxc","hexaware","birlasoft","coforge"}
PRODUCT = {"swiggy","zomato","flipkart","razorpay","cred","meesho","nykaa","phonepe",
           "paytm","ola","uber","google","meta","amazon","microsoft","netflix",
           "linkedin","myntra","hooli","pied piper","stark industries","wayne enterprises",
           "globex","initech","acme corp"}


def _pd(s):
    if not s or not isinstance(s,str): return None
    for f in ("%Y-%m-%d","%Y-%m"):
        try: return datetime.strptime(s,f).date()
        except: pass
    return None


def _is_honeypot(c):
    pr=c.get("profile",{}) or {}
    yoe=float(pr.get("years_of_experience",0) or 0)
    career=c.get("career_history",[]) or c.get("career",[]) or []
    skills=c.get("skills",[]) or []
    today=date.today()
    for ch in career:
        d=ch.get("duration_months",0) or 0
        if yoe>0 and d>yoe*12+6: return True
        sd=_pd(ch.get("start_date")); ed=_pd(ch.get("end_date"))
        if sd and ed and ed<sd: return True
        if sd and sd>today: return True
    expert0=sum(1 for s in skills if s.get("proficiency") in ("expert","advanced") and (s.get("duration_months",0) or 0)==0)
    if expert0>=3: return True
    starts=[_pd(ch.get("start_date")) for ch in career if _pd(ch.get("start_date"))]
    if starts:
        ends=[(_pd(ch.get("end_date")) or today) for ch in career if _pd(ch.get("start_date"))]
        span=(max(ends).year-min(starts).year)*12+(max(ends).month-min(starts).month)
        tot=sum(max(0,ch.get("duration_months",0) or 0) for ch in career)
        if span>0 and tot>span+18: return True
    return False


# Extra vocab for fine-grained differentiation within the ideal tier.
VECTOR_DB=["pinecone","milvus","qdrant","weaviate","faiss","opensearch","elasticsearch","chroma","pgvector"]
RANK_TERMS=["ranking","learning to rank","ltr","re-ranking","reranking","recommendation","relevance"]


def relevance_grade(c):
    """Coarse 0-3 grade (used for binary MAP / P@10)."""
    pr=c.get("profile",{}) or {}
    title=(pr.get("current_title","") or "").lower()
    yoe=float(pr.get("years_of_experience",0) or 0)
    country=(pr.get("country","") or "").lower()
    career=c.get("career_history",[]) or []
    skills_text=" ".join((s.get("name","") or "").lower() for s in (c.get("skills",[]) or []))
    desc=" ".join((ch.get("description","") or "").lower() for ch in career)
    blob=skills_text+" "+desc

    if _is_honeypot(c): return 0
    titles=[(ch.get("title","") or "").lower() for ch in career]+[title]
    on_dom=any(any(d in t for d in ON_DOMAIN) for t in titles)
    off_dom=any(any(d in t for d in OFF_DOMAIN) for t in titles)
    if off_dom and not on_dom: return 0
    comps=[(ch.get("company","") or "").lower() for ch in career]
    if comps and all(any(f in cc for f in CONSULTING) for cc in comps) and not any(any(p==cc or p in cc for p in PRODUCT) for cc in comps):
        return 0
    if country and country!="india":
        return 1 if on_dom else 0
    aiml=sum(1 for k in AIML_EVIDENCE if k in blob)
    ship=sum(1 for k in SHIP if k in desc)
    in_band = 5<=yoe<=9
    near_band = 4<=yoe<=11
    if on_dom and in_band and aiml>=2 and ship>=1:
        return 3
    if on_dom and near_band and aiml>=1:
        return 2
    if on_dom or aiml>=1:
        return 1
    return 0


def fine_gain(c):
    """Continuous relevance gain (0..~15) for NDCG, so ordering WITHIN the ideal
    tier is actually tested rather than everyone collapsing to grade 3. Built only
    from raw, countable profile facts (independent of the ranking model's scoring,
    so high NDCG here is not circular). Scale chosen so a clearly-stronger ideal
    candidate outranks a marginally-ideal one in DCG."""
    g=relevance_grade(c)
    if g==0: return 0.0
    pr=c.get("profile",{}) or {}
    yoe=float(pr.get("years_of_experience",0) or 0)
    career=c.get("career_history",[]) or []
    skills_text=" ".join((s.get("name","") or "").lower() for s in (c.get("skills",[]) or []))
    desc=" ".join((ch.get("description","") or "").lower() for ch in career)
    blob=skills_text+" "+desc
    loc=(pr.get("location","") or "").lower()

    aiml=sum(1 for k in AIML_EVIDENCE if k in blob)          # 0..15
    ship=sum(1 for k in SHIP if k in desc)                   # 0..13
    vec=sum(1 for k in VECTOR_DB if k in skills_text)        # 0..9
    rnk=sum(1 for k in RANK_TERMS if k in blob)              # 0..7
    band_center=max(0.0, 1.0-abs(yoe-7.0)/2.0)               # 0..1 (peak yoe=7)
    hub=1.0 if ("pune" in loc or "noida" in loc) else 0.0
    sig=c.get("redrob_signals",{}) or {}
    assess=sig.get("skill_assessment_scores",{}) or {}
    avg_assess=(sum(v for v in assess.values() if isinstance(v,(int,float)))/len(assess)/100.0) if assess else 0.5

    # Base gain by coarse grade keeps tier separation; fine terms order within tier.
    base={1:1.0, 2:3.0, 3:6.0}[g]
    fine=(0.9*min(aiml,8) + 0.8*min(ship,6) + 0.7*min(vec,4) +
          1.0*min(rnk,4) + 1.5*band_center + 0.6*hub + 0.5*avg_assess)
    return round(base+fine, 6)


def dcg(grades):
    return sum((2**g-1)/math.log2(i+2) for i,g in enumerate(grades))


def ndcg_at(grades, ideal_grades, k):
    d=dcg(grades[:k]); idcg=dcg(sorted(ideal_grades,reverse=True)[:k])
    return d/idcg if idcg>0 else 0.0


def _dcg_lin(gains):
    import math
    return sum(g/math.log2(i+2) for i,g in enumerate(gains))


def ndcg_fine(gains, ideal_gains, k):
    """NDCG with real-valued (linear) gains, so ordering within the ideal tier
    is tested. Ideal = the top-k gains sorted desc across the whole pool."""
    d=_dcg_lin(gains[:k]); idcg=_dcg_lin(sorted(ideal_gains,reverse=True)[:k])
    return d/idcg if idcg>0 else 0.0


def main():
    sub=sys.argv[1] if len(sys.argv)>1 else "submission.csv"
    dsf=sys.argv[2] if len(sys.argv)>2 else "data/candidates.jsonl"
    rows=list(csv.DictReader(open(sub,encoding="utf-8")))
    sub_ids=[r["candidate_id"] for r in rows]

    # grade ALL candidates (for ideal ranking) — stream to keep memory low
    all_grades=[]; all_fine=[]; sub_grade_map={}; sub_fine_map={}
    sub_set=set(sub_ids)
    for line in open(dsf,encoding="utf-8"):
        if not line.strip(): continue
        c=json.loads(line)
        g=relevance_grade(c); fg=fine_gain(c)
        all_grades.append(g); all_fine.append(fg)
        if c["candidate_id"] in sub_set:
            sub_grade_map[c["candidate_id"]]=g
            sub_fine_map[c["candidate_id"]]=fg

    sub_grades=[sub_grade_map.get(cid,0) for cid in sub_ids]
    sub_fine=[sub_fine_map.get(cid,0.0) for cid in sub_ids]

    # NDCG uses the CONTINUOUS fine gain so ordering within the ideal tier matters.
    ndcg10=ndcg_fine(sub_fine, all_fine, 10)
    ndcg50=ndcg_fine(sub_fine, all_fine, 50)
    # MAP and P@10 use binary relevance (grade>=2 = relevant)
    rel=[1 if g>=2 else 0 for g in sub_grades]
    total_rel=sum(1 for g in all_grades if g>=2)
    # average precision over the top 100
    hits=0; ap=0.0
    for i,r in enumerate(rel):
        if r: hits+=1; ap+=hits/(i+1)
    ap = ap/min(total_rel,100) if total_rel>0 else 0.0
    p10=sum(rel[:10])/10.0

    composite=0.50*ndcg10+0.30*ndcg50+0.15*ap+0.05*p10

    print("="*56)
    print("PROXY EVALUATION (independent JD-derived ground truth)")
    print("="*56)
    print(f"  Relevant candidates in pool (grade>=2): {total_rel}")
    print(f"  Grade-3 (ideal) in pool:                {sum(1 for g in all_grades if g==3)}")
    print()
    print(f"  Top-10 grades: {sub_grades[:10]}")
    print(f"  Top-10 ideal:  {sorted(all_grades,reverse=True)[:10]}")
    print()
    print(f"  NDCG@10 = {ndcg10:.4f}   (weight 0.50)")
    print(f"  NDCG@50 = {ndcg50:.4f}   (weight 0.30)")
    print(f"  MAP     = {ap:.4f}   (weight 0.15)")
    print(f"  P@10    = {p10:.4f}   (weight 0.05)")
    print(f"  ------------------------------------")
    print(f"  COMPOSITE = {composite:.4f}")
    print("="*56)


if __name__=="__main__":
    main()
