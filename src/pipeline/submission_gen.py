import csv, re

_ID_RE = re.compile(r"^CAND_[0-9]{7}$")

HEADER=["candidate_id","rank","score","reasoning"]
SD=6

class SubmissionGenerator:
    def __init__(self, output_path="submission.csv", top_n=100):
        self.output_path=output_path; self.top_n=top_n

    def generate(self, candidates):
        seen=set(); deduped=[]
        for c in candidates:
            cid=str(c.get("id") or "")
            # Only emit ids that satisfy the official format; guarantees a
            # validator-clean file even if upstream ever yields a malformed id.
            if cid and _ID_RE.match(cid) and cid not in seen:
                seen.add(cid); deduped.append(c)
        rows=[]
        for c in deduped:
            score=round(float(c.get("finalScore",c.get("final_score",0.0)) or 0.0),SD)
            order=c.get("_order", 10**9)
            reason=str(c.get("aiFit_explanation",c.get("explanation","")) or "Relevant candidate.").replace(chr(10)," ").replace(chr(13)," ").strip()
            rows.append((str(c.get("id") or ""),score,reason,order))
        # Preserve api.py's quality-driven order: sort by score desc, then by the
        # order index api assigned (which already encodes the tie-break), then id.
        rows.sort(key=lambda x:(-x[1],x[3],x[0]))
        with open(self.output_path,"w",newline="",encoding="utf-8") as f:
            wr=csv.writer(f); wr.writerow(HEADER)
            for rank,(cid,score,reason,order) in enumerate(rows[:self.top_n],start=1):
                wr.writerow([cid,rank,f"{score:.{SD}f}",reason])
        print(f"Wrote {min(len(rows),self.top_n)} rows to {self.output_path}.")
