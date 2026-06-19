import argparse,os,sys
ROOT=os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path: sys.path.insert(0,ROOT)
from src.api import rank_candidates
from src.pipeline.submission_gen import SubmissionGenerator

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidates"); ap.add_argument("--jd"); ap.add_argument("--out",default="submission.csv")
    args=ap.parse_args()
    print("Ranking candidates...")
    result=rank_candidates(custom_jd_path=args.jd,custom_cand_path=args.candidates)
    cands=result.get("candidates",[])
    print(f"Ranked {len(cands)} candidates in {result.get('elapsed_time',0):.1f}s.")
    SubmissionGenerator(args.out,top_n=100).generate(cands)
    print(f"Done -> {args.out}")

if __name__=="__main__": main()
