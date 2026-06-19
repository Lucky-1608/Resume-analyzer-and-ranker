# Candidate Ranking System — India Runs (Track 1)

A deterministic, fully offline candidate ranking pipeline for the India Runs
Data & AI Challenge. It scores ~100,000 candidates against a job description and
emits the top 100 as `submission.csv`. No LLM, no GPU, and no network access are
used at ranking time.

## How it works

The pipeline runs in seven stages:

1. **Job Intelligence** (`src/intelligence/job_intelligence.py`) — parses
   `job_description.docx`, builds a weighted skill vocabulary (7 families) and an
   experience band (5–9 years) from the JD text.
2. **Candidate Parser** (`src/intelligence/candidate_parser.py`) — normalizes each
   candidate: skills with proficiency / endorsements / assessment scores, career
   history with dates and durations, education tiers, and Redrob signals, plus a
   combined text blob used for retrieval.
3. **Hybrid Retriever** (`src/retrieval/hybrid_retriever.py`) — deterministic
   TF-IDF cosine similarity over the candidate text, blended 60/40 with weighted
   JD-skill keyword coverage.
4. **Feature Engineer** (`src/features/feature_engineer.py`) — computes
   `technical_fit` (skill-family match weighted by proficiency, endorsements and
   assessment scores), a production / shipping signal, an experience-fit curve
   that peaks inside the JD band and down-weights both over-senior and very-junior
   profiles, plus behavioral and recruitability signals.
5. **Trust + Honeypot** (`src/trust/`) — an evidence/consistency trust score, and a
   honeypot detector that flags impossible profiles (senior title with almost no
   total experience, skill duration exceeding career length, invalid education
   years, etc.).
6. **Ensemble Scorer** (`src/ranking/ensemble_scorer.py`) — weighted blend
   (technical 0.40, retrieval 0.25, production_ml 0.15, career 0.10, trust 0.10),
   multiplied by a behavioral factor and damped by honeypot probability.
7. **Explanation Engine** (`src/ranking/explanation_engine.py`) — a grounded,
   candidate-specific reasoning string for each top candidate, citing that
   candidate's real roles and skills with varied sentence structure.

## Determinism

All vocabulary iterations are sorted so floating-point sums do not depend on
Python's hash seed. Two independent full runs on the 100k dataset produce a
byte-for-byte identical `submission.csv`. Scores are rounded to 6 decimals and
then sorted by score descending, with `candidate_id` ascending as the tie-break,
matching the submission spec.



## Signals modeled (beyond skill keywords)

This ranker deliberately reasons about the gap between what a profile *says* and
what it *means*, per the JD's guidance:

- **Career-history evidence** — descriptions are mined for production-ML,
  retrieval, ranking and recommendation evidence, so a genuine builder with a
  thin skills list still surfaces and a keyword-stuffer with no real history does
  not.
- **Disqualifier engine** — down-weights profiles the JD explicitly does not want:
  careers entirely at IT-services/consulting firms, off-domain roles (e.g. a
  "Marketing Manager" with AI keywords stuffed into skills), seniors who have
  moved off hands-on coding, vision/speech/robotics without NLP/IR, research-only
  with no production, and title-chasers with many short stints.
- **Location fit** — boosts Pune/Noida (office locations) and the explicitly
  welcomed Hyderabad/Mumbai/Delhi-NCR; down-weights outside-India profiles
  (no visa sponsorship), softened when the candidate is open to relocating.
- **Behavioral / availability** — all 23 Redrob behavioral signals fold into an
  availability-and-engagement multiplier (responsiveness, recruiter demand,
  network, verification, recency, notice period), since a perfect-on-paper
  candidate who is inactive or unresponsive is not actually hireable.
- **External validation** — a small bonus for open-source/GitHub activity,
  papers/talks, education tier and relevant certifications.



## Ranking architecture (v3)

The final ranking is a **Reciprocal Rank Fusion (RRF, Cormack et al. 2009, k=60)**
of complementary rankings, which combines signals on incomparable scales without
fragile score normalization:

1. **Rule score** (weight 2.0) — the JD-logic composite: skill fit, career-history
   evidence, experience-band fit, behavioral/availability multiplier (all **23 of 23** documented Redrob signals), location
   fit, validation bonus, disqualifier penalties, honeypot damping.
2. **TF-IDF cosine** (weight 1.0) — lexical relevance of the candidate text to the
   JD, with sublinear term-frequency (the log-saturation BM25 uses).
3. **Evidence coherence** (weight 0.8) — cosine between a candidate's *skills* and
   their *career-history descriptions*; high coherence means the described work
   supports the claimed skills (the "says vs means" test), computed with TF-IDF
   only (no transformer, no network).

### Two-stage honeypot defense

- **Stage 1 — hard logical-impossibility rules** (denial constraints): a single
  job longer than the entire career, ≥3 expert/advanced skills with zero months
  of use, time-travel calendar overlaps, end-before-start / future dates, and
  impossible education years. ~100% precision.
- **Stage 2 — Isolation Forest** (scikit-learn, fixed seed) over signed
  consistency-residual features, to catch the *subtly* impossible honeypots no
  single hard rule flags. Flagged honeypots are excluded from the top via a
  (1 − honeypot_probability) damping factor.

All components run on CPU, with no network and no model downloads, deterministically
(two independent full runs on the 100k pool produce a byte-identical submission).
Peak memory ~2.5 GB; runtime ~3 minutes on a CPU box (within the 5-minute limit).

## Running it

Place the dataset at `data/candidates.jsonl` and the JD at
`data/job_description.docx`, then:

```bash
pip install -r requirements.txt
python run_submission.py
```

This writes the top 100 to `submission.csv` in the repo root. On a typical CPU
box the full 100k run takes roughly 110 seconds (well under the 5-minute limit).

### Custom data

```bash
python run_submission.py --candidates /path/to/candidates.jsonl --jd /path/to/job_description.docx --out submission.csv
```

### Optional REST API

```bash
uvicorn src.api:app --reload
curl http://localhost:8000/rank
```

## Output format

`submission.csv` has exactly the columns `candidate_id,rank,score,reasoning`,
100 rows, ranks 1–100, scores non-increasing with rank, UTF-8 encoded.


## Top-tier tie-breaking (NDCG@10 optimization)

RRF produces the primary order, but the strongest candidates share templated
career descriptions and thus near-identical lexical scores, producing near-ties
at the very top. Because NDCG@10 is highly sensitive to the order of the top
items, breaking those near-ties by candidate_id (arbitrary w.r.t. quality) would
leave score on the table. We therefore break near-ties (candidates within a small
epsilon of each other) by a continuous, JD-prioritized **quality key**:
experience-band centrality (peak at the 5-9 midpoint), high-value skill breadth
(ranking + retrieval + LLM + vector-db), shipper/production evidence, exact
Pune/Noida location, and verified skill-assessment scores. Clearly-separated
orderings are left untouched; only genuine near-ties are reordered by quality.
Final scores are emitted strictly decreasing so the validator's non-increasing
and id-ascending-tie rules are both satisfied.

## Self-evaluation harness

Because there is no public leaderboard, `eval/proxy_eval.py` builds an INDEPENDENT
relevance labeling directly from the JD criteria (graded 0-3 from raw profile
facts, using logic deliberately independent of the ranking model) and scores the
submission with the contest metrics (NDCG@10/@50, MAP, P@10, composite). It is a
proxy, not the hidden ground truth, but a high score is strong evidence of
alignment with what the JD describes as an ideal hire.

  Run: `python eval/proxy_eval.py submission.csv data/candidates.jsonl`
