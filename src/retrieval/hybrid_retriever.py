# Hybrid retriever: deterministic, memory-lean lexical + semantic-lite signals.
#   - TF-IDF cosine with sublinear TF (the log-saturation that gives BM25-like
#     behavior) -> primary lexical relevance
#   - Weighted JD-skill keyword coverage
#   - Skill<->career-description evidence coherence (TF-IDF cosine), the
#     "says vs means" signal computed with NO transformer and NO huge matrices.
# No network, no model downloads. All iterations sorted -> hash-seed independent.
#
# NOTE: a standalone Okapi BM25 object over 100k tokenized docs costs ~1.8GB on
# top of everything else; we instead use TF-IDF with sublinear_tf=True, which
# applies the same 1+log(tf) term-frequency saturation BM25 uses, at a fraction
# of the memory. This keeps the ranking step comfortably inside the RAM budget.
import re
from typing import List, Dict, Any


class HybridRetriever:
    def __init__(self):
        self.profiles = []
        self._vectorizer = None
        self._matrix = None
        self._ids = []

    def fit(self, profiles):
        self.profiles = profiles
        self._ids = [p["id"] for p in profiles]
        from sklearn.feature_extraction.text import TfidfVectorizer
        corpus = [(p.get("text", "") or "") for p in profiles]
        self._vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 1),
            min_df=3, max_features=30000, sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(corpus)
        print(f"Hybrid retriever fit on {len(profiles)} candidates "
              f"(tfidf_vocab={len(self._vectorizer.vocabulary_)}).")

    def _build_query(self, role_config, jd_text=""):
        parts = []
        for fam in sorted(role_config.get("vocabulary", {}).keys()):
            spec = role_config["vocabulary"][fam]
            reps = max(1, int(round(spec.get("weight", 1.0))))
            parts.extend(spec.get("terms", []) * reps)
        if jd_text:
            parts.append(jd_text)
        return " ".join(parts)

    def _evidence_coherence(self):
        """Row-wise cosine between SKILLS text and career DESCRIPTION text in a
        shared TF-IDF space. High = described work supports claimed skills."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        import numpy as np
        skills_docs = [" ".join(s.get("name", "") for s in (p.get("skills", []) or []))
                       for p in self.profiles]
        desc_docs = [(p.get("career_text", "") or "") for p in self.profiles]
        vec = TfidfVectorizer(lowercase=True, stop_words="english",
                              ngram_range=(1, 1), min_df=3, max_features=20000,
                              sublinear_tf=True)
        vec.fit(skills_docs + desc_docs)
        S = normalize(vec.transform(skills_docs))
        D = normalize(vec.transform(desc_docs))
        num = np.asarray(S.multiply(D).sum(axis=1)).ravel()
        out = {}
        for i, cid in enumerate(self._ids):
            out[cid] = round(max(0.0, min(1.0, float(num[i]))), 6)
        return out

    def score_all(self, role_config, jd_text=""):
        """Return cid -> {tfidf, coverage, evidence} raw signals in [0,1]."""
        from sklearn.metrics.pairwise import linear_kernel
        q = self._vectorizer.transform([self._build_query(role_config, jd_text)])
        cos = linear_kernel(q, self._matrix).ravel()
        cmax = float(cos.max()) if cos.size and cos.max() > 0 else 1.0

        vocab = role_config.get("vocabulary", {})
        fam_rx = role_config.get("family_regex", {})
        fam_order = sorted(vocab.keys())
        wsum = sum(vocab[f].get("weight", 1.0) for f in fam_order) or 1.0

        evid = self._evidence_coherence()

        out = {}
        for i, p in enumerate(self.profiles):
            text = (p.get("text", "") or "").lower()
            cov = 0.0
            for f in fam_order:
                rx = fam_rx.get(f)
                if rx and rx.search(text):
                    cov += vocab[f].get("weight", 1.0)
            out[p["id"]] = {
                "tfidf": round(cos[i] / cmax, 6),
                "coverage": round(cov / wsum, 6),
                "evidence": evid.get(p["id"], 0.0),
            }
        return out
