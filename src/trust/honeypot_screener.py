# Stage-2 statistical honeypot screener (Isolation Forest).
#
# Runs once over the whole candidate pool on the consistency-residual features
# from honeypot_detector.residual_features. Catches "subtly impossible" honeypots
# that no single hard rule flags. Deterministic (random_state fixed).
#
# Strategy (per the research report): do NOT trust the contamination parameter;
# rank by anomaly score and flag the worst small fraction, sized to the known
# honeypot budget. We only ADD flags on top of the hard rules and keep the extra
# set small to avoid false-positiving real candidates.

import numpy as np
from src.trust.honeypot_detector import residual_features, HoneypotDetector


class HoneypotScreener:
    def __init__(self, extra_flag_fraction=0.0010, random_state=0):
        # extra_flag_fraction: fraction of the pool to additionally flag by IF
        # (0.10% ~ 100 on 100k; combined with hard rules and capped below).
        self.extra_flag_fraction = extra_flag_fraction
        self.random_state = random_state
        self.hard = HoneypotDetector()

    def screen(self, profiles):
        """
        Returns dict cid -> honeypot_prob in [0,1].
        Hard-rule hits get 0.97. IF-flagged extras get 0.80. Others get a tiny
        base scaled by their (standardized) anomaly score so the ensemble can use
        it as a soft signal without disqualifying anyone.
        """
        ids = [p["id"] for p in profiles]
        # Stage 1: hard rules
        hard_prob = {p["id"]: self.hard.detect_honeypot(p) for p in profiles}

        # Stage 2: Isolation Forest on residual features
        X = np.array([residual_features(p) for p in profiles], dtype=float)
        # Standardize (improves IF recall); guard zero-variance columns.
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd

        import warnings
        warnings.filterwarnings("ignore", message="max_samples.*greater than the total")
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(
            n_estimators=200, max_samples=256,
            random_state=self.random_state, n_jobs=1,
        )
        iso.fit(Xs)
        # higher score_samples = more normal; lower = more anomalous
        scores = iso.score_samples(Xs)

        # Rank ascending (most anomalous first), excluding hard-rule hits.
        order = sorted(range(len(ids)), key=lambda i: (scores[i], ids[i]))
        n_extra = int(round(self.extra_flag_fraction * len(ids)))
        extra = set()
        for i in order:
            if hard_prob[ids[i]] >= 0.95:
                continue
            # Only flag if the residual features actually show SOME inconsistency
            # (avoid flagging perfectly-clean profiles just to fill the quota).
            if X[i].sum() <= 0:
                continue
            extra.add(ids[i])
            if len(extra) >= n_extra:
                break

        out = {}
        for i, cid in enumerate(ids):
            if hard_prob[cid] >= 0.95:
                out[cid] = 0.97
            elif cid in extra:
                out[cid] = 0.80
            else:
                out[cid] = hard_prob[cid]  # ~0.03 base
        return out
