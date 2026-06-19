# Final score = weighted blend of feature scores, modulated by behavioral,
# location and disqualifier multipliers, plus a small validation bonus, and
# damped by honeypot probability.
#
# Deterministic: weight iteration is sorted so the float sum is identical
# across runs (hash-seed independent).


class EnsembleScorer:
    # Core skill/fit weights (must sum to 1.0).
    W = {
        "technical_fit": 0.42,
        "retrieval_score": 0.23,
        "production_ml_score": 0.16,
        "career_quality": 0.19,
    }

    def calculate_final_score(self, features, honeypot_prob, disqualifier_mult=1.0):
        base = 0.0
        for k in sorted(self.W.keys()):
            base += features.get(k, 0.0) * self.W[k]

        # Validation bonus (additive, small) — external validation / education.
        base = base + features.get("validation_bonus", 0.0)

        # Behavioral/availability multiplier (Gap 3): centered ~1.0.
        bm = features.get("behavioral_mult", 1.0)
        # Location multiplier (Gap 4).
        lm = features.get("location_mult", 1.0)
        # Disqualifier penalty (Gap 1): in (0, 1].
        dq = disqualifier_mult if disqualifier_mult else 1.0

        score = base * bm * lm * dq * (1.0 - honeypot_prob)
        return round(score, 6)
