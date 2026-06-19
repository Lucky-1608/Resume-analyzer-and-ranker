"""
Location fit (Gap 4) and education / external-validation signal (Gap 5).

Location maps to the JD's explicit preference:
  "Pune/Noida-preferred but flexible ... Candidates in Hyderabad, Pune, Mumbai,
   Delhi NCR welcome ... Outside India: case-by-case, but we don't sponsor work
   visas."
Returns multipliers/addites used by the feature engineer.
"""

# JD-named primary hubs (offices + explicitly welcomed cities).
PRIMARY_HUBS = ["pune", "noida"]
WELCOME_CITIES = ["hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
                  "noida", "ghaziabad", "faridabad", "navi mumbai", "thane"]
# Other Indian Tier-1 (still in-country, no visa issue, fine).
OTHER_INDIA_TIER1 = ["bangalore", "bengaluru", "chennai", "kolkata", "ahmedabad",
                     "jaipur", "chandigarh", "coimbatore", "kochi", "trivandrum",
                     "indore", "bhubaneswar", "vizag", "visakhapatnam", "nagpur"]


def location_fit(profile):
    """Return a location multiplier in [0.80, 1.08]."""
    pr = profile.get("profile", {}) or {}
    loc = (pr.get("location", "") or "").lower()
    country = (pr.get("country", "") or "").lower()
    sig = profile.get("signals", {}) or {}
    relocate = bool(sig.get("willing_to_relocate"))

    # Outside India: JD says case-by-case + no visa sponsorship -> down-weight,
    # softened if they're willing to relocate.
    if country and country != "india":
        return 0.88 if relocate else 0.80

    # In India.
    if any(h in loc for h in PRIMARY_HUBS):
        return 1.08  # Pune/Noida — exactly where the offices are
    if any(c in loc for c in WELCOME_CITIES):
        return 1.05  # explicitly welcomed (Hyderabad/Mumbai/Delhi NCR)
    if any(c in loc for c in OTHER_INDIA_TIER1):
        return 1.0 if relocate else 0.97  # in India, relocation helps
    # India but unknown city
    return 1.0 if relocate else 0.98


def validation_signal(profile):
    """
    Small additive bonus in [0, 0.10] for external validation / strong education.
    Maps to JD: "external validation (papers, talks, open-source)" and tier.
    """
    bonus = 0.0
    sig = profile.get("signals", {}) or {}
    edu = profile.get("education", []) or []
    certs = profile.get("certifications", []) or []
    text = (profile.get("text", "") or "").lower()

    # Open-source / GitHub activity (external validation the JD names).
    gh = sig.get("github_activity_score", -1)
    if isinstance(gh, (int, float)) and gh >= 50:
        bonus += 0.04
    elif isinstance(gh, (int, float)) and gh >= 20:
        bonus += 0.02

    # Papers / talks / open-source mentions in profile text.
    if any(w in text for w in ["open-source", "open source", "github.com",
                               "published", "paper", "patent", "speaker", "talk at",
                               "maintainer", "contributor"]):
        bonus += 0.02

    # Education tier (tier_1 strong; we have tier_1..tier_3/unknown).
    tiers = [(e.get("tier", "") or "").lower() for e in edu]
    if any(t == "tier_1" for t in tiers):
        bonus += 0.03
    elif any(t == "tier_2" for t in tiers):
        bonus += 0.01

    # Relevant certifications (small).
    if certs:
        bonus += 0.01

    return round(min(0.10, bonus), 6)
