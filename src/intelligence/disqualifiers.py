"""
Disqualifier / down-weight engine (Gap 1).

The JD explicitly names profiles it does NOT want. This module detects those
named anti-signals from career_history, titles, industries and company names,
and returns a multiplicative penalty in (0, 1]. 1.0 = no penalty.

Every check below maps to a specific line in the job description, so each is
defensible in the Stage-5 interview.
"""
from datetime import date, datetime

# Indian IT-services / consulting firms named or implied by the JD's
# "only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant,
# Capgemini, etc.)" disqualifier. Verified present in the dataset.
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "mindtree", "ltimindtree",
    "deloitte", "ibm", "dxc", "hexaware", "birlasoft", "coforge",
}

# Product-company signals (used to confirm someone has product, not only services,
# experience). Real product cos + the dataset's fictional product-co stand-ins.
PRODUCT_COMPANIES = {
    "swiggy", "zomato", "flipkart", "razorpay", "cred", "meesho", "nykaa",
    "phonepe", "paytm", "ola", "uber", "google", "meta", "amazon", "microsoft",
    "netflix", "linkedin", "myntra", "dunzo", "sharechat", "hooli", "pied piper",
    "stark industries", "wayne enterprises", "globex", "initech", "acme corp",
}

SERVICES_INDUSTRIES = {"it services", "consulting", "staffing", "outsourcing"}
PRODUCT_INDUSTRIES = {
    "software", "fintech", "e-commerce", "edtech", "saas", "ai/ml", "adtech",
    "gaming", "healthtech", "healthtech ai", "conversational ai", "ai services",
    "food delivery", "insurance tech", "transportation",
}

# "primary expertise is computer vision, speech, or robotics without significant
# NLP/IR exposure" disqualifier.
CV_SPEECH_ROBOTICS = [
    "computer vision", "image classification", "object detection", "ocr",
    "speech recognition", "asr", "text-to-speech", "tts", "robotics", "slam",
    "autonomous", "lidar", "diffusion models", "gan", "image segmentation",
]
NLP_IR_TERMS = [
    "nlp", "natural language", "information retrieval", "retrieval", "search",
    "ranking", "recommendation", "embedding", "rag", "semantic", "llm",
    "transformer", "text", "language model",
]

# Roles that are clearly off-domain for a Senior AI Engineer (keyword-stuffer trap:
# "a candidate who has all the AI keywords ... but whose title is Marketing Manager
# is not a fit").
OFF_DOMAIN_TITLES = [
    "marketing manager", "hr manager", "human resources", "sales executive",
    "sales manager", "accountant", "civil engineer", "mechanical engineer",
    "content writer", "graphic designer", "customer support", "operations manager",
    "business analyst", "project manager", "recruiter", "office manager",
    "financial analyst", "supply chain",
]
# Engineering titles that ARE on-domain (so we don't penalize a real ML/data role).
ON_DOMAIN_TITLES = [
    "machine learning", "ml engineer", "ai engineer", "ai specialist",
    "data scientist", "data engineer", "applied scientist", "research engineer",
    "nlp engineer", "software engineer", "backend engineer", "full stack",
    "analytics engineer", "platform engineer", "search engineer",
]

# "senior engineer who hasn't written production code in the last 18 months
# because you've moved into architecture / tech lead roles".
NON_CODING_TITLES = [
    "engineering manager", "director", "vp ", "vice president", "head of",
    "chief", "cto", "principal architect", "enterprise architect",
    "solution architect", "delivery manager", "program manager",
]


def _parse_date(s):
    if not s or not isinstance(s, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    return None


def _norm(s):
    return (s or "").strip().lower()


def assess_disqualifiers(profile, role_config):
    """Return (penalty_multiplier in (0,1], list_of_reason_tags)."""
    pr = profile.get("profile", {}) or {}
    career = profile.get("career", []) or []
    text = (profile.get("text", "") or "").lower()
    career_text = (profile.get("career_text", "") or "").lower()
    title = _norm(pr.get("current_title"))
    roles_l = [_norm(r) for r in profile.get("roles", [])]
    all_titles = " ".join([title] + roles_l)

    penalty = 1.0
    tags = []

    companies = [_norm(ch.get("company")) for ch in career] + [_norm(pr.get("current_company"))]
    companies = [c for c in companies if c]
    industries = [_norm(ch.get("industry")) for ch in career] + [_norm(pr.get("current_industry"))]
    industries = [i for i in industries if i]

    def is_consulting(name):
        return any(f in name for f in CONSULTING_FIRMS)

    def is_product(name):
        return any(f == name or f in name for f in PRODUCT_COMPANIES)

    # 1) Career ENTIRELY at consulting/services firms (no product company at all).
    if companies:
        any_product_company = any(is_product(c) for c in companies)
        all_consulting = all(is_consulting(c) for c in companies)
        product_industry_seen = any(i in PRODUCT_INDUSTRIES for i in industries)
        if all_consulting and not any_product_company and not product_industry_seen:
            penalty *= 0.45
            tags.append("services_only_career")
        elif (not any_product_company) and industries and all(i in SERVICES_INDUSTRIES for i in industries):
            penalty *= 0.62
            tags.append("services_industry_only")

    # 2) Off-domain current role (keyword-stuffer trap): off-domain title AND
    #    no on-domain title anywhere in career.
    is_off = any(t in title for t in OFF_DOMAIN_TITLES)
    has_on_domain = any(any(t in rt for t in ON_DOMAIN_TITLES) for rt in [title] + roles_l)
    if is_off and not has_on_domain:
        penalty *= 0.25
        tags.append("off_domain_role")

    # 3) Senior who stopped coding (architecture/lead) — current title non-coding,
    #    and the current role description shows no hands-on build signal.
    cur = next((ch for ch in career if ch.get("is_current")), career[0] if career else None)
    if cur:
        cur_title = _norm(cur.get("title"))
        cur_desc = (cur.get("description", "") or "").lower()
        is_noncoding = any(t in cur_title for t in NON_CODING_TITLES)
        codes = any(w in cur_desc for w in ["built", "implemented", "coded", "developed",
                                            "wrote", "shipped", "deployed", "designed and built",
                                            "prototyped", "engineered"])
        if is_noncoding and not codes:
            penalty *= 0.55
            tags.append("stopped_coding")

    # 4) Primary CV/speech/robotics WITHOUT NLP/IR exposure.
    cv_hits = sum(1 for t in CV_SPEECH_ROBOTICS if t in text)
    nlp_hits = sum(1 for t in NLP_IR_TERMS if t in text)
    if cv_hits >= 3 and nlp_hits == 0:
        penalty *= 0.55
        tags.append("cv_speech_without_nlp")

    # 5) Pure research without production (academic/research-only).
    research_cues = role_config.get("research_content_cues", [])
    research_title = any(c in all_titles for c in role_config.get("research_title_cues", []))
    research_content = any(c in text for c in research_cues)
    shipper = any(c in career_text for c in role_config.get("shipper_cues", []))
    if (research_title or research_content) and not shipper:
        penalty *= 0.50
        tags.append("research_without_production")

    # 6) Title-chaser: many short stints. Count completed roles < 18 months.
    completed = [ch for ch in career if not ch.get("is_current")]
    short = [ch for ch in completed if 0 < (ch.get("duration_months", 0) or 0) < 18]
    if len(completed) >= 3 and len(short) >= 3 and len(short) >= 0.7 * len(completed):
        penalty *= 0.70
        tags.append("job_hopper")

    return round(max(0.05, penalty), 6), tags
