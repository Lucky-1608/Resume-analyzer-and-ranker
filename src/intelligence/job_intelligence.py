import os, re

def _read_docx(path):
    try:
        import docx
        return chr(10).join(p.text for p in docx.Document(path).paragraphs if p.text.strip())
    except:
        try: return open(path, encoding="utf-8", errors="ignore").read()
        except: return ""

class JobIntelligenceLayer:
    DOMAIN_SKILLS = {
        "embeddings_retrieval": {"weight": 3.0, "terms": ["embedding","embeddings","sentence-transformers","sentence transformers","bge","e5","openai embeddings","semantic search","vector search","retrieval","dense retrieval","ann","faiss","approximate nearest"]},
        "vector_db": {"weight": 2.0, "terms": ["pinecone","milvus","qdrant","weaviate","faiss","opensearch","elasticsearch","vespa","vector database","vector db"]},
        "ranking": {"weight": 2.5, "terms": ["ranking","re-ranking","reranking","learning to rank","learning-to-rank","ltr","lambdamart","ndcg","cross-encoder","cross encoder","recommender"]},
        "llm": {"weight": 2.0, "terms": ["llm","large language model","rag","retrieval augmented","fine-tuning","fine tuning","qlora","lora","prompt engineering","transformers","huggingface","hugging face","langchain","llamaindex"]},
        "ml_core": {"weight": 1.8, "terms": ["machine learning","deep learning","pytorch","tensorflow","scikit-learn","sklearn","xgboost","lightgbm","nlp","natural language","feature engineering","model training","mlops","model serving"]},
        "production_eng": {"weight": 1.5, "terms": ["python","production","api","docker","kubernetes","spark","airflow","scalable","distributed","latency","deployment","deployed","shipped","microservice","backend","data pipeline","pipeline"]},
        "evaluation": {"weight": 1.5, "terms": ["evaluation","a/b test","ab test","offline benchmark","benchmark","metrics","experimentation","eval framework"]},
    }
    RESEARCH_TITLE_CUES = ["research scientist","research engineer","phd researcher","postdoc","research fellow","applied scientist"]
    NON_IC_TITLE_CUES = ["manager","director","vp ","vice president","head of","chief","cto","principal architect"]
    SHIPPER_CUES = ["shipped","deployed","production","built","launched","owned","scaled","delivered","implemented"]
    RESEARCH_CONTENT_CUES = ["publication","published","paper","research lab","academia","thesis","citations","arxiv"]

    def __init__(self, jd_path=None, jd_text=None):
        self.jd_path=jd_path; self.jd_text=jd_text or ""; self.ontology={}
        self.exp_band=(5.0,9.0); self.exp_soft_min=3.0; self.exp_soft_max=12.0

    def parse_jd(self):
        if not self.jd_text and self.jd_path and os.path.exists(self.jd_path):
            self.jd_text = _read_docx(self.jd_path)
        text = self.jd_text.lower()
        m = re.search(r"(\d{1,2})\s*[-to]+\s*(\d{1,2})\s*year", text)
        if m:
            lo,hi=float(m.group(1)),float(m.group(2))
            if 0<lo<hi<=30: self.exp_band=(lo,hi); self.exp_soft_min=max(0.0,lo-2.0); self.exp_soft_max=hi+3.0
        present={n:s for n,s in self.DOMAIN_SKILLS.items() if not text or any(t in text for t in s["terms"])}
        self.ontology=present if present else dict(self.DOMAIN_SKILLS)

    def expand_requirements(self):
        if not self.ontology: self.ontology=dict(self.DOMAIN_SKILLS)

    def get_matching_vocabulary(self): return self.ontology if self.ontology else dict(self.DOMAIN_SKILLS)

    def get_role_config(self):
        import re as _r
        v=self.get_matching_vocabulary()
        rx={f:_r.compile("|".join(_r.escape(t) for t in s.get("terms",[]))) for f,s in v.items() if s.get("terms")}
        return {"vocabulary":v,"family_regex":rx,"exp_band":self.exp_band,"exp_soft_min":self.exp_soft_min,"exp_soft_max":self.exp_soft_max,"research_title_cues":self.RESEARCH_TITLE_CUES,"non_ic_title_cues":self.NON_IC_TITLE_CUES,"shipper_cues":self.SHIPPER_CUES,"research_content_cues":self.RESEARCH_CONTENT_CUES}
