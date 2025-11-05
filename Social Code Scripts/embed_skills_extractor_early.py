"""
embed_skills_extractor.py

Usage:
  python embed_skills_extractor.py --text "<requirement text>" --out skills.json
  or
  python embed_skills_extractor.py --file input.txt --out skills.json

What it does:
- Uses a small canonical skill bank for the water/river cleanup domain (extend as needed).
- Builds TF-IDF embeddings over (input text + skill phrases + synonyms), then
  maps input sentences to the closest canonical skills via cosine similarity.
"""

import argparse, json, re
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

CANONICAL_SKILLS: List[str] = [
    "water quality assessment",
    "hydrology",
    "environmental engineering",
    "waste management",
    "river restoration",
    "GIS and remote sensing",
    "microbiology",
    "civil engineering",
    "mechanical systems (pumps/filtration)",
    "policy and regulatory compliance",
    "community outreach",
    "project management",
    "budgeting and procurement",
    "data analysis and reporting",
    "sensor deployment and IoT",
    "safety and risk management",
    "ecology and biodiversity",
    "sediment management",
    "green infrastructure / nature-based solutions"
]

SYNONYMS = {
    "water quality assessment": ["water testing", "pollution monitoring", "turbidity testing", "BOD COD"],
    "hydrology": ["river flow", "watershed", "stream discharge", "catchment"],
    "environmental engineering": ["treatment process", "filtration", "remediation", "aeration"],
    "waste management": ["solid waste", "litter cleanup", "trash removal", "plastics in river"],
    "river restoration": ["riparian restoration", "riverbank stabilization", "desilting", "dredging"],
    "GIS and remote sensing": ["satellite mapping", "geospatial analysis", "drone survey"],
    "microbiology": ["coliform count", "pathogen testing"],
    "civil engineering": ["drainage infrastructure", "culverts", "embankment"],
    "mechanical systems (pumps/filtration)": ["intake pump", "sand filter", "activated carbon filter"],
    "policy and regulatory compliance": ["CPCB standards", "EPA norms", "water quality standards"],
    "community outreach": ["awareness campaign", "stakeholder engagement", "panchayat"],
    "project management": ["timeline planning", "scheduling"],
    "budgeting and procurement": ["tender", "cost estimation", "procurement"],
    "data analysis and reporting": ["dashboard", "analytics report"],
    "sensor deployment and IoT": ["telemetry", "water sensors", "IoT nodes"],
    "safety and risk management": ["PPE", "hazard assessment"],
    "ecology and biodiversity": ["habitat", "flora and fauna"],
    "sediment management": ["silt control", "sediment load"],
    "green infrastructure / nature-based solutions": ["constructed wetlands", "bioremediation", "nature-based"]
}

def _split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts or [text]

def extract_skills_embed(text: str, topk_per_sentence: int = 5, threshold: float = 0.25) -> List[str]:
    sentences = _split_sentences(text.lower())
    # Build corpus: sentences + canonical + synonyms
    corpus = []
    corpus.extend(sentences)
    skill_variants = []
    for s in CANONICAL_SKILLS:
        skill_variants.append(s.lower())
        for syn in SYNONYMS.get(s, []):
            skill_variants.append(syn.lower())
    corpus.extend(skill_variants)

    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words='english')
    X = vec.fit_transform(corpus)

    n_sent = len(sentences)
    sent_mat = X[:n_sent]
    skill_mat = X[n_sent:]  # canonical + synonyms in order

    # Map each row in skill_mat back to canonical skill id
    sk_ids = []
    idx = 0
    for s in CANONICAL_SKILLS:
        sk_ids.append((s, idx)); idx += 1
        for syn in SYNONYMS.get(s, []):
            sk_ids.append((s, idx)); idx += 1

    sims = cosine_similarity(sent_mat, skill_mat)  # (n_sent, n_skill_variants)
    chosen = set()
    for i in range(n_sent):
        row = sims[i]
        top_idx = np.argsort(-row)[:topk_per_sentence]
        for j in top_idx:
            score = row[j]
            if score >= threshold:
                canon, _ = sk_ids[j]
                chosen.add(canon)
    return sorted(chosen)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, help="Raw requirement text")
    ap.add_argument("--file", type=str, help="Path to a .txt file")
    ap.add_argument("--out", type=str, default="skills.json")
    args = ap.parse_args()

    if not args.text and not args.file:
        raise SystemExit("Provide --text or --file")

    text = args.text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()

    skills = extract_skills_embed(text)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(skills, f, ensure_ascii=False, indent=2)
    print(json.dumps(skills, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
