"""
embed_skills_extractor.py  — village-level expanded skills

Usage (PowerShell):
  python embed_skills_extractor.py --text "river is dirty; drains are blocked; need water testing and community awareness" --out skills.json --threshold 0.20 --fallback_if_empty
  # or load extra skills from a JSON file ({"skills":[...], "synonyms":{...}})
  python embed_skills_extractor.py --text "...your text..." --extra_skills_json extra_skills.json --out skills.json

What it does:
- Uses a large canonical skill bank tailored for village-level projects:
  WASH, irrigation, groundwater, watershed, agriculture, livestock, roads, PMAY, SHGs, Panchayat, MGNREGA, health & nutrition, education, microgrids, ICT, surveys, etc.
- Builds TF-IDF embeddings over (input text + canonical skills + synonyms),
  then maps input sentences to the closest canonical skills via cosine similarity.
- Supports: --threshold, --fallback_if_empty, and external extension file.

Tip:
- If your sentences are short, try lower threshold (0.15–0.25) or add --fallback_if_empty.
"""

import argparse, json, re, os
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------
# 1) Canonical skills (expanded for village-level)
# ---------------------------
CANONICAL_SKILLS: List[str] = [
    # Water, Sanitation & Hygiene (WASH)
    "water quality assessment",
    "drinking water source protection",
    "household water treatment and safe storage",
    "handpump repair and maintenance",
    "borewell installation and rehabilitation",
    "rainwater harvesting",
    "greywater management",
    "fecal sludge management",
    "toilet construction and retrofitting",
    "solid waste segregation and composting",
    "solid waste collection logistics",
    "drainage design and de-silting",
    "vector control and sanitation drives",
    "hygiene behavior change communication",

    # Rivers, Irrigation & Groundwater
    "river restoration",
    "canal and minor irrigation maintenance",
    "check dam and nala bund construction",
    "farm pond design and maintenance",
    "aquifer recharge structures",
    "groundwater assessment and monitoring",
    "watershed management",
    "hydrology",

    # Agriculture & Livelihoods
    "soil testing and fertility management",
    "integrated pest management",
    "nursery management and seedling production",
    "kitchen gardens and nutrition gardens",
    "drip and sprinkler irrigation setup",
    "dairy and livestock management",
    "fisheries pond management",
    "beekeeping and pollination support",
    "post-harvest handling and storage",

    # Energy & Infrastructure
    "solar microgrid design and maintenance",
    "solar pumping systems",
    "street lighting installation and maintenance",
    "rural electrification safety and earthing",
    "rural road maintenance and culvert repair",
    "culvert and causeway design",
    "public building maintenance",
    "low-cost housing construction and PMAY support",

    # Governance & Community Institutions
    "panchayat planning and budgeting",
    "gram sabha facilitation",
    "self-help group formation and strengthening",
    "mgnrega works planning and measurement",
    "procurement and tendering",
    "beneficiary identification and targeting",
    "grievance redressal and social audit",
    "policy and regulatory compliance",

    # Health, Nutrition & Education
    "public health outreach",
    "maternal and child health support",
    "malnutrition screening and referral",
    "school wq testing and wash in schools",
    "anganwadi strengthening",
    "community outreach",
    "education and digital literacy",
    "risk communication and community engagement",

    # Environment & Climate
    "tree plantation and survival monitoring",
    "common land development",
    "pasture and fodder management",
    "erosion control and gully plugging",
    "biodiversity and habitat restoration",
    "climate risk assessment and adaptation planning",
    "disaster preparedness and response",

    # Tech, Data & Ops
    "gis and remote sensing",
    "household survey and enumeration",
    "focus group discussions and PRA tools",
    "baseline and endline studies",
    "data analysis and reporting",
    "mobile data collection and dashboards",
    "sensor deployment and iot",
    "asset mapping and village information systems",
    "project management",
    "budgeting and procurement",
    "safety and risk management",

    # Engineering specifics
    "environmental engineering",
    "civil engineering",
    "mechanical systems (pumps/filtration)",
    "microbiology",
    "sediment management",
    "green infrastructure / nature-based solutions",
]

# ---------------------------
# 2) Synonyms / surface forms
# ---------------------------
SYNONYMS: Dict[str, List[str]] = {
    # WASH
    "water quality assessment": ["water testing", "potability test", "turbidity testing", "BOD COD", "contamination check", "chlorination quality"],
    "drinking water source protection": ["spring protection", "well protection", "source catchment protection", "sanitary survey"],
    "household water treatment and safe storage": ["HWT", "safe storage", "water filter use", "chlorine tablets"],
    "handpump repair and maintenance": ["india mark ii repair", "hand pump mechanic", "pump rod replacement", "leak fix"],
    "borewell installation and rehabilitation": ["tube well", "bore well flushing", "bore yield test", "submersible pump install"],
    "rainwater harvesting": ["rooftop rainwater harvesting", "rwh", "rainwater tank", "surface runoff capture"],
    "greywater management": ["soak pit", "kitchen wastewater reuse", "leach pit", "household wastewater management"],
    "fecal sludge management": ["fsm", "septic tank desludging", "fecal sludge treatment", "faecal sludge"],
    "toilet construction and retrofitting": ["izzat ghar", "leach pit toilet", "twin pit", "toilet superstructure", "wc install"],
    "solid waste segregation and composting": ["wet and dry waste separation", "compost pit", "vermicompost", "home composting"],
    "solid waste collection logistics": ["door to door collection", "ghanta gadi", "waste route planning", "material recovery center"],
    "drainage design and de-silting": ["nala cleaning", "drain desilting", "stormwater drain", "line drains"],
    "vector control and sanitation drives": ["anti larval", "mosquito source reduction", "sanitation day", "fogging"],
    "hygiene behavior change communication": ["sbcc", "handwashing campaign", "iec bcc", "menstrual hygiene awareness"],

    # Rivers / Irrigation / Groundwater
    "river restoration": ["river clean-up", "riparian restoration", "riverbank stabilization", "desilting", "dredging"],
    "canal and minor irrigation maintenance": ["field channel repair", "canal desilting", "khal cleaning", "minor irrigation rehab"],
    "check dam and nala bund construction": ["anicut", "stop dam", "nala bunding", "gabion check dam"],
    "farm pond design and maintenance": ["khet talai", "farm pond excavation", "lining of farm pond"],
    "aquifer recharge structures": ["recharge well", "percolation tank", "recharge shaft", "infiltration gallery"],
    "groundwater assessment and monitoring": ["piezometer reading", "water table monitoring", "aquifer mapping"],
    "watershed management": ["ridge to valley treatment", "contour trenches", "bunding", "watershed plan"],
    "hydrology": ["stream discharge", "catchment hydrology", "hydrograph", "watershed hydrology"],

    # Agriculture & Livelihoods
    "soil testing and fertility management": ["soil sampling", "soil health card", "farm nutrient plan", "NPK ratio"],
    "integrated pest management": ["IPM", "biocontrol", "pheromone trap", "pest scouting"],
    "nursery management and seedling production": ["polyhouse nursery", "seed tray", "hardening seedlings"],
    "kitchen gardens and nutrition gardens": ["poshan vatika", "backyard garden", "nutri garden", "vegetable patch"],
    "drip and sprinkler irrigation setup": ["micro irrigation", "drip kit", "sprinkler set", "filter unit"],
    "dairy and livestock management": ["fodder plan", "dairy hygiene", "vaccination schedule", "cattle shed improvement"],
    "fisheries pond management": ["fish seed stocking", "liming", "pond aeration", "netting"],
    "beekeeping and pollination support": ["apiary", "bee box", "honey extraction", "colony splitting"],
    "post-harvest handling and storage": ["packhouse", "grading and sorting", "cold chain", "storage godown"],

    # Energy & Infrastructure
    "solar microgrid design and maintenance": ["solar mini grid", "solar village", "inverter maintenance", "battery bank maintenance"],
    "solar pumping systems": ["solar pump", "pv pump", "solar bore pump", "solar controller"],
    "street lighting installation and maintenance": ["LED street light", "streetlight pole", "solar street light"],
    "rural electrification safety and earthing": ["earthing pit", "breaker", "metering", "household wiring safety"],
    "rural road maintenance and culvert repair": ["patchwork", "pothole filling", "metalling", "shoulder repair"],
    "culvert and causeway design": ["hume pipe", "box culvert", "causeway", "thrie beam guardrail"],
    "public building maintenance": ["school repair", "anganwadi renovation", "panchayat bhavan maintenance"],
    "low-cost housing construction and PMAY support": ["pmay house", "beneficiary assistance", "mason training", "toilet-cum-bath"],

    # Governance & Community Institutions
    "panchayat planning and budgeting": ["gpdp", "gram panchayat development plan", "budget allocation", "work prioritization"],
    "gram sabha facilitation": ["ward sabha", "resolution drafting", "minutes of meeting"],
    "self-help group formation and strengthening": ["shg bookkeeping", "vo/clf strengthening", "bank linkage"],
    "mgnrega works planning and measurement": ["boq for mgnrega", "mate supervision", "muster roll", "measurement book"],
    "procurement and tendering": ["rfp preparation", "quotation process", "e-tender", "work order"],
    "beneficiary identification and targeting": ["secc data", "ration card list", "household eligibility", "verification"],
    "grievance redressal and social audit": ["public hearing", "jansunwai", "social audit report", "ombudsman"],
    "policy and regulatory compliance": ["CPCB standards", "water quality standards", "environmental clearance"],

    # Health, Nutrition & Education
    "public health outreach": ["health camp", "asha coordination", "immunization awareness", "disease surveillance"],
    "maternal and child health support": ["vhnd support", "ifa supplementation", "anc/pnc tracking"],
    "malnutrition screening and referral": ["muac", "severe acute malnutrition", "nutrition counselling"],
    "school wq testing and wash in schools": ["tippy tap", "handwashing station", "school toilets", "mhm in schools"],
    "anganwadi strengthening": ["icds records", "weighing", "growth monitoring"],
    "community outreach": ["awareness campaign", "stakeholder engagement", "panchayat meetings"],
    "education and digital literacy": ["computer class", "smart classroom", "adult literacy"],
    "risk communication and community engagement": ["rcce", "door-to-door messaging", "public address"],

    # Environment & Climate
    "tree plantation and survival monitoring": ["pit digging", "mulching", "watering schedule", "guarding"],
    "common land development": ["charnoi vikas", "silvipasture", "community land treatment"],
    "pasture and fodder management": ["fodder species", "cut and carry", "stall feeding"],
    "erosion control and gully plugging": ["loose boulder check", "gabion structure", "brushwood check"],
    "biodiversity and habitat restoration": ["native species", "invasive control", "habitat enrichment"],
    "climate risk assessment and adaptation planning": ["heat stress mapping", "drought contingency", "vulnerability assessment"],
    "disaster preparedness and response": ["flood warning", "relief distribution", "evacuation plan"],

    # Tech, Data & Ops
    "gis and remote sensing": ["satellite mapping", "geospatial analysis", "drone survey", "gps mapping"],
    "household survey and enumeration": ["door-to-door survey", "house listing", "enumerator training"],
    "focus group discussions and PRA tools": ["transect walk", "social map", "resource map", "seasonal calendar"],
    "baseline and endline studies": ["baseline survey", "endline evaluation", "monitoring plan"],
    "data analysis and reporting": ["excel dashboards", "analytics report", "power bi", "statistical analysis"],
    "mobile data collection and dashboards": ["kobo toolbox", "odk", "commcare", "tableau"],
    "sensor deployment and iot": ["telemetry", "water sensors", "iot nodes", "lorawan"],
    "asset mapping and village information systems": ["village gis", "amenity mapping", "MIS"],
    "project management": ["timeline planning", "gantt", "scheduling", "work breakdown structure"],
    "budgeting and procurement": ["cost estimation", "quotation", "purchase order", "invoice processing"],
    "safety and risk management": ["PPE", "hazard assessment", "job safety analysis"],

    # Engineering specifics
    "environmental engineering": ["treatment process", "filtration", "remediation", "aeration"],
    "civil engineering": ["drainage infrastructure", "culverts", "embankment"],
    "mechanical systems (pumps/filtration)": ["intake pump", "sand filter", "activated carbon filter"],
    "microbiology": ["coliform count", "pathogen testing"],
    "sediment management": ["silt control", "sediment load"],
    "green infrastructure / nature-based solutions": ["constructed wetlands", "bioremediation", "nature-based"],
}

# ---------------------------
# 3) Core extractor
# ---------------------------
def _split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts or [text]

def _merge_external_skills(extra_json_path: str):
    global CANONICAL_SKILLS, SYNONYMS
    if not extra_json_path or not os.path.exists(extra_json_path):
        return
    with open(extra_json_path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    ext_skills = blob.get("skills", [])
    ext_syn = blob.get("synonyms", {})
    for s in ext_skills:
        if s not in CANONICAL_SKILLS:
            CANONICAL_SKILLS.append(s)
    for k, vals in ext_syn.items():
        base = SYNONYMS.get(k, [])
        for v in vals:
            if v not in base:
                base.append(v)
        SYNONYMS[k] = base

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

# Simple domain-aware fallback for very short inputs
VILLAGE_FALLBACK = [
    "water quality assessment",
    "drainage design and de-silting",
    "solid waste segregation and composting",
    "toilet construction and retrofitting",
    "rainwater harvesting",
    "handpump repair and maintenance",
    "watershed management",
    "soil testing and fertility management",
    "panchayat planning and budgeting",
    "self-help group formation and strengthening",
    "public health outreach",
    "education and digital literacy",
    "project management",
    "data analysis and reporting",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, help="Raw requirement text")
    ap.add_argument("--file", type=str, help="Path to a .txt file")
    ap.add_argument("--out", type=str, default="skills.json")
    ap.add_argument("--threshold", type=float, default=0.25, help="cosine threshold (0..1)")
    ap.add_argument("--fallback_if_empty", action="store_true", help="use domain fallback skills when extraction is empty")
    ap.add_argument("--extra_skills_json", type=str, help="JSON file with {'skills':[...], 'synonyms':{...}} to extend the bank")
    args = ap.parse_args()

    if not args.text and not args.file:
        raise SystemExit("Provide --text or --file")

    # External extensions
    if args.extra_skills_json:
        _merge_external_skills(args.extra_skills_json)

    text = args.text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()

    skills = extract_skills_embed(text, topk_per_sentence=7, threshold=args.threshold)

    if args.fallback_if_empty and not skills:
        t = (text or "").lower()
        # if looks village-ish or environmental, add a sensible base set
        if any(k in t for k in [
            "village","gram","panchayat","ward","toilet","drain","waste","river","water",
            "angawadi","anganwadi","school","handpump","borewell","harvesting","mgnrega",
            "shg","pmay","health","nutrition","road","culvert"
        ]):
            skills = VILLAGE_FALLBACK.copy()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(skills, f, ensure_ascii=False, indent=2)
    print(json.dumps(skills, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
