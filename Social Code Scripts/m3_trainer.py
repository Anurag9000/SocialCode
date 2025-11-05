

import argparse, csv, os, pickle, math, re
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from embeddings import embed_texts, embed_with

AVAILABILITY_LEVELS = {
    "rarely available": 0,
    "generally available": 1,
    "immediately available": 2,
}

SEVERITY_LABELS = {0: "low", 1: "normal", 2: "high"}
SEVERITY_KEYWORDS = {
    2: ["urgent", "immediate", "critical", "outbreak", "epidemic", "collapse", "broken", "flood", "drought", "disease", "contamination", "crisis", "emergency"],
    1: ["audit", "survey", "assessment", "monitoring", "planning", "inspection", "review", "repair", "maintenance"],
}

def normalize_phrase(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

def load_village_names(village_locations_path: str) -> List[str]:
    rows = read_csv_norm(village_locations_path)
    names = [get_any(r, ["village_name", "village", "name"], "") for r in rows]
    names = [n for n in names if n]
    names_sorted = sorted(names, key=lambda x: len(x), reverse=True)
    return names_sorted

def load_distance_lookup(village_distances_path: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    rows = read_csv_norm(village_distances_path)
    lookup: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in rows:
        a = get_any(r, ["village_a", "from", "source"])
        b = get_any(r, ["village_b", "to", "destination"])
        if not a or not b:
            continue
        dist = float(get_any(r, ["distance_km", "distance"], 0.0) or 0.0)
        travel = float(get_any(r, ["travel_time_min", "travel_min"], 0.0) or 0.0)
        lookup[(a.lower(), b.lower())] = {"distance": dist, "travel": travel}
        lookup[(b.lower(), a.lower())] = {"distance": dist, "travel": travel}
    return lookup

def extract_location(text: str, village_names: List[str]) -> str:
    if not text:
        return ""
    norm_text = normalize_phrase(text)
    for name in village_names:
        if normalize_phrase(name) in norm_text:
            return name
    return ""

def estimate_severity(text: str) -> int:
    if not text:
        return 1
    text_norm = text.lower()
    for kw in SEVERITY_KEYWORDS[2]:
        if kw in text_norm:
            return 2
    for kw in SEVERITY_KEYWORDS[1]:
        if kw in text_norm:
            return 1
    return 0

def severity_penalty(availability_label: str, severity_level: int) -> float:
    label = (availability_label or "").lower()
    if severity_level >= 2:
        if label == "generally available":
            return 0.2
        if label == "rarely available":
            return 0.4
    if severity_level == 1:
        if label == "rarely available":
            return 0.2
    return 0.0

def lookup_distance_km(origin: str, target: str, distance_lookup: Dict[Tuple[str, str], Dict[str, float]]) -> float:
    if not origin or not target:
        return 0.0
    rec = distance_lookup.get((origin.lower(), target.lower()))
    if rec:
        return float(rec.get("distance", 0.0))
    return 0.0

# ---------------- utils ----------------

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def read_csv_norm(fp: str) -> List[Dict]:
    """Read CSV and normalize keys to lowercase, strip whitespace."""
    rows = []
    with open(fp, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"{fp}: missing header row")
        reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]
        for r in reader:
            rows.append({(k.strip().lower() if k else k): (v.strip() if isinstance(v, str) else v)
                         for k, v in r.items()})
    return rows

def get_any(d: Dict, keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default

def as2d(v):
    """Ensure a single sample row is 2D for cosine_similarity."""
    import numpy as np
    from scipy import sparse
    if 'sparse' in str(type(v)) or sparse.issparse(v):
        return v  # already 2D row in sparse format
    v = np.asarray(v)
    return v.reshape(1, -1) if v.ndim == 1 else v

# -------------- core builder -------------

def build_feature_matrix(props, people, pairs, prop_model, people_model, backend,
                         proposal_locations: Dict[str, str],
                         severity_levels: Dict[str, int],
                         distance_lookup: Dict[Tuple[str, str], Dict[str, float]],
                         distance_scale: float,
                         distance_decay: float):
    # proposals map
    prop_map = {}
    for r in props:
        pid = get_any(r, ["proposal_id", "id"])
        ptxt = get_any(r, ["text", "proposal_text", "description", "body", "content"], "")
        if pid is not None:
            prop_map[pid] = ptxt

    # normalize people
    people_norm = []
    for r in people:
        pid = get_any(r, ["person_id", "student_id", "id"])
        if not pid:
            continue
        txt = get_any(r, ["text", "skills"], "")
        eff = float(get_any(r, ["willingness_eff", "eff", "w_eff"], 0.5))
        bias = float(get_any(r, ["willingness_bias", "bias", "w_bias"], 0.5))
        name = get_any(r, ["name", "person_name", "full_name"], pid)
        people_norm.append({
            "person_id": pid,
            "name": name,
            "text": txt,
            "eff": eff,
            "bias": bias,
            "availability": (get_any(r, ["availability"], "") or "").lower(),
            "home_location": get_any(r, ["home_location", "location", "village"], ""),
        })
    people_map = {r["person_id"]: r for r in people_norm}

    # collect unique ids from pairs
    prop_ids = []
    person_ids = []
    for r in pairs:
        pid = get_any(r, ["proposal_id", "id"])
        sid = get_any(r, ["person_id", "student_id", "id"])
        if pid is not None and sid is not None:
            prop_ids.append(pid)
            person_ids.append(sid)
    prop_ids = sorted(set(prop_ids))
    person_ids = sorted(set(person_ids))

    # embed unique texts
    try:
        prop_texts = [prop_map[pid] for pid in prop_ids]
    except KeyError as e:
        missing = str(e).strip("'")
        raise SystemExit(f"pairs.csv references proposal_id '{missing}' not found in proposals.csv")

    try:
        person_texts = [people_map[sid]["text"] for sid in person_ids]
    except KeyError as e:
        missing = str(e).strip("'")
        raise SystemExit(f"pairs.csv references person_id/student_id '{missing}' not found in people.csv")

    P = embed_with(prop_model, prop_texts, backend)
    S = embed_with(people_model, person_texts, backend)

    p_index = {pid: i for i, pid in enumerate(prop_ids)}
    s_index = {sid: i for i, sid in enumerate(person_ids)}

    X, y = [], []
    for r in pairs:
        pid = get_any(r, ["proposal_id", "id"])
        sid = get_any(r, ["person_id", "student_id", "id"])
        label = int(get_any(r, ["label", "y", "target"], 0))
        if pid not in p_index or sid not in s_index or sid not in people_map:
            # skip malformed rows
            continue

        pv = as2d(P[p_index[pid]])
        sv = as2d(S[s_index[sid]])
        sim = float(cosine_similarity(pv, sv)[0, 0])

        eff = float(people_map[sid]["eff"])
        bias = float(people_map[sid]["bias"])
        base_W = sigmoid(eff + bias)

        availability_label = people_map[sid]["availability"]
        availability_level = AVAILABILITY_LEVELS.get(availability_label, 1)
        severity_level = severity_levels.get(pid, 1)
        prop_location = proposal_locations.get(pid, "")
        person_location = people_map[sid]["home_location"]
        distance_km = lookup_distance_km(person_location, prop_location, distance_lookup)
        distance_norm = min(distance_km / distance_scale, 1.0) if distance_scale > 0 else 0.0
        distance_penalty = math.exp(-distance_km / distance_decay) if distance_decay > 0 else 1.0
        w_after_severity = max(0.0, base_W - severity_penalty(availability_label, severity_level))
        W = max(0.0, min(1.0, w_after_severity * distance_penalty))

        X.append([
            sim,
            sim * W,
            W,
            distance_norm,
            distance_penalty,
            availability_level / 2.0,
            severity_level / 2.0,
        ])
        y.append(label)

    return np.asarray(X), np.asarray(y)

# --------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--people", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", default="model.pkl")
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    default_dataset_root = r"D:\SocialCode\gram_sahayta_dataset_with_locations_and_availability"
    ap.add_argument("--village_locations", default=os.path.join(default_dataset_root, "village_locations.csv"))
    ap.add_argument("--village_distances", default=os.path.join(default_dataset_root, "village_distances.csv"))
    ap.add_argument("--distance_scale", type=float, default=50.0, help="Distance in km mapped to 1.0 in features")
    ap.add_argument("--distance_decay", type=float, default=30.0, help="Decay constant (km) for distance penalty exp(-d/decay)")
    args = ap.parse_args()

    if not os.path.exists(args.proposals):
        raise SystemExit(f"Not found: {args.proposals}")
    if not os.path.exists(args.people):
        raise SystemExit(f"Not found: {args.people}")
    if not os.path.exists(args.pairs):
        raise SystemExit(f"Not found: {args.pairs}")
    if args.village_locations and not os.path.exists(args.village_locations):
        raise SystemExit(f"Not found: {args.village_locations}")
    if args.village_distances and not os.path.exists(args.village_distances):
        raise SystemExit(f"Not found: {args.village_distances}")

    props = read_csv_norm(args.proposals)
    people = read_csv_norm(args.people)
    pairs = read_csv_norm(args.pairs)

    village_names = load_village_names(args.village_locations) if args.village_locations else []
    distance_lookup = load_distance_lookup(args.village_distances) if args.village_distances else {}

    proposal_locations: Dict[str, str] = {}
    severity_levels: Dict[str, int] = {}
    missing_locations = 0
    for r in props:
        pid = get_any(r, ["proposal_id", "id"])
        text = get_any(r, ["text", "proposal_text", "description", "body", "content"], "")
        if pid is None:
            continue
        location = extract_location(text, village_names) if village_names else ""
        if not location:
            missing_locations += 1
        proposal_locations[pid] = location
        severity_levels[pid] = estimate_severity(text)
    if missing_locations and village_names:
        print(f"[trainer] Warning: {missing_locations} proposals did not match a known village name.")

    # Fit embedding backends on corpora
    prop_texts_all = [get_any(r, ["text","proposal_text","description","body","content"], "") for r in props]
    people_texts_all = [get_any(r, ["text","skills"], "") for r in people]

    prop_model, _, backend = embed_texts(prop_texts_all, model_name=args.model_name)
    people_model, _, backend2 = embed_texts(people_texts_all, model_name=args.model_name)

    # Use SHARED TF-IDF vectorizer if either side fell back to TF-IDF
    if backend == "tfidf" or backend2 == "tfidf":
        from embeddings import embed_texts as _embed
        combined = prop_texts_all + people_texts_all
        shared_vec, _, _ = _embed(combined, model_name=args.model_name)
        prop_model = shared_vec
        people_model = shared_vec
        backend = "tfidf"
    else:
        backend = "sentence-transformers"

    X, y = build_feature_matrix(
        props,
        people,
        pairs,
        prop_model,
        people_model,
        backend,
        proposal_locations,
        severity_levels,
        distance_lookup,
        args.distance_scale,
        args.distance_decay,
    )

    n = len(y)
    if n == 0:
        raise SystemExit("No training pairs after normalization. Check your CSV headers/values.")

    # train/val split
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    split = max(1, int(0.8 * n))
    tr, va = idx[:split], idx[split:]
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xtr, ytr)

    if len(Xva) and len(np.unique(yva)) > 1:
        va_pred = clf.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(yva, va_pred)
    else:
        auc = float("nan")

    print(f"Validation AUC: {auc:.3f} (backend={backend})")

    with open(args.out, "wb") as f:
        pickle.dump({
            "model": clf,
            "backend": backend,
            "prop_model": prop_model,
            "people_model": people_model,
            "distance_scale": args.distance_scale,
            "distance_decay": args.distance_decay,
        }, f)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
