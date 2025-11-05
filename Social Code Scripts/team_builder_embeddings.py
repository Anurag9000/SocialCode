"""
team_builder_embeddings.py  (with willingness)

- ULTRA-style team scoring:
  * Coverage (+): similarity-weighted using TF-IDF embeddings + cosine, threshold tau,
                  and now scaled by each member's willingness W
  * k-Robustness (+): k=1 (removing any one member keeps coverage) — recomputed with W
  * Redundancy (-): fraction of required skills held by >=2 members (using W-weighted sims >= tau)
  * Set size (-): normalized by a soft cap
- Goodness = normalized (+coverage +k_robustness −redundancy −set_size) in [0,1].
"""

import argparse, json, csv, itertools, math
from typing import List, Dict, Tuple
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def read_skills(skills_json_path: str) -> List[str]:
    with open(skills_json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _parse_w(v: str, default=0.5) -> float:
    try:
        return float(v)
    except:
        return float(default)

def read_students(students_csv_path: str) -> List[Dict]:
    rows = []
    with open(students_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            skills = [s.strip().lower() for s in r["skills"].split(";") if s.strip()]
            eff = _parse_w(r.get("willingness_eff", "0.5"))
            bias = _parse_w(r.get("willingness_bias", "0.5"))
            W = sigmoid(eff + bias)  # SWAM-style willingness
            rows.append({
                "student_id": r["student_id"],
                "name": r["name"],
                "skills": skills,
                "w_eff": eff,
                "w_bias": bias,
                "W": W
            })
    return rows

def embed_texts(texts: List[str]):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words='english')
    X = vec.fit_transform([t.lower() for t in texts])
    return vec, X

def similarity_coverage(required: List[str], team: List[Dict], tau: float = 0.35):
    """
    W-weighted similarity coverage with member-level aggregation:
      - Embed required phrases and all team phrases with TF-IDF.
      - Compute cosine similarity and scale each phrase column by owner's W.
      - For each member u and requirement r, take max over u's phrases => sim_u[r].
      - Per-req best = max_u sim_u[r].
      - covered_counts[r] counts DISTINCT members u with sim_u[r] >= tau.
    """
    member_phrases = []
    ownersW = []
    member_index = []
    for idx, m in enumerate(team):
        for sk in m["skills"]:
            member_phrases.append(sk)
            ownersW.append(m["W"])
            member_index.append(idx)
    if not required:
        return 0.0, {}, {}
    if not member_phrases:
        return 0.0, {r: 0.0 for r in required}, {r: 0 for r in required}

    corpus = [*required, *member_phrases]
    vec, X = embed_texts(corpus)
    R = X[:len(required)]
    M = X[len(required):]
    S = cosine_similarity(R, M)

    Wcol = np.array(ownersW).reshape(1, -1)
    S_w = S * Wcol

    per_member_best = [np.zeros(len(required), dtype=float) for _ in range(len(team))]
    for col in range(S_w.shape[1]):
        u = member_index[col]
        col_vals = np.array(S_w[:, col]).ravel()
        per_member_best[u] = np.maximum(per_member_best[u], col_vals)

    per_req_best = np.max(np.stack(per_member_best, axis=0), axis=0) if len(team) else np.zeros(len(required))

    covered_counts = {}
    for i, req in enumerate(required):
        count = sum(1 for u in range(len(team)) if per_member_best[u][i] >= tau)
        covered_counts[req] = int(count)

    adj = []
    for s in per_req_best:
        s = float(s)
        if s < tau:
            adj.append(0.0)
        else:
            adj.append((s - tau) / (1.0 - tau))
    coverage = float(np.mean(adj)) if len(adj) > 0 else 0.0
    per_req_map = {required[i]: float(per_req_best[i]) for i in range(len(required))}
    return coverage, per_req_map, covered_counts

def redundancy_metric(required: List[str], covered_counts: Dict[str, int]) -> float:
    if not required:
        return 0.0
    redundant = sum(1 for r in required if covered_counts.get(r, 0) >= 2)
    return redundant / len(required)

def k_robustness(required: List[str], team: List[Dict], tau: float = 0.35, k: int = 1, sample_limit: int = 2000) -> float:
    if not team:
        return 0.0
    _, per_req_best, _ = similarity_coverage(required, team, tau=tau)
    if not all(v >= tau for v in per_req_best.values()):
        return 0.0
    n = len(team)
    k = max(1, min(k, n))
    import itertools, random
    subsets = []
    for r in range(1, k + 1):
        combs = list(itertools.combinations(range(n), r))
        subsets.extend(combs)
        if len(subsets) > sample_limit:
            break
    if len(subsets) > sample_limit:
        random.seed(42)
        subsets = random.sample(subsets, sample_limit)
    ok = 0
    for rem in subsets:
        sub_team = [m for i, m in enumerate(team) if i not in rem]
        _, per_req_best2, _ = similarity_coverage(required, sub_team, tau=tau)
        if all(v >= tau for v in per_req_best2.values()):
            ok += 1
    return ok / len(subsets) if subsets else 0.0

def team_metrics(required: List[str], team: List[Dict], tau: float = 0.35, k: int = 1) -> Dict[str, float]:
    coverage, per_req_best, covered_counts = similarity_coverage(required, team, tau)
    redundancy = redundancy_metric(required, covered_counts)
    soft_cap = max(len(required), 4)
    set_size = min(len(team) / soft_cap, 1.0)
    krob = k_robustness(required, team, tau, k=k)
    w_vals = [float(m.get("W", 0.0)) for m in team]
    willingness_avg = float(np.mean(w_vals)) if w_vals else 0.0
    willingness_min = float(np.min(w_vals)) if w_vals else 0.0
    return {
        "coverage": coverage,
        "redundancy": redundancy,
        "set_size": set_size,
        "k_robustness": krob,
        "willingness_avg": willingness_avg,
        "willingness_min": willingness_min,
    }

def goodness(metrics: Dict[str, float], lambda_red: float = 1.0, lambda_size: float = 1.0, lambda_will: float = 0.5) -> float:
    s = (
        (+1) * metrics["coverage"]
        + (+1) * metrics["k_robustness"]
        + lambda_will * metrics["willingness_avg"]
        - lambda_red * metrics["redundancy"]
        - lambda_size * metrics["set_size"]
    )
    return (s + 2.0) / 4.0

def recommend_teams(required: List[str], students: List[Dict], team_size_min=2, team_size_max=5, top_k=10, tau: float = 0.35, k: int = 1, lambda_red: float = 1.0, lambda_size: float = 1.0, lambda_will: float = 0.5):
    recs = []
    n = len(students)
    for r in range(team_size_min, min(team_size_max, n)+1):
        for combo in itertools.combinations(students, r):
            mets = team_metrics(required, list(combo), tau=tau, k=k)
            score = goodness(mets, lambda_red=lambda_red, lambda_size=lambda_size, lambda_will=lambda_will)
            recs.append({
                "team_ids": [m["student_id"] for m in combo],
                "team_names": [m["name"] for m in combo],
                "metrics": mets,
                "goodness": round(score, 4)
            })
    recs.sort(key=lambda x: (x["goodness"], x["metrics"]["coverage"], -len(x["team_ids"])), reverse=True)
    return recs[:top_k]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", required=True, help="skills.json")
    ap.add_argument("--students", required=True, help="students.csv (now with willingness columns)")
    ap.add_argument("--out", default="teams.csv")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--k_robust", type=int, default=1)
    ap.add_argument("--lambda_red", type=float, default=1.0)
    ap.add_argument("--lambda_size", type=float, default=1.0)
    ap.add_argument("--lambda_will", type=float, default=0.5)
    args = ap.parse_args()

    required = read_skills(args.skills)
    students = read_students(args.students)

    recs = recommend_teams(
        required,
        students,
        team_size_min=2,
        team_size_max=5,
        top_k=args.topk,
        tau=args.tau,
        k=args.k_robust,
        lambda_red=args.lambda_red,
        lambda_size=args.lambda_size,
        lambda_will=args.lambda_will,
    )

    fieldnames = ["rank", "team_ids", "team_names", "goodness", "coverage", "k_robustness", "redundancy", "set_size", "willingness_avg", "willingness_min"]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(recs, start=1):
            w.writerow({
                "rank": i,
                "team_ids": ";".join(r["team_ids"]),
                "team_names": "; ".join(r["team_names"]),
                "goodness": r["goodness"],
                "coverage": round(r["metrics"]["coverage"], 3),
                "k_robustness": round(r["metrics"]["k_robustness"], 3),
                "redundancy": round(r["metrics"]["redundancy"], 3),
                "set_size": round(r["metrics"]["set_size"], 3),
                "willingness_avg": round(r["metrics"]["willingness_avg"], 3),
                "willingness_min": round(r["metrics"]["willingness_min"], 3),
            })
    print(f"Wrote top {len(recs)} teams to {args.out}")

if __name__ == "__main__":
    main()
