"""
m3_recommend.py  (with willingness)

- Candidate ranking uses trained model over features derived from backend embeddings (the model already learned W).
- ULTRA coverage/k-robustness use W-weighted similarity.
"""

import argparse, csv, pickle, math
from typing import List, Dict, Tuple
import numpy as np
from embeddings import embed_with
from sklearn.metrics.pairwise import cosine_similarity

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def read_people(people_csv: str) -> List[Dict]:
    rows = []
    with open(people_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            skills = [s.strip() for s in r["text"].split(";") if s.strip()]
            eff = float(r.get("willingness_eff", 0.5))
            bias = float(r.get("willingness_bias", 0.5))
            W = sigmoid(eff + bias)
            rows.append({
                "person_id": r["person_id"],
                "name": r.get("name", r["person_id"]),
                "text": r["text"],
                "skills": skills,
                "W": W
            })
    return rows

def similarity_coverage(required: List[str], team_members: List[Dict], backend: str, model_or_vec, tau: float = 0.35):
    # Build phrase list and W per phrase owner
    phrases, ownersW = [], []
    for m in team_members:
        for sk in m["skills"]:
            phrases.append(sk)
            ownersW.append(m["W"])
    if not required or not phrases:
        if not required:
            return 0.0, {}, {}
        return 0.0, {r:0.0 for r in required}, {r:0 for r in required}

    R = embed_with(model_or_vec, required, backend)
    M = embed_with(model_or_vec, phrases, backend)
    S = cosine_similarity(R, M)
    Wcol = np.array(ownersW).reshape(1, -1)
    S_w = S * Wcol

    per_req_best = S_w.max(axis=1)
    covered_counts = {}
    for i in range(len(required)):
        row = np.array(S_w[i]).ravel()
        covered_counts[required[i]] = int((row >= tau).sum())
    adj = []
    for s in per_req_best:
        s = float(s)
        if s < tau: adj.append(0.0)
        else: adj.append((s - tau) / (1.0 - tau))
    coverage = float(np.mean(adj)) if len(adj) else 0.0
    return coverage, {required[i]: float(per_req_best[i]) for i in range(len(required))}, covered_counts

def redundancy_metric(required: List[str], covered_counts: Dict[str, int]) -> float:
    if not required: return 0.0
    redundant = sum(1 for r in required if covered_counts.get(r, 0) >= 2)
    return redundant / len(required)

def k_robustness(required: List[str], team_members: List[Dict], backend: str, model_or_vec, tau: float = 0.35) -> float:
    if not team_members: return 0.0
    cov, per_req_best, _ = similarity_coverage(required, team_members, backend, model_or_vec, tau)
    if not all(v >= tau for v in per_req_best.values()): return 0.0
    ok = 0
    for i in range(len(team_members)):
        sub_members = [m for j, m in enumerate(team_members) if j != i]
        _, per_req_best2, _ = similarity_coverage(required, sub_members, backend, model_or_vec, tau)
        if all(v >= tau for v in per_req_best2.values()):
            ok += 1
    return ok / len(team_members)

def goodness(metrics: Dict[str, float]) -> float:
    s = (+1)*metrics["coverage"] + (+1)*metrics["k_robustness"] + (-1)*metrics["redundancy"] + (-1)*metrics["set_size"]
    return (s + 2.0) / 4.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--proposal_text", required=True)
    ap.add_argument("--people", required=True)
    ap.add_argument("--required_skills", nargs="+", help="Optional explicit required skills; otherwise default water/river set.")
    ap.add_argument("--out", default="teams_m3.csv")
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--soft_cap", type=int, default=6)
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    clf = bundle["model"]; backend = bundle["backend"]
    prop_model = bundle["prop_model"]; people_model = bundle["people_model"]

    people = read_people(args.people)

    # Candidate probabilities (model learned W via training features)
    P = embed_with(prop_model, [args.proposal_text], backend)
    S = embed_with(people_model, [p["text"] for p in people], backend)
    sims = cosine_similarity(P, S).ravel()
    X = np.stack([sims, sims*np.array([p["W"] for p in people]), np.array([p["W"] for p in people])], axis=1)
    probs = clf.predict_proba(X)[:,1]
    ranked = sorted(zip(people, probs), key=lambda x: x[1], reverse=True)

    required = args.required_skills if args.required_skills else [
        "village","gram","panchayat","ward","toilet","drain","waste","river","water",
            "angawadi","anganwadi","school","handpump","borewell","harvesting","mgnrega",
            "shg","pmay","health","nutrition","road","culvert"
    ]

    # Greedy build with W-weighted ULTRA coverage
    team = []
    def current_metrics(team_list):
        coverage, _, covered_counts = similarity_coverage(required, team_list, backend, people_model, tau=args.tau)
        redundancy = redundancy_metric(required, covered_counts)
        set_size = min(len(team_list)/max(len(required),4), 1.0)
        krob = k_robustness(required, team_list, backend, people_model, tau=args.tau)
        return {"coverage": coverage, "redundancy": redundancy, "set_size": set_size, "k_robustness": krob}

    best_m = {"coverage":0,"redundancy":0,"set_size":0,"k_robustness":0}
    for person, p in ranked:
        if len(team) >= args.soft_cap: break
        cand = team + [person]
        mets = current_metrics(cand)
        if mets["coverage"] > best_m["coverage"] or goodness(mets) > goodness(best_m):
            team = cand; best_m = mets
        if mets["coverage"] >= 0.999 and mets["k_robustness"] >= 0.999: break

    # Export greedy + simple 1-swap variants
    recs = []
    base_good = goodness(best_m)
    recs.append({
        "team_ids": ";".join([m["person_id"] for m in team]),
        "team_names": "; ".join([m["name"] for m in team]),
        "goodness": round(base_good,4),
        "coverage": round(best_m["coverage"],3),
        "k_robustness": round(best_m["k_robustness"],3),
        "redundancy": round(best_m["redundancy"],3),
        "set_size": round(best_m["set_size"],3)
    })

    for person, p in ranked[:10]:
        if person in team: continue
        for i in range(len(team)):
            variant = team.copy(); variant[i] = person
            mets = current_metrics(variant)
            recs.append({
                "team_ids": ";".join([m["person_id"] for m in variant]),
                "team_names": "; ".join([m["name"] for m in variant]),
                "goodness": round(goodness(mets),4),
                "coverage": round(mets["coverage"],3),
                "k_robustness": round(mets["k_robustness"],3),
                "redundancy": round(mets["redundancy"],3),
                "set_size": round(mets["set_size"],3)
            })

    # dedupe + sort
    dedup = {(r['team_ids'], r['team_names']): r for r in recs}.values()
    recs = sorted(dedup, key=lambda r: (r["goodness"], r["coverage"]), reverse=True)[:10]

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank","team_ids","team_names","goodness","coverage","k_robustness","redundancy","set_size"])
        w.writeheader()
        for i, r in enumerate(recs, start=1):
            r2 = dict(r); r2["rank"] = i; w.writerow(r2)
    print(f"Wrote {len(recs)} teams to {args.out}")

if __name__ == "__main__":
    main()
