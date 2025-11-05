"""
m3_recommend.py  — M3 + ULTRA with willingness, expanded village skills, auto extraction

What’s new:
- Flexible CSV headers: person_id|student_id, text|skills
- Auto skill extraction from proposal_text using embed_skills_extractor.extract_skills_embed if available
- Or pass --skills_json (path to JSON: either ["skill", ...]  OR  {"skills":[...]} )
- Strong village fallback skills if nothing supplied
- W-weighted coverage, k-robustness, redundancy, set-size → goodness

Usage (PowerShell):
  python m3_recommend.py `
    --model model.pkl `
    --people people.csv `
    --proposal_text "village drains blocked; handpump broken; need toilets and awareness" `
    --out teams_m3.csv `
    --tau 0.35 --soft_cap 6 `
    --auto_extract --threshold 0.20

Or if you already have a skills JSON:
  python m3_recommend.py --model model.pkl --people people.csv --proposal_text "..." --skills_json skills.json --out teams_m3.csv
"""

import argparse, csv, json, pickle, math, os, re
from typing import List, Dict, Tuple, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import embed_with
from datetime import datetime, timedelta, timezone
from collections import defaultdict

AVAILABILITY_LEVELS = {
    "rarely available": 0,
    "generally available": 1,
    "immediately available": 2,
}

SEVERITY_LABELS = {0: "LOW", 1: "NORMAL", 2: "HIGH"}
SEVERITY_KEYWORDS = {
    2: ["urgent", "immediate", "critical", "outbreak", "epidemic", "collapse", "broken", "flood", "drought", "disease", "contamination", "crisis", "emergency"],
    1: ["audit", "survey", "assessment", "monitoring", "planning", "inspection", "review", "repair", "maintenance"],
}

SEVERITY_AVAILABILITY_PENALTIES = {
    2: {"generally available": 0.2, "rarely available": 0.4},
    1: {"rarely available": 0.2},
    0: {},
}

# ---------------- utils ----------------

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def read_csv_norm(fp: str) -> List[Dict[str, Any]]:
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

def get_any(d: Dict[str, Any], keys, default=None):
    for k in keys:
        if k in d and d[k] not in ("", None):
            return d[k]
    return default

def normalize_phrase(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

def load_village_names(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    rows = read_csv_norm(path)
    names = [get_any(r, ["village_name", "village", "name"], "") for r in rows]
    names = [n for n in names if n]
    return sorted(names, key=lambda n: len(n), reverse=True)

def load_distance_lookup(path: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    lookup: Dict[Tuple[str, str], Dict[str, float]] = {}
    if not path or not os.path.exists(path):
        return lookup
    rows = read_csv_norm(path)
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
    norm_text = normalize_phrase(text or "")
    for name in village_names:
        if normalize_phrase(name) in norm_text:
            return name
    return ""

def estimate_severity(text: str) -> int:
    text_norm = (text or "").lower()
    for kw in SEVERITY_KEYWORDS[2]:
        if kw in text_norm:
            return 2
    for kw in SEVERITY_KEYWORDS[1]:
        if kw in text_norm:
            return 1
    return 0

def severity_penalty(availability_label: str, severity_level: int) -> float:
    label = (availability_label or "").lower()
    penalties = SEVERITY_AVAILABILITY_PENALTIES.get(severity_level, {})
    return penalties.get(label, 0.0)

def lookup_distance_km(origin: str, target: str, distance_lookup: Dict[Tuple[str, str], Dict[str, float]]) -> float:
    if not origin or not target:
        return 0.0
    rec = distance_lookup.get((origin.lower(), target.lower()))
    if rec:
        return float(rec.get("distance", 0.0))
    return 0.0

# ------------- expanded fallback skills -------------
# Canonical, skill-like phrases (not just keywords) for village projects
VILLAGE_FALLBACK_SKILLS = [
    # WASH
    "water quality assessment",
    "drainage design and de-silting",
    "handpump repair and maintenance",
    "borewell installation and rehabilitation",
    "rainwater harvesting",
    "fecal sludge management",
    "toilet construction and retrofitting",
    "solid waste segregation and composting",
    "hygiene behavior change communication",
    # Rivers / irrigation / groundwater
    "river restoration",
    "watershed management",
    "groundwater assessment and monitoring",
    "check dam and nala bund construction",
    "farm pond design and maintenance",
    # Agriculture & livelihoods
    "soil testing and fertility management",
    "integrated pest management",
    "drip and sprinkler irrigation setup",
    "dairy and livestock management",
    "fisheries pond management",
    # Energy & infrastructure
    "solar microgrid design and maintenance",
    "solar pumping systems",
    "rural road maintenance and culvert repair",
    "culvert and causeway design",
    "low-cost housing construction and PMAY support",
    # Governance & community
    "panchayat planning and budgeting",
    "gram sabha facilitation",
    "self-help group formation and strengthening",
    "mgnrega works planning and measurement",
    "beneficiary identification and targeting",
    # Health, education
    "public health outreach",
    "school wq testing and wash in schools",
    "anganwadi strengthening",
    "education and digital literacy",
    # Environment & climate
    "tree plantation and survival monitoring",
    "erosion control and gully plugging",
    "biodiversity and habitat restoration",
    "disaster preparedness and response",
    # Data & ops
    "gis and remote sensing",
    "household survey and enumeration",
    "data analysis and reporting",
    "mobile data collection and dashboards",
    "sensor deployment and iot",
    "project management",
    "safety and risk management",
]

# -------- reading people / roster (robust) --------

def read_people(people_csv: str) -> List[Dict]:
    rows = read_csv_norm(people_csv)
    out = []
    for r in rows:
        pid = get_any(r, ["person_id","student_id","id"])
        if not pid:
            # skip rows without id
            continue
        name = get_any(r, ["name","person_name","full_name"], pid)
        text = get_any(r, ["text","skills"], "")
        # turn either "text" or "skills" into a list of phrases; accept both sentencey or ; separated
        if "skills" in r and r["skills"]:
            raw = r["skills"]
        else:
            raw = text
        # split on semicolons if present; otherwise keep sentences/phrases by splitting on commas as a fallback
        if ";" in raw:
            skills = [s.strip() for s in raw.split(";") if s.strip()]
        else:
            skills = [s.strip() for s in raw.replace("  ", " ").split(",") if s.strip()]
        # willingness
        try:
            eff = float(get_any(r, ["willingness_eff","eff","w_eff"], 0.5))
        except Exception:
            eff = 0.5
        try:
            bias = float(get_any(r, ["willingness_bias","bias","w_bias"], 0.5))
        except Exception:
            bias = 0.5
        W = sigmoid(eff + bias)
        out.append({
            "person_id": pid,
            "name": name,
            "text": text,
            "skills": skills,
            "W": W,
            "availability": (get_any(r, ["availability"], "") or "").lower(),
            "home_location": get_any(r, ["home_location", "location", "village"], ""),
        })
    return out

# --------- ULTRA metrics with willingness ----------

def similarity_coverage(required: List[str], team_members: List[Dict], backend: str, model_or_vec, tau: float = 0.35):
    """
    W-weighted similarity coverage with member-level aggregation.
    """
    member_phrases = []
    ownersW = []
    member_index = []
    for idx, m in enumerate(team_members):
        for sk in m["skills"]:
            member_phrases.append(sk)
            ownersW.append(m["W"])
            member_index.append(idx)
    if not required:
        return 0.0, {}, {}
    if not member_phrases:
        return 0.0, {r: 0.0 for r in required}, {r: 0 for r in required}

    R = embed_with(model_or_vec, required, backend)
    M = embed_with(model_or_vec, member_phrases, backend)
    S = cosine_similarity(R, M)
    Wcol = np.array(ownersW).reshape(1, -1)
    S_w = S * Wcol

    per_member_best = [np.zeros(len(required), dtype=float) for _ in range(len(team_members))]
    for col in range(S_w.shape[1]):
        owner = member_index[col]
        col_vals = np.array(S_w[:, col]).ravel()
        per_member_best[owner] = np.maximum(per_member_best[owner], col_vals)

    per_req_best = np.max(np.stack(per_member_best, axis=0), axis=0) if len(team_members) else np.zeros(len(required))

    covered_counts = {}
    for i, req in enumerate(required):
        count = sum(1 for u in range(len(team_members)) if per_member_best[u][i] >= tau)
        covered_counts[req] = int(count)

    adj = []
    for s in per_req_best:
        s = float(s)
        if s < tau:
            adj.append(0.0)
        else:
            adj.append((s - tau) / (1.0 - tau))
    coverage = float(np.mean(adj)) if len(adj) else 0.0
    return coverage, {required[i]: float(per_req_best[i]) for i in range(len(required))}, covered_counts

def redundancy_metric(required: List[str], covered_counts: Dict[str, int]) -> float:
    if not required: return 0.0
    redundant = sum(1 for r in required if covered_counts.get(r, 0) >= 2)
    return redundant / len(required)

def k_robustness(required: List[str], team_members: List[Dict], backend: str, model_or_vec, tau: float = 0.35, k: int = 1, sample_limit: int = 2000) -> float:
    if not team_members:
        return 0.0
    _, per_req_best, _ = similarity_coverage(required, team_members, backend, model_or_vec, tau)
    if not all(v >= tau for v in per_req_best.values()):
        return 0.0
    n = len(team_members)
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
        sub_members = [m for i, m in enumerate(team_members) if i not in rem]
        _, per_req_best2, _ = similarity_coverage(required, sub_members, backend, model_or_vec, tau)
        if all(v >= tau for v in per_req_best2.values()):
            ok += 1
    return ok / len(subsets) if subsets else 0.0

def team_metrics(required: List[str], team_members: List[Dict], backend: str, model_or_vec, tau: float = 0.35, k: int = 1) -> Dict[str, float]:
    coverage, per_req_best, covered_counts = similarity_coverage(required, team_members, backend, model_or_vec, tau)
    redundancy = redundancy_metric(required, covered_counts)
    soft_cap = max(len(required), 4)
    set_size = min(len(team_members) / soft_cap, 1.0)
    krob = k_robustness(required, team_members, backend, model_or_vec, tau, k=k)
    w_vals = [float(m.get("W", 0.0)) for m in team_members]
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

# ------------- size-bucket selection -------------

DEFAULT_SIZE_BUCKETS = "small:2-10:10,medium:11-50:10,large:51-200:10"

def parse_size_buckets(spec: str):
    """
    Parse a size bucket specification like:
      "small:2-3:20,medium:4-6:100,high:7-999:500"
    meaning label:min-max:limit, min/max inclusive, max can be 'inf'.
    """
    buckets = []
    if not spec:
        return buckets
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        try:
            label, size_range, limit = [x.strip() for x in part.split(":")]
        except ValueError:
            raise SystemExit(f"Invalid --size_buckets entry '{part}'. Expected label:min-max:limit.")
        if "-" in size_range:
            min_s, max_s = [x.strip() for x in size_range.split("-", 1)]
        else:
            min_s = max_s = size_range.strip()
        try:
            min_size = int(min_s)
        except ValueError:
            raise SystemExit(f"Invalid min size '{min_s}' in size bucket '{part}'.")
        if max_s.lower() in ("inf", "infinity", "*", "max"):
            max_size = math.inf
        else:
            try:
                max_size = int(max_s)
            except ValueError:
                raise SystemExit(f"Invalid max size '{max_s}' in size bucket '{part}'.")
        try:
            limit_int = int(limit)
        except ValueError:
            raise SystemExit(f"Invalid limit '{limit}' in size bucket '{part}'.")
        if limit_int < 0:
            raise SystemExit(f"Limit must be >= 0 in size bucket '{part}'.")
        buckets.append({
            "label": label,
            "min": min_size,
            "max": max_size,
            "limit": limit_int,
        })
    return buckets

def select_top_teams_by_size(teams: List[Dict], buckets):
    if not buckets:
        return teams
    grouped = {b["label"]: [] for b in buckets}
    for team in teams:
        size = team.get("team_size")
        if size is None:
            size = team.get("team_ids", "").count(";") + 1 if team.get("team_ids") else 0
        assigned = False
        for bucket in buckets:
            min_size = bucket["min"]
            max_size = bucket["max"]
            if size < min_size or (max_size is not math.inf and size > max_size):
                continue
            label = bucket["label"]
            if len(grouped[label]) < bucket["limit"]:
                grouped[label].append(team)
            assigned = True
            break
        if not assigned:
            # Unbucketed teams are ignored for now.
            pass
    ordered = []
    for bucket in buckets:
        ordered.extend(grouped[bucket["label"]])
    return ordered

# ------------- scheduling & workload helpers -------------

def parse_datetime(value: str, label: str) -> datetime:
    if not value:
        raise SystemExit(f"{label} is required.")
    value = value.strip()
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise SystemExit(f"{label} '{value}' is not a valid ISO-8601 timestamp.")
    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt

def split_hours_by_week(start: datetime, end: datetime) -> Dict[Tuple[int, int], float]:
    if end <= start:
        return {}
    hours_by_week: Dict[Tuple[int, int], float] = defaultdict(float)
    cursor = start
    while cursor < end:
        iso_year, iso_week, _ = cursor.isocalendar()
        week_key = (iso_year, iso_week)
        week_start = cursor - timedelta(days=cursor.weekday())
        week_end = week_start + timedelta(days=7)
        segment_end = min(end, week_end)
        hours = (segment_end - cursor).total_seconds() / 3600.0
        hours_by_week[week_key] += hours
        cursor = segment_end
    return dict(hours_by_week)

def intervals_overlap(intervals: List[Tuple[datetime, datetime]], new_interval: Tuple[datetime, datetime]) -> bool:
    ns, ne = new_interval
    for s, e in intervals:
        if s < ne and ns < e:
            return True
    return False

def parse_schedule_csv(path: str) -> Dict[str, Dict[str, Any]]:
    rows = read_csv_norm(path)
    schedule: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        pid = get_any(row, ["person_id", "student_id", "volunteer_id", "id"])
        if not pid:
            continue
        start_raw = get_any(row, ["start", "start_time", "begin", "start_datetime"])
        end_raw = get_any(row, ["end", "end_time", "finish", "finish_time", "end_datetime"])
        if not start_raw or not end_raw:
            continue
        try:
            start_dt = parse_datetime(start_raw, f"schedule start for {pid}")
            end_dt = parse_datetime(end_raw, f"schedule end for {pid}")
        except SystemExit as exc:
            raise
        if end_dt <= start_dt:
            continue
        info = schedule.setdefault(pid, {"intervals": [], "week_hours": defaultdict(float)})
        info["intervals"].append((start_dt, end_dt))
        week_hours = split_hours_by_week(start_dt, end_dt)
        for wk, hrs in week_hours.items():
            info["week_hours"][wk] += hrs
    # normalize defaultdicts to dicts
    for info in schedule.values():
        if isinstance(info.get("week_hours"), defaultdict):
            info["week_hours"] = dict(info["week_hours"])
    return schedule

# ------------- skill acquisition logic -------------

def _load_skills_json(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    if isinstance(blob, list):
        return [str(x) for x in blob]
    if isinstance(blob, dict) and "skills" in blob and isinstance(blob["skills"], list):
        return [str(x) for x in blob["skills"]]
    raise SystemExit(f"--skills_json must be a JSON array or an object with a 'skills' array.")

def _auto_extract_skills(text: str, threshold: float) -> List[str]:
    """
    Try to import your expanded extractor. If not present, return a compact
    set from VILLAGE_FALLBACK_SKILLS filtered by simple keyword heuristics.
    """
    try:
        # Attempt to use your expanded extractor file in the same directory
        import embed_skills_extractor as ex
        # ex.extract_skills_embed(text, topk_per_sentence=7, threshold=threshold)
        return ex.extract_skills_embed(text, topk_per_sentence=7, threshold=threshold)
    except Exception:
        # Heuristic fallback: pick subset of village skills based on text keywords
        t = (text or "").lower()
        keys = ["village","gram","panchayat","ward","toilet","drain","waste","river","water",
                "anganwadi","school","handpump","borewell","harvesting","mgnrega","shg",
                "pmay","health","nutrition","road","culvert","solar","gis","survey","iot"]
        if any(k in t for k in keys):
            # return a reasonable, diverse subset
            base = [
                "drainage design and de-silting",
                "handpump repair and maintenance",
                "toilet construction and retrofitting",
                "solid waste segregation and composting",
                "water quality assessment",
                "rainwater harvesting",
                "watershed management",
                "soil testing and fertility management",
                "panchayat planning and budgeting",
                "self-help group formation and strengthening",
                "public health outreach",
                "education and digital literacy",
                "gis and remote sensing",
                "data analysis and reporting",
                "project management",
            ]
            return base
        # absolute fallback
        return VILLAGE_FALLBACK_SKILLS[:12]

# ------------------------ main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--proposal_text", help="Raw project/proposal text (use --proposal_file to read from file)")
    ap.add_argument("--proposal_file", help="Path to a text file containing the proposal")
    ap.add_argument("--people", required=True)
    ap.add_argument("--required_skills", nargs="+", help="Explicit required skills (overrides auto/JSON)")
    ap.add_argument("--skills_json", help="Path to JSON with ['skill', ...] or {'skills':[...]} to use as required")
    ap.add_argument("--auto_extract", action="store_true", help="Auto extract skills from proposal_text using extractor or fallback")
    ap.add_argument("--threshold", type=float, default=0.25, help="cosine threshold for extraction")
    ap.add_argument("--out", default="teams_m3.csv")
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--task_start", required=True, help="ISO timestamp for when the task/team work begins")
    ap.add_argument("--task_end", required=True, help="ISO timestamp for when the task/team work ends")
    default_dataset_root = r"D:\SocialCode\gram_sahayta_dataset_with_locations_and_availability"
    ap.add_argument("--village_locations", default=os.path.join(default_dataset_root, "village_locations.csv"))
    ap.add_argument("--distance_csv", default=os.path.join(default_dataset_root, "village_distances.csv"))
    ap.add_argument("--distance_scale", type=float, default=50.0, help="Distance in km mapped to 1.0 in features")
    ap.add_argument("--distance_decay", type=float, default=30.0, help="Decay constant (km) for distance penalty exp(-d/decay)")
    ap.add_argument("--severity", choices=["LOW", "NORMAL", "HIGH"], help="Override detected severity (default inferred from text)")
    ap.add_argument("--schedule_csv", help="Existing volunteer schedule CSV with columns person_id,start,end[,hours]")
    ap.add_argument("--weekly_quota", type=float, default=5.0, help="Weekly hour quota before overwork penalties")
    ap.add_argument("--overwork_penalty", type=float, default=0.1, help="Penalty multiplier per overwork hour")
    ap.add_argument("--soft_cap", type=int, default=6)
    ap.add_argument("--topk_swap", type=int, default=10, help="How many top alternatives to consider for 1-swap variants")
    ap.add_argument("--k_robust", type=int, default=1, help="Robustness level k (removals) to preserve coverage")
    ap.add_argument("--lambda_red", type=float, default=1.0, help="Weight for redundancy penalty")
    ap.add_argument("--lambda_size", type=float, default=1.0, help="Weight for set-size penalty")
    ap.add_argument("--lambda_will", type=float, default=0.5, help="Weight for willingness encouragement")
    ap.add_argument("--size_buckets", default=DEFAULT_SIZE_BUCKETS,
                    help="Comma separated label:min-max:limit groups for size-based top-K selection "
                         "(default 'small:2-10:10,medium:11-50:10,large:51-200:10')")
    args = ap.parse_args()

    if not args.proposal_text and not args.proposal_file:
        raise SystemExit("Provide --proposal_text or --proposal_file")

    if args.proposal_file and not os.path.exists(args.proposal_file):
        raise SystemExit(f"Not found: {args.proposal_file}")

    if not os.path.exists(args.model):
        raise SystemExit(f"Not found: {args.model}")
    if not os.path.exists(args.people):
        raise SystemExit(f"Not found: {args.people}")
    if args.schedule_csv and not os.path.exists(args.schedule_csv):
        raise SystemExit(f"Not found: {args.schedule_csv}")
    if args.village_locations and not os.path.exists(args.village_locations):
        raise SystemExit(f"Not found: {args.village_locations}")
    if args.distance_csv and not os.path.exists(args.distance_csv):
        raise SystemExit(f"Not found: {args.distance_csv}")

    text = args.proposal_text
    if args.proposal_file:
        with open(args.proposal_file, "r", encoding="utf-8") as f:
            text = f.read()

    village_names = load_village_names(args.village_locations)
    distance_lookup = load_distance_lookup(args.distance_csv)
    proposal_location = extract_location(text, village_names)
    if proposal_location:
        print(f"Detected proposal location: {proposal_location}")
    elif village_names:
        print("[recommender] Warning: proposal text did not match a known village name; distance penalties will be zero.")
    if args.severity:
        severity_level = {"LOW": 0, "NORMAL": 1, "HIGH": 2}[args.severity]
    else:
        severity_level = estimate_severity(text)
    severity_label = SEVERITY_LABELS.get(severity_level, "NORMAL")
    print(f"Detected severity: {severity_label}")

    task_start = parse_datetime(args.task_start, "--task_start")
    task_end = parse_datetime(args.task_end, "--task_end")
    if task_end <= task_start:
        raise SystemExit("--task_end must be after --task_start")
    task_interval = (task_start, task_end)
    task_week_hours = split_hours_by_week(task_start, task_end)
    if not task_week_hours:
        raise SystemExit("Task duration must be positive.")
    schedule_map = parse_schedule_csv(args.schedule_csv) if args.schedule_csv else {}

    # Load model bundle
    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    clf = bundle["model"]
    backend = bundle["backend"]
    prop_model = bundle["prop_model"]
    people_model = bundle["people_model"]
    distance_scale = bundle.get("distance_scale", args.distance_scale)
    distance_decay = bundle.get("distance_decay", args.distance_decay)

    # People / roster
    people = read_people(args.people)
    if not people:
        raise SystemExit("No valid rows found in people CSV.")
    filtered_people: List[Dict[str, Any]] = []
    conflicts = 0
    for person in people:
        sched_info = schedule_map.get(person["person_id"])
        intervals = sched_info.get("intervals", []) if sched_info else []
        if intervals and intervals_overlap(intervals, task_interval):
            conflicts += 1
            continue
        week_hours_map = sched_info.get("week_hours", {}) if sched_info else {}
        overwork_total = 0.0
        for week_key, hrs in task_week_hours.items():
            existing = float(week_hours_map.get(week_key, 0.0))
            total_hours = existing + hrs
            overwork_total += max(0.0, total_hours - args.weekly_quota)
        adjusted = dict(person)
        base_W = person["W"]
        penalty_overwork = args.overwork_penalty * overwork_total
        adjusted_W = max(0.0, min(1.0, base_W - penalty_overwork))
        adjusted["W_original"] = base_W
        adjusted["W_base"] = adjusted_W
        adjusted["W"] = adjusted_W
        adjusted["overwork_hours"] = overwork_total
        adjusted["availability"] = (adjusted.get("availability") or "").lower()
        filtered_people.append(adjusted)
    if conflicts:
        print(f"Excluded {conflicts} volunteers due to overlapping assignments.")
    people = filtered_people
    if not people:
        raise SystemExit("No available volunteers after applying schedule and workload constraints.")

    # Acquire required skills
    if args.required_skills:
        required = [s for s in args.required_skills if s.strip()]
    elif args.skills_json:
        required = _load_skills_json(args.skills_json)
    elif args.auto_extract:
        required = _auto_extract_skills(text, args.threshold)
    else:
        required = VILLAGE_FALLBACK_SKILLS

    if not required:
        required = VILLAGE_FALLBACK_SKILLS

    # Candidate probabilities (model learned W via training features)
    # Embed proposal (1,d) and roster texts (n,d)
    P = embed_with(prop_model, [text], backend)
    S = embed_with(people_model, [p["text"] for p in people], backend)
    sims = cosine_similarity(P, S).ravel()
    features = []
    for idx, person in enumerate(people):
        avail_label = (person.get("availability") or "").lower()
        availability_level = AVAILABILITY_LEVELS.get(avail_label, 1)
        base_W = person.get("W_base", person["W"])
        sev_pen = severity_penalty(avail_label, severity_level)
        W_after_severity = max(0.0, min(1.0, base_W - sev_pen))
        distance_km = lookup_distance_km(person.get("home_location"), proposal_location, distance_lookup)
        distance_norm = min(distance_km / distance_scale, 1.0) if distance_scale > 0 else 0.0
        distance_penalty = math.exp(-distance_km / distance_decay) if distance_decay > 0 else 1.0
        W_adjusted = max(0.0, min(1.0, W_after_severity * distance_penalty))
        person["W"] = W_adjusted
        person["distance_km"] = distance_km
        person["distance_penalty"] = distance_penalty
        person["availability_level"] = availability_level
        person["severity_level"] = severity_level
        person["severity_penalty"] = sev_pen
        features.append([
            sims[idx],
            sims[idx] * W_adjusted,
            W_adjusted,
            distance_norm,
            distance_penalty,
            availability_level / 2.0,
            severity_level / 2.0,
        ])
    X = np.asarray(features)
    probs = clf.predict_proba(X)[:, 1]
    ranked = sorted(zip(people, probs), key=lambda x: x[1], reverse=True)

    def compute_metrics(team_list: List[Dict]) -> Dict[str, float]:
        return team_metrics(required, team_list, backend, people_model, tau=args.tau, k=args.k_robust)

    def evaluate(team_list: List[Dict]) -> Tuple[float, Dict[str, float]]:
        mets = compute_metrics(team_list)
        return goodness(
            mets,
            lambda_red=args.lambda_red,
            lambda_size=args.lambda_size,
            lambda_will=args.lambda_will,
        ), mets

    # Greedy build with robustness-aware marginal gain
    team: List[Dict] = []
    team_ids = set()
    team_score, best_m = evaluate(team)

    while len(team) < args.soft_cap:
        best_candidate = None
        best_candidate_score = None
        best_candidate_metrics = None
        best_candidate_prob = -1.0
        best_delta = 0.0
        for person, prob in ranked:
            if person["person_id"] in team_ids:
                continue
            candidate_team = team + [person]
            cand_score, cand_metrics = evaluate(candidate_team)
            delta = cand_score - team_score
            if delta > best_delta + 1e-9 or (
                abs(delta - best_delta) <= 1e-9 and (
                    cand_metrics["coverage"] > (best_candidate_metrics["coverage"] if best_candidate_metrics else -1.0) or
                    (
                        abs(cand_metrics["coverage"] - (best_candidate_metrics["coverage"] if best_candidate_metrics else -1.0)) <= 1e-9
                        and prob > best_candidate_prob
                    )
                )
            ):
                best_candidate = person
                best_candidate_score = cand_score
                best_candidate_metrics = cand_metrics
                best_candidate_prob = prob
                best_delta = delta
        if best_candidate is None or best_delta <= 1e-9:
            break
        team.append(best_candidate)
        team_ids.add(best_candidate["person_id"])
        team_score = best_candidate_score
        best_m = best_candidate_metrics
        if best_m["coverage"] >= 0.999 and best_m["k_robustness"] >= 0.999:
            break

    # Export greedy + 1-swap neighborhood variants
    def score_team(tlist: List[Dict]):
        return evaluate(tlist)

    def enforce_unique_volunteers(recs_list: List[Dict]) -> List[Dict]:
        assigned: Dict[str, bool] = {}
        resolved: List[Dict] = []
        for rec in recs_list:
            members = list(rec.get("members", []))
            keep_members = []
            removed_any = False
            for member in members:
                pid = member["person_id"]
                if pid in assigned:
                    removed_any = True
                    continue
                keep_members.append(member)
            if not keep_members:
                continue
            if removed_any:
                g, m = score_team(keep_members)
                rec = dict(rec)
                rec["members"] = keep_members
                rec["team_ids"] = ";".join([mbr["person_id"] for mbr in keep_members])
                rec["team_names"] = "; ".join([mbr["name"] for mbr in keep_members])
                rec["team_size"] = len(keep_members)
                rec["goodness"] = round(g, 4)
                rec["coverage"] = round(m["coverage"], 3)
                rec["k_robustness"] = round(m["k_robustness"], 3)
                rec["redundancy"] = round(m["redundancy"], 3)
                rec["set_size"] = round(m["set_size"], 3)
                rec["willingness_avg"] = round(m["willingness_avg"], 3)
                rec["willingness_min"] = round(m["willingness_min"], 3)
            else:
                rec = dict(rec)
                rec["members"] = keep_members
            for member in keep_members:
                assigned[member["person_id"]] = True
            resolved.append(rec)
        return resolved

    recs = []
    base_good, base_m = score_team(team)
    base_size = len(team)
    recs.append({
        "team_ids": ";".join([m["person_id"] for m in team]),
        "team_names": "; ".join([m["name"] for m in team]),
        "team_size": base_size,
        "goodness": round(base_good, 4),
        "coverage": round(base_m["coverage"], 3),
        "k_robustness": round(base_m["k_robustness"], 3),
        "redundancy": round(base_m["redundancy"], 3),
        "set_size": round(base_m["set_size"], 3),
        "willingness_avg": round(base_m["willingness_avg"], 3),
        "willingness_min": round(base_m["willingness_min"], 3),
        "members": list(team),
    })

    team_ids = {m["person_id"] for m in team}
    for person, _p in ranked[:max(1, args.topk_swap)]:
        if person["person_id"] in team_ids:
            continue
        for i in range(len(team)):
            variant = team.copy()
            variant[i] = person
            g, m = score_team(variant)
            recs.append({
                "team_ids": ";".join([mm["person_id"] for mm in variant]),
                "team_names": "; ".join([mm["name"] for mm in variant]),
                "team_size": len(variant),
                "goodness": round(g, 4),
                "coverage": round(m["coverage"], 3),
                "k_robustness": round(m["k_robustness"], 3),
                "redundancy": round(m["redundancy"], 3),
                "set_size": round(m["set_size"], 3),
                "willingness_avg": round(m["willingness_avg"], 3),
                "willingness_min": round(m["willingness_min"], 3),
                "members": list(variant),
            })

    # dedupe + sort
    dedup = {(r['team_ids'], r['team_names']): r for r in recs}.values()
    sorted_recs = sorted(dedup, key=lambda r: (r["goodness"], r["coverage"]), reverse=True)
    buckets = parse_size_buckets(args.size_buckets)
    final = select_top_teams_by_size(sorted_recs, buckets)
    if not final:
        final = sorted_recs[:10]
    final = enforce_unique_volunteers(final)

    # write CSV
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "team_ids",
                "team_names",
                "team_size",
                "goodness",
                "coverage",
                "k_robustness",
                "redundancy",
                "set_size",
                "willingness_avg",
                "willingness_min",
            ],
        )
        w.writeheader()
        for i, r in enumerate(final, start=1):
            row = dict(r)
            row.pop("members", None)
            row["rank"] = i
            w.writerow(row)

    print(f"Wrote {len(final)} teams to {args.out}")
    print(f"Required skills used ({len(required)}): {', '.join(required)}")

if __name__ == "__main__":
    main()
