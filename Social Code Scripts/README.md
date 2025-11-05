# Social Code Scripts - CLI & API Parameters

Every script in this repository is a self-contained CLI tool. The flags documented here map directly to parameters you can surface through a future API. Defaults target the Gram Sahayta dataset at `D:\SocialCode\gram_sahayta_dataset_with_locations_and_availability`.

---

## `m3_trainer.py`
Train the GradientBoosting classifier that scores volunteer-to-proposal compatibility using embeddings plus structured fairness features.

- `--proposals` **(required)**: CSV with proposal text (`text|proposal_text|description|body|content`).  
- `--people` **(required)**: CSV with volunteer profiles, willingness signals, availability category, and home village.  
- `--pairs` **(required)**: CSV of labelled proposal-person pairs (`label|y|target`).  
- `--out` (default `model.pkl`): Output bundle containing the classifier and embedding backends.  
- `--model_name` (default `sentence-transformers/all-MiniLM-L6-v2`): Transformer to try before falling back to TF-IDF.  
- `--village_locations` (default `...\village_locations.csv`): Master village list for proposal location parsing.  
- `--village_distances` (default `...\village_distances.csv`): Pairwise village distances (km plus travel minutes).  
- `--distance_scale` (default `50.0` km): Normalisation scale so `distance_km / distance_scale` is clipped to `[0, 1]`.  
- `--distance_decay` (default `30.0` km): Decay constant for the distance penalty `exp(-distance / distance_decay)` applied to willingness.

The trainer automatically finds the village mentioned in each proposal, estimates severity (LOW/NORMAL/HIGH) from keywords, maps availability strings to levels, and augments the feature matrix with distance and severity-aware penalties. The chosen distance hyperparameters are saved inside `model.pkl` so the recommender stays in sync.

---

## `m3_recommend.py`
Build teams for a new proposal while respecting skills, willingness, geography, availability fairness, pre-existing schedules, and weekly workload ceilings.

- `--model` **(required)**: Path to `model.pkl` from training.  
- `--proposal_text` / `--proposal_file`: Inline text or file path for the problem statement (mutually exclusive).  
- `--people` **(required)**: Volunteer roster (flexible headers for IDs, text/skills, willingness, availability, home location).  
- `--required_skills`: Explicit list of skills (overrides auto extraction).  
- `--skills_json`: JSON file with `["skill", ...]` or `{"skills":[...]}`.  
- `--auto_extract`: Enable automatic skill extraction (falls back to heuristics).  
- `--threshold` (default `0.25`): Cosine threshold for auto extraction.  
- `--out` (default `teams_m3.csv`): Output CSV.  
- `--tau` (default `0.35`): Coverage threshold for similarity metrics.  
- `--task_start` **(required)** / `--task_end` **(required)**: ISO-8601 timestamps defining when the task runs.  
- `--village_locations` (default `...\village_locations.csv`): Village list to locate the proposal.  
- `--distance_csv` (default `...\village_distances.csv`): Distance table for travel penalties.  
- `--distance_scale` (default `50.0` km) / `--distance_decay` (default `30.0` km): Match the training settings.  
- `--severity` (`LOW|NORMAL|HIGH`): Override the auto severity classifier (defaults to keyword inference).  
- `--schedule_csv`: Existing volunteer schedule (`person_id,start,end[,hours]`) to avoid clashes.  
- `--weekly_quota` (default `5.0` hours): Weekly hour budget before overwork penalties apply.  
- `--overwork_penalty` (default `0.1`): Willingness deduction per hour above the weekly quota.  
- `--soft_cap` (default `6`): Maximum team size considered during greedy selection.  
- `--topk_swap` (default `10`): Number of alternatives inspected for one-swap improvements.  
- `--k_robust` (default `1`): Required robustness level (team survives loss of any `k` members).  
- `--lambda_red`, `--lambda_size`, `--lambda_will` (defaults `1.0`, `1.0`, `0.5`): Weights for redundancy, size, and average willingness in the goodness score.  
- `--size_buckets` (default `small:2-10:10,medium:11-50:10,large:51-200:10`): `label:min-max:limit` rules for how many teams to return per size band.

Runtime pipeline:
1. Detect proposal village and severity (LOW/NORMAL/HIGH). HIGH severity penalises "generally available" volunteers moderately and "rarely available" volunteers heavily; NORMAL penalises "rarely available" only.  
2. Drop volunteers already booked during the specified window (`--schedule_csv`).  
3. Apply weekly workload penalties via `--weekly_quota` and `--overwork_penalty`.  
4. Apply distance decay (`exp(-distance / distance_decay)`) and severity-aware penalties to willingness before scoring with the model.  
5. Greedily assemble the best team under `--soft_cap`, then explore one-swap variants.  
6. Enforce per-bucket top-k limits via `--size_buckets`.  
7. Remove any volunteer who appears in multiple recommended teams; the lower-ranked team is recomputed without them so the final list is collision-free.

Output columns include `team_size`, `goodness`, `coverage`, `k_robustness`, `redundancy`, `set_size`, `willingness_avg`, and `willingness_min`. Distance and severity effects are reflected in the aggregated willingness metrics and goodness scores.

---

## `team_builder_embeddings.py`
Exhaustive enumerator for small rosters (useful for validation or debugging).

- `--skills` **(required)**: JSON list of required skills.  
- `--students` **(required)**: CSV roster with willingness columns.  
- `--out` (default `teams.csv`).  
- `--topk` (default `10`).  
- `--tau` (default `0.35`).  
- `--k_robust` (default `1`).  
- `--lambda_red`, `--lambda_size`, `--lambda_will` (defaults `1.0`, `1.0`, `0.5`): Align weights with the recommender for comparable scores.

Outputs mirror the recommender metrics, including willingness averages/minima.

---

## `embed_skills_extractor.py`
Map free-form proposal text to canonical Gram Sahayta skill phrases.

- `--text` or `--file`: Inline text or path to a `.txt` file.  
- `--out` (default `skills.json`): Destination JSON.  
- `--threshold` (default `0.25`): Cosine threshold to accept a skill.  
- `--fallback_if_empty`: Use the domain fallback skill list when nothing clears the threshold.  
- `--extra_skills_json`: JSON with `{"skills": [...], "synonyms": {...}}` to extend the skill bank.

---

## Legacy Helpers

- `m3_recommend_early.py`: Early recommender variant (flags: `--model`, `--proposal_text`, `--people`, `--required_skills`, `--out`, `--tau`, `--soft_cap`).  
- `embed_skills_extractor_early.py`: Minimal extractor (flags: `--text`, `--file`, `--out`). Prefer the main extractor for production.

---

### Integration Notes
- CLI flags are API-ready; no environment variables are consumed. Boolean flags are simple presence switches.  
- Time arguments must be ISO-8601; the scheduler converts to UTC and keeps volunteers collision-free.  
- Severity/availability penalties follow a fixed heuristic (HIGH penalises "generally" and "rarely"; NORMAL penalises "rarely"; LOW applies no penalty). Override `--severity` or adjust the source if policy changes.  
- Distance penalties depend on the supplied village assets; point the CLI to different CSVs if you deploy in another geography.  
- Goodness weights (`lambda_*`), workload penalties, and distance parameters are hot-swappable so you can tune fairness without code edits.

Keep this document in sync whenever CLI options change; downstream services will reference it as the parameter contract.
