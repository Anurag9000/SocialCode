# AI Assistant Instructions for Social Code Scripts

## Project Overview
This codebase implements a sophisticated team recommendation system using ML-based skill matching and team optimization. The system helps form effective teams for village/community development projects by matching required skills with available team members while considering their willingness to participate.

## Key Components

### 1. Skill Extraction and Management
- Main component: `embed_skills_extractor.py`
- Uses TF-IDF embeddings to map input text to canonical skills
- Extensive skill bank for village-level projects (WASH, irrigation, agriculture, etc.)
- Supports external skill extensions via JSON (`extra_skills_json`)
- Fallback mechanism for short inputs with domain-specific heuristics

### 2. Team Recommendation Engine
- Core file: `m3_recommend.py` (M3 + ULTRA with willingness)
- Key metrics:
  - Coverage: Similarity-weighted skill matching (using embeddings)
  - k-Robustness: Team resilience to member removal
  - Redundancy: Overlap in skill coverage
  - Set size: Team size optimization
  - Goodness score: Combined metric `(coverage + k_robustness - redundancy - set_size + 2)/4`

### 3. Model Training
- Implemented in `m3_trainer.py`
- Uses GradientBoostingClassifier with embedding-based features
- Supports both sentence-transformers and TF-IDF backends
- Handles train/val split and model persistence

## Data Formats

### Input CSVs
- `people.csv`/`students.csv`: Team member profiles with skills and willingness
  - Required columns: person_id|student_id, name, skills (semicolon-separated)
  - Optional: willingness_eff, willingness_bias (default 0.5)
- `proposals.csv`: Project descriptions
  - Flexible headers: text|proposal_text|description|body|content

## Common Workflows

### 1. Team Recommendation
```powershell
python m3_recommend.py `
  --model model.pkl `
  --people people.csv `
  --proposal_text "project description" `
  --out teams_m3.csv `
  --tau 0.35 --soft_cap 6 `
  --auto_extract --threshold 0.20
```

### 2. Skill Extraction
```powershell
python embed_skills_extractor.py `
  --text "requirement text" `
  --out skills.json `
  --threshold 0.20 `
  --fallback_if_empty
```

## Project Conventions

### Skill Processing
- Skills are case-insensitive and normalized
- Skills can be semicolon-separated or comma-separated phrases
- SWAM-style willingness calculation: `W = sigmoid(effectiveness + bias)`

### Embedding Usage
- Default model: sentence-transformers/all-MiniLM-L6-v2
- Fallback to TF-IDF when needed (with shared vectorizer)
- Cosine similarity threshold (tau) typically 0.35 for skill matching

### Error Handling
- Robust CSV header handling with flexible column names
- Fallback mechanisms for missing/empty skills
- Input validation with helpful error messages

## Key Files Reference
- `m3_recommend.py`: Main recommendation engine
- `embed_skills_extractor.py`: Skill extraction and management
- `m3_trainer.py`: Model training pipeline
- `team_builder_embeddings.py`: Core team optimization logic
- `embeddings.py`: Shared embedding utilities