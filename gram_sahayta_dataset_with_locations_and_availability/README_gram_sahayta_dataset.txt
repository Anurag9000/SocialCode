Gram Sahayta Project Matching Dataset (NSS / Rural Thematic)

FILES IN THIS ZIP
=================

1) proposals.csv
----------------
Columns:
- proposal_id : String ID like 'P00001'.
- text        : 2–4 sentence description of a real-world Gram Sahayta style project
                (WASH, irrigation, agriculture, MGNREGA, governance, health, environment,
                 tech/data, engineering). Each proposal is situated in a specific gram / village.

How to interpret:
- Each proposal is a small, implementable local project idea for NSS / village-level work.
- You should treat 'text' as the input description to your ML model (e.g. text embedding).


2) people.csv
-------------
Columns:
- person_id        : String ID like 'U00001'.
- name             : Synthetic volunteer name.
- text             : 2–3 sentences describing their background, skills and Gram Sahayta interests,
                     including the skills you specified (e.g. water quality assessment,
                     mgnrega works planning and measurement, gis and remote sensing, etc.).
- willingness_eff  : Float in [0.0, 1.0] as a continuous proxy of effective willingness / capacity
                     for Gram Sahayta work (higher = more willing / more capacity).
- willingness_bias : Float in [0.0, 1.0], an auxiliary signal you can choose to use or ignore.
- availability     : Categorical text indicating coarse availability (see legend below).
- home_location    : Village / gram where the volunteer is assumed to live.

Availability categories (synonyms used)
--------------------------------------
We derived an availability label based on simulated willingness and constraints. The exact strings
used in the dataset are:

- "immediately available"
    ~ synonym of "readily available right away".
    Volunteer is effectively ready right away for intensive Gram Sahayta fieldwork and coordination.
- "generally available"
    ~ synonym of "normally available".
    Volunteer is normally available alongside regular academics and can take up most assignments.
- "rarely available"
    ~ synonym of "scarcely available".
    Volunteer is scarcely available due to exams, jobs, or other constraints and prefers short,
    well-defined tasks.

You can one-hot encode these availability values or map them to numerical levels
(e.g. 2 = immediately available, 1 = generally available, 0 = rarely available).


3) pairs.csv
------------
Columns:
- proposal_id : Must exist in proposals.csv.
- person_id   : Must exist in people.csv.
- label       : 1 if the person is a good candidate for that proposal,
                0 otherwise ("hard negative").

How labels were generated:
- Positives (label = 1) were chosen where:
  * The person has skills overlapping the proposal's theme, AND
  * Their willingness_eff is relatively high, AND
  * Their availability is "immediately available" or "generally available".
- Negatives (label = 0) are hard negatives:
  * Often have some overlapping skills, but
  * Low willingness_eff and/or "rarely available",
  * Or skill profile not very aligned with that specific proposal.

This makes the dataset suitable for training a non-trivial matching model rather than random labels.


4) village_locations.csv
------------------------
Columns:
- village_name        : Name of the gram / village used in proposals and home_location.
- district_placeholder: Placeholder text for district (you can replace with real districts).
- state_placeholder   : Placeholder text for state (you can replace with real states).

Use this file if you want to:
- Join volunteers and proposals via village names.
- Attach more realistic coordinates, districts, or states later.


5) village_distances.csv
------------------------
This is the extra "location between any two graams mapped" database.

Columns:
- village_a       : First village name.
- village_b       : Second village name (village_a != village_b).
- distance_km     : Synthetic road distance in kilometers between the two villages.
- travel_time_min : Approximate travel time in minutes assuming ~30 km/h with small variation.

Notes:
- Only one row per unordered pair (village_a, village_b) is provided.
- If you need symmetric entries in code, you can expand it by duplicating rows with swapped columns.
- Can be used to build features like "distance between volunteer home_location and proposal village".


6) availability_legend.csv
--------------------------
Columns:
- availability_label : One of "immediately available", "generally available", "rarely available".
- description        : English explanation of what that level means.

This is just a machine-readable summary of the availability categories described above.


HOW TO READ / USE THE DATA
==========================

General
-------
All files are plain CSV with headers and UTF-8 encoding. You can open them in Excel, LibreOffice,
or read them programmatically in Python, R, etc.

Example in Python (pandas)
--------------------------
import pandas as pd

proposals = pd.read_csv("proposals.csv")
people    = pd.read_csv("people.csv")
pairs     = pd.read_csv("pairs.csv")
villages  = pd.read_csv("village_locations.csv")
distances = pd.read_csv("village_distances.csv")
avail_leg = pd.read_csv("availability_legend.csv")

# Join proposals and pairs
pairs_full = pairs.merge(proposals, on="proposal_id").merge(people, on="person_id")


Example in R
------------
library(readr)
proposals <- read_csv("proposals.csv")
people    <- read_csv("people.csv")
pairs     <- read_csv("pairs.csv")


Possible ML usage
-----------------
- Input features:
  * Text embeddings from proposals.text and people.text.
  * Numerical features from willingness_eff, willingness_bias.
  * Categorical availability mapped to integers.
  * Distance between proposal village and volunteer home_location derived from village_distances.csv.
- Target:
  * label in pairs.csv as a binary target for matching models (logistic regression, gradient boosting,
    neural models, etc.).

File naming and formats
-----------------------
- proposals.csv            : Proposal master data (id + description).
- people.csv               : Volunteer master data (id + profile, willingness, availability, home_location).
- pairs.csv                : Labelled proposal–person pairs.
- village_locations.csv    : Master list of villages / graams used in the dataset.
- village_distances.csv    : Approximate distances between any two villages.
- availability_legend.csv  : Legend of availability categories used in people.csv.

All files are independent CSVs but are designed to be joined using:
- proposal_id between proposals.csv and pairs.csv,
- person_id between people.csv and pairs.csv,
- village_name / home_location between people.csv, proposals (via text) and village_locations.csv,
- village_a / village_b with home_location / proposal village via string matching.

