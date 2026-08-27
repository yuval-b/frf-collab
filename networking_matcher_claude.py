"""
Networking event matcher: multi-facet embedding pipeline
=========================================================

Embeddings: run locally via sentence-transformers (no API key needed)
Labels:     Claude API (Haiku 4.5) for "why you should meet" text

Facet scores per pair:
  1. Interests       (symmetric): shared research area
  2. Complementarity (asymmetric): A needs what B offers, or vice versa
  3. Skill exchange  (asymmetric): A wants to learn what B knows

Outputs:
  - adjacency_matrix.csv   (full N x N, all people, for network viz)
  - attendees.csv          (name + RSVP flag, for filtering the viz)
  - edges_ranked.csv       (long-form edge list with facet breakdown + RSVP flags)
  - match_cards.md         (RSVP'd attendees only, matched to other attendees)

Requirements:
    pip install anthropic sentence-transformers pandas numpy
    export ANTHROPIC_API_KEY=sk-ant-...
"""

import os
import numpy as np
import pandas as pd
from anthropic import Anthropic
import requests
from sentence_transformers import SentenceTransformer
from itertools import combinations

# Load API key
exec(open("api_key.py").read())

# ============================================================
# CONFIGURATION  --  edit this block for your data
# ============================================================
 
INPUT_CSV = "responses.csv"
OUTPUT_DIR = "output"
 
# Column name mapping. Set a value to None if you don't have that field.
COLUMN_MAP = {
    "first_name":     "First Name",
    "last_name":      "Last Name",
    "institution":    "Institution (other than FRF)",
    "specialty":      "Specialty area (as brief as possible)",
    "keywords":       "Keywords (choose up to 6 keywords/phrases, separated by a comma)",
    "skills":         "Skills (lab techniques, programming Languages, methodologies etc.)",
    "connections":    "Existing connections outside academia (industry, government, etc.)",
    "looking_for":    "Looking for collaborators with expertise in...",
    "wants_to_learn": "What skills would you like to develop further? Learn from someone at this event?",
    "side_project":   "Any side projects you don't have time for but would love to make happen?",
    "rsvp":           "RSVP",   # <-- CHANGE THIS to your actual RSVP column name
}
 
# Values in the RSVP column that count as "attending"
RSVP_YES_VALUES = {"yes", "unknown"}
 
WEIGHTS = {
    "interests":       0.40,
    "complementarity": 0.40,
    "skill_exchange":  0.20,
}
 

TOP_K_ATTENDING     = 4   # matches shown per card for people attending tonight
TOP_K_NOT_ATTENDING = 1   # extra matches with people who couldn't make it
 
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LABEL_MODEL      = "claude-sonnet-5"
 
# ============================================================
# STEP 1: LOAD + BUILD FACET TEXT BUNDLES
# ============================================================
 
df = pd.read_csv(INPUT_CSV)
 
# --- Deduplicate by name (case- and whitespace-insensitive) ---
_fn, _ln = COLUMN_MAP["first_name"], COLUMN_MAP["last_name"]
_key = (df[_fn].fillna("").str.strip().str.lower() + "|" +
        df[_ln].fillna("").str.strip().str.lower())
 
# Warn about empty names
_empty = df[_key == "|"]
if len(_empty):
    print(f"WARNING: {len(_empty)} row(s) with empty names \u2014 dropping them.")
    df = df[_key != "|"].reset_index(drop=True)
    _key = (df[_fn].fillna("").str.strip().str.lower() + "|" +
            df[_ln].fillna("").str.strip().str.lower())
 
# Warn about duplicate names
_dupes = _key.value_counts()
_dupes = _dupes[_dupes > 1]
if len(_dupes):
    print(f"WARNING: {len(_dupes)} name(s) appear multiple times \u2014 keeping the last submission for each:")
    for name_key, count in _dupes.items():
        fn, ln = name_key.split("|")
        print(f"  {fn.title()} {ln.title()}  ({count} entries)")
    df = df.drop_duplicates(subset=[_fn, _ln], keep="last").reset_index(drop=True)
    # Note: keep="last" assumes form order is chronological. Change to "first" if not.
 
def col(key):
    c = COLUMN_MAP.get(key)
    return c if (c and c in df.columns) else None
 
def get(row, key):
    c = col(key)
    if c is None or pd.isna(row[c]):
        return ""
    return str(row[c]).strip()
 
def full_name(row):
    parts = [get(row, "first_name"), get(row, "last_name")]
    return " ".join(p for p in parts if p)
 
def parse_rsvp(val):
    if pd.isna(val):
        return False
    return str(val).strip().lower() in {v.strip().lower() for v in RSVP_YES_VALUES}
 
def build_facets(row):
    """Bundle raw fields into the 5 facet strings we'll embed."""
    return {
        # Symmetric side: what you work on
        "interests": " | ".join(filter(None, [
            get(row, "specialty"),
            get(row, "keywords"),
        ])),
        # What you bring: expertise, skills, and external network
        "offers": " | ".join(filter(None, [
            get(row, "specialty"),
            get(row, "skills"),
            get(row, "connections"),
        ])),
        "looking_for":    get(row, "looking_for"),
        "wants_to_learn": get(row, "wants_to_learn"),
        "side_project":   get(row, "side_project"),
    }
 
people = []
for _, row in df.iterrows():
    p = build_facets(row)
    p["name"] = full_name(row)
    p["institution"] = get(row, "institution")
    p["attending"] = parse_rsvp(row[col("rsvp")]) if col("rsvp") else True
    people.append(p)
 
N = len(people)
n_attending = sum(1 for p in people if p["attending"])
 
# Diagnose RSVP parsing so silent zero-attendance never happens again
if col("rsvp"):
    print(f"\nRSVP column: {col('rsvp')!r}")
    print("Unique values found (count in brackets, tick = counted as attending):")
    for val, count in df[col("rsvp")].value_counts(dropna=False).items():
        attending = parse_rsvp(val)
        mark = "\u2713" if attending else "\u2717"
        print(f"  {mark}  {val!r:30s}  [{count}]")
else:
    print(f"\nNo RSVP column found (looking for {COLUMN_MAP['rsvp']!r}). All {N} people treated as attending.")
 
print(f"\nLoaded {N} people ({n_attending} attending).")
 
if n_attending == 0:
    raise SystemExit(
        "\nERROR: 0 attendees detected \u2014 match cards would be empty.\n"
        "Check the RSVP diagnostic above and either:\n"
        f"  (a) add your 'attending' values to RSVP_YES_VALUES (currently {sorted(RSVP_YES_VALUES)}), or\n"
        "  (b) fix COLUMN_MAP['rsvp'] if the column name is wrong, or\n"
        "  (c) set COLUMN_MAP['rsvp'] = None to treat everyone as attending."
    )
 
# ============================================================
# STEP 2: EMBED ALL FACETS LOCALLY
# ============================================================
 
FACET_KEYS = ["interests", "offers", "looking_for", "wants_to_learn", "side_project"]
 
texts, index = [], []
for i, p in enumerate(people):
    for f in FACET_KEYS:
        if p[f]:
            texts.append(p[f])
            index.append((i, f))
 
print(f"Loading embedding model ({EMBED_MODEL_NAME})...")
embedder = SentenceTransformer(EMBED_MODEL_NAME)
 
print(f"Embedding {len(texts)} facet strings...")
vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
 
E = {i: {} for i in range(N)}
for (i, f), v in zip(index, vectors):
    E[i][f] = v
 
# ============================================================
# STEP 3: PAIRWISE FACET SCORES
# ============================================================
 
def cos(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))  # already normalised
 
def facet_scores(i, j):
    Ei, Ej = E[i], E[j]
    s_interests = cos(Ei.get("interests"), Ej.get("interests"))
    comp_ij = max(
        cos(Ei.get("looking_for"),   Ej.get("offers")),
        cos(Ei.get("side_project"),  Ej.get("looking_for")),
    )
    comp_ji = max(
        cos(Ej.get("looking_for"),   Ei.get("offers")),
        cos(Ej.get("side_project"),  Ei.get("looking_for")),
    )
    s_comp = max(comp_ij, comp_ji)
    s_learn = max(
        cos(Ei.get("wants_to_learn"), Ej.get("offers")),
        cos(Ej.get("wants_to_learn"), Ei.get("offers")),
    )
    return {
        "interests":       s_interests,
        "complementarity": s_comp,
        "skill_exchange":  s_learn,
    }

adj = np.zeros((N, N))
edge_details = {}
 
for i, j in combinations(range(N), 2):
    s = facet_scores(i, j)
    total = (WEIGHTS["interests"]       * s["interests"] +
             WEIGHTS["complementarity"] * s["complementarity"] +
             WEIGHTS["skill_exchange"]  * s["skill_exchange"])
    adj[i, j] = adj[j, i] = total
    edge_details[(i, j)] = s
 
# Sanity: diagonal must be zero, and no two rows share a name
names = [p["name"] for p in people]
assert np.all(np.diag(adj) == 0), "Diagonal should be zero \u2014 no self-edges"
assert len(set(names)) == len(names), f"Duplicate names in output: {[n for n in names if names.count(n) > 1]}"
 
# ============================================================
# STEP 4: SAVE MATRIX + ATTENDEE FLAGS + EDGE LIST
# ============================================================
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# Full adjacency matrix (includes non-attendees; use attendees.csv to grey them out)
pd.DataFrame(adj, index=names, columns=names).to_csv(
    f"{OUTPUT_DIR}/adjacency_matrix.csv"
)
 
# Attendee metadata for the visualisation layer
pd.DataFrame([{
    "name": p["name"],
    "institution": p["institution"],
    "attending": p["attending"],
} for p in people]).to_csv(f"{OUTPUT_DIR}/attendees.csv", index=False)
 
# Long-form edges with RSVP flags on both endpoints
edge_rows = []
for (i, j), s in edge_details.items():
    edge_rows.append({
        "person_a":        names[i],
        "person_b":        names[j],
        "attending_a":     people[i]["attending"],
        "attending_b":     people[j]["attending"],
        "total":           adj[i, j],
        "interests":       s["interests"],
        "complementarity": s["complementarity"],
        "skill_exchange":  s["skill_exchange"],
    })
edges_df = pd.DataFrame(edge_rows).sort_values("total", ascending=False)
edges_df.to_csv(f"{OUTPUT_DIR}/edges_ranked.csv", index=False)
 
# ============================================================
# STEP 5: TOP-K MATCHES PER ATTENDEE (attendees only)
# ============================================================
 
attending_ids     = [i for i in range(N) if people[i]["attending"]]
not_attending_ids = [i for i in range(N) if not people[i]["attending"]]
 
top_attending     = {}   # top K matches AMONG the people attending tonight
top_not_attending = {}   # top K matches AMONG people who couldn't make it
for i in attending_ids:
    att_scores = [(j, adj[i, j]) for j in attending_ids if j != i]
    att_scores.sort(key=lambda x: -x[1])
    top_attending[i] = att_scores[:TOP_K_ATTENDING]
 
    absent_scores = [(j, adj[i, j]) for j in not_attending_ids]
    absent_scores.sort(key=lambda x: -x[1])
    top_not_attending[i] = absent_scores[:TOP_K_NOT_ATTENDING]
 
# ============================================================
# STEP 6: CLAUDE LABELS FOR EDGES THAT WILL APPEAR ON CARDS
# ============================================================
 
def label_prompt(pa, pb, scores):
    return f"""Two researchers at a networking event. Suggest why they should meet.

CRITICAL RULES:
1. Identify the SINGLE strongest concrete overlap. Do NOT stitch together weak connections across unrelated fields.
2. Do NOT invent links between unrelated topics. Example of what NOT to do: if Person A is an astrophysicist whose side project is investing, and Person B knows finance — the connection is investing. Do NOT link astrophysics to finance.
3. If the strongest overlap is genuinely modest, be honest: "Both work on X, though in different domains" is fine. Better an honest weak match than a fabricated strong one.
4. Be specific — name the actual technique, topic, project, or skill. Don't say "could collaborate".
5. One sentence, max 26 words, no preamble.

PERSON A: {pa['name']} ({pa['institution']})
- Specialty/keywords: {pa['interests']}
- Skills/expertise/connections: {pa['offers']}
- Looking for collaborators in: {pa['looking_for']}
- Wants to learn: {pa['wants_to_learn']}
- Side project: {pa['side_project']}

PERSON B: {pb['name']} ({pb['institution']})
- Specialty/keywords: {pb['interests']}
- Skills/expertise/connections: {pb['offers']}
- Looking for collaborators in: {pb['looking_for']}
- Wants to learn: {pb['wants_to_learn']}
- Side project: {pb['side_project']}

Facet scores (guide only, don't quote): interests={scores['interests']:.2f}, complementarity={scores['complementarity']:.2f}, skill_exchange={scores['skill_exchange']:.2f}
"""
 
def call_claude(prompt: str) -> str:
    """Direct HTTP call to the Anthropic messages API. Bypasses the SDK/httpx2 stack."""
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": LABEL_MODEL,
            "max_tokens": 80,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["content"][0]["text"].strip()
 
edges_to_label = set()
for i in attending_ids:
    for j, _ in top_attending[i] + top_not_attending[i]:
        edges_to_label.add(tuple(sorted([i, j])))
 
print(f"\nAttending: {len(attending_ids)} \u2192 {len(attending_ids)} match cards")
print(f"Top matches per card: {TOP_K_ATTENDING} attending + {TOP_K_NOT_ATTENDING} to reach out to later")
print(f"Total unique edges to label: {len(edges_to_label)}")
 
print(f"Generating Claude labels for {len(edges_to_label)} edges...")
labels = {}
label_failures = 0
for k, (i, j) in enumerate(edges_to_label, 1):
    try:
        labels[(i, j)] = call_claude(label_prompt(people[i], people[j], edge_details[(i, j)]))
    except Exception as e:
        label_failures += 1
        if label_failures <= 3:
            print(f"  Label call failed for ({names[i]}, {names[j]}): {e}")
    if k % 20 == 0:
        print(f"  {k}/{len(edges_to_label)}")
 
if label_failures:
    print(f"WARNING: {label_failures} label call(s) failed. Cards will still be written without those labels.")
 
# ============================================================
# STEP 7: WRITE MATCH CARDS (attendees only)
# ============================================================
 
print(f"\nWriting match cards for {len(attending_ids)} attendees to {OUTPUT_DIR}/match_cards.md ...")
 
def write_match(f, i, j, score):
    pair = tuple(sorted([i, j]))
    label = labels.get(pair)
    s = edge_details[pair]
    f.write(f"**{people[j]['name']}** \u2014 *{people[j]['institution']}*  \n")
    f.write(f"Score: {score:.2f} ")
    if label:
        f.write(f"> {label}\n")
    f.write("\n")
 
with open(f"{OUTPUT_DIR}/match_cards.md", "w") as f:
    for i in attending_ids:
        p = people[i]
        f.write(f"# {p['name']}\n*{p['institution']}*\n\n")
 
        f.write(f"## Your top {TOP_K_ATTENDING} matches tonight\n\n")
        for j, score in top_attending[i]:
            write_match(f, i, j, score)
 
        if top_not_attending[i]:
            f.write(f"## Worth reaching out to after (not attending)\n\n")
            for j, score in top_not_attending[i]:
                write_match(f, i, j, score)
 
        f.write("\n---\n\n")
 
print(f"\nDone. Outputs written to {OUTPUT_DIR}/")
print(f"  - adjacency_matrix.csv   (all {N} people)")
print(f"  - attendees.csv          (name + RSVP flag)")
print(f"  - edges_ranked.csv       (all pairs with RSVP flags on endpoints)")
print(f"  - match_cards.md         ({n_attending} attendees)")