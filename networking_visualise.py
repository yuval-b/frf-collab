"""
Networking event visualisation
==============================

Reads the outputs from networking_matcher_v2.py and produces:
  - network_interactive.html   (Pyvis, zoomable and hoverable)
  - network_static.png         (matplotlib, high-res for slides)

Only shows each person's top K strongest edges to keep the network readable.
Attending people are highlighted; those who couldn't make it are greyed out.

Requirements:
    pip install pyvis matplotlib networkx pandas
"""

import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "output"

EDGE_TOP_K_PER_PERSON = 5   # each person contributes their top K edges to the graph
MIN_EDGE_SCORE        = 0.2  # additional floor to drop very weak edges

# Colours
COLOUR_ATTENDING     = "#b2cb3c"   # Forrest green
COLOUR_NOT_ATTENDING = "#888888"   # muted grey
COLOUR_EDGE          = "#BBBBBB"
COLOUR_BG_STATIC     = "white"
COLOUR_BG_INTERACTIVE = "#1a1a1a"
COLOUR_FONT_INTERACTIVE = "white"

# ============================================================
# LOAD DATA
# ============================================================

adj       = pd.read_csv(f"{OUTPUT_DIR}/adjacency_matrix.csv", index_col=0)
attendees = pd.read_csv(f"{OUTPUT_DIR}/attendees.csv")
edges_df  = pd.read_csv(f"{OUTPUT_DIR}/edges_ranked.csv")

# Convenience: dict[name] -> row of attendee info
attendee_info = {row["name"]: row for _, row in attendees.iterrows()}

# Convenience: dict[frozenset({name_a, name_b})] -> facet scores
edge_lookup = {
    frozenset([row["person_a"], row["person_b"]]): row
    for _, row in edges_df.iterrows()
}

# ============================================================
# BUILD GRAPH
# ============================================================

G = nx.Graph()

# Nodes
for _, row in attendees.iterrows():
    G.add_node(
        row["name"],
        attending=bool(row["attending"]),
        institution=row["institution"] if pd.notna(row["institution"]) else "",
    )

# Edges: union of each person's top K (avoids one-sided sparsity)
names = adj.index.tolist()
for name in names:
    row = adj.loc[name].drop(name).sort_values(ascending=False)
    for other, score in row.head(EDGE_TOP_K_PER_PERSON).items():
        if score < MIN_EDGE_SCORE:
            break
        if not G.has_edge(name, other):
            G.add_edge(name, other, weight=float(score))

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Precompute top-3 matches per person for tooltips
top_matches_lookup = {}
for name in names:
    row = adj.loc[name].drop(name).sort_values(ascending=False)
    top_matches_lookup[name] = [(other, float(s)) for other, s in row.head(3).items()]

# ============================================================
# INTERACTIVE (Pyvis)
# ============================================================

net = Network(
    height="850px",
    width="100%",
    bgcolor=COLOUR_BG_INTERACTIVE,
    font_color=COLOUR_FONT_INTERACTIVE,
    notebook=False,
    cdn_resources="remote",
)

# Force-directed layout with nice settling behaviour
net.set_options("""
{
  "nodes": {
    "borderWidth": 2,
    "borderWidthSelected": 4,
    "font": {"size": 14}
  },
  "edges": {
    "color": {"inherit": false},
    "smooth": {"enabled": true, "type": "continuous"},
    "scaling": {"min": 1, "max": 8}
  },
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -80,
      "centralGravity": 0.01,
      "springLength": 120,
      "springConstant": 0.08,
      "damping": 0.4
    },
    "solver": "forceAtlas2Based",
    "stabilization": {"iterations": 200}
  },
  "interaction": {"hover": true, "tooltipDelay": 200}
}
""")

for name in G.nodes():
    data = G.nodes[name]
    #attending = data["attending"]

    # Rich tooltip
    top3 = top_matches_lookup.get(name, [])
    top3_html = "".join(
        f"<br>&nbsp;&nbsp;{other} ({s:.2f})" for other, s in top3
    )
    tooltip = (
        f"{name}\n"
        f"{data['institution']}\n"
        #f"{'Attending' if attending else 'Not attending'}\n\n"
        f"Top matches:\n" +
        "\n".join(f"  • {other} ({s:.2f})" for other, s in top3)
    )

    net.add_node(
        name,
        label=name,
        title=tooltip,
        color=COLOUR_ATTENDING,
        size=28,
        borderWidth=2,
    )

for u, v, d in G.edges(data=True):
    score = d["weight"]
    edge_row = edge_lookup.get(frozenset([u, v]))
    if edge_row is not None:
        tooltip = f"{u} ↔ {v}\nScore: {score:.2f}"
    else:
        tooltip = f"Score: {score:.2f}"

    net.add_edge(u, v, value=score, title=tooltip, color=COLOUR_EDGE)

net.write_html(f"{OUTPUT_DIR}/network_interactive.html", notebook=False, open_browser=False)
print(f"Interactive network:  {OUTPUT_DIR}/network_interactive.html")

# ============================================================
# STATIC (matplotlib)
# ============================================================

fig, ax = plt.subplots(figsize=(18, 13))
fig.patch.set_facecolor(COLOUR_BG_STATIC)

# Reproducible layout
pos = nx.spring_layout(G, k=1.8, iterations=300, seed=42, weight="weight")

# Edges: width by score, alpha by score
edge_weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
edge_range   = np.ptp(edge_weights) + 1e-9
edge_widths  = 0.5 + 4.0 * (edge_weights - edge_weights.min()) / edge_range
edge_alphas  = 0.15 + 0.55 * (edge_weights - edge_weights.min()) / edge_range

for (u, v), w, a in zip(G.edges(), edge_widths, edge_alphas):
    nx.draw_networkx_edges(
        G, pos, edgelist=[(u, v)],
        width=w, alpha=a, edge_color=COLOUR_EDGE, ax=ax,
    )

# Nodes: attending vs not
nodes = [n for n, d in G.nodes(data=True)]
#absent_nodes    = [n for n, d in G.nodes(data=True) if not d["attending"]]

nx.draw_networkx_nodes(
    G, pos, nodelist=nodes,
    node_color=COLOUR_ATTENDING, node_size=650,
    alpha=0.95, edgecolors="white", linewidths=2, ax=ax,
)

# Labels: only for attending (keeps the figure readable at print scale)
nx.draw_networkx_labels(
    G, pos, labels={n: n for n in nodes},
    font_size=9, font_color="#111", font_weight="bold", ax=ax,
)

logo = Image.open("logo.png")

imagebox = OffsetImage(logo, zoom=0.3)
annotation = AnnotationBbox(
    imagebox,
    (0.9, 0.9),                 # position in axes coordinates
    xycoords="axes fraction",
    frameon=False,
    zorder=10
)
ax.add_artist(annotation)

ax.axis("off")
ax.set_title("Forrest Research Foundation Networking Map", fontsize=18, pad=20, weight="bold")

# legend_elements = [
#     mpatches.Patch(color=COLOUR_ATTENDING, label=f"Attending ({len(attending_nodes)})"),
#     mpatches.Patch(color=COLOUR_NOT_ATTENDING, label=f"Not attending ({len(absent_nodes)})"),
# ]
# ax.legend(handles=legend_elements, loc="upper left", frameon=False, fontsize=11)

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/network_static.png",
    dpi=220, bbox_inches="tight", facecolor=COLOUR_BG_STATIC,
)
plt.close()
print(f"Static network:       {OUTPUT_DIR}/network_static.png")

print("\nDone. To view the interactive version, just open the HTML file in a browser.")