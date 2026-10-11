"""Random Voronoi fractures of a baked asset's cells, for comparison with its
fracture-mode Level 1: N seeds drawn uniformly by volume inside the solid,
each cell assigned to its nearest seed, disconnected parts of a Voronoi
region split into separate pieces (as clipping Voronoi cells to the mesh
does). Writes one JSON array of group ids per cell for `prefracture preview
--cell-groups`.

With --highlight-spanning, pieces that span more than one authored part are
coloured red and the rest grey (the file then holds {"groups", "colors"}).

Usage: random_voronoi.py ASSET.json OUT.json PIECES SEED [--highlight-spanning]"""
import json, sys
import numpy as np

asset, out, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
a = json.load(open(asset))
P = np.array([c["mass"]["com"] for c in a["cells"]])
V = np.array([c["mass"]["volume"] for c in a["cells"]])
rng = np.random.default_rng(seed)
seeds = P[rng.choice(len(P), size=n, replace=False, p=V / V.sum())]
lab = np.argmin(((P[:, None, :] - seeds[None, :, :]) ** 2).sum(-1), axis=1)
# split each Voronoi region into its connected parts (cell adjacency)
adj = [[] for _ in P]
for it in a["interfaces"]:
    b = it["cells"][1]
    if isinstance(b, dict) and "Cell" in b:
        x, y = it["cells"][0], b["Cell"]
        adj[x].append(y); adj[y].append(x)
group = -np.ones(len(P), int); g = 0
for s in range(len(P)):
    if group[s] >= 0: continue
    stack = [s]; group[s] = g
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if group[v] < 0 and lab[v] == lab[u]:
                group[v] = g; stack.append(v)
    g += 1
if "--highlight-spanning" in sys.argv:
    comp = np.array([c["component"] for c in a["cells"]])
    span = [len(set(comp[group == k])) > 1 for k in range(g)]
    shades = [[0.86, 0.22, 0.25], [0.95, 0.42, 0.30], [0.78, 0.15, 0.35], [0.98, 0.55, 0.45]]
    colors = {str(k): (shades[k % 4] if span[k] else [0.80, 0.80, 0.83]) for k in range(g)}
    json.dump({"groups": group.tolist(), "colors": colors}, open(out, "w"))
    print(out, n, "seeds ->", g, "pieces,", sum(span), "span more than one part")
else:
    json.dump(group.tolist(), open(out, "w"))
    print(out, n, "seeds ->", g, "pieces")
