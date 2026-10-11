"""Random Voronoi fractures of a baked asset's cells, for comparison with its
fracture-mode Level 1: N seeds drawn uniformly by volume inside the solid,
each cell assigned to its nearest seed, disconnected parts of a Voronoi
region split into separate pieces (as clipping Voronoi cells to the mesh
does). Writes one JSON array of group ids per cell for `prefracture preview
--cell-groups`.

Usage: random_voronoi.py ASSET.json OUT.json PIECES SEED"""
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
json.dump(group.tolist(), open(out, "w"))
print(out, n, "seeds ->", g, "pieces")
