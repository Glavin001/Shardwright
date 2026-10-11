"""Bond views of a baked asset from `frac-sim --analyze` (tools/sim), for
`prefracture preview`:

  OUT_graph.json   the support graph: every fragment of the analysed level is
                   a node at its centre of mass, every bond an edge through
                   the bond's centroid (where the pieces touch), coloured
                   by what it is (inside a part, joint between parts, mortar,
                   bearing; anchors run down to the ground in black),
                   thickness ~ sqrt(bond area)          -> --graph
  OUT_load.json    the self-weight load path on the solid: each fragment
                   coloured by the largest force through its bonds
                   (square-root scale)                  -> --sim
  OUT_removed.json the same structure with one column removed: fragments
                   coloured by their worst bond's utilization, red where a
                   bond is now over capacity, the removed column pale pink
                                                        -> --sim
  OUT_rest.json    the level at rest in plain grey, for ghost.py -> --sim
  OUT_stats.json   counts for captions

Usage: bond_graph.py ASSET.json ANALYSIS.json OUT_PREFIX"""
import json, math, sys
from collections import defaultdict
import numpy as np

asset, analysis, out = sys.argv[1], sys.argv[2], sys.argv[3]
a = json.load(open(asset))
d = json.load(open(analysis))
level = d["level"]
frags = a["hierarchy"]["fragments"]
com = {f["id"]: f["mass"]["com"] for f in frags}
removed = set(d["removed_chunks"])
rows = d["bonds"]   # [bond, a, b (-1 ground), area, kind, centroid, intact [u,|F|,Fn], removed [..] or None]

KIND = {  # colour and legend name per bond kind
    "Monolithic": ([0.62, 0.66, 0.74], "inside a part"),
    "ColdJoint": ([0.95, 0.55, 0.12], "joint between parts"),
    "ComponentConnection": ([0.95, 0.55, 0.12], "joint between parts"),
    "Bearing": ([0.55, 0.30, 0.75], "bearing (compression only)"),
    "MortarJoint": ([0.78, 0.22, 0.20], "mortar joint"),
    "Anchor": ([0.08, 0.08, 0.10], "anchor to the ground"),
}
NODE = [0.30, 0.31, 0.36]
PINK = [0.97, 0.78, 0.78]
RED = [0.86, 0.10, 0.12]

def lerp(stops, t):
    t = min(max(t, 0.0), 1.0)
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            s = (t - t0) / (t1 - t0)
            return [c0[i] + (c1[i] - c0[i]) * s for i in range(3)]
    return list(stops[-1][1])
BLUES = [(0.0, (0.90, 0.91, 0.93)), (0.45, (0.55, 0.68, 0.90)), (1.0, (0.12, 0.16, 0.55))]

# 1. support graph
area = np.array([r[3] for r in rows])
amax = np.percentile(area, 98)
used = sorted({r[1] for r in rows} | {r[2] for r in rows if r[2] >= 0})
scale = 1.0 if level == 0 else 0.6
# each edge runs node -> bond centroid (where the two pieces touch) -> node,
# so the graph follows the structure instead of crossing it
edges, contacts = [], []
for r in rows:
    p, c = com[r[1]], r[5]
    q = [c[0], 0.0, c[2]] if r[2] < 0 else com[r[2]]
    rad = scale * (0.03 + 0.07 * math.sqrt(min(r[3] / amax, 1.0)))
    col = KIND.get(r[4], KIND["Monolithic"])[0]
    edges += [[*p, *c, rad, *col], [*c, *q, rad, *col]]
    contacts.append([*c, rad * 1.6, *col])
nodes = [[*com[f], scale * 0.14, *NODE] for f in used] + contacts
json.dump({"nodes": nodes, "edges": edges}, open(out + "_graph.json", "w"))

# 2. + 3. per fragment: largest force through its bonds (intact), worst
# utilization (column removed)
force, util = defaultdict(float), defaultdict(float)
for r in rows:
    for f in (r[1], r[2]):
        if f < 0:
            continue
        force[f] = max(force[f], r[6][1])
        if r[7] is not None:
            util[f] = max(util[f], r[7][0])
hi = np.percentile(list(force.values()), 99)
# one-frame "sims" of the level at rest, coloured per fragment (rendered with
# `prefracture preview --sim`, the camera of the graph view)
def rest(colour_of):
    frames = [{"t": 0, "bodies": [[f, *com[f], 0.0, 0.0, 0.0, 1.0] for f in sorted(com) if frags[f]["level"] == level],
               "broken": 0, "refined": 0, "islands": 1}]
    cols = {str(f): colour_of(f) for f in sorted(com) if frags[f]["level"] == level}
    return {"fps": 1, "level": level, "frames": frames, "colors": cols}
json.dump(rest(lambda f: lerp(BLUES, math.sqrt(force[f] / hi))), open(out + "_load.json", "w"))
def over_colour(f):
    if f in removed:
        return PINK
    u = util[f]
    return RED if u > 1.0 else lerp(BLUES, 0.85 * u)
json.dump(rest(over_colour), open(out + "_removed.json", "w"))
# and plain, for a ghost of the solid behind the graph
json.dump(rest(lambda f: [0.78, 0.79, 0.82]), open(out + "_rest.json", "w"))

legend = {}
for r in rows:
    c, name = KIND.get(r[4], KIND["Monolithic"])
    legend.setdefault(name, [c, 0])[1] += 1
over_b = sum(1 for r in rows if r[7] is not None and r[7][0] > 1.0)
stats = {"level": level, "removed": d["removed"], "nodes": len(used), "bonds": len(rows), "legend": legend,
         "force_p99_N": round(hi), "intact_max_utilization": round(max(r[6][0] for r in rows), 3),
         "over_capacity_bonds": over_b, "over_capacity_fragments": sum(1 for f in util if util[f] > 1.0 and f not in removed)}
json.dump(stats, open(out + "_stats.json", "w"), indent=1)
print(json.dumps(stats))

