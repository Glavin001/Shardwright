"""Measure every number the explainer and the README's evidence section use,
from the bakes made by make.sh. Writes out/why/data/evidence.json and
partial_counts.json. Needs numpy + scipy (the oracle harness Python)."""
import json, math, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT + "/tools/harness")
from crack_oracle import sample_interfaces  # noqa: E402

BIN = ROOT + "/target/release/prefracture"
SW, UN, SC = (ROOT + p for p in ("/out/why/shardwright", "/out/why/uniform", "/out/support_clusters"))
DATA = ROOT + "/out/why/data"
os.makedirs(DATA, exist_ok=True)
load = lambda p: json.load(open(p))

def cell_pairs(a):
    for it in a["interfaces"]:
        b = it["cells"][1]
        if isinstance(b, dict) and "Cell" in b:
            yield it, it["cells"][0], b["Cell"]

def cut_ids(a, level):
    cf = a["hierarchy"]["cell_fragment"][level]
    return [it["id"] for it, x, y in cell_pairs(a) if cf[x] != cf[y]]

def cut_area(a, level):
    cf = a["hierarchy"]["cell_fragment"][level]
    return sum(it["area"] for it, x, y in cell_pairs(a) if cf[x] != cf[y])

ev = {}
# 2. column: Level-1 recall of the frozen FEM crack oracle (golden draw)
z = np.load(ROOT + "/benchmarks/golden/rc_column/cracks.npz")
rec = {}
for name, path in (("shardwright", f"{SW}/rc_column.asset.json"), ("uniform", f"{UN}/rc_column.asset.json")):
    a = load(path)
    cv = sorted(c["mass"]["volume"] for c in a["cells"])
    tau = 0.25 * (6 * cv[len(cv) // 2] / np.pi) ** (1 / 3)
    t = cKDTree(sample_interfaces(a, cut_ids(a, 1), tau / 3.0, np.random.default_rng(3)))
    rec[name] = round(float(np.mean([(t.query(z[k])[0] <= tau).mean() for k in z.files])), 3)
ev["column_crack_recall_l1"] = rec

# 3. materials
def pane_sizes(a):
    V = np.array([c["mass"]["volume"] for c in a["cells"]])
    ext = np.array([np.array(c["aabb"]["max"]) - np.array(c["aabb"]["min"]) for c in a["cells"]])
    return np.sqrt(V / np.median(ext.min(1)))
meta = load(ROOT + "/benchmarks/assets/glass_annealed_pane.glb.meta.json")
imp = np.array([v for v in meta["parts"].values() if v.get("impact_center")][0]["impact_center"])
prof = {}
for name, root in (("shardwright", SW), ("uniform", UN)):
    a = load(f"{root}/glass_annealed_pane.asset.json")
    P = np.array([c["mass"]["com"] for c in a["cells"]]); d = np.linalg.norm(P - imp, axis=1); size = pane_sizes(a)
    bins = np.linspace(0, d.max(), 9); idx = np.clip(np.digitize(d, bins) - 1, 0, 7)
    prof[name] = [round(float(np.median(size[idx == k])), 3) if (idx == k).any() else None for k in range(8)]
ev["glass_annealed_median_size_m_by_distance"] = prof
tm = {}
for name, root in (("shardwright", SW), ("uniform", UN)):
    s = pane_sizes(load(f"{root}/glass_tempered_pane.asset.json"))
    tm[f"{name}_pieces"] = int(len(s)); tm[f"{name}_median_m"] = round(float(np.median(s)), 3)
ev["tempered"] = tm
g = np.array([v for v in load(ROOT + "/benchmarks/assets/timber_beam.glb.meta.json")["parts"].values() if v.get("grain")][0]["grain"], float)
g /= np.linalg.norm(g)
wood = {}
for name, root in (("shardwright", SW), ("uniform", UN)):
    r = []
    for c in load(f"{root}/timber_beam.asset.json")["cells"]:
        I = np.array(c["mass"]["inertia"]).reshape(3, 3) / c["mass"]["mass"]
        C = np.trace(I) / 2 * np.eye(3) - I          # second moments of the cell
        along = g @ C @ g; across = (np.trace(C) - along) / 2
        if along > 0 and across > 0: r.append(math.sqrt(along / across))
    wood[name] = round(float(np.median(r)), 2)
ev["wood_elongation_along_grain"] = wood
sw_wall = load(f"{SW}/brick_wall_window.asset.json")
joints = cKDTree(sample_interfaces(sw_wall, [it["id"] for it, _, _ in cell_pairs(sw_wall) if it["kind"] == "MortarJoint"], 0.004, np.random.default_rng(1)))
brick = {}
for name, a in (("shardwright", sw_wall), ("uniform", load(f"{UN}/brick_wall_window.asset.json"))):
    s = sample_interfaces(a, [it["id"] for it, _, _ in cell_pairs(a)], 0.01, np.random.default_rng(1))
    brick[name] = round(float((joints.query(s)[0] < 0.005).mean()), 3)   # within 5 mm of a mortar joint
ev["brick_break_surface_on_mortar_joints"] = brick

# 4. bonds: patch test with exact vs approximated bond geometry
out = subprocess.run([ROOT + "/target/release/examples/bond_geometry", f"{SW}/rc_column.asset.json"], capture_output=True, text=True, check=True).stdout
p95 = {}
for line in out.splitlines()[1:]:
    v, strain, _, _, q95, _ = line.split("\t")
    p95.setdefault(v.replace(" ", "_").replace("-", "_"), []).append(float(q95))
ev["bond_patch_test_p95"] = {"exact": p95["exact"], "centre_line_normals": p95["centre_line_normals"], "adjacency_only": p95["adjacency_only"]}

# 5. hierarchy: bodies during a partial break (same schedule as render.py)
n = 44
def count(i):
    t = i / (n - 1); t = 0.5 - 0.5 * math.cos(math.pi * t)
    tmp = f"{DATA}/count_{i:03d}.png"
    r = subprocess.run([BIN, "preview", "--input", f"{SC}/two_storey_building.asset.json", "--out", tmp, "--impact", "4.1,3.3,8.2",
                        "--radius", f"{0.05 + 3.2 * t:.3f}", "--width", "160", "--height", "120"], capture_output=True, text=True, check=True)
    os.remove(re.search(r"wrote (\S+)", r.stdout).group(1))
    return [int(x.split()[0]) for x in re.search(r"partial break: (.*?)\)", r.stdout).group(1).split(" + ")]
with ThreadPoolExecutor(4) as ex:
    counts = list(ex.map(count, range(n)))
json.dump(counts, open(f"{DATA}/partial_counts.json", "w"))
ev["rigid_bodies_partial_break"] = {"hierarchical": sum(counts[-1]), "flat_leaf": len(load(f"{SC}/two_storey_building.asset.json")["cells"])}

# 6. colliders vs upstream CoACD (docs/VALIDATION.md, rc_column L1, equal budget)
ev["coacd_equal_budget"] = {"outside_volume_median": [0.060, 0.209], "overlap": ["no", "yes"]}

json.dump(ev, open(f"{DATA}/evidence.json", "w"), indent=1)
print(json.dumps(ev, indent=1))
