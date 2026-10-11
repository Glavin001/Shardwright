"""Pre-render every 3D sequence the explainer uses (prefracture preview)."""
import os, subprocess, math, shutil
from concurrent.futures import ThreadPoolExecutor
B = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); BIN = B + "/target/release/prefracture"
D = B + "/out/why/render"
R, U, C = B + "/out/why/shardwright", B + "/out/why/uniform", B + "/out/support_clusters"
jobs = []
def seq(name, asset, n, w, h, level=None, explode=(0, 0), az=(35, 35), el=25, extra=()):
    os.makedirs(f"{D}/{name}", exist_ok=True)
    for i in range(n):
        t = i / max(1, n - 1); t = 0.5 - 0.5 * math.cos(math.pi * t)
        a = [BIN, "preview", "--input", asset, "--out", f"{D}/{name}/{i:03d}.png", "--width", str(w), "--height", str(h),
             "--explode", f"{explode[0] + (explode[1] - explode[0]) * t:.4f}", "--azimuth", f"{az[0] + (az[1] - az[0]) * t:.3f}",
             "--elevation", str(el)] + (["--level", str(level)] if level is not None else []) + list(extra)
        jobs.append(a)
# problem: uniform building, slow orbit
seq("problem", f"{U}/two_storey_building.asset.json", 72, 640, 520, level=1, explode=(0.0, 0.06), az=(25, 70))
# fracture modes: input (L0), three random Voronoi fractures, fracture modes coming apart
M = B + "/out/why/models"
for m, az, el, w, h, ex in (("bracket", 35, 18, 480, 900, 0.25), ("bone", 20, 8, 480, 900, 0.25),
                            ("chair", 35, 18, 480, 900, 0.2), ("creature", 30, 38, 1040, 900, 0.16)):
    seq(f"model_{m}_input", f"{M}/modes/{m}.asset.json", 1, w, h, level=0, az=(az, az), el=el)
    for k in (1, 2, 3):
        seq(f"model_{m}_r{k}", f"{M}/modes/{m}.asset.json", 1, w, h, explode=(0.12, 0.12), az=(az, az), el=el,
            extra=("--cell-groups", f"{M}/random/{m}_{k}.json"))
    seq(f"model_{m}_sw", f"{M}/modes/{m}.asset.json", 18, w, h, level=1, explode=(0.0, ex), az=(az, az), el=el)
# materials: finest level, gentle explode
for m, az, el in (("glass_annealed_pane", 30, 18), ("glass_tempered_pane", 30, 18), ("timber_beam", 22, 32), ("brick_wall_window", 35, 22)):
    for v, root in (("sw", R), ("un", U)):
        seq(f"{m}_{v}", f"{root}/{m}.asset.json", 12, 560, 380, level=3, explode=(0.0, 0.05), az=(az, az + 8), el=el)
# bonds by kind (support graph L2), slow orbit
seq("bonds", f"{C}/two_storey_building.asset.json", 48, 700, 560, level=2, az=(30, 50), extra=("--bonds", "kind"))
# problem: a random Voronoi shatter of the whole building (ignores parts and
# joints) and the flat leaf level (every cell a body)
W_ = B + "/out/why"
seq("problem_random", f"{C}/two_storey_building.asset.json", 1, 640, 520, explode=(0.07, 0.07), az=(35, 35),
    extra=("--cell-groups", f"{W_}/data/building_random.json"))
seq("problem_leaf", f"{C}/two_storey_building.asset.json", 1, 640, 520, level=3, explode=(0.05, 0.05), az=(35, 35))
# inputs steer the result: one pane, three materials and a moved impact point
for nm, path in (("annealed", f"{R}/glass_annealed_pane"), ("tempered", f"{W_}/inputs/pane_tempered"),
                 ("concrete", f"{W_}/inputs/pane_concrete"), ("corner", f"{W_}/inputs/pane_corner")):
    seq(f"in_pane_{nm}", f"{path}.asset.json", 1, 480, 520, level=3, explode=(0.03, 0.03), az=(25, 25), el=18)
# convex decomposition: non-convex fracture-mode pieces and their hulls
M = B + "/out/why/models"
seq("hull_pieces", f"{M}/modes/creature.asset.json", 1, 900, 700, level=1, explode=(0.12, 0.12), az=(30, 30), el=38)
seq("hull_hulls", f"{M}/modes/creature.asset.json", 1, 900, 700, level=1, explode=(0.12, 0.12), az=(30, 30), el=38, extra=("--hulls",))
# the end-to-end chain on the two-storey building
seq("chain_cells", f"{C}/two_storey_building.asset.json", 1, 600, 500, level=3, az=(35, 35))
seq("chain_l1", f"{C}/two_storey_building.asset.json", 1, 600, 500, level=1, explode=(0.06, 0.06), az=(35, 35))
seq("chain_l2", f"{C}/two_storey_building.asset.json", 1, 600, 500, level=2, explode=(0.04, 0.04), az=(35, 35))
seq("chain_hulls", f"{C}/two_storey_building.asset.json", 1, 600, 500, level=2, explode=(0.04, 0.04), az=(35, 35), extra=("--hulls",))
seq("chain_bonds", f"{C}/two_storey_building.asset.json", 1, 600, 500, level=2, az=(35, 35), extra=("--bonds", "weakening"))
os.makedirs(f"{D}/chain_input", exist_ok=True)
jobs.append([BIN, "preview", "--input", B + "/benchmarks/assets/two_storey_building.glb", "--out", f"{D}/chain_input/000.png",
             "--width", "600", "--height", "500", "--azimuth", "35", "--elevation", "25"])
# title: Shardwright building L1, slow orbit
seq("title", f"{C}/two_storey_building.asset.json", 36, 640, 520, level=1, explode=(0.0, 0.08), az=(30, 55))
if os.path.isdir(D):
    for root, _, files in os.walk(D):
        for f in files: os.remove(os.path.join(root, f))
def run(a):
    os.makedirs(os.path.dirname(a[a.index("--out") + 1]), exist_ok=True)
    subprocess.run(a, check=True, capture_output=True)
with ThreadPoolExecutor(4) as ex: list(ex.map(run, jobs))
print(len(jobs), "frames")
