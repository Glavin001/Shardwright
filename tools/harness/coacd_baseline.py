#!/usr/bin/env python3
"""Baseline parity of the stand-alone CoACD port against upstream CoACD.

Runs our Rust port (`prefracture decompose`: `coacd::decompose`, no
cell-complex adaptations, no non-overlap step, no hull budget) and the
upstream `coacd` Python package with the same (upstream default) parameters
on identical closed meshes: standard non-convex shapes (L block, U block,
square ring, torus, the ceramic bowl) and the largest fragments of baked
assets. Both outputs are scored by the same evaluator, in CoACD's
normalized frame (longest bounding-box side = 2):

  * hulls       number of convex hulls;
  * h           CoACD concavity of the decomposition, max(Rv, Hb):
                Rv = 0.3·∛(3|V(mesh) − V(∪hulls)|/4π), Hb = symmetric
                Hausdorff distance between the mesh surface and the surface of
                ∪hulls (40k samples each);
  * vol_ratio   Σ V(hull) / V(mesh) (overlap counted, like CoACD's merge Rv);
  * seconds     wall time of the decomposition call.

Usage: coacd_baseline.py --cli target/release/prefracture --asset out/rc_column.asset.json:1:3 \\
                         [--asset ...] [--threshold 0.05] [--cache DIR] [--json out.json]
(`--asset PATH:LEVEL:COUNT` adds the COUNT largest fragments of LEVEL.)
"""
import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coacd_diff import fragment_mesh, hull_manifold, to_manifold  # noqa: E402


def boxes_mesh(boxes):
    import manifold3d as m3
    u = None
    for lo, hi in boxes:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        c = m3.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))
        u = c if u is None else u + c
    mm = u.to_mesh()
    return np.asarray(mm.vert_properties, float)[:, :3], np.asarray(mm.tri_verts, int)


def shapes():
    import trimesh
    out = {
        "L_block": boxes_mesh([((0, 0, 0), (2, 1, 1)), ((0, 0, 0), (1, 2, 1))]),
        "U_block": boxes_mesh([((0, 0, 0), (3, 1, 1)), ((0, 0, 0), (1, 3, 1)), ((2, 0, 0), (3, 3, 1))]),
        "square_ring": boxes_mesh([((0, 0, 0), (3, 1, 1)), ((0, 2, 0), (3, 3, 1)), ((0, 0, 0), (1, 3, 1)), ((2, 0, 0), (3, 3, 1))]),
    }
    t = trimesh.creation.torus(major_radius=1.0, minor_radius=0.3, major_sections=48, minor_sections=16)
    out["torus"] = (np.asarray(t.vertices, float), np.asarray(t.faces, int))
    return out


def key_of(V, T, extra):
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(V, np.float64).tobytes())
    h.update(np.ascontiguousarray(T, np.int64).tobytes())
    h.update(extra.encode())
    return h.hexdigest()


def run_ours(cli, V, T, threshold, cache):
    k = key_of(V, T, f"ours/{threshold}")
    path = os.path.join(cache, k + ".ours.json")
    if not os.path.exists(path):
        mp = os.path.join(cache, k + ".mesh.json")
        json.dump({"vertices": np.asarray(V, float).tolist(), "faces": np.asarray(T, int).tolist()}, open(mp, "w"))
        subprocess.run([cli, "decompose", "--mesh", mp, "--out", path, "--threshold", str(threshold), "--seed", "0"], check=True, stdout=subprocess.DEVNULL)
        os.remove(mp)
    d = json.load(open(path))
    return d["hulls"], d["seconds"]


def run_upstream(V, T, threshold, cache):
    k = key_of(V, T, f"coacd/{threshold}")
    path = os.path.join(cache, k + ".coacd.json")
    if not os.path.exists(path):
        import coacd
        coacd.set_log_level("error")
        t0 = time.perf_counter()
        parts = coacd.run_coacd(coacd.Mesh(V, T), threshold=threshold, seed=0)
        secs = time.perf_counter() - t0
        json.dump({"hulls": [np.asarray(p[0], float).tolist() for p in parts], "seconds": secs}, open(path, "w"))
    d = json.load(open(path))
    return d["hulls"], d["seconds"]


def evaluate(V, T, hulls, rng):
    import trimesh
    from scipy.spatial import cKDTree
    L = float((V.max(0) - V.min(0)).max())
    s = 2.0 / max(L, 1e-300)
    mesh = to_manifold(V, T)
    vm = mesh.volume()
    hs = [hull_manifold(h) for h in hulls if len(h) >= 4]
    hs = [h for h in hs if h.volume() > 0]
    if not hs or vm <= 0:
        return None
    union = hs[0]
    for h in hs[1:]:
        union = union + h
    rv = 0.3 * np.cbrt(3.0 * abs(vm - union.volume()) / (4.0 * np.pi)) * s
    tm = trimesh.Trimesh(V, T, process=False)
    um = union.to_mesh()
    ut = trimesh.Trimesh(np.asarray(um.vert_properties)[:, :3], np.asarray(um.tri_verts), process=False)
    seed = int(rng.integers(1 << 31))
    fs, _ = trimesh.sample.sample_surface(tm, 40000, seed=seed)
    us, _ = trimesh.sample.sample_surface(ut, 40000, seed=seed + 1)
    hb = max(cKDTree(np.vstack([fs, V])).query(us)[0].max(), cKDTree(np.vstack([us, ut.vertices])).query(fs)[0].max()) * s
    return {"hulls": len(hulls), "h": float(max(rv, hb)), "rv": float(rv), "hb": float(hb), "vol_ratio": float(sum(h.volume() for h in hs) / vm)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--asset", action="append", default=[])
    ap.add_argument("--bowl", default="", help="ceramic_bowl asset: adds the whole bowl (level 0 fragment)")
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--cache", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--no-shapes", action="store_true")
    args = ap.parse_args()
    cache = args.cache or os.path.join(os.getcwd(), ".coacd_baseline_cache")
    os.makedirs(cache, exist_ok=True)
    meshes = [] if args.no_shapes else list(shapes().items())
    if args.bowl:
        a = json.load(open(args.bowl))
        f = [f for f in a["hierarchy"]["fragments"] if f["level"] == 0][0]
        meshes.append(("ceramic_bowl", fragment_mesh(a, f)))
    for spec in args.asset:
        path, level, count = spec.rsplit(":", 2)
        a = json.load(open(path))
        frs = sorted([f for f in a["hierarchy"]["fragments"] if f["level"] == int(level)], key=lambda f: -f["mass"]["volume"])[: int(count)]
        for f in frs:
            meshes.append((f"{a['meta']['name']} L{level} #{f['id']}", fragment_mesh(a, f)))
    rng = np.random.default_rng(1)
    rows = []
    for name, (V, T) in meshes:
        ho, so = run_ours(args.cli, V, T, args.threshold, cache)
        hu, su = run_upstream(V, T, args.threshold, cache)
        eo, eu = evaluate(V, T, ho, rng), evaluate(V, T, hu, rng)
        if eo and eu:
            eo["seconds"], eu["seconds"] = so, su
            rows.append({"mesh": name, "tris": int(len(T)), "ours": eo, "coacd": eu})
            print(f"{name}: ours {eo['hulls']} hulls h={eo['h']:.4f} ({so:.1f}s) | coacd {eu['hulls']} hulls h={eu['h']:.4f} ({su:.1f}s)", file=sys.stderr)
    print(f"### Stand-alone port vs upstream CoACD 1.0.14 (threshold {args.threshold}, upstream defaults)\n")
    print("| Mesh | Tris | Hulls ours / CoACD | h ours / CoACD | Rv ours / CoACD | Hb ours / CoACD | Σ hull vol / V ours / CoACD | Seconds ours / CoACD |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        o, c = r["ours"], r["coacd"]
        print(f"| {r['mesh']} | {r['tris']} | {o['hulls']} / {c['hulls']} | {o['h']:.4f} / {c['h']:.4f} | {o['rv']:.4f} / {c['rv']:.4f} | "
              f"{o['hb']:.4f} / {c['hb']:.4f} | {o['vol_ratio']:.3f} / {c['vol_ratio']:.3f} | {o['seconds']:.1f} / {c['seconds']:.1f} |")
    if rows:
        med = lambda k, m: float(np.median([r[m][k] for r in rows]))
        print(f"| **median** | | {med('hulls','ours'):.0f} / {med('hulls','coacd'):.0f} | {med('h','ours'):.4f} / {med('h','coacd'):.4f} | "
              f"{med('rv','ours'):.4f} / {med('rv','coacd'):.4f} | {med('hb','ours'):.4f} / {med('hb','coacd'):.4f} | "
              f"{med('vol_ratio','ours'):.3f} / {med('vol_ratio','coacd'):.3f} | {med('seconds','ours'):.1f} / {med('seconds','coacd'):.1f} |")
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    sys.exit(main())
