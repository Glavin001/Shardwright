#!/usr/bin/env python3
"""Convex-decomposition differential test (spec §13.2 / §13.6).

Compares the built-in decomposition (`frac-collision`, the hulls stored in a
baked asset) against upstream CoACD (python `coacd`) on the same clean
fragment meshes. Per fragment and method:
  * hull count;
  * coverage  = vol(∪hulls ∩ fragment) / vol(fragment)   (1 = no holes);
  * outside   = vol(∪hulls − fragment) / vol(fragment)   (overshoot);
  * concavity = symmetric Hausdorff distance between the surface of ∪hulls
                and the fragment surface / fragment diameter (from 40k surface
                samples each; CoACD's boundary term);
  * concavity_norm = the same distance in CoACD's normalized units (divided by
                half the longest bounding-box side), comparable to the CoACD
                threshold;
  * cw_concavity = collision-aware variant: max distance from the surface of
                (∪hulls − fragment) to the fragment surface and from the
                surface of (fragment − ∪hulls) to the hull surface / diameter
                (ignores crevices between hulls inside the fragment, which
                dominate `concavity` for many-hull decompositions); cw_norm in
                CoACD units.

Budget modes:
  * equal (default): CoACD runs with `max_convex_hull` = our hull count for the
    fragment;
  * `--no-budget`: CoACD runs without a hull limit at `--threshold`. Pass an
    asset whose hulls were recomputed without a budget at the same threshold:
    `prefracture hulls --asset X.asset.json --coacd-threshold 0.03 --max-hulls 0
    --out Y.asset.json`.

Our hulls are non-overlapping and margin-shrunk, so their coverage is
slightly below 1 by construction.

Upstream results are cached (`--cache`, default `<asset dir>/.coacd_cache`)
keyed by the fragment mesh, threshold, budget and seed, and computed by
`--jobs` worker processes.

Usage: coacd_diff.py --asset X.asset.json [--level 1] [--max 40] [--threshold 0.03]
                     [--no-budget] [--jobs 2] [--cache DIR] [--json out.json]
"""
import argparse, hashlib, json, os, sys
import numpy as np


def fragment_mesh(asset, frag):
    """Boundary mesh of a fragment's cells (outward), like cells_boundary_mesh."""
    order = asset["hierarchy"]["cell_order"]
    rng = frag["cells"]
    cells = set(order[rng["start"]:rng["end"]])
    verts_all, tris_all, off = [], [], 0
    comps = sorted({asset["cells"][c]["component"] for c in cells})
    for ci in comps:
        g = asset["components"][ci]["geometry"]
        V = np.array(g["verts"], float)
        T = []
        for e in g["ext_polys"]:
            if e["cell"] in cells:
                T.extend(e["tris"])
        for p in g["patches"]:
            a, b = p["cells"][0] in cells, p["cells"][1] in cells
            if a and not b:
                T.extend(p["tris"])
            elif b and not a:
                T.extend([[t[0], t[2], t[1]] for t in p["tris"]])
        T = np.array(T, int).reshape(-1, 3)
        used = np.unique(T)
        remap = -np.ones(len(V), int)
        remap[used] = np.arange(len(used))
        verts_all.append(V[used])
        tris_all.append(remap[T] + off)
        off += len(used)
    return np.vstack(verts_all), np.vstack(tris_all)


def to_manifold(V, T):
    import manifold3d as m3
    mesh = m3.Mesh(vert_properties=np.asarray(V, np.float32), tri_verts=np.asarray(T, np.uint32))
    mesh.merge()
    return m3.Manifold(mesh)


def hull_manifold(points):
    import manifold3d as m3
    return m3.Manifold.hull_points(np.asarray(points, np.float64))


def metrics(V, T, hulls, rng):
    import trimesh
    frag = to_manifold(V, T)
    fv = frag.volume()
    if not (fv > 0) or not hulls:
        return None
    union = hulls[0]
    for h in hulls[1:]:
        union = union + h
    inside = (union ^ frag).volume()
    outside = (union - frag).volume()
    from scipy.spatial import cKDTree
    tm = trimesh.Trimesh(V, T, process=False)
    diam = float(np.linalg.norm(V.max(0) - V.min(0)))
    um = union.to_mesh()
    ut = trimesh.Trimesh(np.asarray(um.vert_properties)[:, :3], np.asarray(um.tri_verts), process=False)
    # symmetric Hausdorff between the union-of-hulls surface and the fragment
    # surface, from dense surface samples (bias ≈ sample spacing)
    seed = int(rng.integers(1 << 31))
    fs, _ = trimesh.sample.sample_surface(tm, 40000, seed=seed)
    us, _ = trimesh.sample.sample_surface(ut, 40000, seed=seed + 1)
    d_out = cKDTree(np.vstack([fs, V])).query(us)[0].max()
    d_in = cKDTree(np.vstack([us, ut.vertices])).query(fs)[0].max()
    worst = max(float(d_out), float(d_in))
    half_longest = 0.5 * float((V.max(0) - V.min(0)).max())
    # collision-aware deviation (ignores crevices between hulls inside the
    # fragment): surface of (∪hulls − fragment) to the fragment surface, and
    # surface of (fragment − ∪hulls) to the hull surface
    cw = 0.0
    for region, ref in ((union - frag, np.vstack([fs, V])), (frag - union, np.vstack([us, ut.vertices]))):
        if region.volume() <= 0:
            continue
        rm = region.to_mesh()
        rt = trimesh.Trimesh(np.asarray(rm.vert_properties)[:, :3], np.asarray(rm.tri_verts), process=False)
        if rt.area <= 0:
            continue
        rs, _ = trimesh.sample.sample_surface(rt, 20000, seed=seed + 2)
        cw = max(cw, float(cKDTree(ref).query(rs)[0].max()))
    return {"hulls": len(hulls), "coverage": inside / fv, "outside": outside / fv, "concavity": worst / max(diam, 1e-12),
            "concavity_norm": worst / max(half_longest, 1e-12), "cw_concavity": cw / max(diam, 1e-12),
            "cw_norm": cw / max(half_longest, 1e-12)}


def run_upstream(job):
    """Upstream CoACD on one fragment mesh (cached). Returns hull vertex lists."""
    V, T, threshold, budget, cache = job
    key = hashlib.sha1()
    key.update(np.ascontiguousarray(V, np.float64).tobytes())
    key.update(np.ascontiguousarray(T, np.int64).tobytes())
    key.update(f"{threshold:.6g}/{budget}/seed1".encode())
    path = os.path.join(cache, key.hexdigest() + ".json") if cache else ""
    if path and os.path.exists(path):
        return json.load(open(path))
    import coacd
    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(V, T), threshold=threshold, seed=1, max_convex_hull=budget)
    res = [np.asarray(p[0], float).tolist() for p in parts]
    if path:
        os.makedirs(cache, exist_ok=True)
        tmp = path + f".{os.getpid()}.tmp"
        json.dump(res, open(tmp, "w"))
        os.replace(tmp, path)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--json", default="")
    ap.add_argument("--no-budget", action="store_true", help="run CoACD without a hull limit")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    cache = args.cache if args.cache is not None else os.path.join(os.path.dirname(os.path.abspath(args.asset)), ".coacd_cache")
    asset = json.load(open(args.asset))
    frags = [f for f in asset["hierarchy"]["fragments"] if f["level"] == args.level]
    # deterministic subsample, largest fragments first (the non-convex ones)
    frags.sort(key=lambda f: -f["mass"]["volume"])
    frags = frags[: args.max]
    hulls_by_frag = {}
    for h in asset["hulls"]:
        hulls_by_frag.setdefault(h["fragment"], []).append(h)
    rng = np.random.default_rng(1)
    meshes = [fragment_mesh(asset, f) for f in frags]
    # same hull budget as ours (CoACD merges down to max_convex_hull), or none
    budgets = [-1 if args.no_budget else max(len(hulls_by_frag.get(f["id"], [])), 1) for f in frags]
    jobs = [(V, T, args.threshold, b, cache) for (V, T), b in zip(meshes, budgets)]
    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            upstream = pool.map(run_upstream, jobs, chunksize=1)
    else:
        upstream = [run_upstream(j) for j in jobs]
    rows = []
    for f, (V, T), parts in zip(frags, meshes, upstream):
        ours = [hull_manifold(h["vertices"]) for h in hulls_by_frag.get(f["id"], [])]
        theirs = [hull_manifold(p) for p in parts]
        mo, mt = metrics(V, T, ours, rng), metrics(V, T, theirs, rng)
        if mo and mt:
            rows.append((f["id"], mo, mt))
    if not rows:
        print("no fragments compared")
        return

    def agg(k, i, fn=np.median):
        return float(fn([r[i][k] for r in rows]))

    out = {"asset": asset["meta"]["name"], "level": args.level, "fragments": len(rows), "threshold": args.threshold,
           "budget": "none" if args.no_budget else "equal"}
    keys = ("hulls", "coverage", "outside", "concavity", "concavity_norm", "cw_concavity", "cw_norm")
    for name, i in (("ours", 1), ("coacd", 2)):
        out[name] = {k: {"median": agg(k, i), "p95": agg(k, i, lambda x: np.percentile(x, 95)),
                         "total" if k == "hulls" else "max": agg(k, i, np.sum if k == "hulls" else np.max)} for k in keys}
    out["per_fragment"] = [{"id": r[0], "ours": r[1], "coacd": r[2]} for r in rows]
    mode = "no hull limit" if args.no_budget else "equal hull budget"
    print(f"### Convex decomposition vs upstream CoACD: {out['asset']} L{args.level} ({len(rows)} largest fragments, "
          f"CoACD threshold {args.threshold}, {mode})\n")
    print("| Method | Hulls (total) | Hulls/frag median | Coverage median | Outside median | Outside p95 | Concavity median | Concavity p95 | Conc. norm max | CW conc. median | CW conc. p95 | CW norm max |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in ("ours", "coacd"):
        o = out[name]
        print(f"| {name} | {o['hulls']['total']:.0f} | {o['hulls']['median']:.0f} | {o['coverage']['median']:.4f} | "
              f"{o['outside']['median']:.4f} | {o['outside']['p95']:.4f} | {o['concavity']['median']:.4f} | "
              f"{o['concavity']['p95']:.4f} | {o['concavity_norm']['max']:.4f} | {o['cw_concavity']['median']:.4f} | "
              f"{o['cw_concavity']['p95']:.4f} | {o['cw_norm']['max']:.4f} |")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    sys.exit(main())
