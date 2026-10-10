#!/usr/bin/env python3
"""Crack-placement oracle (spec §13.4), Rankine variant.

The spec's physics oracle is Kratos FEM-DEM, which is not distributed for
the installed Kratos (10.4); we substitute a brittle *Rankine* oracle on the
same Kratos linear FEM of the UNFRACTURED solid:

  * load cases: cantilever bending in the two transverse directions and
    torsion (clamped at the axis-min face, as in the bond-fidelity harness),
    plus `--impacts` surface impacts (inward pressure on a disc of radius
    0.1 × the smallest box extent at deterministic surface points, both end
    faces held);
  * initiation: the element with maximum principal stress σ1, excluding
    two element sizes around supports and load patches (singularities);
  * crack surface: the plane through that point ⊥ σ1 (mode I), restricted
    to the connected region of the plane inside the solid where the normal
    stress on the plane exceeds 25% of the initiation stress.

Metrics against our interfaces at Level 1 (between different L1 fragments)
and Level 3 (all fine interfaces), by area-weighted point sampling:
  recall@τ, precision@τ, F-score (τ = 0.25 × median cell diameter, plus a
  τ curve) and, for Level 1, the Spearman correlation between per-analysis-
  cell oracle crack density and our Level-1 cut density.

For components with a grain direction (wood) the criterion is anisotropic:
the crack plane maximizes σ_nn(n)/f(n) with f(n) = f_min (1 + (r−1)(n·g)²),
r = sqrt(G_f,across / G_f,along) from the material library (planes parallel
to the grain are weakest). The FEM itself is isotropic with E = E_long.

Usage: crack_oracle.py --asset X.asset.json [--network X.network.json] --cache DIR [--impacts 6]
"""
import argparse, hashlib, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kratos_fem import mesh_solid, solve_static  # noqa: E402
from bond_fidelity import StressField, polygon_quadrature  # noqa: E402

TETFACES = [(0, 1, 2, 4, 5, 6), (0, 1, 3, 4, 8, 7), (1, 2, 3, 5, 9, 8), (0, 2, 3, 6, 9, 7)]


def boundary_faces(tets):
    """Boundary P2 faces (corner ids + mid-side ids) and their tets."""
    key = {}
    for e, t in enumerate(tets):
        for f in TETFACES:
            ids = t[list(f)]
            k = tuple(sorted(ids[:3]))
            key.setdefault(k, []).append((e, ids))
    out = [v[0][1] for v in key.values() if len(v) == 1]
    return np.array(out)


def locate_inside(field, P):
    """Containing element and inside flag per point."""
    k = min(12, len(field.cent))
    _, idx = field.tree.query(P, k=k)
    idx = idx.reshape(len(P), k)
    v = field.coords[field.tets[idx]]
    T = np.stack([v[:, :, 1] - v[:, :, 0], v[:, :, 2] - v[:, :, 0], v[:, :, 3] - v[:, :, 0]], axis=-1)
    ok = np.abs(np.linalg.det(T)) > 1e-300
    T[~ok] = np.eye(3)
    lam = np.linalg.solve(T, (P[:, None, :] - v[:, :, 0])[..., None])[..., 0]
    inside = ok & (lam.min(axis=-1) >= -1e-9) & (lam.sum(axis=-1) <= 1 + 1e-9)
    anyin = inside.any(axis=1)
    first = np.where(anyin, inside.argmax(axis=1), 0)
    return idx[np.arange(len(P)), first], anyin


def plane_basis(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def rankine_crack(field, x0, n, s0, D, spacing, excl):
    """Connected tensile region of the plane (x0, n) inside the solid."""
    u, v = plane_basis(n)
    m = int(np.ceil(D / spacing))
    g = (np.arange(-m, m + 1) * spacing)
    U, V = np.meshgrid(g, g, indexing="ij")
    P = x0 + U.reshape(-1, 1) * u + V.reshape(-1, 1) * v
    ok = np.zeros(len(P), bool)
    sn = np.zeros(len(P))
    for c in range(0, len(P), 200000):
        e, ins = locate_inside(field, P[c:c + 200000])
        S = field.eval_many(e, P[c:c + 200000])
        sn[c:c + 200000] = np.einsum("i,pij,j->p", n, S, n)
        ok[c:c + 200000] = ins
    good = (ok & (sn >= 0.25 * s0) & ~excl(P)).reshape(U.shape)
    # flood fill from the grid cell nearest x0
    seen = np.zeros_like(good)
    stack = [(m, m)]
    if not good[m, m]:
        # nearest good cell to the center
        gi = np.argwhere(good)
        if len(gi) == 0:
            return np.zeros((0, 3))
        d = ((gi - m) ** 2).sum(1)
        stack = [tuple(gi[d.argmin()])]
    while stack:
        i, j = stack.pop()
        if i < 0 or j < 0 or i >= good.shape[0] or j >= good.shape[1] or seen[i, j] or not good[i, j]:
            continue
        seen[i, j] = True
        stack.extend([(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)])
    return P[seen.reshape(-1)]


def sample_interfaces(asset, ids, spacing, rng):
    pts = []
    for i in ids:
        for poly in asset["interfaces"][i]["polygons"]:
            q, w = polygon_quadrature(poly["loops"], np.array(poly["normal"], float), sub=1)
            if len(q) == 0:
                continue
            # jittered: each fan triangle contributes ~area/spacing² points
            area = w.sum()
            k = max(1, int(round(area / spacing ** 2)))
            loops = [np.array(l, float) for l in poly["loops"]]
            r = loops[0][0]
            tri = []
            for L in loops:
                for a, b in zip(L, np.roll(L, -1, axis=0)):
                    ar = 0.5 * np.dot(np.cross(a - r, b - r), poly["normal"])
                    if ar > 0:
                        tri.append((a, b, ar))
            if not tri:
                continue
            ar = np.array([t[2] for t in tri])
            ch = rng.choice(len(tri), size=k, p=ar / ar.sum())
            s1, s2 = rng.random(k), rng.random(k)
            flip = s1 + s2 > 1
            s1[flip], s2[flip] = 1 - s1[flip], 1 - s2[flip]
            A = np.array([tri[c][0] for c in ch])
            B = np.array([tri[c][1] for c in ch])
            pts.append(r + (A - r) * s1[:, None] + (B - r) * s2[:, None])
    return np.vstack(pts) if pts else np.zeros((0, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--network", default="", help="network.json from `prefracture validate` (axis/material); derived from the asset if omitted")
    ap.add_argument("--materials", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../crates/frac-material/materials.toml"))
    ap.add_argument("--cache", required=True)
    ap.add_argument("--impacts", type=int, default=6)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
    asset = json.load(open(args.asset))
    os.makedirs(args.cache, exist_ok=True)
    comps = asset["components"]
    if len(comps) != 1:
        print(f"crack oracle: {asset['meta']['name']}: {len(comps)} components — single-component assets only; skipped\n")
        return
    solid = comps[0]["solid"]
    verts, tris = np.array(solid["verts"], float), np.array(solid["tris"], int)
    import tomllib
    lib = tomllib.load(open(args.materials, "rb"))["material"]
    m = lib[comps[0]["material"]]
    aniso = m.get("anisotropy")
    E = m.get("youngs_modulus") or (aniso or {}).get("E_long", 1e9)
    mat = {"E": float(E), "nu": float(m.get("poisson", 0.2)), "rho": float(m.get("density", 1000.0))}
    if args.network:
        net = json.load(open(args.network))
        axis, lo, hi, L = net["axis"], net["fixed_plane"], net["loaded_plane"], net["length"]
    else:
        axis = int(np.argmax(verts.max(0) - verts.min(0)))
        lo, hi = float(verts[:, axis].min()), float(verts[:, axis].max())
        L = hi - lo
    # strength anisotropy (wood): planes parallel to the grain are weak;
    # f(n) = f_min + (f_max - f_min)(n·g)², f_max/f_min = sqrt(Gf_across/Gf_along)
    grain = comps[0].get("grain")
    ratio = 1.0
    if grain is not None and m.get("fracture_energy_across_grain") and m.get("fracture_energy_along_grain"):
        ratio = float(np.sqrt(m["fracture_energy_across_grain"] / m["fracture_energy_along_grain"]))
        grain = np.array(grain, float) / np.linalg.norm(grain)
    else:
        grain = None
    dirs = None
    if grain is not None:
        k = 400
        i = np.arange(k) + 0.5
        phi = np.arccos(1 - i / k)  # hemisphere
        th = np.pi * (1 + 5 ** 0.5) * i
        dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)
        strength = 1.0 + (ratio - 1.0) * (dirs @ grain) ** 2
    ext = verts.max(0) - verts.min(0)
    D = float(np.linalg.norm(ext))
    h = L / 40.0
    key = hashlib.sha1(json.dumps([solid, h]).encode()).hexdigest()[:16]
    mpath = os.path.join(args.cache, f"mesh_{key}.npz")
    if os.path.exists(mpath):
        z = np.load(mpath)
        coords, tets = z["coords"], z["tets"]
    else:
        coords, tets = mesh_solid(verts, tris, h)
        np.savez(mpath, coords=coords, tets=tets)
    tol = 1e-6 * L
    fixed = np.abs(coords[:, axis] - lo) <= tol
    bfaces = boundary_faces(tets)
    fc = coords[bfaces[:, :3]].mean(axis=1)
    fn = np.cross(coords[bfaces[:, 1]] - coords[bfaces[:, 0]], coords[bfaces[:, 2]] - coords[bfaces[:, 0]])
    fa = 0.5 * np.linalg.norm(fn, axis=1)
    fn /= np.maximum(2 * fa[:, None], 1e-300)
    # orient outward: tets' 4th vertex is inside
    # (boundary faces from gmsh are consistently oriented outward for the
    # right-handed tets Kratos uses; fix any inward ones via the solid center)
    c0 = coords.mean(axis=0)
    flip = ((fc - c0) * fn).sum(1) < 0
    # load cases
    tdirs = [i for i in range(3) if i != axis]
    cases = []
    for t in tdirs:
        d = np.zeros(3)
        d[t] = 1.0
        cases.append({"name": f"bending_{'xyz'[t]}", "kind": "face", "direction": d})
    d = np.zeros(3)
    d[axis] = 1.0
    cases.append({"name": "torsion", "kind": "torque", "direction": d})
    rng = np.random.default_rng(7)
    side = ~((np.abs(fc[:, axis] - lo) <= tol) | (np.abs(fc[:, axis] - hi) <= tol))
    mid = side & (fc[:, axis] > lo + 0.2 * L) & (fc[:, axis] < hi - 0.2 * L)
    cand = np.nonzero(mid)[0]
    if len(cand):
        for k, fi in enumerate(rng.choice(cand, size=min(args.impacts, len(cand)), replace=False, p=fa[cand] / fa[cand].sum())):
            nrm = fn[fi] * (-1.0 if flip[fi] else 1.0)
            cases.append({"name": f"impact_{k}", "kind": "impact", "center": fc[fi], "direction": -nrm})
    r_imp = 0.1 * ext.min()
    on_hi = np.abs(coords[:, axis] - hi) <= tol
    elem_h = h
    cent_all = coords[tets[:, :4]].mean(axis=1)
    cracks = {}
    mag = 1e6
    for case in cases:
        F = np.zeros_like(coords)
        if case["kind"] in ("face", "torque"):
            from bond_fidelity import face_loads
            lc = {"kind": "torque" if case["kind"] == "torque" else "force", "direction": list(case["direction"]), "magnitude": mag}
            F, *_ = face_loads(coords, tets, axis, hi, tol, lc)
            load_zone = lambda P: np.abs(P[:, axis] - hi) <= 2 * elem_h  # noqa: E731
        else:
            sel = np.nonzero(np.linalg.norm(fc - case["center"], axis=1) <= r_imp)[0]
            area = fa[sel].sum()
            for fi in sel:
                for kk in bfaces[fi][3:]:
                    F[kk] += case["direction"] * mag * fa[fi] / area / 3.0
            cc = case["center"]
            load_zone = lambda P, cc=cc: np.linalg.norm(P - cc, axis=1) <= r_imp + 2 * elem_h  # noqa: E731
        ckey = hashlib.sha1(json.dumps([key, "v2", case["name"], [list(map(float, case.get("direction", []))), list(map(float, case.get("center", [])))], mat]).encode()).hexdigest()[:16]
        cpath = os.path.join(args.cache, f"crack_{ckey}.npz")
        if os.path.exists(cpath):
            z = np.load(cpath)
            S = z["S"]
        else:
            # impacts: both end faces held (column/beam between supports);
            # end-face loads: cantilever clamped at the axis-min face
            sup = (fixed | on_hi) if case["kind"] == "impact" else fixed
            _, S, _ = solve_static(coords, tets, mat["E"], mat["nu"], mat["rho"], sup, F)
            np.savez(cpath, S=S)
        field = StressField(coords, tets, S)
        Sc = field.eval_many(np.arange(len(tets)), cent_all)
        w, V = np.linalg.eigh(Sc)
        s1 = w[:, 2]
        n_best = V[:, :, 2]
        if dirs is not None:
            # failure index max_n σ_nn(n) / f(n) and its plane normal
            snn = np.einsum("ki,eij,kj->ek", dirs, Sc, dirs) / strength[None, :]
            kb = snn.argmax(axis=1)
            s1 = snn[np.arange(len(Sc)), kb]
            n_best = dirs[kb]
        if case["kind"] == "impact":
            excl = lambda P, lz=load_zone: (np.abs(P[:, axis] - lo) <= 2 * elem_h) | (np.abs(P[:, axis] - hi) <= 2 * elem_h) | lz(P)  # noqa: E731
        else:
            excl = lambda P, lz=load_zone: (np.abs(P[:, axis] - lo) <= 2 * elem_h) | lz(P)  # noqa: E731
        s1m = np.where(excl(cent_all), -np.inf, s1)
        e0 = int(np.argmax(s1m))
        if not np.isfinite(s1m[e0]) or s1m[e0] <= 0:
            continue
        x0, n1 = cent_all[e0], n_best[e0]
        cells_vol = sorted(c["mass"]["volume"] for c in asset["cells"])
        cell_d = (6 * cells_vol[len(cells_vol) // 2] / np.pi) ** (1 / 3)
        spacing = max(0.00316, D / 400.0)
        s0 = s1m[e0] * (1.0 + (ratio - 1.0) * float(n1 @ grain) ** 2 if grain is not None else 1.0)
        cracks[case["name"]] = rankine_crack(field, x0, n1, s0, D, spacing, excl)
    # our interfaces
    cells_vol = sorted(c["mass"]["volume"] for c in asset["cells"])
    cell_d = (6 * cells_vol[len(cells_vol) // 2] / np.pi) ** (1 / 3)
    tau = 0.25 * cell_d
    cf = asset["hierarchy"]["cell_fragment"]
    lv1 = 1
    ids3, ids1 = [], []
    for it in asset["interfaces"]:
        b = it["cells"][1]
        if not isinstance(b, dict) or "Cell" not in b:
            continue
        a, b = it["cells"][0], b["Cell"]
        ids3.append(it["id"])
        if cf[lv1][a] != cf[lv1][b]:
            ids1.append(it["id"])
    srng = np.random.default_rng(3)
    ours_spacing = tau / 3.0
    S3 = sample_interfaces(asset, ids3, ours_spacing, srng)
    S1 = sample_interfaces(asset, ids1, ours_spacing, srng)
    allcr = np.vstack([c for c in cracks.values() if len(c)]) if cracks else np.zeros((0, 3))
    out = {"asset": asset["meta"]["name"], "strength_anisotropy": ratio, "tau": tau, "median_cell_diameter": cell_d, "cases": {k: int(len(v)) for k, v in cracks.items()}, "levels": {}}
    taus = [0.125, 0.25, 0.5, 1.0]
    for name, ours in (("L1", S1), ("L3", S3)):
        if len(ours) == 0 or len(allcr) == 0:
            continue
        t_ours = cKDTree(ours)
        t_cr = cKDTree(allcr)
        d_rec = t_ours.query(allcr)[0]
        d_pre = t_cr.query(ours)[0]
        curve = []
        for f in taus:
            R = float((d_rec <= f * cell_d).mean())
            P = float((d_pre <= f * cell_d).mean())
            curve.append({"tau_over_cell": f, "recall": R, "precision": P, "f": 2 * P * R / max(P + R, 1e-12)})
        per_case = {}
        for k, c in cracks.items():
            if len(c):
                per_case[k] = float((t_ours.query(c)[0] <= tau).mean())
        R = float((d_rec <= tau).mean())
        P = float((d_pre <= tau).mean())
        out["levels"][name] = {"recall": R, "precision": P, "f": 2 * P * R / max(P + R, 1e-12), "curve": curve, "recall_by_case": per_case, "interface_area_samples": int(len(ours))}
    # weak-region agreement at L1: per analysis cell densities
    cent = np.array([c["mass"]["com"] for c in asset["cells"]])
    ac = np.array([c["analysis_cell"] for c in asset["cells"]])
    nac = int(ac.max()) + 1
    vol = np.bincount(ac, weights=[c["mass"]["volume"] for c in asset["cells"]], minlength=nac)
    tc = cKDTree(cent)
    if len(allcr) and len(S1):
        spacing = max(0.00316, D / 400.0)
        oc = np.bincount(ac[tc.query(allcr)[1]], minlength=nac) * spacing ** 2 / np.maximum(vol, 1e-300)
        oursd = np.bincount(ac[tc.query(S1)[1]], minlength=nac) * ours_spacing ** 2 / np.maximum(vol, 1e-300)
        rho = spearmanr(oc, oursd).correlation
        out["weak_region_spearman"] = float(rho) if np.isfinite(rho) else float("nan")
    np.savez(os.path.join(args.cache, f"{out['asset']}.cracks.npz"), **{k: v for k, v in cracks.items()})
    if args.json:
        json.dump(out, open(args.json, "w"), indent=2)
    json.dump(out, open(os.path.join(args.cache, f"{out['asset']}.crack_oracle.json"), "w"), indent=2)
    print(f"### Crack placement (Rankine oracle): {out['asset']}\n")
    print(f"Oracle cracks: {', '.join(f'{k} ({v} samples)' for k, v in out['cases'].items())}; τ = 0.25 × median cell diameter = {tau:.4f} m.\n")
    print("| Interfaces | Recall@τ | Precision@τ | F-score | F@τ/2 | F@2τ |\n|---|---|---|---|---|---|")
    for name, l in out["levels"].items():
        print(f"| {name} | {l['recall']:.3f} | {l['precision']:.3f} | {l['f']:.3f} | {l['curve'][0]['f']:.3f} | {l['curve'][2]['f']:.3f} |")
    if "weak_region_spearman" in out:
        print(f"\nWeak-region agreement (Spearman, per analysis cell, L1): {out['weak_region_spearman']:.3f}\n")


if __name__ == "__main__":
    main()
