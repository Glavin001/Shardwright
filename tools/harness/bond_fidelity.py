#!/usr/bin/env python3
"""Bond-fidelity oracle (spec §13.3).

Compares the reference bond-network solver (exported by
`prefracture validate`) against a finite-element oracle on the UNFRACTURED
solid: Kratos Multiphysics StructuralMechanicsApplication with quadratic
tetrahedra (gmsh). Metrics per level and load case:
  * interface traction error: network F_n/A vs FEM (1/A)∫ n·σ·n dA over the
    exact bond polygons (p50/p95 with an epsilon floor);
  * effective stiffness error at the loaded end;
  * first natural frequencies (scikit-fem P2 eigen oracle);
  * convergence L1 -> L2 -> L3.
Usage: bond_fidelity.py --asset X.asset.json --network X.network.json --cache DIR
"""
import argparse, hashlib, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kratos_fem import mesh_solid, solve_static  # noqa: E402


def polygon_quadrature(loops, normal, sub=2):
    """Signed fan quadrature points/weights over a planar polygon with holes."""
    r = np.array(loops[0][0])
    pts, wts = [], []
    for loop in loops:
        L = np.array(loop)
        for k in range(len(L)):
            a, b = L[k], L[(k + 1) % len(L)]
            area = 0.5 * np.dot(np.cross(a - r, b - r), normal)
            if area == 0.0:
                continue
            # barycentric subdivision of the fan triangle (r, a, b)
            for i in range(sub):
                for j in range(sub - i):
                    for up in (0, 1):
                        if up and i + j > sub - 2:
                            continue
                        if not up:
                            bc = np.array([(i + 1 / 3), (j + 1 / 3)]) / sub
                        else:
                            bc = np.array([(i + 2 / 3), (j + 2 / 3)]) / sub
                        p = r + (a - r) * bc[0] + (b - r) * bc[1]
                        pts.append(p)
                        wts.append(area / sub ** 2)
    return np.array(pts), np.array(wts)


def voigt_to_tensor(s):
    # Kratos Voigt: xx, yy, zz, xy, yz, xz
    return np.array([[s[0], s[3], s[5]], [s[3], s[1], s[4]], [s[5], s[4], s[2]]])


class StressField:
    """Piecewise-linear FEM stress (per-element linear fit of the
    integration-point stresses)."""

    def __init__(self, coords, tets, coeffs):
        from scipy.spatial import cKDTree
        self.coords, self.tets, self.C = coords, tets[:, :4], coeffs
        self.cent = coords[self.tets].mean(axis=1)
        self.tree = cKDTree(self.cent)

    def eval(self, e, p):
        co = self.C[e]
        return voigt_to_tensor(co[0] + (p - self.cent[e]) @ co[1:4])

    def at(self, p):
        _, idx = self.tree.query(p, k=12)
        for e in np.atleast_1d(idx):
            v = self.coords[self.tets[e]]
            T = np.column_stack([v[1] - v[0], v[2] - v[0], v[3] - v[0]])
            try:
                l = np.linalg.solve(T, p - v[0])
            except np.linalg.LinAlgError:
                continue
            if l.min() >= -1e-9 and l.sum() <= 1 + 1e-9:
                return self.eval(e, p)
        e = np.atleast_1d(idx)[0]
        return self.eval(e, p)


def face_loads(coords, tets, axis, plane, tol, case, face_c=None):
    """Consistent nodal loads on the boundary face x[axis] = plane for a
    total force (or torque) applied as uniform (or ∝ r) traction."""
    faces = [(0, 1, 2, 4, 5, 6), (0, 1, 3, 4, 9, 7), (1, 2, 3, 5, 8, 9), (0, 2, 3, 6, 8, 7)]
    on = np.abs(coords[:, axis] - plane) <= tol
    tris = []
    for t in tets:
        for f in faces:
            ids = t[list(f)]
            if on[ids[:3]].all():
                tris.append(ids)
    tris = np.array(tris)
    A = np.array([0.5 * np.linalg.norm(np.cross(coords[t[1]] - coords[t[0]], coords[t[2]] - coords[t[0]])) for t in tris])
    cen = np.array([coords[t[:3]].mean(axis=0) for t in tris])
    d = np.array(case["direction"], float)
    F = np.zeros_like(coords)
    if case["kind"] == "torque":
        c0 = (cen * A[:, None]).sum(0) / A.sum()
        r = cen - c0
        r -= np.outer(r @ d, d)
        J = (A * (r * r).sum(1)).sum()
        trac = np.cross(d, r) * (case["magnitude"] / J)
    else:
        trac = np.tile(d * case["magnitude"] / A.sum(), (len(tris), 1))
    for t, a, tr in zip(tris, A, trac):
        for k in (3, 4, 5):  # P2 triangle under uniform traction: mid-side nodes 1/3 each
            F[t[k]] += tr * a / 3.0
    return F, tris, A, cen


def response(disp, tris, A, cen, case):
    d = np.array(case["direction"], float)
    u = np.array([disp[t].mean(axis=0) for t in tris])
    if case["kind"] == "torque":
        c0 = (cen * A[:, None]).sum(0) / A.sum()
        r = cen - c0
        r -= np.outer(r @ d, d)
        tang = np.cross(d, r)
        return float((A * (u * tang).sum(1)).sum() / (A * (r * r).sum(1)).sum())
    return float((A * (u @ d)).sum() / A.sum())


def pct(x, q):
    return float(np.percentile(np.asarray(x), q)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--network", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--h", type=float, default=0.0, help="FEM element size (default: length/40)")
    args = ap.parse_args()
    asset = json.load(open(args.asset))
    net = json.load(open(args.network))
    os.makedirs(args.cache, exist_ok=True)
    comps = asset["components"]
    if len(comps) != 1:
        print(f"bond fidelity: {asset['meta']['name']}: {len(comps)} components — single-component assets only; skipped\n")
        return
    mat = net["materials"][0]
    solid = comps[0]["solid"]
    verts = np.array(solid["verts"], float)
    tris = np.array(solid["tris"], int)
    axis, lo, hi, L = net["axis"], net["fixed_plane"], net["loaded_plane"], net["length"]
    # oracle resolution: well below the finest bond scale
    cells = asset["cells"]
    vols = sorted(c["mass"]["volume"] for c in cells)
    cell_size = vols[len(vols) // 2] ** (1.0 / 3.0)
    h = args.h or min(L / 40.0, cell_size / 2.5)
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
    interfaces = asset["interfaces"]
    # quadrature per interface (cached)
    iq = {}
    out = {"asset": asset["meta"]["name"], "fem_nodes": int(len(coords)), "fem_tets": int(len(tets)), "levels": []}
    fem_cases = {}
    for case in net["load_cases"]:
        F, ftris, A, cen = face_loads(coords, tets, axis, hi, tol, case)
        ckey = hashlib.sha1(json.dumps([key, case, mat]).encode()).hexdigest()[:16]
        cpath = os.path.join(args.cache, f"fem2_{ckey}.npz")
        if os.path.exists(cpath):
            z = np.load(cpath)
            disp, S, C = z["disp"], z["S"], z["C"]
        else:
            disp, S, C = solve_static(coords, tets, mat["E"], mat["nu"], mat["rho"], fixed, F)
            np.savez(cpath, disp=disp, S=S, C=C)
        fem_cases[case["name"]] = (StressField(coords, tets, S), response(disp, ftris, A, cen, case))
    for lv in net["levels"]:
        lrep = {"level": lv["level"], "fragments": lv["fragments"], "cases": []}
        for res in lv["results"]:
            name = res["case"]
            if name not in fem_cases:
                continue
            field, fem_resp = fem_cases[name]
            t_net, t_fem, areas = [], [], []
            v_raw, v_rec, v_fem = [], [], []
            for b in res["bonds"]:
                n = np.array(b["normal"], float)
                num = 0.0
                numv = np.zeros(3)
                den = 0.0
                for i in b["interfaces"]:
                    if i not in iq:
                        pts, wts = [], []
                        for poly in interfaces[i]["polygons"]:
                            p, w = polygon_quadrature(poly["loops"], np.array(poly["normal"], float))
                            pts.append(p)
                            wts.append(w)
                        iq[i] = (np.vstack(pts) if pts else np.zeros((0, 3)), np.concatenate(wts) if wts else np.zeros(0))
                    pts, wts = iq[i]
                    for p, w in zip(pts, wts):
                        Sg = field.at(p)
                        num += w * (n @ Sg @ n)
                        numv += w * (Sg @ n)
                        den += w
                if den <= 0:
                    continue
                t_net.append(b["traction"])
                t_fem.append(num / den)
                areas.append(b["area"])
                v_raw.append(b.get("traction_vec", [0, 0, 0]))
                v_rec.append(b.get("recovered", [0, 0, 0]))
                v_fem.append(numv / den)
            t_net, t_fem = np.array(t_net), np.array(t_fem)
            v_raw, v_rec, v_fem = np.array(v_raw), np.array(v_rec), np.array(v_fem)
            eps = 0.05 * (np.abs(t_fem).max() if len(t_fem) else 1.0)
            err_normal = np.abs(t_net - t_fem) / np.maximum(np.abs(t_fem), eps) if len(t_fem) else np.array([])
            fem_mag = np.linalg.norm(v_fem, axis=1) if len(v_fem) else np.array([])
            epsv = 0.05 * (fem_mag.max() if len(fem_mag) else 1.0)
            err_raw = np.linalg.norm(v_raw - v_fem, axis=1) / np.maximum(fem_mag, epsv) if len(v_fem) else np.array([])
            err = np.linalg.norm(v_rec - v_fem, axis=1) / np.maximum(fem_mag, epsv) if len(v_fem) else np.array([])
            stiff_err = abs(res["response"] / fem_resp - 1.0) if fem_resp else float("nan")
            lrep["cases"].append({"case": name, "bonds": int(len(err)), "traction_err_p50": pct(err, 50), "traction_err_p95": pct(err, 95),
                                  "raw_vector_err_p50": pct(err_raw, 50), "raw_vector_err_p95": pct(err_raw, 95),
                                  "raw_normal_err_p50": pct(err_normal, 50), "raw_normal_err_p95": pct(err_normal, 95), "stiffness_err": stiff_err, "network_response": res["response"], "fem_response": fem_resp})
        out["levels"].append(lrep)
    # modal
    try:
        from skfem_modal import modal_frequencies
        mkey = os.path.join(args.cache, f"modal_{hashlib.sha1(json.dumps([key, mat]).encode()).hexdigest()[:16]}.json")
        if os.path.exists(mkey):
            fem_f = json.load(open(mkey))
        else:
            fem_f = [float(x) for x in modal_frequencies(coords, tets, mat["E"], mat["nu"], mat["rho"], fixed, 10)]
            json.dump(fem_f, open(mkey, "w"))
        out["fem_modal_hz"] = fem_f
        for lrep, lv in zip(out["levels"], net["levels"]):
            nf = lv.get("modal_hz", [])
            m = min(len(nf), len(fem_f))
            lrep["modal_err"] = [abs(nf[k] / fem_f[k] - 1.0) for k in range(m)]
    except Exception as e:  # pragma: no cover
        out["modal_error"] = str(e)
    # convergence: errors decrease from coarse to fine
    def series(key):
        return [np.nanmean([c[key] for c in l["cases"]]) for l in out["levels"]]
    for key in ("traction_err_p95", "stiffness_err"):
        s = series(key)
        out[f"{key}_by_level"] = s
        out[f"{key}_monotone"] = bool(all(s[i + 1] <= s[i] + 1e-12 for i in range(len(s) - 1)))
    json.dump(out, open(os.path.join(args.cache, f"{out['asset']}.bond_fidelity.json"), "w"), indent=2)
    # markdown summary
    print(f"### Bond fidelity: {out['asset']} (FEM: {out['fem_tets']} P2 tets)\n")
    print(f"Network stiffness model: {net.get('stiffness_model', '?')}. Traction error = |t_net - t_FEM| / max(|t_FEM|, 5% of max) per bond;")
    print("'recovered' uses the Love–Weber stress of the two fragments, 'raw' uses F_b/A_b (spec definition).\n")
    print("| Level | Case | Bonds | Recovered p50 | Recovered p95 | Raw F/A p50 | Raw F/A p95 | Raw normal p50 | Stiffness err |\n|---|---|---|---|---|---|---|---|---|")
    for l in out["levels"]:
        for c in l["cases"]:
            print(f"| L{l['level']} | {c['case']} | {c['bonds']} | {c['traction_err_p50']:.3f} | {c['traction_err_p95']:.3f} | {c['raw_vector_err_p50']:.3f} | {c['raw_vector_err_p95']:.3f} | {c['raw_normal_err_p50']:.3f} | {c['stiffness_err']:.3f} |")
    if "fem_modal_hz" in out:
        print("\nFEM modal (Hz): " + ", ".join(f"{f:.2f}" for f in out["fem_modal_hz"][:6]))
        for l in out["levels"]:
            if "modal_err" in l and l["modal_err"]:
                print(f"- L{l['level']} modal rel err (first {len(l['modal_err'])}): max {max(l['modal_err']):.3f}")
    print(f"\nMonotone convergence: traction p95 {out['traction_err_p95_monotone']}, stiffness {out['stiffness_err_monotone']}\n")


if __name__ == "__main__":
    main()
