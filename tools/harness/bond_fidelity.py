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
import golden  # noqa: E402


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

    def locate(self, P):
        """Containing element of each point (vectorized; nearest-centroid
        element when a point lies outside every candidate)."""
        k = min(12, len(self.cent))
        _, idx = self.tree.query(P, k=k)
        idx = idx.reshape(len(P), k)
        v = self.coords[self.tets[idx]]  # [P, k, 4, 3]
        T = np.stack([v[:, :, 1] - v[:, :, 0], v[:, :, 2] - v[:, :, 0], v[:, :, 3] - v[:, :, 0]], axis=-1)
        rhs = P[:, None, :] - v[:, :, 0]
        det = np.linalg.det(T)
        ok = np.abs(det) > 1e-300
        T[~ok] = np.eye(3)
        lam = np.linalg.solve(T, rhs[..., None])[..., 0]
        inside = ok & (lam.min(axis=-1) >= -1e-9) & (lam.sum(axis=-1) <= 1 + 1e-9)
        first = np.where(inside.any(axis=1), inside.argmax(axis=1), 0)
        return idx[np.arange(len(P)), first]

    def eval_many(self, E, P):
        co = self.C[E]  # [P, 4, 6]
        s = co[:, 0] + np.einsum("pi,pij->pj", P - self.cent[E], co[:, 1:4])
        S = np.empty((len(P), 3, 3))
        S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
        S[:, 0, 1] = S[:, 1, 0] = s[:, 3]
        S[:, 1, 2] = S[:, 2, 1] = s[:, 4]
        S[:, 0, 2] = S[:, 2, 0] = s[:, 5]
        return S


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


def oracle_modal(cache, solid, verts, tris, h, L, axis, lo, tol, mat):
    """First 10 natural frequencies of the clamped solid (scikit-fem P2) on
    its own coarser mesh (global modes converge fast); cached."""
    from skfem_modal import modal_frequencies
    hm = max(h, L / 40.0)
    mkey_m = hashlib.sha1(json.dumps([solid, hm]).encode()).hexdigest()[:16]
    mpath_m = os.path.join(cache, f"mesh_{mkey_m}.npz")
    if os.path.exists(mpath_m):
        z = np.load(mpath_m)
        mcoords, mtets = z["coords"], z["tets"]
    else:
        mcoords, mtets = mesh_solid(verts, tris, hm)
        np.savez(mpath_m, coords=mcoords, tets=mtets)
    mfixed = np.abs(mcoords[:, axis] - lo) <= tol
    mkey = os.path.join(cache, f"modal_{hashlib.sha1(json.dumps([mkey_m, mat]).encode()).hexdigest()[:16]}.json")
    if os.path.exists(mkey):
        return json.load(open(mkey))
    fem_f = [float(x) for x in modal_frequencies(mcoords, mtets, mat["E"], mat["nu"], mat["rho"], mfixed, 10)]
    json.dump(fem_f, open(mkey, "w"))
    return fem_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--network", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--h", type=float, default=0.0, help="FEM element size (default: min(length/40, median cell size/2.5))")
    ap.add_argument("--golden", default="", help="golden oracle directory (default: benchmarks/golden/<asset>); used when it matches the solid, material and load cases")
    ap.add_argument("--no-golden", action="store_true", help="always run the FEM oracle")
    ap.add_argument("--write-golden", action="store_true", help="freeze the computed oracle into the golden directory")
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
    gdir = args.golden or golden.default_dir(asset["meta"]["name"])
    man = golden.load_manifest(gdir)
    use_golden = not args.no_golden and golden.matches(man, solid, mat, net["load_cases"])
    if use_golden:
        h = man["fem"]["h"]
    key = hashlib.sha1(json.dumps([solid, h]).encode()).hexdigest()[:16]
    mpath = os.path.join(args.cache, f"mesh_{key}.npz")
    if use_golden:
        coords, tets = golden.load_mesh(gdir)
    elif os.path.exists(mpath):
        z = np.load(mpath)
        coords, tets = z["coords"], z["tets"]
    else:
        coords, tets = mesh_solid(verts, tris, h)
        np.savez(mpath, coords=coords, tets=tets)
    tol = 1e-6 * L
    fixed = np.abs(coords[:, axis] - lo) <= tol
    interfaces = asset["interfaces"]
    # quadrature over every interface polygon, located once in the FEM mesh
    qp, qw, qowner = [], [], []
    for i, itf in enumerate(interfaces):
        for poly in itf["polygons"]:
            pp, ww = polygon_quadrature(poly["loops"], np.array(poly["normal"], float))
            if len(pp):
                qp.append(pp)
                qw.append(ww)
                qowner.append(np.full(len(ww), i))
    qp = np.vstack(qp) if qp else np.zeros((0, 3))
    qw = np.concatenate(qw) if qw else np.zeros(0)
    qowner = np.concatenate(qowner) if qowner else np.zeros(0, int)
    iarea = np.bincount(qowner, weights=qw, minlength=len(interfaces))
    locator = StressField(coords, tets, np.zeros((len(tets), 4, 6)))
    qelem = locator.locate(qp)
    out = {"asset": asset["meta"]["name"], "oracle_source": f"golden ({gdir})" if use_golden else "computed", "fem_nodes": int(len(coords)), "fem_tets": int(len(tets)), "fem_h": h, "levels": []}
    fem_cases = {}
    frozen = []
    for case in net["load_cases"]:
        if use_golden:
            S = golden.load_fem(gdir, case["name"])
            resp = man["fem_response"][case["name"]]
        else:
            F, ftris, A, cen = face_loads(coords, tets, axis, hi, tol, case)
            ckey = hashlib.sha1(json.dumps([key, case, mat]).encode()).hexdigest()[:16]
            cpath = os.path.join(args.cache, f"fem2_{ckey}.npz")
            if os.path.exists(cpath):
                z = np.load(cpath)
                disp, S, C = z["disp"], z["S"], z["C"]
            else:
                disp, S, C = solve_static(coords, tets, mat["E"], mat["nu"], mat["rho"], fixed, F)
                np.savez(cpath, disp=disp, S=S, C=C)
            resp = response(disp, ftris, A, cen, case)
            frozen.append((case, (S, resp)))
        field = StressField(coords, tets, S)
        # ∫ σ dA per interface
        Sq = field.eval_many(qelem, qp) * qw[:, None, None]
        isig = np.zeros((len(interfaces), 3, 3))
        np.add.at(isig, qowner, Sq)
        fem_cases[case["name"]] = (isig, resp)
    dump = {}
    for lv in net["levels"]:
        lrep = {"level": lv["level"], "fragments": lv["fragments"], "cases": []}
        for res in lv["results"]:
            name = res["case"]
            if name not in fem_cases:
                continue
            isig, fem_resp = fem_cases[name]
            t_net, t_fem, areas = [], [], []
            v_raw, v_rec, v_fem, b_pos, b_nrm = [], [], [], [], []
            for b in res["bonds"]:
                n = np.array(b["normal"], float)
                ids = np.asarray(b["interfaces"], int)
                den = iarea[ids].sum()
                if den <= 0:
                    continue
                Sint = isig[ids].sum(axis=0)
                t_net.append(b["traction"])
                t_fem.append(n @ Sint @ n / den)
                areas.append(b["area"])
                v_raw.append(b.get("traction_vec", [0, 0, 0]))
                v_rec.append(b.get("recovered", [0, 0, 0]))
                v_fem.append(Sint @ n / den)
                b_pos.append(b.get("centroid", [0, 0, 0]))
                b_nrm.append(n)
            t_net, t_fem = np.array(t_net), np.array(t_fem)
            v_raw, v_rec, v_fem = np.array(v_raw), np.array(v_rec), np.array(v_fem)
            eps = 0.05 * (np.abs(t_fem).max() if len(t_fem) else 1.0)
            err_normal = np.abs(t_net - t_fem) / np.maximum(np.abs(t_fem), eps) if len(t_fem) else np.array([])
            fem_mag = np.linalg.norm(v_fem, axis=1) if len(v_fem) else np.array([])
            epsv = 0.05 * (fem_mag.max() if len(fem_mag) else 1.0)
            err_raw = np.linalg.norm(v_raw - v_fem, axis=1) / np.maximum(fem_mag, epsv) if len(v_fem) else np.array([])
            err = np.linalg.norm(v_rec - v_fem, axis=1) / np.maximum(fem_mag, epsv) if len(v_fem) else np.array([])
            stiff_err = abs(res["response"] / fem_resp - 1.0) if fem_resp else float("nan")
            dump[f"L{lv['level']}_{name}"] = dict(pos=np.array(b_pos), normal=np.array(b_nrm), area=np.array(areas), raw=v_raw, rec=v_rec, fem=v_fem, err_raw=err_raw, err_rec=err)
            lrep["cases"].append({"case": name, "bonds": int(len(err)), "traction_err_p50": pct(err, 50), "traction_err_p95": pct(err, 95),
                                  "raw_vector_err_p50": pct(err_raw, 50), "raw_vector_err_p95": pct(err_raw, 95),
                                  "raw_normal_err_p50": pct(err_normal, 50), "raw_normal_err_p95": pct(err_normal, 95), "stiffness_err": stiff_err, "network_response": res["response"], "fem_response": fem_resp})
        out["levels"].append(lrep)
    # modal
    try:
        fem_f = man["modal_hz"] if use_golden and man.get("modal_hz") else oracle_modal(args.cache, solid, verts, tris, h, L, axis, lo, tol, mat)
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
    for key in ("raw_vector_err_p95", "traction_err_p95", "stiffness_err"):
        s = series(key)
        out[f"{key}_by_level"] = s
        out[f"{key}_monotone"] = bool(all(s[i + 1] <= s[i] + 1e-12 for i in range(len(s) - 1)))
    if args.write_golden and not use_golden and frozen:
        golden.write_bond(gdir, out["asset"], solid, mat, axis, lo, hi, L, h, coords, tets, frozen, out.get("fem_modal_hz"))
        out["golden_written"] = gdir
    json.dump(out, open(os.path.join(args.cache, f"{out['asset']}.bond_fidelity.json"), "w"), indent=2)
    np.savez(os.path.join(args.cache, f"{out['asset']}.bond_errors.npz"), **{f"{k}__{f}": v for k, d in dump.items() for f, v in d.items()})
    # markdown summary
    print(f"### Bond fidelity: {out['asset']} (FEM: {out['fem_tets']} P2 tets, h = {h:.4f} m; oracle: {out['oracle_source']})\n")
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
    print(f"\nMonotone convergence (mean over cases): raw F/A p95 {out['raw_vector_err_p95_monotone']}, recovered p95 {out['traction_err_p95_monotone']}, stiffness {out['stiffness_err_monotone']}\n")


if __name__ == "__main__":
    main()
