#!/usr/bin/env python3
"""Fracture modes: differential test against the authors' reference
implementation (spec §13.2), geometry only (w_g = 1).

Paper: S. Sellán, J. Luong, L. Mattos Da Silva, A. Ramakrishnan, Y. Yang,
A. Jacobson. "Breaking Good: Fracture Modes for Realtime Destruction."
ACM Transactions on Graphics 42(1), 2023.

The reference implementation (github.com/sgsellan/fracture-modes) is
academic / non-commercial. With the authors' permission it is used here as a
TEST-ONLY ORACLE: it is cloned outside this repository
($FRACTURE_MODES_REF, else $ORACLES/fracture-modes, else
/opt/oracles/fracture-modes; `tools/setup.sh --with-fracture-modes-ref`)
and imported at run time. None of its code is copied here; everything in
this file is our own harness code.

MOSEK substitution. The reference solves its conic subproblem with MOSEK
(commercial). `clarabel_conic_solve` below is our own formulation of the
same second-order-cone program for the open interior-point solver Clarabel,
installed in place of the reference's solve at import time (the module
attribute `compute_fracture_modes.conic_solve` is replaced). The reference's
`sparse_sqrt` (CHOLMOD) result is never used by its mode computation, so it
is replaced by a no-op to avoid the scikit-sparse dependency; its GUI
(polyscope) and meshing (gpytoolbox, tetgen) modules are never imported.

Both implementations get the SAME tet mesh (V, T), generated once by our
`frac_fem::tetrahedralize` through the `modes_from_mesh` example, normalized
to a unit bounding box like the reference's own pipeline does.

Commands:

  # live comparison (needs the reference + venv; prints a markdown table)
  $ORACLES/fmref-venv/bin/python tools/harness/fracture_modes_ref.py run \
      [--cases notched_bar,l_shape,plate_hole,bunny] [--k 6] [--json out.json] \
      [--configs full:uniform,...] [--fields] [--freeze] [--golden]

  --fields  runs our side from a copy of this workspace with
            tools/harness/patches/*.patch applied (our own proposed changes
            to frac-modes, not reference code): 01 exposes the mode
            displacement fields needed for principal angles (the public
            `ModesOutput` has jumps and energies only), 02/03 add the
            per-cell translation model `cell-p0` of the paper's §3.6.
  --freeze  writes the golden dataset benchmarks/golden/fracture_modes/
            (meshes, reference modes and per-mode pieces; npz in Git LFS).
  --golden  uses the frozen meshes and reference outputs instead of running
            the reference.

  # frozen comparison (no reference needed): see test_fracture_modes_golden.py

Results, formulation differences and diagnosis: docs/VALIDATION.md,
"Fracture modes vs reference implementation".
"""
import argparse
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
GOLDEN = os.path.join(ROOT, "benchmarks", "golden", "fracture_modes")
PATCHES = os.path.join(HERE, "patches")
ORACLES = os.environ.get("ORACLES", "/opt/oracles")
REF_DIR = os.environ.get("FRACTURE_MODES_REF", os.path.join(ORACLES, "fracture-modes"))
# commit the golden dataset was generated with (tools/setup.sh pins the same)
REF_COMMIT = "bdf5051fb0d78d787f49fabbf75b54bbc6698b17"

# §13.2 targets
MAX_ANGLE_DEG = 5.0
MIN_ARI = 0.9

# ------------------------------------------------------------- shapes ----


def _triangulate(poly):
    """Ear clipping of a simple CCW polygon (list of (x, y))."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    idx = list(range(len(poly)))
    tris = []
    while len(idx) > 3:
        n = len(idx)
        for i in range(n):
            a, b, c = idx[(i - 1) % n], idx[i], idx[(i + 1) % n]
            if cross(poly[a], poly[b], poly[c]) <= 1e-14:
                continue
            if any(j not in (a, b, c) and cross(poly[a], poly[b], poly[j]) >= 0
                   and cross(poly[b], poly[c], poly[j]) >= 0 and cross(poly[c], poly[a], poly[j]) >= 0 for j in idx):
                continue
            tris.append((a, b, c))
            idx.pop(i)
            break
        else:
            raise RuntimeError("ear clipping failed")
    tris.append(tuple(idx))
    return tris


def extrude(poly, z0, z1):
    n = len(poly)
    v = [(x, y, z0) for x, y in poly] + [(x, y, z1) for x, y in poly]
    f = []
    for a, b, c in _triangulate(poly):
        f.append((a, c, b))
        f.append((a + n, b + n, c + n))
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, j + n), (i, j + n, i + n)]
    return np.array(v, float), np.array(f, int)


def extrude_ring(outer, inner, z0, z1):
    n = len(outer)
    v = []
    for z in (z0, z1):
        v += [(x, y, z) for x, y in outer] + [(x, y, z) for x, y in inner]
    ob, ib, ot, it = 0, n, 2 * n, 3 * n
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [(ot + i, ot + j, it + j), (ot + i, it + j, it + i),
              (ob + i, ib + j, ob + j), (ob + i, ib + i, ib + j),
              (ob + i, ob + j, ot + j), (ob + i, ot + j, ot + i),
              (ib + i, it + j, ib + j), (ib + i, it + i, it + j)]
    return np.array(v, float), np.array(f, int)


def read_obj(path):
    v, f = [], []
    for line in open(path):
        s = line.split()
        if not s:
            continue
        if s[0] == "v":
            v.append([float(x) for x in s[1:4]])
        elif s[0] == "f":
            f.append([int(x.split("/")[0]) - 1 for x in s[1:4]])
    return np.array(v), np.array(f, int)


def normalize(v):
    """Unit bounding box (largest extent 1), centered at the origin: the
    reference pipeline's normalization, which its absolute 0.1 piece
    threshold assumes. Our formulation is scale-invariant."""
    v = v - v.min(axis=0)
    v = v / v.max()
    return v - 0.5 * v.max(axis=0)


def case_surfaces():
    """Known-answer shapes of crates/frac-modes/tests/known_answer.rs (same
    geometry) plus the reference's bundled bunny. Values: (verts, tris, h)."""
    out = {}
    notch = [(0, 0), (4, 0), (4, 1), (2.1, 1), (2.1, 0.5), (1.9, 0.5), (1.9, 1), (0, 1)]
    out["notched_bar"] = extrude(notch, 0.0, 1.0) + (0.085,)
    lsh = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    out["l_shape"] = extrude(lsh, 0.0, 0.5) + (0.11,)
    n = 48
    outer, inner = [], []
    for i in range(n):
        a = 2 * np.pi * (i + 0.5) / n
        s, c = np.sin(a), np.cos(a)
        t = 1 / max(abs(c), abs(s))
        outer.append((c * t, s * t))
        inner.append((0.3 * c, 0.3 * s))
    out["plate_hole"] = extrude_ring(outer, inner, 0.0, 0.3) + (0.095,)
    bunny = os.path.join(REF_DIR, "data", "bunny_oded.obj")
    if os.path.exists(bunny):
        v, f = read_obj(bunny)
        out["bunny"] = (v, f, 0.115)
    return out


# ------------------------------------------------------------ helpers ----


def tet_volumes(V, T):
    p = V[T]
    return np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6.0


def orient(V, T):
    """Positive orientation (libigl's signed volume convention)."""
    T = T.copy()
    neg = tet_volumes(V, T) < 0
    T[neg, 0], T[neg, 1] = T[neg, 1].copy(), T[neg, 0].copy()
    return T


def interior_faces(T):
    """Interior faces as (t1, t2, v0, v1, v2) with t1 < t2."""
    loc = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
    keys = {}
    pairs = []
    for t, tt in enumerate(T):
        for l in loc:
            k = tuple(sorted(tt[list(l)]))
            if k in keys:
                pairs.append((keys.pop(k), t) + k)
            else:
                keys[k] = t
    return np.array(pairs, int).reshape(-1, 5)


def tri_areas(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def segment_level1(n_cells, groups, max_jump, target):
    """Python port of our own `frac_modes::segment_level1` (deterministic
    bisection on the distinct jump values; largest σ with ≥ target
    fragments, then the closer of it and the next σ, ties → larger σ)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    groups = np.asarray(groups)
    max_jump = np.asarray(max_jump, float)
    vals = np.unique(np.concatenate([[0.0], max_jump[(max_jump > 0) & np.isfinite(max_jump)]]))

    def sigma_at(j):
        return 0.5 * (vals[j] + vals[j + 1]) if j + 1 < len(vals) else vals[j]

    def ev(sigma):
        keep = ~(max_jump > sigma)
        g = groups[keep]
        a = coo_matrix((np.ones(len(g)), (g[:, 0], g[:, 1])), shape=(n_cells, n_cells))
        n, lab = connected_components(a, directed=False)
        # canonical labels: order of first appearance
        _, first = np.unique(lab, return_index=True)
        order = np.argsort(first)
        remap = np.empty(n, int)
        remap[order] = np.arange(n)
        return n, remap[lab]

    last = len(vals) - 1
    lo, hi = 0, last
    if ev(sigma_at(0))[0] < target:
        j = 0
    elif ev(sigma_at(hi))[0] >= target:
        j = hi
    else:
        while hi - lo > 1:
            mid = lo + (hi - lo) // 2
            if ev(sigma_at(mid))[0] >= target:
                lo = mid
            else:
                hi = mid
        j = lo
    fa, la = ev(sigma_at(j))
    best = (sigma_at(j), fa, la)
    if j < last and fa > target:
        fb, lb = ev(sigma_at(j + 1))
        if abs(fb - target) <= abs(fa - target):
            best = (sigma_at(j + 1), fb, lb)
    return best[2], best[1]


def adjusted_rand_index(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    n = len(a)
    cont = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(cont, (ai, bi), 1)
    c2 = lambda x: x * (x - 1) / 2.0
    s = c2(cont).sum()
    sa = c2(cont.sum(1)).sum()
    sb = c2(cont.sum(0)).sum()
    exp = sa * sb / c2(n)
    mx = 0.5 * (sa + sb)
    if mx == exp:
        return 1.0
    return (s - exp) / (mx - exp)


def voxel_tets(V, T, res=48):
    """Tet containing each voxel center of a res^3 grid over the bounding box
    (voxels outside the mesh dropped): §13.2 compares voxelized labels."""
    lo, hi = V.min(0), V.max(0)
    h = (hi - lo).max() / res
    axes = [np.arange(lo[k] + h / 2, hi[k], h) for k in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    owner = -np.ones(len(P), int)
    p = V[T]
    tmin, tmax = p.min(1), p.max(1)
    for t in range(len(T)):
        i0 = np.maximum(np.floor((tmin[t] - lo - h / 2) / h).astype(int), 0)
        i1 = np.ceil((tmax[t] - lo - h / 2) / h).astype(int)
        sl = np.stack(np.meshgrid(*[np.arange(i0[k], min(i1[k], len(axes[k]) - 1) + 1) for k in range(3)], indexing="ij"), -1).reshape(-1, 3)
        if len(sl) == 0:
            continue
        flat = (sl[:, 0] * len(axes[1]) + sl[:, 1]) * len(axes[2]) + sl[:, 2]
        A = np.stack([p[t, 1] - p[t, 0], p[t, 2] - p[t, 0], p[t, 3] - p[t, 0]], 1)
        bc = np.linalg.solve(A, (P[flat] - p[t, 0]).T).T
        inside = (bc >= -1e-12).all(1) & (bc.sum(1) <= 1 + 1e-12)
        owner[flat[inside]] = t
    return owner[owner >= 0]


def principal_angles_deg(A, B, w):
    """Principal angles between span(A) and span(B) (columns) in the inner
    product <x, y> = Σ w x y."""
    sw = np.sqrt(w)[:, None]
    qa, _ = np.linalg.qr(A * sw)
    qb, _ = np.linalg.qr(B * sw)
    s = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), -1, 1)
    return np.degrees(np.arccos(s))


# ------------------------------------------------------ our implementation ----


def build_example(root=ROOT, target_dir=None, rustflags=None):
    env = dict(os.environ)
    if target_dir:
        env["CARGO_TARGET_DIR"] = target_dir
    if rustflags:
        env["RUSTFLAGS"] = (env.get("RUSTFLAGS", "") + " " + rustflags).strip()
    subprocess.run(["cargo", "build", "--release", "-q", "-p", "frac-modes", "--example", "modes_from_mesh"],
                   cwd=root, env=env, check=True)
    td = target_dir or os.path.join(root, "target")
    return os.path.join(td, "release", "examples", "modes_from_mesh")


def run_example(exe, payload):
    with tempfile.TemporaryDirectory() as d:
        i, o = os.path.join(d, "in.json"), os.path.join(d, "out.json")
        with open(i, "w") as f:
            json.dump(payload, f)
        r = subprocess.run([exe, i, o], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"modes_from_mesh failed: {r.stderr}")
        sys.stderr.write(r.stderr)
        with open(o) as fh:
            return json.load(fh)


def make_mesh(exe, verts, tris, h):
    out = run_example(exe, {"solid": {"verts": normalize(verts).tolist(), "tris": np.asarray(tris).tolist()},
                            "h": h, "modes": False})
    V = np.array(out["mesh"]["verts"], float)
    T = orient(V, np.array(out["mesh"]["tets"], int))
    return V, T


# Configurations of our implementation compared with the reference:
# "<discretization>:<group weight>". full / cell-p1 are the current library
# (spec §4: P1 cell-exploded field, linear-elastic Q, w_g = 1); with every
# tet its own cell, cell-p1 spans the same space as full and exercises the
# production solver path (hybrid ADMM + Clarabel). cell-p0 is the proposed
# per-cell-translation model of the paper's §3.6 (needs the patches in
# tools/harness/patches/, i.e. --fields); with sqrt_area weights its
# per-face penalty equals the reference implementation's (∝ A_f·|jump|).
DEFAULT_CONFIGS = ("full:uniform", "cell-p1:uniform")
PATCHED_CONFIGS = ("cell-p0:uniform", "cell-p0:sqrt_area")


def ours_params(k, omega=1e-3, disc="full"):
    # spec defaults (eps 1e-4, 50 iterations); the interior-point solver for
    # the full problem so that solver tolerance is not a factor, the
    # production path (Auto: hybrid ADMM + Clarabel) for cell-p1
    solver = "auto" if disc == "cell-p1" else "clarabel"
    return {"k": k, "omega": omega, "eps": 1e-4, "max_iters": 50, "solver": solver}


def run_config(exe, V, T, k, config, targets=(2, 3, 4), omega=1e-3):
    disc, weight = config.split(":")
    out = run_example(exe, {"verts": V.tolist(), "tets": T.tolist(), "params": ours_params(k, omega, disc),
                            "discretizations": [disc], "level1_targets": list(targets), "group_weight": weight})
    run = out["runs"][0]
    run["config"] = config
    return run


# ------------------------------------------------------------- reference ----


def clarabel_conic_solve(D, M, Us, c, d, verbose=False):
    """Our open-solver formulation of the reference's conic subproblem

        min_u  Σ_e ‖(D u)_e‖₂   s.t.  Uᵢᵀ M u = 0 (previous modes),  cᵀ M u = 1,

    where (D u)_e stacks rows e, e+p, …, e+(d-1)p of D (p = rows/d). As a
    Clarabel conic program over x = [u; t]: minimize Σ t_e subject to the
    zero cone for the equalities and one second-order cone (t_e, (D u)_e)
    per interior face."""
    import clarabel

    q, A, b, n_eq, p, n = _conic_data(D, M, Us, c, d)
    P = csc_matrix_zeros(n + p)
    cones = [clarabel.ZeroConeT(n_eq)] + [clarabel.SecondOrderConeT(d + 1)] * p
    st = clarabel.DefaultSettings()
    st.verbose = bool(verbose)
    sol = clarabel.DefaultSolver(P, q, A, b, cones, st).solve()
    status = str(sol.status)
    if "Solved" not in status:
        sys.stderr.write(f"[clarabel] status {status}\n")
    clarabel_conic_solve.log.append((status, sol.iterations, sol.obj_val))
    return np.asarray(sol.x)[:n]


clarabel_conic_solve.log = []


def scs_conic_solve(D, M, Us, c, d, verbose=False):
    """Same conic program solved by SCS (first-order, independent code base);
    used only to cross-check the Clarabel substitution."""
    import scs
    q, A, b, n_eq, p, n = _conic_data(D, M, Us, c, d)
    data = {"A": A, "b": b, "c": q}
    sol = scs.SCS(data, {"z": n_eq, "q": [d + 1] * p}, verbose=bool(verbose), eps_abs=1e-9, eps_rel=1e-9,
                  max_iters=200000).solve()
    clarabel_conic_solve.log.append((sol["info"]["status"], sol["info"]["iter"], sol["info"]["pobj"]))
    return np.asarray(sol["x"])[:n]


def csc_matrix_zeros(n):
    from scipy.sparse import csc_matrix
    return csc_matrix((n, n))


def _conic_data(D, M, Us, c, d):
    from scipy.sparse import coo_matrix, csc_matrix, hstack, vstack

    D = coo_matrix(D)
    p = D.shape[0] // d
    n = D.shape[1]
    Mc = M @ np.asarray(c)
    eq = [np.asarray(M @ U).ravel() for U in Us] + [np.asarray(Mc).ravel()]
    A_eq = hstack([csc_matrix(np.vstack(eq)), csc_matrix((len(eq), p))])
    b_eq = np.zeros(len(eq))
    b_eq[-1] = 1.0
    # cone rows: face e occupies rows e*(d+1) .. e*(d+1)+d: (t_e, D_e,0, ..., D_e,d-1)
    rows = (D.row % p) * (d + 1) + 1 + D.row // p
    A_d = coo_matrix((-D.data, (rows, D.col)), shape=(p * (d + 1), n + p))
    A_t = coo_matrix((-np.ones(p), (np.arange(p) * (d + 1), n + np.arange(p))), shape=(p * (d + 1), n + p))
    A = vstack([A_eq, (A_d + A_t)]).tocsc()
    b = np.concatenate([b_eq, np.zeros(p * (d + 1))])
    q = np.concatenate([np.zeros(n), np.ones(p)])
    return q, A, b, len(eq), p, n


def load_reference(ref_dir=REF_DIR, solve=None):
    """Imports the reference's mode computation without its package
    __init__ (which pulls in GUI and meshing dependencies) and installs the
    Clarabel adapter (or `solve`) in place of the MOSEK solve."""
    if not os.path.isdir(os.path.join(ref_dir, "fracture_utility")):
        raise RuntimeError(f"reference implementation not found at {ref_dir} (tools/setup.sh --with-fracture-modes-ref)")
    if "mosek" not in sys.modules:
        sys.modules["mosek"] = types.ModuleType("mosek")  # never called: solve replaced below
    try:
        import sksparse.cholmod  # noqa: F401
    except ImportError:
        sk = types.ModuleType("sksparse")
        ch = types.ModuleType("sksparse.cholmod")
        ch.cholesky = None
        sk.cholmod = ch
        sys.modules["sksparse"], sys.modules["sksparse.cholmod"] = sk, ch
    pkg = types.ModuleType("fracture_utility")
    pkg.__path__ = [os.path.join(ref_dir, "fracture_utility")]
    sys.modules["fracture_utility"] = pkg
    import importlib
    cfm = importlib.import_module("fracture_utility.compute_fracture_modes")
    prm = importlib.import_module("fracture_utility.fracture_modes_parameters")
    cfm.conic_solve = solve or clarabel_conic_solve
    cfm.sparse_sqrt = lambda A: A  # result unused by the mode computation
    return cfm, prm


def run_reference(V, T, k_nontrivial, d=3, max_iter=10, tol=1e-4, solve=None):
    """Reference modes, d = 3 (vector displacements). The reference's first d
    modes are the global translations (its initial eigenvectors include the
    constant null space); k_nontrivial further modes are requested. Returns
    per-tet displacements (n_tets, 3, k) of the non-trivial modes, M-unit
    with M = tet volumes as in the reference, plus its per-mode labels."""
    cfm, prm = load_reference(solve=solve)
    params = prm.fracture_modes_parameters(num_modes=k_nontrivial + d, verbose=True, d=d, max_iter=max_iter, tol=tol)
    clarabel_conic_solve.log = []
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        res = cfm.compute_fracture_modes(V, T, params)
    secs = time.time() - t0
    modes, labels = res[2], res[3]
    nt = len(T)
    U = np.asarray(modes).reshape(d, nt, -1).transpose(1, 0, 2)  # (nt, d, modes)
    iters = [int(x) for x in re.findall(r"using (\d+) iterations", buf.getvalue())]
    pieces = [int(x) for x in re.findall(r"into (\d+) pieces", buf.getvalue())]
    vol = tet_volumes(V, T)
    # identify the trivial (translation) modes: constant fields
    mean = np.einsum("t,tdk->dk", vol, U) / vol.sum()
    const = np.einsum("t,tdk->k", vol, (U - mean[None]) ** 2) / np.maximum(np.einsum("t,tdk->k", vol, U ** 2), 1e-300)
    trivial = const < 1e-6
    keep = np.nonzero(~trivial)[0][:k_nontrivial]
    return {
        "U": U[:, :, keep], "labels": np.asarray(labels)[:, keep].astype(int), "iterations": np.array(iters)[keep],
        "pieces": np.array(pieces)[keep], "n_trivial": int(trivial[:d].sum()), "trivial_first": bool(trivial[:d].all()),
        "seconds": secs, "solver_log": list(clarabel_conic_solve.log), "max_iter": max_iter, "tol": tol,
    }


# --------------------------------------------------------------- metrics ----


def exploded_weights(V, T):
    """Lumped M̂ weights of the fully exploded P1 space (per tet corner,
    total 1), and the per-tet M̂ weights."""
    vol = tet_volumes(V, T)
    return np.repeat(vol / (4 * vol.sum()), 4), vol / vol.sum()


def lift_ref(U, V, T):
    """Reference per-tet fields (nt, 3, k), M-unit with M = volumes, to the
    exploded P1 space (4 corners per tet), M̂-unit (total mass 1)."""
    vol = tet_volumes(V, T)
    Uc = U * np.sqrt(vol.sum())
    return np.repeat(Uc[:, None, :, :], 4, axis=1).reshape(len(T) * 4 * 3, -1)


def ours_fields_exploded(run, T):
    """Our mode fields (patched build) per (vertex, cell = tet) node mapped to
    the exploded P1 layout (tet, corner, xyz)."""
    nodes = [tuple(x) for x in run["nodes"]]
    index = {nv: i for i, nv in enumerate(nodes)}
    k = len(run["mode_fields"])
    F = np.array(run["mode_fields"]).reshape(k, -1, 3)
    out = np.zeros((len(T), 4, 3, k))
    for t, tt in enumerate(T):
        for c, v in enumerate(tt):
            out[t, c] = F[:, index[(int(v), t)], :].T
    return out.reshape(len(T) * 12, k)


def ref_face_jumps(U, V, T, faces):
    """|jump| per interior face of M̂-unit reference modes: (k, n_faces)."""
    vol = tet_volumes(V, T)
    Uc = U * np.sqrt(vol.sum())
    return np.linalg.norm(Uc[faces[:, 0]] - Uc[faces[:, 1]], axis=1).T


def compare(V, T, ref, ours_runs, targets=(2, 3, 4), omega=1e-3, voxels=None):
    """§13.2 metrics of each of our runs against the reference (one row per
    run; a run's "config" names its group weights)."""
    return [_compare_one(V, T, ref, run, targets, omega, voxels if voxels is not None else voxel_tets(V, T))
            for run in ours_runs]


def _compare_one(V, T, ref, run, targets, omega, owner):
    weight = run.get("config", "full:uniform").split(":")[1]
    faces = interior_faces(T)
    areas = tri_areas(V, faces[:, 2:])
    L = abs(tet_volumes(V, T).sum()) ** (1 / 3)
    k = ref["U"].shape[2]
    jr = ref_face_jumps(ref["U"], V, T, faces)
    # reference modes under OUR energy: per-tet constants have zero strain
    # energy and a constant jump per face, so ‖B̂_f U‖ = √(A_f/L²)·|jump_f|;
    # plus the δ/2 regularization of our solvers
    wf = np.sqrt(areas / areas.mean()) if weight == "sqrt_area" else np.ones_like(areas)
    e_ref_ours = omega * (wf * np.sqrt(areas / L ** 2)[None, :] * jr).sum(1) + 0.5e-8
    # the reference implementation's functional: Σ_f A_f |jump_f| / L²
    e_ref_refn = (areas[None, :] / L ** 2 * jr).sum(1)
    # our groups (tet pairs) onto the face list
    fidx = {(a, b): i for i, (a, b) in enumerate(faces[:, :2])}
    order = np.array([fidx[(a, b)] for a, b in np.array(run["groups"], int)])
    jo = np.zeros((len(run["jumps"]), len(faces)))
    jo[:, order] = np.array(run["jumps"])
    kk = min(k, len(run["energies"]))
    e_ours = np.array(run["energies"][:kk])
    # Σ A_f rms_f: equals Σ A_f |jump_f| for per-tet constant fields, an upper
    # bound otherwise (only RMS jumps are public)
    e_ours_refn = (areas[None, :] / L ** 2 * jo[:kk]).sum(1)
    row = {
        "config": run.get("config", run["discretization"] + ":uniform"), "discretization": run["discretization"],
        "solver": run["solver_used"], "n_dofs": run["n_dofs"], "iterations": run["iterations"],
        "converged": run["converged"], "seconds": run.get("timings_ms", {}).get("total", float("nan")) / 1e3,
        "energy_ours": e_ours.tolist(), "energy_ref_in_our_functional": e_ref_ours[:kk].tolist(),
        "energy_rel_err": ((e_ours - e_ref_ours[:kk]) / e_ref_ours[:kk]).tolist(),
        "energy_ref_functional_ours": e_ours_refn.tolist(), "energy_ref_functional_ref": e_ref_refn[:kk].tolist(),
        "energy_ref_functional_rel_err": ((e_ours_refn - e_ref_refn[:kk]) / e_ref_refn[:kk]).tolist(),
    }
    # Level-1: the same procedure (our segment_level1) on both max-jump fields
    ari = {}
    for t in targets:
        lo, no = segment_level1(len(T), faces[:, :2], jo[:kk].max(0), t)
        lr, nr = segment_level1(len(T), faces[:, :2], jr[:kk].max(0), t)
        mine = next((l for l in run["level1"] if l["target"] == t), None)
        if mine is not None and kk == len(run["jumps"]):
            assert (np.array(mine["labels"], int) == lo).all(), "python port of segment_level1 disagrees"
        ari[str(t)] = {"ari_voxel": adjusted_rand_index(lo[owner], lr[owner]), "ari_tet": adjusted_rand_index(lo, lr),
                       "n_ours": int(no), "n_ref": int(nr)}
    row["level1"] = ari
    # the reference's own pieces of its first non-trivial mode (|Δc| < 0.1)
    n1 = int(ref["labels"][:, 0].max() + 1)
    lo1, _ = segment_level1(len(T), faces[:, :2], jo[:kk].max(0), n1)
    row["ref_mode1_native"] = {"n_pieces": n1, "ari_voxel": adjusted_rand_index(lo1[owner], ref["labels"][owner, 0])}
    if "mode_fields" in run:
        wv, wt = exploded_weights(V, T)
        Uo = ours_fields_exploded(run, T)[:, :kk]
        Ur = lift_ref(ref["U"], V, T)[:, :kk]
        ang = principal_angles_deg(Uo, Ur, np.repeat(wv, 3))
        # per-tet means (M̂-projection onto per-tet constants) remove the
        # within-tet rotation/strain part the reference cannot represent
        Uo_c = Uo.reshape(len(T), 4, 3, kk).mean(1).reshape(-1, kk)
        Ur_c = Ur.reshape(len(T), 4, 3, kk).mean(1).reshape(-1, kk)
        row["principal_angles_deg"] = ang.tolist()
        row["max_angle_deg"] = float(ang.max())
        row["principal_angles_tetmean_deg"] = principal_angles_deg(Uo_c, Ur_c, np.repeat(wt, 3)).tolist()
        row["tetmean_norm2"] = (np.repeat(wt, 3)[:, None] * Uo_c ** 2).sum(0).tolist()
    return row


def markdown(results):
    lines = ["| Mesh | Tets | Ours | Time (s) | Principal angles (°) | Energy rel. err, our E (median / max \\|·\\|) | Energy rel. err, reference E (median / max \\|·\\|) | ARI L1, T = 2 / 3 / 4 |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['n_tets']} | reference (Clarabel) | {r['reference']['seconds']:.1f} | — | — | — | — |")
        for row in r["ours"]:
            ma = ", ".join(f"{a:.1f}" for a in row["principal_angles_deg"]) if "principal_angles_deg" in row else "n/a (fields not public)"
            ee = np.abs(row["energy_rel_err"])
            er = np.abs(row["energy_ref_functional_rel_err"])
            ar = " / ".join(f"{row['level1'][t]['ari_voxel']:.2f}" for t in sorted(row["level1"], key=int))
            lines.append(f"| {name} | {r['n_tets']} | {row['config']} | {row['seconds']:.1f} | {ma} | "
                         f"{np.median(ee):.2g} / {ee.max():.2g} | {np.median(er):.2g} / {er.max():.2g} | {ar} |")
    return "\n".join(lines)




# ---------------------------------------------------------------- golden ----


def freeze(name, V, T, ref, h, k):
    """Golden dataset entry: the mesh both implementations ran on and the
    reference outputs (non-trivial modes as per-tet displacements, M-unit
    with M = tet volumes; per-mode piece labels)."""
    os.makedirs(GOLDEN, exist_ok=True)
    np.savez_compressed(os.path.join(GOLDEN, f"{name}.npz"), verts=V, tets=T.astype(np.int32),
                        ref_modes=ref["U"].astype(np.float64), ref_labels=ref["labels"].astype(np.int32))
    return {"n_tets": int(len(T)), "h": h, "k": k, "ref_iterations": ref["iterations"].tolist(),
            "ref_pieces": ref["pieces"].tolist(), "ref_seconds": round(ref["seconds"], 2),
            "ref_trivial_first": ref["trivial_first"]}


def load_golden(name):
    z = np.load(os.path.join(GOLDEN, f"{name}.npz"))
    return z["verts"], z["tets"].astype(int), {"U": z["ref_modes"], "labels": z["ref_labels"]}


def patched_workspace(dst):
    """Copy of this workspace with tools/harness/patches/*.patch applied
    (mode fields for the principal-angle metric; the proposed per-cell
    translation model)."""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns("target", ".git", "benchmarks", "out*"))
    for p in sorted(os.listdir(PATCHES)):
        subprocess.run(["patch", "-p1", "-s", "-i", os.path.join(PATCHES, p)], cwd=dst, check=True)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--cases", default="notched_bar,l_shape,plate_hole,bunny")
    ap.add_argument("--k", type=int, default=6,
                    help="non-trivial modes (a multiple of 3: the reference's modes come in degenerate direction triples)")
    ap.add_argument("--omega", type=float, default=1e-3)
    ap.add_argument("--configs", default=None, help="comma-separated disc:weight list (default: all available)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--fields", action="store_true")
    ap.add_argument("--golden", action="store_true", help="use the frozen meshes and reference outputs (no reference run)")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--work", default=os.path.join(tempfile.gettempdir(), "fmref-work"))
    a = ap.parse_args()
    exe = build_example()
    if a.fields:
        ws = patched_workspace(os.path.join(a.work, "ws"))
        exe = build_example(ws, os.path.join(a.work, "target"), "--cfg frac_modes_fields")
    configs = a.configs.split(",") if a.configs else list(DEFAULT_CONFIGS) + (list(PATCHED_CONFIGS) if a.fields else [])
    surfaces = case_surfaces()
    results = {}
    manifest = {"reference": {"repo": "https://github.com/sgsellan/fracture-modes", "commit": REF_COMMIT,
                              "conic_solver": "clarabel (substituted for MOSEK)", "d": 3, "max_iter": 10, "tol": 1e-4},
                "paper": "Sellán, Luong, Mattos Da Silva, Ramakrishnan, Yang, Jacobson. Breaking Good: Fracture Modes "
                         "for Realtime Destruction. ACM TOG 42(1), 2023",
                "mesher": "frac_fem::tetrahedralize on the normalized surface (unit bounding box)",
                "meshes": {
                    "notched_bar, l_shape, plate_hole": "known-answer solids of crates/frac-modes/tests/known_answer.rs",
                    "bunny": "bunny_oded.obj bundled with the reference repository (Stanford bunny), tetrahedralized by us"},
                "arrays": "verts, tets: the mesh both ran on; ref_modes: non-trivial reference modes as per-tet "
                          "displacements (tets, 3, k), M-unit with M = tet volumes; ref_labels: the reference's "
                          "per-mode pieces (tets, k)",
                "k": a.k, "omega": a.omega, "cases": {},
                "ours_params": {d: ours_params(a.k, a.omega, d) for d in ("full", "cell-p1", "cell-p0")}}
    mpath = os.path.join(GOLDEN, "manifest.json")
    if a.freeze and os.path.exists(mpath):
        old = json.load(open(mpath))
        if old.get("k") == a.k and old.get("reference", {}).get("commit") == REF_COMMIT:
            manifest["cases"] = old.get("cases", {})  # extend the existing dataset

    def write_manifest():
        os.makedirs(GOLDEN, exist_ok=True)
        with open(mpath, "w") as fh:
            json.dump(manifest, fh, indent=1)

    for name in a.cases.split(","):
        if a.golden:
            V, T, ref = load_golden(name)
            with open(mpath) as fh:
                frozen = json.load(fh)["cases"].get(name, {})
            ref_info = {"seconds": frozen.get("ref_seconds", float("nan")), "frozen": True}
        else:
            v, f, h = surfaces[name]
            V, T = make_mesh(exe, v, f, h)
            print(f"== {name}: {len(V)} verts, {len(T)} tets", file=sys.stderr)
            ref = run_reference(V, T, a.k)
            print(f"   reference: {ref['seconds']:.1f}s iterations {ref['iterations'].tolist()} pieces {ref['pieces'].tolist()} "
                  f"trivial-first {ref['trivial_first']}", file=sys.stderr)
            ref_info = {"iterations": ref["iterations"].tolist(), "pieces": ref["pieces"].tolist(),
                        "seconds": ref["seconds"], "trivial_first": ref["trivial_first"],
                        "solver_statuses": sorted(set(s for s, _, _ in ref["solver_log"]))}
            if a.freeze:
                manifest["cases"][name] = freeze(name, V, T, ref, h, a.k)
                write_manifest()
        owner = voxel_tets(V, T)
        rows = []
        for cfg in configs:
            t0 = time.time()
            run = run_config(exe, V, T, a.k, cfg, omega=a.omega)
            print(f"   ours {cfg}: {time.time() - t0:.1f}s", file=sys.stderr)
            rows.append(_compare_one(V, T, ref, run, (2, 3, 4), a.omega, owner))
        results[name] = {"n_tets": len(T), "ours": rows, "reference": ref_info}
        if a.json:
            with open(a.json, "w") as fh:
                json.dump(results, fh, indent=1)
    print(markdown(results))


if __name__ == "__main__":
    main()
