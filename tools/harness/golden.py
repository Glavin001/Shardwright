"""Golden oracle dataset (versioned under benchmarks/golden/<asset>/).

FEM oracle solutions depend only on the UNFRACTURED solid, its material,
the load cases and the FEM mesh, never on the fracture output. They are
expensive (minutes to tens of minutes per asset) and are therefore frozen
once and reused by every later bake. Scoring against a golden set needs only
numpy/scipy (no Kratos, gmsh or scikit-fem).

Layout of benchmarks/golden/<asset>/ (npz files are tracked by Git LFS):
  manifest.json   solid sha1, material, axis/planes, FEM h, load cases,
                  loaded-face responses, modal frequencies, oracle versions
  mesh.npz        coords (float64 [N,3]), tets (int32 [E,10], gmsh order)
  fem_<case>.npz  C: per-element linear stress fit (float32 [E,4,6]; row 0 =
                  value at the element centroid, rows 1..3 = gradient)
  cracks.npz      Rankine crack-oracle samples per load case (float32 [n,3])
"""
import datetime
import hashlib
import json
import os

import numpy as np

GOLDEN_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarks", "golden"))
FORMAT = 1


def solid_sha(solid):
    return hashlib.sha1(json.dumps(solid).encode()).hexdigest()


def default_dir(asset_name):
    return os.path.join(GOLDEN_ROOT, asset_name)


def _close(a, b, rel=1e-9):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], rel) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)
    return a == b


def load_manifest(directory):
    p = os.path.join(directory, "manifest.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    return m if m.get("format") == FORMAT else None


def matches(manifest, solid, mat, load_cases=None):
    """True when the golden set was computed for this solid, material and
    (optionally) these load cases."""
    if manifest is None or manifest.get("solid_sha1") != solid_sha(solid):
        return False
    if not _close(manifest.get("material", {}), {k: mat[k] for k in ("E", "nu", "rho")}):
        return False
    if load_cases is not None:
        stored = {c["name"]: c for c in manifest.get("load_cases", [])}
        for c in load_cases:
            if c["name"] not in stored or not _close(stored[c["name"]], c):
                return False
    return True


def load_fem(directory, case_name):
    z = np.load(os.path.join(directory, f"fem_{case_name}.npz"))
    return z["C"].astype(np.float64)


def load_mesh(directory):
    z = np.load(os.path.join(directory, "mesh.npz"))
    return z["coords"], z["tets"].astype(np.int64)


def _versions():
    v = {}
    for mod in ("KratosMultiphysics", "gmsh", "skfem", "numpy", "scipy"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", None) or getattr(m, "GMSH_API_VERSION", None) or "?"
        except Exception:
            pass
    try:
        from importlib.metadata import version
        v["KratosMultiphysics"] = version("KratosMultiphysics")
    except Exception:
        pass
    return v


def write_bond(directory, asset_name, solid, mat, axis, lo, hi, length, h, coords, tets, cases, modal_hz):
    """Freeze the bond-fidelity oracle. `cases` maps a load-case dict (with
    'name') to (C [E,4,6], fem_response)."""
    os.makedirs(directory, exist_ok=True)
    np.savez_compressed(os.path.join(directory, "mesh.npz"), coords=np.asarray(coords, np.float64), tets=np.asarray(tets, np.int32))
    lcs = []
    for case, (C, resp) in cases:
        np.savez_compressed(os.path.join(directory, f"fem_{case['name']}.npz"), C=np.asarray(C, np.float32))
        lcs.append(dict(case, fem_response=float(resp)))
    old = load_manifest(directory) or {}
    m = {
        "format": FORMAT,
        "asset": asset_name,
        "solid_sha1": solid_sha(solid),
        "material": {k: mat[k] for k in ("E", "nu", "rho")},
        "axis": axis,
        "fixed_plane": lo,
        "loaded_plane": hi,
        "length": length,
        "fem": {"h": h, "nodes": int(len(coords)), "tets": int(len(tets)), "element": "SmallDisplacementElement3D10N", "solver": "AMGCL CG 1e-10 (LU below 60k dofs)"},
        "load_cases": [{k: v for k, v in c.items() if k != "fem_response"} for c in lcs],
        "fem_response": {c["name"]: c["fem_response"] for c in lcs},
        "modal_hz": modal_hz,
        "versions": _versions(),
        "created": datetime.date.today().isoformat(),
    }
    for k in ("crack_cases", "crack_tau_reference"):
        if k in old:
            m[k] = old[k]
    json.dump(m, open(os.path.join(directory, "manifest.json"), "w"), indent=2)


def write_cracks(directory, solid, cracks, case_defs):
    """Freeze the crack-oracle surfaces (sample points per load case)."""
    os.makedirs(directory, exist_ok=True)
    np.savez_compressed(os.path.join(directory, "cracks.npz"), **{k: np.asarray(v, np.float32) for k, v in cracks.items()})
    m = load_manifest(directory) or {"format": FORMAT, "solid_sha1": solid_sha(solid)}
    m["crack_cases"] = case_defs
    m["crack_solid_sha1"] = solid_sha(solid)
    json.dump(m, open(os.path.join(directory, "manifest.json"), "w"), indent=2)


def load_cracks(directory, solid):
    m = load_manifest(directory)
    p = os.path.join(directory, "cracks.npz")
    if m is None or m.get("crack_solid_sha1") != solid_sha(solid) or not os.path.exists(p):
        return None
    z = np.load(p)
    return {k: z[k].astype(np.float64) for k in z.files}
