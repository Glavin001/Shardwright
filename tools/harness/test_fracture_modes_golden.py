#!/usr/bin/env python3
"""Fracture modes vs the frozen reference outputs (spec §13.2), without the
reference installed.

benchmarks/golden/fracture_modes/ holds, per test mesh, the tet mesh both
implementations ran on and the outputs of the authors' reference
implementation of "Breaking Good: Fracture Modes for Realtime Destruction"
(Sellán et al., ACM TOG 2023), produced by tools/harness/fracture_modes_ref.py
(MOSEK replaced by Clarabel, d = 3, w_g = 1, k = 6). This test runs OUR
implementation (crates/frac-modes/examples/modes_from_mesh.rs) on the same
meshes and scores it with the §13.2 metrics and the spec's thresholds:

  * Level-1 segmentation agreement: adjusted Rand index ≥ 0.9 on voxelized
    labels (T = 2, 3, 4 fragments, the same segmentation procedure on both);
  * principal angles between the mode subspaces ≤ 5°. This needs the mode
    displacement fields, which the public ModesOutput does not expose. It is
    checked only when FM_EXAMPLE points to a build with
    tools/harness/patches/ applied (`fracture_modes_ref.py run --fields`
    builds one);
  * relative error of each mode's energy: reported, since §13.2 sets no
    bound.

The current library fails these checks. The reasons are formulation
differences, see docs/VALIDATION.md, "Fracture modes vs reference
implementation". Known failures are skipped without running anything, and
the skip reason gives the measured numbers (like #[ignore]).

  python3 tools/harness/test_fracture_modes_golden.py -v
      dataset consistency only (seconds)
  FM_GOLDEN_RUN=1 python3 tools/harness/test_fracture_modes_golden.py -v
      also run our implementation: 10-55 min per mesh for the current
      full / cell-p1 discretizations on these fully exploded meshes
  FM_GOLDEN_KNOWN_FAILURES=1   also run and assert the known failures
  FM_GOLDEN_CONFIGS=full:uniform,cell-p1:uniform   (default: cell-p1:uniform)
  FM_GOLDEN_CASES=notched_bar,l_shape              (default: all)
  FM_EXAMPLE=/path/to/modes_from_mesh              (default: cargo build)

  # the proposed per-cell translation model (patched build), ~30 s in total:
  FM_GOLDEN_RUN=1 FM_EXAMPLE=$WORK/target/release/examples/modes_from_mesh \\
      FM_GOLDEN_CONFIGS=cell-p0:sqrt_area python3 tools/harness/test_fracture_modes_golden.py -v

Needs numpy and scipy (e.g. /opt/fracenv/bin/python).
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fracture_modes_ref as F  # noqa: E402  (does not import the reference itself)

MANIFEST = os.path.join(F.GOLDEN, "manifest.json")
RUN = os.environ.get("FM_GOLDEN_RUN") == "1"
KNOWN_FAILURES = os.environ.get("FM_GOLDEN_KNOWN_FAILURES") == "1"
CONFIGS = os.environ.get("FM_GOLDEN_CONFIGS", "cell-p1:uniform").split(",")
CASES = os.environ.get("FM_GOLDEN_CASES")
TARGETS = (2, 3, 4)

# Measured (docs/VALIDATION.md, k = 6): principal angles in degrees, ARI at
# T = 2 / 3 / 4. full and cell-p1 span the same space on these meshes and
# agree wherever both were run (cell-p1 was not run on plate_hole and bunny).
_CURRENT = {
    "notched_bar": "angles 12.0, 41.1, 41.1, 60.9, 69.6, 90.0; ARI 0.29 / 0.73 / 0.48; energies 43-79% below the reference's",
    "l_shape": "angles 45.5, 64.9, 65.2, 71.8, 90.0, 90.0; ARI 0.00 / 0.00 / 0.00; energies 26-62% below",
    "plate_hole": "angles 31.3, 81.8, 84.5, 88.8, 89.5, 89.9; ARI 0.00 / 0.00 / 0.06; energies 22-71% below",
    "bunny": "angles 33.4, 34.7, 38.5, 44.2, 89.5, 89.7; ARI -0.01 / 0.97 / 0.97; energies 52-89% below",
}
KNOWN_FAILING = {
    "full:uniform": _CURRENT,
    "cell-p1:uniform": _CURRENT,
    # proposed model, paper patch weights (patched build)
    "cell-p0:uniform": {
        "notched_bar": "angles 5.0 (x3), 10.9 (x3); ARI -0.06 / 0.47 / 0.98",
        "l_shape": "angles 20.2 (x3), 32.6 (x3); ARI -0.02 / 0.29 / 0.80",
        "plate_hole": "angles 22.9, 22.9, 35.8, 40.3, 40.3, 40.4; ARI 0.00 / 0.64 / 0.54",
        "bunny": "angles 0.0 (x3), 80.5 (x3); ARI -0.01 / 0.10 / 0.10",
    },
    # proposed model with the reference's per-face weighting: exact (0.0°,
    # ARI 1, energies to 1e-7) except on the 4-fold symmetric plate, where
    # ICCM from a different basis of the degenerate initial eigenspace ends in
    # a different local minimum (and T = 2 is unreachable for both)
    "cell-p0:sqrt_area": {
        "plate_hole": "angles 22.9, 22.9, 35.8, 40.3, 40.3, 40.4; ARI 0.00 / 0.39 / 0.54; energies -4.2% to +2.6%",
    },
}

_cache = {}


def manifest():
    with open(MANIFEST) as fh:
        return json.load(fh)


def cases():
    return [c for c in manifest()["cases"] if CASES is None or c in CASES.split(",")]


def example():
    exe = os.environ.get("FM_EXAMPLE")
    if exe:
        return exe
    if "exe" not in _cache:
        _cache["exe"] = F.build_example()
    return _cache["exe"]


def results(case, config):
    key = (case, config)
    if key not in _cache:
        m = manifest()
        V, T, ref = F.load_golden(case)
        run = F.run_config(example(), V, T, m["k"], config, TARGETS, m["omega"])
        if config.startswith("cell-p0") and not run["solver_used"].startswith("cell-p0"):
            raise unittest.SkipTest("cell-p0 needs a build with tools/harness/patches applied (FM_EXAMPLE)")
        _cache[key] = F._compare_one(V, T, ref, run, TARGETS, m["omega"], F.voxel_tets(V, T))
    return _cache[key]


def known_failure(case, config):
    return KNOWN_FAILING.get(config, {}).get(case)


@unittest.skipUnless(os.path.exists(MANIFEST), "golden dataset benchmarks/golden/fracture_modes missing")
class GoldenDataset(unittest.TestCase):
    """Cheap consistency checks of the frozen reference outputs."""

    def test_reference_modes_are_mass_orthonormal_and_nontrivial(self):
        for case in cases():
            V, T, ref = F.load_golden(case)
            vol = F.tet_volumes(V, T)
            self.assertTrue((vol > 0).all(), case)
            U = ref["U"]  # (tets, 3, k), M-unit with M = tet volumes
            G = np.einsum("t,tdi,tdj->ij", vol, U, U)
            np.testing.assert_allclose(G, np.eye(U.shape[2]), atol=1e-5, err_msg=case)
            # the global translations (the reference's first 3 modes) were dropped
            mean = np.einsum("t,tdk->dk", vol, U) / vol.sum()
            self.assertLess(np.abs(mean).max() * np.sqrt(vol.sum()), 1e-4, case)

    def test_reference_labels_follow_its_piece_threshold(self):
        # reference pieces: tets connected when their displacements differ by < 0.1
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        for case in cases():
            V, T, ref = F.load_golden(case)
            faces = F.interior_faces(T)
            for i in range(ref["U"].shape[2]):
                c = ref["U"][:, :, i]
                g = faces[np.linalg.norm(c[faces[:, 0]] - c[faces[:, 1]], axis=1) < 0.1, :2]
                _, lab = connected_components(coo_matrix((np.ones(len(g)), (g[:, 0], g[:, 1])), shape=(len(T),) * 2),
                                              directed=False)
                self.assertEqual(F.adjusted_rand_index(lab, ref["labels"][:, i]), 1.0, f"{case} mode {i}")


@unittest.skipUnless(os.path.exists(MANIFEST), "golden dataset benchmarks/golden/fracture_modes missing")
class OursVsReference(unittest.TestCase):

    def check(self, metric):
        for case in cases():
            for cfg in CONFIGS:
                with self.subTest(case=case, config=cfg):
                    reason = known_failure(case, cfg)
                    if reason and not KNOWN_FAILURES:
                        self.skipTest(f"known §13.2 failure ({cfg}, {case}): {reason}")
                    if not RUN:
                        self.skipTest("set FM_GOLDEN_RUN=1 to run our implementation")
                    metric(case, cfg, results(case, cfg))

    def test_energies_reported(self):
        def metric(case, cfg, r):
            e = np.array(r["energy_rel_err"])
            print(f"\n  {case} {cfg}: energy rel. err {np.round(e, 4).tolist()} (our functional), "
                  f"{np.round(r['energy_ref_functional_rel_err'], 4).tolist()} (reference functional)", file=sys.stderr)
            self.assertTrue(np.isfinite(e).all())
            self.assertTrue(all(r["converged"]), "ICCM did not converge")
        for case in cases():
            for cfg in CONFIGS:
                with self.subTest(case=case, config=cfg):
                    if not RUN:
                        self.skipTest("set FM_GOLDEN_RUN=1 to run our implementation")
                    metric(case, cfg, results(case, cfg))

    def test_level1_adjusted_rand_index(self):
        def metric(case, cfg, r):
            for t, v in r["level1"].items():
                self.assertGreaterEqual(v["ari_voxel"], F.MIN_ARI, f"T={t}")
        self.check(metric)

    def test_principal_angles(self):
        def metric(case, cfg, r):
            if "principal_angles_deg" not in r:
                self.skipTest("mode fields not exposed by this build (public ModesOutput has no displacements)")
            self.assertLessEqual(r["max_angle_deg"], F.MAX_ANGLE_DEG, str(r["principal_angles_deg"]))
        self.check(metric)


if __name__ == "__main__":
    unittest.main()
