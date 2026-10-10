# Testing strategy

Some oracles take minutes to an hour per asset: Kratos FEM on 100k P2 tets,
upstream CoACD, crack-oracle FEM solves. They cannot run on every change.
Their answers depend only on the *unfractured* input (solid, material, load
cases, mesh), never on our fracture output. So they are computed once,
**frozen as a golden dataset**, and every later build is scored against the
frozen answers in seconds.

## Tiers

| Tier | When | Time | What | Command |
|---|---|---|---|---|
| 0 | every commit | minutes | unit and property tests; exact-answer tests; differential tests against frozen oracle outputs; determinism | `cargo test --release --workspace`, `tools/ci/determinism.sh` |
| 1 | every PR | ~1 min per asset | bake each golden asset, score it against the frozen oracles, compare with the committed baseline | `tools/ci/golden.sh` |
| 2 | nightly / release | hours | regenerate the oracles from scratch (Docker image), full benchmark suite incl. the building, performance targets, re-freeze when oracle versions change | `docker build -t shardwright .`, `prefracture validate --no-golden --write-golden …` |

Tier 0 exact-answer tests:

* Network patch tests: uniform stress must be reproduced per bond, both in
  the interior and next to free surfaces
  (`crates/frac-pipeline/tests/network_patch.rs`).
* An exact pure-bending elasticity solution (the `network_diag` example).
* Fracture-mode known-answer cases: a notched bar, an L-shape, a plate with
  a hole, material weights and forbidden zones (`crates/frac-modes/tests`).
* Clipping property tests (volume conservation, watertightness,
  degeneracies).
* Pattern-library tests.

Tier 0 differential tests: Voro++ (frozen outputs in
`crates/frac-cells/tests/golden/`; the live binary is cross-checked when
installed). Fracture modes against the authors' reference implementation
(`tools/harness/test_fracture_modes_golden.py`, `FM_GOLDEN_RUN=1`): our
default model against the frozen reference outputs, with the §13.2
thresholds, in about 5 s. The linear-elastic P1 configurations take 10–55 min
per mesh (tier 2). See `VALIDATION.md`.

## Golden oracle dataset

`benchmarks/golden/<asset>/` holds the following (`.npz` files in Git LFS):

| File | Contents |
|---|---|
| `manifest.json` | solid SHA-1, material, axis and planes, FEM mesh size, load cases, loaded-face responses, modal frequencies, oracle versions |
| `mesh.npz`, `fem_<case>.npz` | Kratos P2 mesh and per-element linear stress fields (float32) |
| `cracks.npz` | Rankine crack-oracle surfaces |
| `test.json` | input asset and bake config used by `tools/ci/golden.sh` |
| `expected.json` | baseline metrics (tolerance: 5% relative or 0.005 absolute) |

The harness uses a golden set only when the solid hash, material and load
cases match. Otherwise it recomputes, so a stale answer is never reused.
`prefracture validate` scores automatically against a matching golden set
and needs only numpy and scipy for that.

| Oracle | Frozen as | Status |
|---|---|---|
| Kratos FEM (statics) | stress fields per load case | rc_column |
| scikit-fem (modal) | first 10 frequencies | rc_column |
| Rankine crack oracle | crack surfaces per load case | rc_column |
| Voro++ | cell volumes and vertices | 2 cases |
| fracture-modes reference (Sellán et al.; MOSEK → Clarabel) | tet meshes, 6 modes and per-mode pieces (`benchmarks/golden/fracture_modes/`) | 4 meshes |
| upstream CoACD | hulls on fixed fragment meshes | planned |
| Khronos glTF validator, flatc | — (cheap; run live) | — |

Adding an asset:

```sh
prefracture bake --input benchmarks/assets/X.glb --config benchmarks/configs/fast.toml --out out/
prefracture validate --input out/X.asset.json --oracle-cache /tmp/oc --write-golden   # FEM: minutes–hours
python tools/harness/crack_oracle.py --asset out/X.asset.json --cache /tmp/oc --write-golden
echo '{"input": "benchmarks/assets/X.glb", "config": "benchmarks/configs/fast.toml", "oracles": ["bond", "crack"]}' > benchmarks/golden/X/test.json
UPDATE=1 tools/ci/golden.sh X     # write the baseline
```

A change that improves a metric should re-baseline (`UPDATE=1`) in the same
PR, so the improvement becomes the new floor. `STRICT=1` also enforces the
spec targets.
