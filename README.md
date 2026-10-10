# Shardwright

A toolkit that fractures any mesh along its real material and structural weak
points. It produces render-ready fragments and exact bond graphs for
destruction solvers.

`prefracture` is an offline Rust library and CLI that implements the
*Pre-Fracture Library — Engineering Spec v1.0*. It takes a glTF/OBJ/PLY/STL
scene plus authoring metadata and produces:

* a **glTF render payload** (`.glb`): one node per fragment, shared displaced
  interior surfaces, chipping, triplanar interior UVs, LODs (`MSFT_lod`),
  meshopt compression and rebar stubs;
* a **physics payload** (`.fracphys`, FlatBuffers, `schemas/frac.fbs`): the
  L0–L3 fragment hierarchy, exact mass properties, non-overlapping convex
  hulls, and bonds with exact area, centroid, normal, second moments, frame,
  extent, composition, rebar crossings, boundary loops, Weibull strength and
  parent/child links;
* a **report** (`.report.md` / `.report.json`): hard gates (§13.1), scorecard
  metrics and stage timings.

## Quick start

```sh
cargo build --release
# bake one asset (sidecar metadata <input>.meta.json is picked up automatically)
./target/release/prefracture bake --input benchmarks/assets/rc_column.glb \
    --config benchmarks/configs/bake.toml --out out/ [--check-determinism]
# re-validate and run the FEM bond-fidelity oracle (needs the Python harness env)
./target/release/prefracture validate --input out/rc_column.asset.json --oracle-cache /tmp/oracle
./target/release/prefracture inspect --input out/rc_column.fracphys
./target/release/prefracture diff --a out/a.fracphys --b out/b.fracphys
./target/release/prefracture patterns --out patterns.fracpat   # runtime pattern library (M5)
./target/release/prefracture gen-bench                         # regenerate benchmarks/assets
```

Configs: `benchmarks/configs/bake.toml` is the spec defaults with fracture
modes enabled. `fast.toml` disables modes and uses agglomeration for Level 1.
`ablation_geometric.toml` and `ablation_fallback.toml` are the M3 ablations.

## Layout

See [`docs/DESIGN.md`](docs/DESIGN.md) for the crate map, algorithms and
every deviation from the spec. See [`docs/VALIDATION.md`](docs/VALIDATION.md)
for oracle results: the benchmark suite, FEM bond fidelity, the convex
decomposition against upstream CoACD, and crack placement.

| Path | Contents |
|---|---|
| `crates/` | the 16 workspace crates (`frac-core` … `frac-cli`) |
| `schemas/` | FlatBuffers schemas (`frac.fbs`, `fracpat.fbs`) and TypeScript bindings |
| `benchmarks/` | benchmark assets (9 + 3 held-out) and configs |
| `tools/harness/` | oracles: Kratos FEM bond fidelity, scikit-fem modal, CoACD differential, Rankine crack oracle |
| `tools/oracles/` | Voro++ differential oracle |
| `tools/gltf_validate/` | Khronos glTF validator wrapper |
| `tools/ci/` | determinism script (also run by `.github/workflows/ci.yml` on Linux and macOS) |

## Oracle environment

The harness expects Python with Kratos Multiphysics 10.4
(StructuralMechanics, LinearSolvers), gmsh, scikit-fem, scipy, coacd,
manifold3d and trimesh. It looks for `/opt/fracenv/bin/python`; override with
`PREFRACTURE_PYTHON`. The glTF gate uses Node with the Khronos validator in
`tools/gltf_validate`.
