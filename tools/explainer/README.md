# Explainer

`make.sh` rebuilds the "Why Shardwright?" explainer and every number in the
README's [evidence section](../../README.md#the-evidence):

| Step | Script | Output |
|---|---|---|
| Comparison bakes (Shardwright vs uniform baseline) | `make.sh` | `out/why/{shardwright,uniform}/` |
| Fracture-modes models (bracket, bone, creature, chair; manifold3d) | `models.py` | `out/why/models/` |
| Random Voronoi fractures of the same cells | `random_voronoi.py` | `out/why/models/random/` |
| Measurements | `metrics.py` | `out/why/data/evidence.json` |
| Rapier collapse (hierarchy scene, README hero and runtime section) | `../sim/make.sh` | `out/sim/`, `docs/readme/hero_collapse.gif`, `collapse_closeup.gif`, `collapse_two_storey.mp4` |
| Support-graph views (bonds scene) | `support.sh` → `frac-sim --analyze`, `bond_graph.py`, `ghost.py` | `out/why/graph/`, `out/why/render/support_*/` |
| Renders | `render.py`, `render_zoom.py` | `out/why/render/` |
| Video frames and README stills | `compose.py` | `out/why/frames/`, `docs/readme/why_*.png` |
| MP4 and GIF | `make.sh` (ffmpeg) | `docs/readme/why_shardwright.{mp4,gif}` |
| One scene as a README GIF | `snippet.py` | `docs/readme/why_hierarchy.gif` |

The **uniform baseline** is the same pipeline and meshes with an isotropic
Voronoi recipe on every part and geometry-only clustering
(`benchmarks/configs/ablation_fallback.toml`) instead of fracture modes.

The bond comparison is the `bond_geometry` example
(`crates/frac-pipeline/examples/bond_geometry.rs`). It runs a
uniform-strain patch test on the same pieces with:
* the exact bonds;
* bonds whose normals and centroids come from the line between piece centres;
* bonds that keep only adjacency.

The collider numbers come from `tools/harness/coacd_diff.py`
([`docs/VALIDATION.md`](../../docs/VALIDATION.md)).

The **support-graph views** (`support.sh`):

1. `frac-sim --analyze` solves the design-checked bond network of the
   two-storey building under self-weight at Level 0 and Level 2. It then
   solves again with the column the wrecking ball hits removed
   (`--remove f0_col_22`), and records every bond's utilization and force
   (`out/why/data/bonds_L*.json`).
2. `bond_graph.py` turns those records into:
   * the support graph, as spheres and cylinders for
     `prefracture preview --graph`;
   * one-frame "sims" coloured per chunk by force or utilization, for
     `prefracture preview --sim`, which uses the same camera.
3. `ghost.py` lays the graph over a faded render of the parts.
