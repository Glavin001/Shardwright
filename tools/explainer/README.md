# Explainer

`make.sh` rebuilds the "Why Shardwright?" explainer and every number in the
README's [evidence section](../../README.md#the-evidence):

| Step | Script | Output |
|---|---|---|
| Comparison bakes (Shardwright vs uniform baseline) | `make.sh` | `out/why/{shardwright,uniform}/` |
| Fracture-modes models (bracket, bone, creature, chair; manifold3d) | `models.py` | `out/why/models/` |
| Random Voronoi fractures of the same cells | `random_voronoi.py` | `out/why/models/random/` |
| Measurements | `metrics.py` | `out/why/data/evidence.json` |
| Renders | `render.py`, `render_zoom.py` | `out/why/render/` |
| Video frames and README stills | `compose.py` | `out/why/frames/`, `docs/readme/why_*.png` |
| MP4 and GIF | `make.sh` (ffmpeg) | `docs/readme/why_shardwright.{mp4,gif}` |

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
