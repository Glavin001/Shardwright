# frac-collision

Collision hulls for every fragment of every hierarchy level: a pure-Rust
port of CoACD's approximate convex decomposition (Wei et al., "Approximate
Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree
Search", SIGGRAPH 2022; MIT-licensed reference code
<https://github.com/SarahWeiii/CoACD>, see `THIRD_PARTY_NOTICES.md`) adapted
to the fracture cell complex, plus the non-overlap guarantee CoACD does not
provide.

## Pipeline (`build_hulls`)

1. **Atoms.** Each leaf cell is an exact convex piece when convex (Voronoi
   cells of convex solids: the common case). Non-convex cells (curved
   shells, scans, arches) are cut by the CoACD port (`coacd::cut`) until
   each part's concavity `h = max(Rv, Hb)` is below the threshold (at most
   `2 × max_hulls_per_fragment` parts per cell, breadth first).
2. **Bottom-up merging.** For each fragment (finest level first, fragments
   in parallel) the pieces — atoms at the leaf level, the children's carried
   pieces above — are merged greedily by minimal merge cost (CoACD's
   `MergeConvexHulls`), until the hull budget is met and the cheapest merge
   exceeds the threshold (`concavity × diameter`). The pieces present when
   the threshold phase ends (at most `carry_cap` = 32) are carried to the
   parent level; a fragment's surface is a subset of its children's, so the
   carried pieces' costs stay valid upper bounds.
3. **Non-overlap** (not in CoACD). Overlapping hull pairs of bonded
   fragments at each level are detected in parallel and each pair is cut by
   one plane — the optimal remedy for two convex sets — chosen among the
   bond / interface normals, the face normals of the overlap region and the
   centroid direction, at the offset (uniform samples + golden section over
   the overlap interval) that removes the least of the fragments' own atom
   volume (screened with a smoothstep volume estimate, the best three
   refined with exact clipping). Independent pairs are resolved
   concurrently in an order-preserving way, so the result equals the
   sequential one. Hulls are then shrunk by `margin`.
4. Particle candidates keep a single hull.

## Merge cost (collision-aware)

For two pieces `a, b` with merged hull `H` (all in world units, compared
with the world threshold):

* `Rv = k·∛(3·(V(H) − V_a − V_b)/4π)`, `k = 0.3`, with the true volumes of
  the geometry the pieces cover (CoACD's volume term);
* covered-surface term: max distance from the fragment-surface samples of
  `a ∪ b` (spacing `max(threshold/2, √(area/6000))`) to the boundary of `H`
  (exact for points inside a convex polytope);
* hull-surface term: max distance from the part of `∂H` outside the
  fragment to the fragment surface, by best-first branch and bound over
  recursively split hull triangles (the distance field is 1-Lipschitz: a
  triangle with centroid distance `d` and circumradius `ρ` is bounded by
  `d + ρ`, and an inside centroid deeper than `ρ` proves the triangle
  inside), ε-optimal to half the sample spacing; side tests use
  angle-weighted pseudo-normals;
* neighbour intrusion: `0.6·∛(3V/4π)` for the volume `V` of `H` inside the
  atoms of bonded neighbour fragments (geometry the non-overlap step would
  remove again);
* the pieces' own costs, so a merge never hides earlier ones.

The merge is **lazy**: every candidate pair holds a lower bound refined in
stages (own costs → hull + Rv + covered-surface term → exact) only while it
competes for the minimum; the selected merges equal eager evaluation.

## The CoACD port (`coacd.rs`)

Ported: normalization to the [-1, 1] frame, `Clip` (exact shared cut
vertices, boundary loops, outer/hole nesting, constrained Delaunay caps),
`ComputeRv` / `ComputeHb` / `ComputeHCost`, the Monte-Carlo tree search
(`tree_policy`, `expand`, `default_policy`, `best_child`, `backup`) with
axis-aligned candidate planes, `ComputeBestRvClippingPlane`, `clip_by_path`,
`TernaryMCTS`, the iterative cut loop `Compute`, and `MergeConvexHulls`.
`coacd::decompose` runs the whole stand-alone pipeline on a closed mesh.

Deviations from upstream:

* `Hb` uses exact point-to-triangle distances (BVH) instead of the
  10-nearest-sample approximation, recognizes hull faces lying on the part
  surface (and part triangles lying on the hull) as zero-distance, and uses
  deterministic low-discrepancy samples instead of random + Sobol samples.
* Inside the tree search, halves are produced by a fast clip with fan caps
  (exact signed volume and convex hull, the only quantities the Rv-based
  search uses); the parts kept are clipped exactly with CDT caps.
* The merge cost is measured against the original geometry (as in
  upstream's current master, not the hull-vs-hull cost of 1.0.x), plus the
  terms above; upstream's `MeshDist < 0.01` candidate filter is replaced by
  cell adjacency in the pipeline (kept in `decompose`).
* Randomness (plane shuffles) uses ChaCha8 seeded per cell from the bake
  seed; no wall clock; parallel loops collect in index order. Output is
  bit-identical across runs and thread counts.
* No manifold preprocessing: fragment and cell meshes are already clean,
  closed and consistently oriented. Decimation / extrusion options are not
  ported (hulls stay exact, non-overlap is handled separately).

## Search effort (`[collision]` settings)

| setting | default | upstream CoACD |
|---|---|---|
| `mcts_iterations` | 12 | 150 |
| `mcts_depth` | 1 | 3 |
| `mcts_nodes` | 20 | 20 |
| `resolution` | 2000 | 2000 |

The tree search only runs on non-convex cells. On the ceramic bowl (40
curved-shell cells) upstream effort costs ~20× more clips and hull
evaluations than the defaults for no measurable difference after merging
(L1 symmetric Hausdorff/diameter median 0.0270 vs 0.0282, p95 0.0333 vs
0.0326); raise the effort for offline quality runs.

PARITY_TABLES

## Reproducing

```
prefracture bake --input benchmarks/assets/rc_column.glb --config benchmarks/configs/fast.toml --out out/
python tools/harness/coacd_diff.py --asset out/rc_column.asset.json --level 1            # equal budget
prefracture hulls --asset out/rc_column.asset.json --config benchmarks/configs/fast.toml \
    --coacd-threshold 0.03 --max-hulls 0 --levels 1 --out out/rc_column.nb.asset.json
python tools/harness/coacd_diff.py --asset out/rc_column.nb.asset.json --level 1 --no-budget
```

`FRAC_PROFILE=1` prints per-level timings and cost-component counters.
