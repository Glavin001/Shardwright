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

### Fast path (no decomposition for convex shapes)

A solid is treated as **convex** when the volume of its convex hull exceeds
its exact volume by at most a relative `CONVEX_TOL = 1e-6` (cell volumes
are exact polyhedral integrals; the hull is the floating-point Quickhull of
the cell vertices). Convex leaf cells become atoms directly (no tree search),
and any fragment that is convex as a whole (or a particle candidate) gets
its single hull without merging. Only non-convex cells are cut and only
non-convex fragments are merged. `FRAC_PROFILE=1` prints, per level, how
many fragments take the fast path vs decomposition (see the table below).

### Hull vertex cap (`collision.max_hull_vertices`, default 64)

Every output hull has at most 64 vertices (physics-engine limit). A hull
over the limit is replaced by the hull of a vertex subset grown greedily
from its axis-extreme vertices, always adding the vertex farthest outside
the current hull (a budgeted Quickhull: most volume per vertex). This is an
inner approximation — it adds no overshoot and preserves separations — and
is applied before the non-overlap cuts and again after the cuts and the
margin shrink (plane clipping adds vertices). Measured effect at equal
budget (L1): bowl Hausdorff 0.0271/0.0326 → 0.0266/0.0325, messy_scan
0.0889/0.1186 → 0.0886/0.1192 (median/p95), coverage within 0.001.

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

Two estimator settings (`CoacdParams::hb`, `CoacdParams::merge_cost`):

* **faithful** (default of the stand-alone `coacd::decompose` /
  `prefracture decompose`): upstream's `ComputeHb` estimator —
  `ExtractPointSet` sampling (small triangles sampled every other one) and
  `face_hausdorff_distance` (distance to the triangles of the 10 nearest
  samples of the other surface) — and the 1.0.x hull-vs-hull merge cost
  `ComputeHCost(cvx1, cvx2, CH)`. This is the 1:1 port checked against
  coacd 1.0.14 below.
* **pipeline** (`build_hulls`): exact point-to-triangle distances (BVH)
  with zero-distance face recognition (hull faces lying on the part surface
  and part triangles lying on the hull), deterministic low-discrepancy
  samples, and the collision-aware merge cost above (as upstream's current
  master, the cost is measured against the original geometry). The upstream
  estimator overestimates distances wherever triangles are small (the true
  nearest triangle is often not among the 10 sampled ones), so upstream
  cuts more: on SnowFlake the exact estimator stops at 28 parts before
  merging, the upstream one (ours) at 58 (upstream 60).

Other deviations, in both settings:

* Inside the tree search, halves are produced by a fast clip with fan caps
  (exact signed volume and convex hull, the only quantities the Rv-based
  search uses); the parts kept are clipped exactly with CDT caps; when a
  cut section cannot be triangulated (pinched loops through vertices) the
  plane is retried with offsets up to 1e-3 (upstream throws).
* Randomness (plane shuffles, upstream-style samples) uses ChaCha8 seeded
  from `seed` (per cell from the bake seed in the pipeline) instead of
  mt19937, so the plane order differs from upstream's run for run; no wall
  clock; parallel loops collect in index order. Output is bit-identical
  across runs and thread counts.
* No manifold preprocessing (fragment and cell meshes are already clean and
  closed; the baseline script thickens open example meshes for both
  methods). Upstream's `MeshDist < 0.01` merge-candidate filter is kept in
  `decompose` and replaced by cell adjacency in the pipeline. Decimation /
  extrusion options are not ported (the 64-vertex cap replaces decimation;
  non-overlap is handled separately).

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
