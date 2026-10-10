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

## Parity with upstream CoACD

### 1. Baseline: stand-alone port vs coacd 1.0.14

`tools/harness/coacd_baseline.py`: `coacd::decompose` with upstream
defaults (threshold 0.05, 150 MCTS iterations, depth 3, 20 nodes,
resolution 2000, merge on, `--hb upstream --merge-cost upstream`) against
`coacd.run_coacd(threshold=0.05, seed=0)` (coacd 1.0.14), both on the same
closed input mesh. Inputs: standard shapes, the upstream example meshes
(`examples/*.obj`; Bottle and Kettle are open surfaces and are thickened
into thin solids first: offset 1% and grid 2% of the longest side), the
whole ceramic bowl and the largest fragments of our assets.

Both results are scored by one evaluator, deterministic per mesh. Units are
CoACD's, with the longest side normalized to 2. The columns:
- h = max(Rv, Hb).
- Rv = 0.3·∛(3|V − V(∪hulls)|/4π).
- Hb is the collision-aware symmetric deviation: the surface of
  ∪hulls − mesh measured to the mesh, and the surface of mesh − ∪hulls
  measured to the hull union. It uses 20k samples against a dense
  200k-sample reference, with a bias of about 0.002.

Seconds are wall time on a shared, loaded 4-core machine.

| Mesh | Tris | Hulls ours / CoACD | h ours / CoACD | Rv ours / CoACD | Hb ours / CoACD | Σ hull vol / V ours / CoACD | Seconds ours / CoACD |
|---|---|---|---|---|---|---|---|
| L_block | 28 | 2 / 2 | 0.0140 / 0.0163 | 0.0063 / 0.0127 | 0.0140 / 0.0163 | 1.000 / 1.000 | 7.6 / 0.5 |
| U_block | 44 | 3 / 3 | 0.0000 / 0.0175 | 0.0000 / 0.0091 | 0.0000 / 0.0175 | 1.000 / 1.000 | 6.1 / 0.9 |
| square_ring | 64 | 4 / 6 | 0.0145 / 0.0166 | 0.0106 / 0.0135 | 0.0145 / 0.0166 | 1.000 / 1.000 | 15.9 / 1.7 |
| torus | 1536 | 12 / 11 | 0.0548 / 0.0609 | 0.0548 / 0.0609 | 0.0284 / 0.0448 | 1.033 / 1.045 | 6.3 / 7.9 |
| SnowFlake.obj | 2208 | 40 / 47 | 0.0635 / 0.0556 | 0.0563 / 0.0556 | 0.0635 / 0.0529 | 1.310 / 1.314 | 13.1 / 45.6 |
| Bottle.obj (thickened) | 77540 | 86 / 106 | 0.1573 / 0.1356 | 0.1232 / 0.1356 | 0.1573 / 0.0608 | 1.873 / 2.139 | 181.7 / 450.1 |
| Kettle.obj (thickened) | 62242 | 95 / 74 | 0.6179 / 0.1312 | 0.1886 / 0.1312 | 0.6179 / 0.0605 | 4.481 / 2.160 | 323.5 / 268.2 |
| Octocat-v2.obj | 40246 | 53 / 62 | 0.0570 / 0.0522 | 0.0539 / 0.0522 | 0.0570 / 0.0486 | 1.110 / 1.142 | 47.4 / 101.3 |
| ceramic_bowl | 8548 | 33 / 42 | 0.0953 / 0.0848 | 0.0953 / 0.0848 | 0.0798 / 0.0502 | 1.928 / 1.654 | 18.6 / 33.4 |
| rc_column L1 #10 | 1054 | 65 / 99 | 0.0751 / 0.0668 | 0.0751 / 0.0668 | 0.0666 / 0.0515 | 1.053 / 1.042 | 26.0 / 61.8 |
| rc_column L1 #5 | 772 | 56 / 83 | 0.0752 / 0.0671 | 0.0744 / 0.0671 | 0.0752 / 0.0553 | 1.030 / 1.020 | 34.5 / 50.2 |
| rc_column L2 #66 | 326 | 49 / 70 | 0.0794 / 0.0648 | 0.0753 / 0.0648 | 0.0794 / 0.0503 | 1.038 / 1.026 | 25.0 / 55.7 |
| rc_column L2 #77 | 278 | 46 / 64 | 0.0736 / 0.0636 | 0.0736 / 0.0636 | 0.0721 / 0.0510 | 1.034 / 1.023 | 24.6 / 43.7 |
| messy_scan L1 #4 | 15788 | 55 / 71 | 0.0835 / 0.1047 | 0.0835 / 0.1047 | 0.0684 / 0.0623 | 1.077 / 1.147 | 27.5 / 284.4 |
| stone_arch L1 #2 | 1034 | 15 / 22 | 0.0606 / 0.0503 | 0.0535 / 0.0503 | 0.0606 / 0.0487 | 1.020 / 1.018 | 10.0 / 17.5 |
| brick_wall_window L1 #8 | 1160 | 16 / 9 | 0.0470 / 0.0741 | 0.0344 / 0.0741 | 0.0470 / 0.0547 | 1.019 / 1.188 | 3.4 / 37.5 |
| **median** | | 43 / 54 | 0.0685 / 0.0642 | 0.0649 / 0.0642 | 0.0650 / 0.0507 | 1.036 / 1.043 | 21.6 / 44.6 |

| Mesh | max hull vertices ours / CoACD | h ours, ≤64 vertices | Σ hull vol / V ours, ≤64 vertices | exact Hb + collision-aware merge: hulls / h / seconds |
|---|---|---|---|---|
| L_block | 10 / 10 | 0.0140 | 1.000 | 2 / 0.0140 / 0.0 |
| U_block | 10 / 8 | 0.0000 | 1.000 | 3 / 0.0000 / 0.1 |
| square_ring | 11 / 10 | 0.0145 | 1.000 | 4 / 0.0145 / 0.4 |
| torus | 103 / 126 | 0.0522 | 1.028 | 12 / 0.0548 / 4.8 |
| SnowFlake.obj | 28 / 29 | 0.0635 | 1.310 | 24 / 0.0629 / 5.2 |
| Bottle.obj (thickened) | 215 / 313 | 0.1566 | 1.870 | 136 / 0.1452 / 237.9 |
| Kettle.obj (thickened) | 1326 / 550 | 0.6155 | 4.421 | 251 / 0.6226 / 352.0 |
| Octocat-v2.obj | 1474 / 987 | 0.0568 | 1.103 | 61 / 0.0511 / 58.1 |
| ceramic_bowl | 229 / 198 | 0.0951 | 1.921 | 43 / 0.0838 / 28.6 |
| rc_column L1 #10 | 33 / 37 | 0.0751 | 1.053 | 68 / 0.0705 / 15.8 |
| rc_column L1 #5 | 23 / 25 | 0.0752 | 1.030 | 62 / 0.0688 / 9.2 |
| rc_column L2 #66 | 32 / 28 | 0.0794 | 1.038 | 52 / 0.0709 / 7.0 |
| rc_column L2 #77 | 24 / 23 | 0.0736 | 1.034 | 43 / 0.0698 / 4.4 |
| messy_scan L1 #4 | 108 / 318 | 0.0835 | 1.077 | 56 / 0.0774 / 26.2 |
| stone_arch L1 #2 | 21 / 26 | 0.0606 | 1.020 | 17 / 0.0511 / 3.2 |
| brick_wall_window L1 #8 | 19 / 116 | 0.0470 | 1.019 | 17 / 0.0471 / 3.0 |

**Systematic differences, and why:**
- **Hull counts.** We produce fewer hulls on 11 of 16 meshes, equal
  counts on 2 and more on 3 (median 43 vs 54). Upstream's distance
  estimator (`face_hausdorff_distance`: each sample's distance to the
  triangles of its 10 nearest samples on the other surface) overestimates
  distances whenever the nearest samples miss the nearest triangle.
  Upstream therefore cuts more and refuses more merges. The port uses the
  same estimator in faithful mode. Before the merge the counts nearly
  match (SnowFlake 58 vs 60). The remaining gap comes from random sample
  positions and the MCTS random rollouts: we use ChaCha8 and upstream uses
  mt19937, so the two can never be bit-identical.
- **Concavity.** Our scored h is slightly higher (median 0.0685 vs 0.0642).
  That follows from merging further, because both stop at the same
  estimated threshold (0.05). On messy_scan and brick_wall_window our h
  is lower. There, upstream's manifold preprocessing (`preprocess_mode
  auto`, voxel remesh) changes the input. We skip that step because our
  fragments are already closed.
- **Kettle and Bottle.** These are thickened open shells about 2% thick.
  Both methods are far above the threshold, and we are higher (Kettle
  0.62, with 95 vs 74 hulls). Upstream remeshes the thin parts away. On a
  2%-thick shell, a cut that leaves part of the handle in a hull spanning
  the spout is cheap under the sampled estimator. The exact-Hb variant
  does not fix this: the shell is narrower than the sample spacing.
- **Volume ratio** (Σ hull volume / V): equal within ±0.04 except on the
  thin shells.
- **The 64-vertex cap** (second table) changes h by less than 0.003 on
  every mesh. The capped hulls are inner approximations, so they cannot
  add outside volume.
- **Exact Hb + collision-aware merge** (the estimator the pipeline uses)
  gives lower h on most meshes. It is also several times faster than the
  10-NN estimator, because the faithful estimator samples at least 1000
  points per hull for every merge candidate. That cost is why our
  faithful mode is slower than upstream on the tiny shapes.
- **Floor.** Upstream h does not reach 0 on L_block and U_block (0.016–0.018)
  because its float32 boolean and remesh leave seams. We are exact there
  (h ≤ 0.015, and 0 for U_block).

### 2. Pipeline decomposition (cell-complex head start) vs upstream CoACD

`tools/harness/coacd_diff.py`, upstream run on the same fragment meshes
(threshold 0.03, seed 1), hulls ≤ 64 vertices on our side (upstream hulls
uncapped). "Before" = the previous built-in decomposition (commit
810642c). Symmetric Hausdorff between the union of hulls and the fragment
surface / bounding-box diagonal ("Haus."), outside volume and coverage as
fractions of the fragment volume; medians unless noted.

**Equal hull budget** (`max_convex_hull` = our hull count per fragment):

| Asset | Method | Hulls | Coverage | Outside med / p95 | Haus. median | Haus. p95 |
|---|---|---|---|---|---|---|
| rc_column L1 (12) | before | 95 | 0.869 | 0.060 / 0.089 | 0.0890 | 0.1290 |
| | **ours** | 96 | 0.933 | 0.039 / 0.060 | **0.0619** | **0.0895** |
| | CoACD | 96 | 1.000 | 0.208 / 0.318 | 0.0999 | 0.1149 |
| rc_column L2 (40) | before | 299 | 0.945 | 0.012 / 0.039 | 0.1221 | 0.1556 |
| | **ours** | 317 | 0.970 | 0.002 / 0.010 | **0.1162** | 0.1480 |
| | CoACD | 317 | 1.000 | 0.154 / 0.225 | 0.1200 | 0.1462 |
| ceramic_bowl L1 (5) | before | 40 | 0.825 | 0.512 / 0.757 | 0.0290 | 0.0463 |
| | **ours** | 40 | 0.831 | 0.542 / 0.790 | **0.0276** | **0.0326** |
| | CoACD | 40 | 1.000 | 0.753 / 1.135 | 0.0362 | 0.0418 |
| brick_wall_window L1 (12) | before | 87 | 0.916 | 0.022 / 0.045 | 0.0828 | 0.2528 |
| | **ours** | 88 | 0.964 | 0.015 / 0.051 | **0.0608** | **0.0737** |
| | CoACD | 88 | 1.000 | 0.140 / 0.298 | 0.0611 | 0.0756 |
| messy_scan L1 (12) | before | 94 | 0.839 | 0.056 / 0.094 | 0.1139 | 0.1302 |
| | **ours** | 96 | 0.944 | 0.050 / 0.064 | **0.0814** | **0.1196** |
| | CoACD | 96 | 1.000 | 0.283 / 0.369 | 0.0991 | 0.1268 |
| stone_arch L1 (12, held out) | before | 75 | 0.829 | 0.039 / 0.113 | 0.1020 | 0.1764 |
| | **ours** | 83 | 0.953 | 0.023 / 0.111 | **0.0582** | **0.1274** |
| | CoACD | 83 | 1.000 | 0.090 / 0.434 | 0.1134 | 0.1725 |

Our coverage is below 1 by construction (non-overlap with neighbours plus a
0.5 mm margin; on the thin bowl shell the margin alone costs 17%: coverage
without the non-overlap step is 0.832). The collision-aware deviation
(`cw_concavity`: surfaces of hulls − fragment and fragment − hulls only)
is lower than CoACD's everywhere except the stone_arch p95 (0.0913 vs
0.0628), where coverage holes left by the separating planes dominate.

**Unlimited budget**, threshold 0.03 in CoACD units for both (ours:
`prefracture hulls --coacd-threshold 0.03 --max-hulls 0`). Criterion: our
hull count ≤ 1.1× CoACD's.

| Asset | Hulls ours / CoACD | ratio | Coverage ours | Outside med ours / CoACD | CW deviation med ours / CoACD | CW max (CoACD units) ours / CoACD |
|---|---|---|---|---|---|---|
| rc_column L1 | 680 / 1731 | 0.39 | 0.968 | 0.000 / 0.020 | 0.0151 / 0.0166 | 0.054 / 0.069 |
| rc_column L2 | 428 / 3948 | 0.11 | 0.968 | 0.000 / 0.009 | 0.0119 / 0.0155 | 0.047 / 0.058 |
| ceramic_bowl L1 | 122 / 141 | 0.87 | 0.784 | 0.119 / 0.230 | 0.0096 / 0.0123 | 0.028 / 0.041 |
| brick_wall_window L1 | 193 / 219 | 0.88 | 0.981 | 0.000 / 0.102 | 0.0128 / 0.0142 | 0.040 / 0.041 |
| messy_scan L1 | 839 / 1092 | 0.77 | 0.962 | 0.018 / 0.124 | 0.0139 / 0.0143 | 0.053 / 0.045 |
| stone_arch L1 | 702 / 681 | 1.03 | 0.969 | 0.001 / 0.041 | 0.0142 / 0.0159 | 0.063 / 0.049 |

Neither method keeps every fragment's collision-aware deviation below the
threshold (0.03 in CoACD units) at the max; medians are well below for
both. (The symmetric Hausdorff of the union is ~0.11–0.13 for both here: it
is dominated by crevices between the many hulls inside the fragment.)

### Fast path vs decomposition (fragments per level)

Convex = hull volume within 1e-6 of the exact volume; leaf rows count
cells (convex atoms vs cells cut by the tree search).

| Asset | atoms (cells) | L3 | L2 | L1 | L0 |
|---|---|---|---|---|---|
| rc_column | 764 / 0 | 764 / 0 | 0 / 96 | 0 / 12 | 1 / 0 |
| ceramic_bowl | 0 / 40 | 0 / 40 | 0 / 5 | 0 / 5 | 0 / 1 |
| brick_wall_window | 1652 / 8 | 1652 / 8 | 1338 / 8 | 1 / 11 | 0 / 1 |
| messy_scan | 57 / 144 | 57 / 144 | 2 / 39 | 0 / 12 | 0 / 1 |
| stone_arch | 922 / 100 | 922 / 100 | 23 / 181 | 1 / 11 | 0 / 1 |
| building_v0 (building_lite) | 6892 / 0 | 6892 / 0 | 36 / 1559 | 26 / 976 | 132 / 0 |
| two_storey_building | 60888 / 32 | 60888 / 32 | 4132 / 7165 | 6 / 570 | 44 / 4 |

(fast path / decomposed)

PERF_TABLE


## Reproducing

```
prefracture bake --input benchmarks/assets/rc_column.glb --config benchmarks/configs/fast.toml --out out/
python tools/harness/coacd_diff.py --asset out/rc_column.asset.json --level 1            # equal budget
prefracture hulls --asset out/rc_column.asset.json --config benchmarks/configs/fast.toml \
    --coacd-threshold 0.03 --max-hulls 0 --levels 1 --out out/rc_column.nb.asset.json
python tools/harness/coacd_diff.py --asset out/rc_column.nb.asset.json --level 1 --no-budget
```

`FRAC_PROFILE=1` prints per-level timings and cost-component counters.
