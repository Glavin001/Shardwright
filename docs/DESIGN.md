# Shardwright `prefracture` — design notes and spec deviations

This document records how the implementation maps onto the
*Pre-Fracture Library — Engineering Spec (Rust) v1.0*, and every place where
it deliberately differs. The validation results live in
[`VALIDATION.md`](VALIDATION.md).

## Crate map (spec §14)

| Crate | Role |
|---|---|
| `frac-core` | Data model (`Asset`, cells, interfaces, fragments, bonds, hulls), IDs, errors, `bake.toml` settings, determinism utilities (BLAKE3 stable hashes, ChaCha8 streams keyed by stable IDs) |
| `frac-geom` | Exact kernel: Shewchuk expansions + error-bounded float filter, `orient3d`/`insphere` (+SoS), polygon and polyhedral integrals, BVH, generalized winding number, Quickhull, convex polytopes |
| `frac-cells` | Delaunay (Bowyer–Watson, exact SoS), Voronoi and box complexes, **exact symbolic clipping** of the solid against the cell complex, recipes and seeding, sliver/island post-processing |
| `frac-material` | Material library (`materials.toml`) parsing, validation, fracture-energy anisotropy, Weibull statistics |
| `frac-ingest` | Import repair, weld, solidify (fast winding number + marching tetrahedra), component split, connections, anchors |
| `frac-fem` | Linear tetrahedral FEM (BCC lattice mesher, isotropic and transversely isotropic elasticity) |
| `frac-modes` | Fracture modes (Sellán et al. 2023) with material weights, Level-1 segmentation |
| `frac-hierarchy` | L0–L3 hierarchy, agglomeration fallback (method B), contiguous child ordering |
| `frac-bonds` | Exact bond properties: area, centroid, normal, planarity, second moments, frame, extent, composition, rebar, loops, Weibull strength, parent/child |
| `frac-collision` | CoACD-style convex decomposition (built in), exact hulls, non-overlap enforcement |
| `frac-render` | Shared displaced interior surfaces, chipping, UVs, LODs (meshopt), rebar stubs |
| `frac-io` | glTF/OBJ/PLY/STL import; glTF export (`MSFT_lod`, `EXT_meshopt_compression`); FlatBuffers `.fracphys`; JSON debug |
| `frac-patterns` | Runtime pattern library `.fracpat` (M5) |
| `frac-validate` | Hard gates, scorecard metrics, reference bond-network solver |
| `frac-pipeline` | Stage orchestration, assembly, Level-1, export, report, oracle export |
| `frac-cli` | `prefracture bake / validate / inspect / diff / patterns / gen-bench` |

## Geometry kernel: exact, symbolic, no Manifold FFI

The spec proposes clipping with Manifold. We instead clip with an exact
symbolic kernel written for this problem:

* Every plane of the cell complex is an **exact function of input points**:
  Voronoi bisectors `f_ij(x) = |x−s_i|² − |x−s_j|²`, axis-aligned box planes.
  Because the bisectors are exact, the planes through a Voronoi vertex are
  exactly concurrent, so the complex has no "nearly coincident" vertices.
* Degeneracies are resolved by **Simulation of Simplicity** consistent with
  the Delaunay lifted-weight perturbation (rank = point index).
* All predicates are evaluated on **symbolic vertex keys**
  (`Orig`, edge∩plane, triangle∩line, complex vertex) with an error-bounded
  float filter and an expansion-arithmetic fallback. Vertex coordinates are a
  pure function of the key, so both sides of an interface get bit-identical
  vertices (render "no gaps" holds by construction).
* Inside/outside state is propagated by parity BFS over the complex
  1-skeleton; face regions are triangulated with a constrained Delaunay
  triangulation (spade), with T-junction-aware facet subdivision for box
  complexes (masonry).
* Coincident symbolic vertices produced by different keys (exactly the same
  point, e.g. masonry corners) are welded with a 1e-11·diag tolerance after
  clipping.

Differential tests against Voro++ (`tools/oracles/voro_oracle.cc`,
`crates/frac-cells/tests/voro_diff.rs`) cover the Voronoi complex (§13.2), and
property tests cover the clipping (volume conservation to 1e-12, watertight
manifold output, quantized degenerate inputs).

## Convex decomposition: built-in CoACD-style, not the FFI

The spec integrates upstream CoACD via FFI. The container has no CoACD build,
and per-fragment inputs are already small polyhedral cells, so we implemented
the CoACD algorithm (concavity metric = max(boundary Hausdorff, interior
distance); plane-search splits; greedy merge of pieces with cached pair costs)
in `frac-collision`. The non-overlap gate is enforced exactly: hulls of
neighbouring fragments are clipped by the shared bond plane (planar bonds) or a
separating plane, then shrunk by the margin.

## Tetrahedral meshing and fracture modes

* fTetWild is replaced by a BCC-lattice mesher clipped to the solid
  (`frac-fem`), which is robust for the cell-exploded solids we feed it.
* The fracture-modes solve uses a **cell-polynomial reduction** (displacement
  affine per analysis cell) for large problems, solved with a hybrid
  ADMM + Clarabel scheme. `Discretization::Full` keeps the unreduced problem.
  This is ~100× faster with the same Level-1 segmentation on the benchmarks.
* Material weights `w_g = sqrt(G_f / G_ref)` scale the cut cost of each
  analysis-cell interface; the geometric (`w_g = 1`) and fallback
  agglomeration paths remain available for the §13.4 ablation.

## Reference bond-network solver (§13.3)

`frac-validate::network` implements three stiffness models behind the
`BondNetworkSolver` trait:

* `Spec`: the §6 reference formulas verbatim.
* `Calibrated`: Eliáš (2016) `α = (1−4ν)/(1+ν)`, `E0 = E(4+α)/(2+3α)`.
* `Tensorial` (default for validation): deviatoric springs
  `k_n = k_s = 2μA/h` plus a volumetric pressure `λ tr ε̃`, with `ε̃` a
  least-squares strain fitted from neighbour displacements. Three details make
  it reproduce uniform stress states exactly at the bond level:
  1. **Kinematic centers.** Rigid-body networks are exact under uniform
     stress only if every connector `p_b − p_a` is parallel to the bond
     normal. Mass centroids are not (median misalignment 8% on the column);
     Voronoi sites are. `kinematic_centers` recovers such points for any
     complex by least squares (tangential connector residual + weak normal
     length anchor + centroid regularizer); on rc_column L3 the residual
     misalignment drops to 0.5%. Levels where the fit does not help
     (clustered L1/L2) keep the centroids.
  2. **Finite-volume pressure.** In statics the volumetric pressure is
     transmitted through the bonds as face forces
     `½(λ_a tr ε̃_a + λ_b tr ε̃_b) A n` at the bond centroid. Then any affine
     motion is an exact equilibrium, including cells cut by the free surface
     (the symmetric energy form is consistent kinematically but not in
     equilibrium on irregular cells: 2.6% → 0.3% interior patch error). The
     operator is non-symmetric (dense LU); modal analysis keeps the symmetric
     energy form.
  3. **Loads like the FEM.** End loads are applied at each fragment's
     loaded-patch centroid with the moment about the COM, and torque loads
     include each patch's own couple `κ (tr(I_f) I − I_f) d`.

  Patch tests (`crates/frac-pipeline/tests/network_patch.rs` and the
  `network_diag` example) assert per-bond traction errors under uniform
  stress below 0.5% (interior and next to free surfaces). Under linearly
  varying stress (pure bending, exact elasticity solution) the error is first
  order in the cell size, about 5% of σ_max at the column's L3 resolution.
  Its source is the offset between each face centroid and the connector
  foot.

Bond tractions are reported both as the spec's `F_b/A_b` and as Love–Weber
recovered stress `½(σ_a + σ_b)·n`.

## Schema deviations (`schemas/frac.fbs`)

* `Bond.extent_half` is a `Vec2` (half extents along the bond frame).
* `Hull.fragment` back-reference added.
* `Bond.interface_count` added (bonds may aggregate several fine interfaces).
* `Asset.bond_children` added (flattened parent→child bond map for the
  bond-hierarchy gate and runtime).

## Determinism

All randomness comes from ChaCha8 streams seeded by `stable_hash(seed, stable
IDs)`; all parallel stages reduce in a fixed order; output ordering is by
Morton code with stable tie-breaks. `prefracture bake --check-determinism`
bakes twice in-process and compares output bytes.
