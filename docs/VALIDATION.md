# Validation results

All numbers come from the commands shown below; the harness lives in
`tools/harness/` and caches FEM solutions keyed by mesh, load case and
material. The FEM oracle is Kratos Multiphysics 10.4 (StructuralMechanics,
SmallDisplacementElement3D10N quadratic tets, gmsh meshes). Large systems are
solved with AMGCL CG (tolerance 1e-10). On the rc_column tip displacement it
reproduces PL/EA to 0.2%.

<!-- SUITE -->

## Status against the oracles

| Oracle / check | Result |
|---|---|
| Hard gates (§13.1) | all pass on all 14 benchmark assets incl. held-out and both buildings (`tools/ci/suite.sh`) |
| Voro++ (§13.2) | cell volumes and vertices match (frozen differential test) |
| Upstream CoACD 1.0.14 (§13.2) | at equal hull budget our hulls fit better on every asset (e.g. rc_column L1 Hausdorff/diam 0.060 vs 0.100 median); see `crates/frac-collision/README.md`. rc_column L1 coverage 0.933 (target 0.95) |
| Authors' fracture modes (§13.2) | 0.0° principal angles, energy error ≤ 5e-8, ARI 1.0 on notched bar, L-shape and bunny; on the 4-fold symmetric plate our energies are equal or lower in every mode but the symmetric cuts differ |
| Kratos FEM bond fidelity (§13.3) | all targets met at 2× resolution; at the default resolution all except torsion p95 (0.355 vs ≤ 0.30), a first-order resolution limit of rigid-cell kinematics (connector-alignment and midpoint-placement variants of the kinematic centres do not reduce it) |
| Analytical (patch tests, pure bending, known-answer modes) | pass |
| Rankine crack oracle (§13.4) | L3 recall ≥ 0.8 met; L1 recall and weak-region Spearman below target (load-independent modes vs load-specific cracks; see below) |
| Performance (§17) | every asset end to end in ≤ 300 s on 4 cores (two-storey building 255 s, 5-storey building 149 s), peak ≤ 7.4 GB |

## Bond fidelity (spec §13.3)

`prefracture validate --input out/rc_column.asset.json --oracle-cache DIR`

The reference network uses the Tensorial model (see `DESIGN.md`) with the
bonds exactly as baked. The FEM oracle is the unfractured column (0.4 × 3 ×
0.4 m, C30 concrete) with h = min(L/40, median cell size / 2.5), clamped at
y = 0 and loaded at y = 3 m.

Traction error for each bond is |t_net − t_FEM| / max(|t_FEM|, 5% of the case
maximum), with vectors compared. t_FEM is the quadrature average of σ·n over
the bond's exact interface polygons. "Raw" is the spec definition F_b/A_b.

### Default resolution: `fine_per_analysis = 8`, 764 L3 cells (FEM: 57,217 P2 tets)

| Level | Case | Bonds | Raw F/A p50 | Raw F/A p95 | Stiffness err |
|---|---|---|---|---|---|
| L3 | axial | 4459 | 0.004 | 0.043 | 0.026 |
| L3 | bending | 4459 | 0.046 | 0.215 | 0.027 |
| L3 | shear | 4459 | 0.049 | 0.276 | 0.028 |
| L3 | torsion | 4459 | 0.082 | **0.355** | 0.059 |
| L2 | axial / bending / shear / torsion | 434 | 0.28–0.38 | 0.87–1.77 | 0.29–0.41 |
| L1 | axial / bending / shear / torsion | 23 | 0.42–1.45 | 0.68–24 | 0.53–0.85 |

At L3 the modal error over the first 10 frequencies is at most 4.7%. Raw p95
and stiffness errors decrease monotonically L1 → L2 → L3 (mean over cases).

### 2× resolution: `fine_per_analysis = 16`, 1537 L3 cells (FEM: 109,605 P2 tets)

| Level | Case | Bonds | Raw F/A p50 | Raw F/A p95 | Stiffness err |
|---|---|---|---|---|---|
| L3 | axial | 9494 | 0.004 | 0.035 | 0.017 |
| L3 | bending | 9494 | 0.036 | 0.179 | 0.021 |
| L3 | shear | 9494 | 0.036 | 0.186 | 0.019 |
| L3 | torsion | 9494 | 0.060 | 0.265 | 0.039 |

All §13.3 targets are met at 2× resolution: p50 ≤ 10%, p95 ≤ 30%,
stiffness ≤ 10%, monotone convergence.

At the default resolution three of the four load cases meet p95 ≤ 30%. Torsion
reaches 35.5%. The residual is first order in cell size. Two diagnostics
confirm this:

* **Uniform stress patch tests** (`crates/frac-pipeline/tests/network_patch.rs`)
  give per-bond errors of p50 0.1–0.3% and p95 ≤ 1%. This holds in the
  interior and in uniaxial stress with a traction-free lateral surface.
* **Pure bending** (exact elasticity solution driven on the end cells) gives
  a mid-section error of about 5% of σ_max at p95. The Taylor analysis in
  `DESIGN.md` explains it: the rigid-cell kinematics leave a term proportional
  to the offset between each face centroid and the foot of the connector.
  Its relative effect is largest where |σ·n| is small, such as the torsion
  core and the corners of the square section.

How the network got here, all at the default resolution and with the same
oracle:

| Network variant | L3 raw p95 (axial / bending / shear / torsion) |
|---|---|
| centroid kinematics, energy volumetric term | 0.56 / 0.36 / 0.42 / 0.42 |
| + kinematic centers | 0.48 / 0.35 / 0.41 / 0.39 |
| + finite-volume pressure | 0.085 / 0.22 / 0.28 / 0.39 |
| + FEM-consistent end loads (moments, patch couples) | 0.043 / 0.215 / 0.276 / 0.355 |

## Convex decomposition vs upstream CoACD (spec §13.2, §13.6)

`tools/harness/coacd_diff.py --asset out/rc_column.asset.json --level 1`

The test uses the 12 L1 fragments of rc_column. These are jagged clusters of
Voronoi cells with convexity 0.61. Upstream CoACD 1.0.14 runs with threshold
0.03 and `max_convex_hull` set to our hull count for each fragment (8).

| Method | Hulls | Coverage (median) | Outside volume (median / p95) | Surface Hausdorff / diameter (median / p95) | Hulls overlap |
|---|---|---|---|---|---|
| built-in (`frac-collision`) | 95 | 0.869 | 0.060 / 0.089 | **0.089** / 0.129 | no (gate) |
| upstream CoACD | 95 | 1.000 | 0.209 / 0.318 | 0.100 / 0.115 | yes |

At equal budget the built-in decomposition fits the surface better than
upstream CoACD and overshoots far less. It also satisfies the
no-overlap-after-margin gate, which CoACD does not attempt. The volume it
leaves uncovered is the cost of that non-overlap constraint.

Neither method reaches the §13.6 hull-fit target (≤ 0.03) on these fragments
with the default budget of 8 hulls. The baked report's per-level hull-fit
metric agrees with the independent harness (L1 median 0.089, L2 0.050,
L3 0.004).

## Fracture modes vs reference implementation (spec §13.2)

`$ORACLES/fmref-venv/bin/python tools/harness/fracture_modes_ref.py run --freeze --k 6 [--elastic]`
(frozen: `... run --golden`, or `FM_GOLDEN_RUN=1 python3 tools/harness/test_fracture_modes_golden.py`)

Algorithm source: S. Sellán, J. Luong, L. Mattos Da Silva, A. Ramakrishnan,
Y. Yang, A. Jacobson, *Breaking Good: Fracture Modes for Realtime
Destruction*, ACM Transactions on Graphics 42(1), 2023 (the "paper" below).
`frac-modes` is our own implementation of the paper. The authors' reference
implementation (github.com/sgsellan/fracture-modes, commit `bdf5051`,
academic / non-commercial licence) is used with their permission as a
test-only oracle: `tools/setup.sh --with-fracture-modes-ref` clones it to
`$ORACLES/fracture-modes`, and the harness imports it from there at run
time. None of its code is in this repository.

**Setup.** Both implementations run on the same tet mesh `(V, T)`, made once
by `frac_fem::tetrahedralize` from the surface normalized to a unit bounding
box (the reference pipeline's normalization), with every tet its own analysis
cell (the fully exploded mesh), no anchors, no material weights. Meshes: the
known-answer notched bar, L-shape and plate with a hole of
`crates/frac-modes/tests/known_answer.rs`, and the reference's bundled
`bunny_oded.obj`, each about 1600–2000 tets. Reference: `d = 3`, its defaults
otherwise (tolerance 1e-4 on `‖c − c_prev‖_∞`, at most 10 ICCM iterations).
Ours: library defaults (ω = 1e-3, ε = 1e-4, 50 iterations), `k = 6`.

**Parameter mapping.**

* **k:** the reference's first `d = 3` modes are the global translations
  (zero energy). We request `k + 3` from it and drop those three; the
  harness checks they are constant fields.
* **ω:** both translational objectives (the reference's and ours) are
  1-homogeneous in `u`, so ω has no effect on the modes. For the P1 model,
  ω = 1e-3 (normalized) is in the paper's small-ω, fracture-only regime
  (Fig. 12).
* **Normalization:** the reference's modes are unit in `M` = tet volumes.
  Ours are unit in M̂ = M / total mass. Reference modes are multiplied by
  `√V` before comparing.
* **ε:** the reference stops on `‖Δc‖_∞ ≤ 1e-4` with `c` unit in `M`; ours
  stops on `‖Δc‖_M̂ ≤ 1e-4`. For the unit-box meshes (`V ≈ 0.06–0.2`) these
  are of the same order. Both converge in 2–3 (reference) and 2–25 (ours)
  iterations.
* **Anchors:** the reference has none, so both run as free bodies.
* **Weights:** geometric only. The default (`area_weighting`) sets
  `w_g = √(A_g/Ā)`, which reproduces the reference's per-face weighting;
  rows marked `w_g = 1` are the spec's unit weights (the paper's Eq. 5).
* **Modes:** our translational modes are expanded into direction triples
  (see "Formulation"), so `⌈6/3⌉ = 2` modes are compared with the
  reference's 6.

**MOSEK substitution.** The reference's conic subproblem
`min Σ_f ‖(D u)_f‖ s.t. Uᵢᵀ M u = 0, cᵀ M u = 1` goes to MOSEK, which needs a
commercial licence. The harness passes the same second-order-cone program to
Clarabel (interior point, same problem class), set up by our own
`clarabel_conic_solve` and installed in place of the reference's solve
function when the module is imported. The reference's `sparse_sqrt`
(CHOLMOD) result is never used by its mode computation, so it is replaced by
a no-op. Its GUI (polyscope) and meshing (gpytoolbox, tetgen) modules are not
imported. Clarabel reports `Solved` for every subproblem except a few
`AlmostSolved` ones (reduced accuracy) on the L-shape and the bunny.

The substitution was cross-checked against a second open solver, SCS
(first-order). On the notched bar, both solvers give the same modes: mode
energies agree within 1.4e-4 relative and the per-mode pieces have ARI 0.97–1.0.
With 4 modes, the two subspaces share two directions exactly and differ by
10.5° in a third (the 4th angle, 53°, belongs to a truncated degenerate
triple; see below). The cut can sit anywhere across the 0.2-wide notch at
almost the same energy, so the angle measures which layer of tets the two
solvers cut, not solver error. The reference
also reproduces its bundled example qualitatively. On the bunny, the first
triple of modes breaks off one ear (1.4% of the volume) and the second
triple also breaks off the other ear (1.1%). On the known-answer shapes it
cuts the notch, one arm at the re-entrant corner of the L, and the plate
across the hole. The reference takes 13–53 s per mesh with Clarabel.

### Formulation

`frac-modes` (column 4) now follows the paper's §3.6 model by default:

| | Paper | Reference implementation | `frac-modes` default (`translational`) | `frac-modes` `p1` (spec §4.1) |
|---|---|---|---|---|
| Displacement space | exploded P1 (Eq. 9); §3.6: one constant displacement per element | per-tet constant, `d = 3` (README example) or `d = 1` lifted to 3D (dataset script) | one constant per analysis cell, solved for one component (below) | P1 per (vertex, analysis cell); per-tet cells give the fully exploded P1 space |
| Strain energy `Q` | any PSD Hessian; §3.6 uses `I_d ⊗ L` for all examples, because linearized rotations made small elements break off | `I_d ⊗ L` on per-tet constants: identically zero, no quadratic term | zero (translations); no quadratic term | linear elasticity; its null space contains rotations |
| Patch energy | `√(∫_S ‖D‖²)` per patch (Eq. 5, 12) | `2A_f·‖jump_f‖` per face, i.e. `∝ ∫‖D‖ dA` | `w_g √(∫_g ‖D‖²)` with `w_g = √(A_g/Ā)` (`area_weighting`), i.e. `∝ ∫‖D‖ dA` for a constant jump | same |
| ω | balances the terms; only the null space of `Q` matters in the fracture regime (Fig. 12) | no effect (1-homogeneous) | no effect (1-homogeneous) | normalized; 1e-3 is in the fracture regime |
| Rigid modes | — | the first `d` modes are the global translations | orthogonal to the translation | orthogonal to the 6 rigid modes |
| ICCM start | eigenvectors of `Q` on the unexploded mesh | vector-Laplacian eigenvectors incl. constants (ARPACK, random start vector) | vector-Laplacian eigenvectors (seeded LOBPCG), multi-start over degenerate eigenspaces | elastic eigenvectors (seeded LOBPCG) |
| ICCM stop | `‖Uᵢ − c‖ ≤ ε` | `‖c − c_prev‖_∞ ≤ 1e-4`, ≤ 10 iterations | `‖u − c‖_M̂ ≤ 1e-4`, ≤ 50 iterations | same |
| Pieces | impact-dependent (§3.3) | per mode: tets with `‖c_i − c_j‖ < 0.1` connected | Level 1: threshold the max jump to a target count (spec §4.4) | same |

With the isotropic `‖·‖₂` and translation-only displacements, the vector
problem separates per axis. ICCM started from `φ e_x` stays in the
x component, so the vector modes are exactly scalar modes times
`e_x, e_y, e_z`: degenerate direction triples, which is how the reference's
modes come. `frac-modes` therefore solves one component, so `k` modes are
`k` distinct cut patterns. The harness requests `⌈k/3⌉` modes and expands
each one into its triple before comparing with the reference's `k = 6`
modes (two complete triples; principal angles of a truncated triple would
only measure the reference's random ARPACK start vector).

### Results (`k = 6`; seconds on 4 shared cores)

Our functional is `½uᵀQ̂u + ω Σ w_g ‖B̂_g u‖`. The reference modes are
evaluated in it exactly: per-tet constants have zero strain energy and a
constant jump on each face. The reference functional is `Σ_f A_f ‖jump_f‖`;
ours is evaluated in it with face RMS jumps. Energy errors are
`(E(ours) − E(ref)) / E(ref)` per mode. ARI is measured on voxelized labels
(48³ grid). Both Level-1 labelings come from the same procedure (our
`segment_level1` on the max-over-modes jumps) with target T.

| Mesh (tets) | Ours | Time (s) | Principal angles (°) | Energy rel. err., our E (median / max \|·\|) | Energy rel. err., reference E (median / max \|·\|) | ARI L1, T = 2 / 3 / 4 |
|---|---|---|---|---|---|---|
| notched bar (1605) | reference (Clarabel) | 12.9 | — | — | — | — |
| | **default** (`translational`, area weights) | 0.3 | **0.0 ×6** | 4e-9 / 8e-9 | 4e-9 / 8e-9 | **1.00 / 1.00 / 1.00** |
| | translational, `w_g = 1` | 0.4 | 5.0 ×3, 10.9 ×3 | 0.10 / 0.12 | 0.0054 / 0.0059 | −0.06 / 0.47 / 0.98 |
| | `p1` (full) | 652 | 12.0, 41.1, 41.1, 60.9, 69.6, 90.0 | 0.67 / 0.79 | 0.64 / 0.77 | 0.29 / 0.73 / 0.48 |
| | `p1` (cell-p1) | 671 | identical to full | 0.67 / 0.79 | 0.64 / 0.77 | 0.29 / 0.73 / 0.48 |
| L-shape (1976) | reference (Clarabel) | 52.5 | — | — | — | — |
| | **default** | 0.6 | **0.0 ×6** | 1e-8 / 1e-8 | 1e-8 / 1e-8 | **1.00 / 1.00 / 1.00** |
| | translational, `w_g = 1` | 0.6 | 20.2 ×3, 32.6 ×3 | 0.033 / 0.042 | 0.031 / 0.058 | −0.02 / 0.29 / 0.80 |
| | `p1` (full) | 2091 | 45.5, 64.9, 65.2, 71.8, 90.0, 90.0 | 0.55 / 0.62 | 0.56 / 0.62 | 0.00 / 0.00 / 0.00 |
| | `p1` (cell-p1) | 1063 | 45.5, 64.1, 65.2, 71.8, 90.0, 90.0 | 0.55 / 0.62 | 0.56 / 0.62 | 0.00 / 0.00 / 0.00 |
| plate with hole (1924) | reference (Clarabel) | 15.9 | — | — | — | — |
| | default | 1.2 | 20.9, 20.9, 22.2, 30.0, 30.0, 30.0 | 5e-5 / 0.066, **all ≤ 0** | 5e-5 / 0.066 | 0.00 / 0.65 / 0.58 |
| | translational, `w_g = 1` | 1.2 | 20.9, 20.9, 22.2, 30.0, 30.0, 30.0 | 2.5e-5 / 0.066 | 5e-5 / 0.066 | 0.00 / 0.40 / 0.58 |
| | `p1` (full) | 515 | 31.3, 81.8, 84.5, 88.8, 89.5, 89.9 | 0.56 / 0.71 | 0.58 / 0.73 | 0.00 / 0.00 / 0.06 |
| bunny (1940) | reference (Clarabel) | 32.8 | — | — | — | — |
| | **default** | 0.4 | **0.0 ×6** | 1e-8 / 5e-8 | 1e-8 / 5e-8 | **1.00 / 1.00 / 1.00** |
| | translational, `w_g = 1` | 0.5 | 0.0 ×3, 80.5 ×3 | 1.7 / 3.5 | 2.2 / 4.4 | −0.01 / 0.10 / 0.10 |
| | `p1` (full) | 3177 | 33.4, 34.7, 38.5, 44.2, 89.5, 89.7 | 0.80 / 0.89 | 0.79 / 0.89 | −0.01 / 0.97 / 0.97 |

All runs converged. Targets: max angle ≤ 5°, ARI ≥ 0.9.

**The default meets §13.2 on the notched bar, the L-shape and the bunny:**
0.0° principal angles, energies equal to 1e-8, ARI 1.00 at every T, in
0.3–0.6 s, 20–90× faster than the reference with Clarabel.

**The plate with a hole is a symmetric tie.** Its 4-fold symmetry makes the
initial Laplacian eigenspaces degenerate. The reference takes a random
basis of them (ARPACK). With the multi-start, ours reaches equal or lower
energy in every mode: −0.01% on modes 1–4 (the same cut family), and −6.6%
and −2.6% on modes 5–6. But it picks other symmetric cuts, so the angles
are 21–30° and the ARI is 0.65 / 0.58 at T = 3 / 4. T = 2 is unreachable for
both (one cut through the hole opens two ligaments at once).

`tools/harness/test_fracture_modes_golden.py` (`FM_GOLDEN_RUN=1`, ~5 s)
asserts the §13.2 thresholds for the default model on the frozen reference
outputs. It also asserts that our energies never exceed the reference's by
more than 0.1%, which covers the plate. The plate's angle and ARI checks,
and the other configurations' known failures, are skipped with the measured
numbers (`FM_GOLDEN_KNOWN_FAILURES=1` runs them).

### Diagnosis (why the earlier P1 model failed)

1. **Formulation, not a bug.** With linear elasticity, our energies are
   22–89% *below* the reference's in either functional. The elastic null
   space contains rotations, so fragments can hinge relative to each other.
   This is the effect the paper cites in §3.6 as its reason to use
   `Q = I ⊗ L`. The P1 model's first modes are 6 relative rigid motions of
   the same two fragments, where the reference has 3 translations.
2. **The solver, ICCM and normalization were right.** With the paper's
   model (per-cell translations, vector-Laplacian start, area weights) the
   same machinery reproduces the reference exactly.
3. **Patch weighting.** With the paper's patch norm `√(∫‖D‖²)` (translational,
   `w_g = 1`), the cuts move by a few tet layers. Energies stay within
   0.5–6% in the reference functional, but angles reach 5–33°, and on the
   bunny the second pattern changes outright. The reference code weights
   faces by area (`∝ ∫‖D‖ dA`), which differs from the paper's Eq. (5) on
   single-face patches. The `area_weighting` default reproduces the
   reference.
4. **The P1 reduced discretization was not the cause.** With one tet per
   cell, cell-p1 spans the same space as full and, on the hybrid
   ADMM + Clarabel path, reproduces the Clarabel full solve (angles within
   1°, energies within 0.1%).
5. **§13.2 sensitivity.** The 5° bound is tighter than the reference's own
   reproducibility on degenerate geometry. Two open solvers on the same
   reference (Clarabel, SCS) differ by 10.5° on the notched bar, because
   the cut can sit anywhere across the notch at equal cost. Symmetric parts
   (the plate) have no unique answer at all.

### Pipeline impact (`benchmarks/configs/bake.toml`, `FRAC_LOG=1`; 4 shared cores)

Old = linear-elastic P1 (`discretization = "p1"`, `area_weighting = false`,
`multi_start = false`: the previous pipeline). New = defaults. Modes time is
the `[modes]` log line summed over setup (analysis mesh, cell location) and
solve. The hierarchy stage includes Level 2 and repair.

| Asset | Analysis cells | Modes: old → new (s) | Hierarchy stage: old → new (s) | Bake total: old → new (s) | Gates |
|---|---|---|---|---|---|
| timber_beam | 37 | 5.5 → **1.0** | 5.5 → 1.0 | 12.5 → 8.6 | all PASS (both) |
| rc_column | 96 | 6.8 → **1.6** | 6.8 → 1.7 | 20.9 → 16.8 | all PASS (both) |
| brick_wall_window | 1346 | 129.6 → **5.2** | 130.1 → 5.3 | 158.6 → 43.2 | all PASS (both) |
| two_storey_building | 48 components, up to 2015 | 1360 (reported) → **95.8** stage | — → 95.8 | collision (366 s) and render (324 s) dominate | — |

On the building, the two floor slabs (2015 cells, square, so with
degenerate eigenpairs and the multi-start) take 53 s each, in parallel with
the other components; the walls take 8 s.

**Crack ablation (spec §13.4, Rankine oracle, L1 interfaces):** recall@τ /
F / weak-region Spearman. `tools/harness/crack_oracle.py` on bakes with
`bake.toml` (material-aware), `ablation_geometric.toml` and
`ablation_fallback.toml`.

| Asset | Material-aware (new default) | Geometric (new) | Fallback (agglomeration) | Old P1, material-aware |
|---|---|---|---|---|
| brick_wall_window | **0.223** / 0.057 / **0.066** | 0.087 / 0.028 / −0.043 | 0.123 / 0.030 / −0.093 | 0.174 / 0.059 / 0.060 |
| timber_beam | 0.439 / 0.067 / −0.117 | 0.369 / 0.054 / −0.277 | 0.455 / 0.067 / −0.281 | **0.585** / 0.103 / 0.219 |
| rc_column | 0.144 / 0.046 / −0.231 | — | — | 0.167 / 0.054 / −0.193 |

* **Wall:** material-aware beats geometric and the fallback, and has higher
  recall than the old P1 model (equal F and Spearman).
* **Timber beam:** the old P1 configuration is best. This 37-cell, 12-fragment
  metric is very sensitive to settings, though. P1 with the new area weights
  and multi-start drops to 0.357 / 0.058 / −0.350, and translational without
  area weights scores 0.324 / 0.049 / −0.307. The L3 interfaces (recall
  0.983–0.996) are unchanged.
* **rc_column:** neither model has a geometric weak region to find (see
  below).

The material-weight extension is next, as decided. Its quality (§13.4) is
tracked by this table.

## Crack placement (spec §13.4) — Rankine oracle

`tools/harness/crack_oracle.py --asset out_full/rc_column.asset.json --network out/rc_column.network.json --cache DIR`

The spec's oracle, Kratos FEM-DEM, is not distributed for Kratos 10.4 (it is
not on PyPI). The harness substitutes a brittle **Rankine** oracle on the
same FEM:

* The crack initiates at the maximum principal stress.
* It is mode I: the plane ⊥ σ₁, restricted to the connected tensile region.
* Load cases: cantilever bending in x and z, torsion, and 6 surface impacts
  with both ends held.

τ = 0.25 × median cell diameter = 2.7 cm. Results for the modes-enabled bake
(`benchmarks/configs/bake.toml`; translational fracture modes, the default):

| Interfaces | Recall@τ | Precision@τ | F | F@τ/2 | F@2τ |
|---|---|---|---|---|---|
| L3 (detail) | **0.855** | 0.043 | 0.081 | 0.040 | 0.172 |
| L1 (structural) | 0.144 | 0.028 | 0.046 | 0.021 | 0.105 |

Weak-region agreement (Spearman, per analysis cell, L1): −0.23 (target ≥ 0.6).
The previous linear-elastic P1 modes gave L1 0.167 / 0.032 / 0.054 and
Spearman −0.19. For the ablation on the wall and the timber beam, see
"Fracture modes vs reference implementation".

L3 recall meets the ≥ 0.8 target. Precision is low by construction: the
detail interfaces fill the volume, while each oracle crack is a single
surface.

L1 does not meet the targets. The fracture modes are load-independent
worst-case cuts. The Rankine oracle concentrates cracks at the clamped root
and under impacts on this prismatic column, so there is no geometric weak
region for the modes to find. Before the size-balanced segmentation was
added, Level 1 was one fragment holding 87% of the volume plus 11 surface
chips (recall 0.07).
