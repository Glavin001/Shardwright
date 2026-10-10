# Validation results

All numbers come from the commands shown below; the harness lives in
`tools/harness/` and caches FEM solutions keyed by mesh, load case and
material. The FEM oracle is Kratos Multiphysics 10.4 (StructuralMechanics,
SmallDisplacementElement3D10N quadratic tets, gmsh meshes). Large systems are
solved with AMGCL CG (tolerance 1e-10). On the rc_column tip displacement it
reproduces PL/EA to 0.2%.

<!-- SUITE -->

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

`$ORACLES/fmref-venv/bin/python tools/harness/fracture_modes_ref.py run --fields --freeze --k 6`

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
cell (the fully exploded mesh), no anchors, `w_g = 1`. Meshes: the
known-answer notched bar, L-shape and plate with a hole of
`crates/frac-modes/tests/known_answer.rs`, and the reference's bundled
`bunny_oded.obj`, each about 1600–2000 tets. Reference: `d = 3`, its defaults
otherwise (tolerance 1e-4 on `‖c − c_prev‖_∞`, at most 10 ICCM iterations).
Ours: spec defaults (ω = 1e-3, ε = 1e-4, 50 iterations), `k = 6`.

**Parameter mapping.**

* **k:** the reference's first `d = 3` modes are the global translations
  (zero energy). We request `k + 3` from it and drop those three; the
  harness checks they are constant fields.
* **ω:** the reference's objective has no quadratic term and is
  1-homogeneous in `u`, so its ω (default 0.01) has no effect. It
  corresponds to the small-ω, fracture-only limit of the paper's energy
  (Fig. 12). Our normalized ω = 1e-3 is in that regime: our own modes are
  piecewise rigid.
* **Normalization:** the reference's modes are unit in `M` = tet volumes.
  Ours are unit in M̂ = M / total mass. Reference modes are multiplied by
  `√V` before comparing.
* **ε:** the reference stops on `‖Δc‖_∞ ≤ 1e-4` with `c` unit in `M`; ours
  stops on `‖Δc‖_M̂ ≤ 1e-4`. For the unit-box meshes (`V ≈ 0.06–0.2`) these
  are of the same order. Both converge in 2–3 (reference) and 2–25 (ours)
  iterations.
* **Anchors:** the reference has none, so both run as free bodies.
* **Weights:** `w_g = 1`. The `sqrt_area` rows set `w_g = √(A_g/Ā)`, a
  purely geometric weight that reproduces the reference's per-face
  weighting (see the next table).

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

### Formulation differences found

| | Paper | Reference implementation | `frac-modes` (spec §4) |
|---|---|---|---|
| Displacement space | exploded P1 (Eq. 9); §3.6: one constant displacement per element | per-tet constant, `d = 3` (README example) or `d = 1` lifted to 3D (dataset script) | P1 per (vertex, analysis cell); per-tet cells give the fully exploded P1 space |
| Strain energy `Q` | any PSD Hessian. §3.6 uses `I_d ⊗ L` (cotangent Laplacian, only translations in its null space) for all examples, because linearized rotations made small elements break off | `I_d ⊗ L` restricted to per-tet constants. This is identically zero, so the conic problem has no quadratic term | linear elasticity (spec §4.1). Its null space contains rotations |
| Patch energy | `√(∫_S ‖D‖²)` per patch (Eq. 5, 12) | `2A_f·‖jump_f‖` per face (one quadrature point, `D` scaled by double area), i.e. `∝ ∫‖D‖ dA` | `√(∫_g ‖D‖²)` per cell-pair group, i.e. `√A_f·rms` per face when cells are tets |
| ω | balances the two terms; in the fracture regime only the null space of `Q` matters (Fig. 12) | has no effect (the objective is 1-homogeneous) | normalized; 1e-3 is in the fracture regime |
| Rigid modes | — | the first `d` modes are the global translations; later modes are orthogonal to them | orthogonality to the 6 rigid modes is enforced; they are not reported |
| ICCM start | eigenvectors of `Q` on the unexploded mesh | vector-Laplacian eigenvectors, including the constants (ARPACK, random start vector) | elastic eigenvectors with rigid modes skipped (seeded LOBPCG) |
| ICCM stop | `‖Uᵢ − c‖ ≤ ε` | `‖c − c_prev‖_∞ ≤ 1e-4`, at most 10 iterations | `‖u − c‖_M̂ ≤ ε`, at most 50 iterations |
| Pieces | impact-dependent (§3.3) | per mode: tets with `‖c_i − c_j‖ < 0.1` connected | Level 1: threshold `max_i` jump to a target count (spec §4.4) |

Because of the reference's isotropic `‖·‖₂` and its translation-only space,
every cut pattern appears as a degenerate triple of modes (one per
displacement direction). The harness therefore uses `k = 6` (two complete
triples). Principal angles of a truncated triple would only measure the
reference's random ARPACK start vector.

### Results (`k = 6`; seconds on 4 shared cores)

Our functional is `½uᵀQ̂u + ω Σ w_g ‖B̂_g u‖`. The reference modes are
evaluated in it exactly: per-tet constants have zero strain energy and a
constant jump on each face. The reference functional is `Σ_f A_f ‖jump_f‖`;
ours is evaluated in it with face RMS jumps. Energy errors are
`(E(ours) − E(ref)) / E(ref)` per mode. ARI is measured on voxelized labels
(48³ grid). Both Level-1 labelings come from the same procedure (our
`segment_level1` on the max-over-modes jumps) with target T. Rows
`cell-p0:*` use the proposed change below (`--fields` build).

| Mesh (tets) | Ours | Time (s) | Principal angles (°) | Energy rel. err., our E (median / max \|·\|) | Energy rel. err., reference E (median / max \|·\|) | ARI L1, T = 2 / 3 / 4 |
|---|---|---|---|---|---|---|
| notched bar (1605) | reference (Clarabel) | 12.9 | — | — | — | — |
| | full | 652 | 12.0, 41.1, 41.1, 60.9, 69.6, 90.0 | 0.67 / 0.79 | 0.64 / 0.77 | 0.29 / 0.73 / 0.48 |
| | cell-p1 | 671 | 12.0, 41.1, 41.1, 60.9, 69.6, 90.0 | 0.67 / 0.79 | 0.64 / 0.77 | 0.29 / 0.73 / 0.48 |
| | cell-p0:uniform | 3.9 | 5.0, 5.0, 5.0, 10.9, 10.9, 10.9 | 0.10 / 0.12 | 0.0054 / 0.0059 | −0.06 / 0.47 / 0.98 |
| | **cell-p0:sqrt_area** | 4.1 | **0.0 ×6** | 6e-9 / 8e-9 | 6e-9 / 8e-9 | **1.00 / 1.00 / 1.00** |
| L-shape (1976) | reference (Clarabel) | 52.5 | — | — | — | — |
| | full | 2091 | 45.5, 64.9, 65.2, 71.8, 90.0, 90.0 | 0.55 / 0.62 | 0.56 / 0.62 | 0.00 / 0.00 / 0.00 |
| | cell-p1 | 1063 | 45.5, 64.1, 65.2, 71.8, 90.0, 90.0 | 0.55 / 0.62 | 0.56 / 0.62 | 0.00 / 0.00 / 0.00 |
| | cell-p0:uniform | 8.5 | 20.2 ×3, 32.6 ×3 | 0.033 / 0.042 | 0.031 / 0.058 | −0.02 / 0.29 / 0.80 |
| | **cell-p0:sqrt_area** | 9.3 | **0.0 ×6** | 1e-9 / 1e-9 | 1e-9 / 1e-9 | **1.00 / 1.00 / 1.00** |
| plate with hole (1924) | reference (Clarabel) | 15.9 | — | — | — | — |
| | full | 515 | 31.3, 81.8, 84.5, 88.8, 89.5, 89.9 | 0.56 / 0.71 | 0.58 / 0.73 | 0.00 / 0.00 / 0.06 |
| | cell-p0:uniform | 3.6 | 22.9, 22.9, 35.8, 40.3, 40.3, 40.4 | 0.012 / 0.055 | 0.026 / 0.042 | 0.00 / 0.64 / 0.54 |
| | cell-p0:sqrt_area | 3.4 | 22.9, 22.9, 35.8, 40.3, 40.3, 40.4 | 0.026 / 0.042 | 0.026 / 0.042 | 0.00 / 0.39 / 0.54 |
| bunny (1940) | reference (Clarabel) | 32.8 | — | — | — | — |
| | full | 3177 | 33.4, 34.7, 38.5, 44.2, 89.5, 89.7 | 0.80 / 0.89 | 0.79 / 0.89 | −0.01 / 0.97 / 0.97 |
| | cell-p0:uniform | 11.0 | 0.0 ×3, 80.5 ×3 | 1.7 / 3.5 | 2.2 / 4.4 | −0.01 / 0.10 / 0.10 |
| | **cell-p0:sqrt_area** | 10.9 | **0.0 ×6** | 1e-8 / 5e-8 | 1e-8 / 5e-8 | **1.00 / 1.00 / 1.00** |

cell-p1 was not run on the plate and the bunny (it costs as much as full and
matches it). All our runs converged. Targets: max angle ≤ 5°, ARI ≥ 0.9.

**The current library (full, cell-p1) does not meet §13.2.** The
frozen-reference test (`tools/harness/test_fracture_modes_golden.py`) skips
these checks with the measured numbers. It does not weaken the thresholds.

### Diagnosis

1. **Formulation, not a bug.** Our energies are 22–89% *below* the
   reference's, measured in either functional. Elasticity has rotations in
   its null space, so our fragments can rotate relative to each other
   (hinge-type openings with a jump that vanishes along an axis). This is
   the effect the paper describes in §3.6 and the reason it switched to
   `Q = I ⊗ L`. Because of this, our first modes are 6 relative rigid
   motions of the same two fragments where the reference has 3 (notched
   bar: all four of our first modes cut the notch, where the reference's
   4th mode already has a second cut). The mode subspaces therefore differ
   by construction, even where the cuts agree. On the bunny, our Level-1
   cuts (T = 3, 4) match the reference with ARI 0.97. The ARI against the
   reference's own first-mode pieces is 0.96 (notched bar) and 0.94
   (bunny). On the L-shape and the plate, the cuts differ entirely.
2. **The solver, ICCM and normalization are correct.** We restricted our
   pipeline to the reference's model. That means per-cell translations
   (`cell-p0`), ICCM started from vector-Laplacian eigenvectors, and
   `w_g = √(A_g/Ā)`, which turns our per-face `√A·rms` into the reference's
   `A·|jump|`. With these three changes it reproduces the reference exactly
   on 3 of 4 meshes: 0.0° principal angles, energies equal to 1e-7, ARI 1.00
   at every T. The remaining mesh is the 4-fold symmetric plate. There, both
   implementations start ICCM from a 2-D degenerate Laplacian eigenspace
   (ARPACK picks a random basis of it), and ICCM is a local method, so they
   settle in different local minima: different straight cuts through the
   hole. Ours is 2.6% higher in energy on modes 1–4 and 4.2% / 0.1% lower on
   modes 5–6. The total over the 6 modes is within 1%.
3. **Patch weighting.** With the paper's patch norm `√(∫‖D‖²)` per face
   (`cell-p0:uniform`), the cuts move by a few tet layers. Energies stay
   within 0.5–6% in the reference functional, but angles reach 5–40°. On the
   bunny, the second cut pattern changes outright (80.5°). The reference
   code weights faces by area (`∝ ∫‖D‖ dA`), which differs from the paper's
   Eq. (5) on single-face patches. For cell-pair groups, `frac-modes`
   follows the paper.
4. **The reduced discretization is not the cause.** With one tet per cell,
   cell-p1 spans the same space as full, and on the production solver path
   (hybrid ADMM + Clarabel) it reproduces the Clarabel full solve (angles
   within 1°, energies within 0.1%). Both are slow on fully exploded
   meshes: 9–53 minutes against 13–53 s for the reference. The per-cell
   translation model takes 3–11 s.
5. **§13.2 sensitivity.** The 5° bound is tighter than the reference's own
   reproducibility on degenerate geometry. Two open solvers on the same
   reference (Clarabel and SCS) already differ by 10.5° on the notched bar,
   because the cut can sit anywhere across the notch at equal cost. The
   Level-1 procedure (max over modes of mode-normalized jumps, spec §4.4)
   also lets high modes dominate. At T = 2 it isolates an end segment of
   the notched bar rather than splitting at the notch, for both
   implementations, and on the plate T = 2 is unreachable (one cut through
   the hole opens two ligaments at once).

### Proposed changes to `frac-modes` (not applied; patches in `tools/harness/patches/`)

These are our own Rust code, written from the paper:

* `01-frac-modes-mode-fields.patch`: `ModesOutput::{nodes, mode_fields}`
  gives each mode's displacement per exploded `(vertex, cell)` node,
  prolongated for reduced discretizations. §13.2's principal angles need
  it, and so does the paper's runtime impact projection
  `w* = Σ Uᵢ Uᵢᵀ M̃ w` (§3.3).
* `02-frac-modes-cell-constant.patch`: `Discretization::CellPolynomial(0)`,
  one translation per (super-)cell (the paper's §3.6). The strain term
  vanishes, and only the rigid modes contained in the subspace (the
  translations) are constrained. It has 3 unknowns per cell instead of 12
  and is 150–290× faster than full on these meshes.
* `03-frac-modes-laplacian-init.patch`: for degree 0, ICCM starts from
  eigenvectors of the model's own strain Hessian, the vector Laplacian
  `I₃ ⊗ L` (scalar cotangent Laplacian, lumped mass, constants deflated;
  seeded LOBPCG, so it is deterministic), as the paper prescribes. Elastic
  eigenvectors projected to translations sent mode 3 of the notched bar to an
  end piece at 2.7× the energy, which did not converge in 50 iterations.
* Not prototyped:
  * try several bases of a degenerate initial eigenspace and keep the
    lowest-energy ICCM result, so symmetric parts (the plate) get the
    better local minimum deterministically;
  * an option for the reference's area weighting of fault faces;
  * make the translational model the default for Level 1, if the spec owner
    accepts deviating from §4.1's "linear-elastic `Q`". The paper's
    observation that only the null space of `Q` matters (Fig. 12) argues
    for it.

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
(`benchmarks/configs/bake.toml`):

| Interfaces | Recall@τ | Precision@τ | F | F@τ/2 | F@2τ |
|---|---|---|---|---|---|
| L3 (detail) | **0.855** | 0.043 | 0.081 | 0.040 | 0.172 |
| L1 (structural) | 0.167 | 0.032 | 0.054 | 0.026 | 0.118 |

Weak-region agreement (Spearman, per analysis cell, L1): −0.19 (target ≥ 0.6).

L3 recall meets the ≥ 0.8 target. Precision is low by construction: the
detail interfaces fill the volume, while each oracle crack is a single
surface.

L1 does not meet the targets. The fracture modes are load-independent
worst-case cuts. The Rankine oracle concentrates cracks at the clamped root
and under impacts on this prismatic column, so there is no geometric weak
region for the modes to find. Before the size-balanced segmentation was
added, Level 1 was one fragment holding 87% of the volume plus 11 surface
chips (recall 0.07).
