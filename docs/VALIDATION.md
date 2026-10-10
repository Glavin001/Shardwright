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
