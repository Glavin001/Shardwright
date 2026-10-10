//! Weak-region analysis (spec Stage 4): a clean-room reimplementation of
//! "Breaking Good: Fracture Modes for Realtime Destruction" (Sellán et al.,
//! ACM TOG 2022) on a cell-exploded tetrahedral mesh, with material-aware
//! interface weights.
//!
//! # Discretization (§4.1)
//! One displacement node per (mesh vertex, analysis cell) pair: DOFs are
//! shared inside a cell and duplicated across *fault faces* (faces between
//! tets of different cells). `Q` is the P1 linear-elastic stiffness of this
//! exploded mesh with per-tet materials, `M̃` its lumped mass. Faces are
//! grouped by analysis-cell pair; `B_g` evaluates the displacement jump
//! `u_a − u_b` at the three edge midpoints of every face of group `g`
//! (weights `area/3`, exact for the quadratic `‖D‖²`), scaled so that
//! `‖B_g u‖² = ∫_g ‖D(u,x)‖² dA` exactly.
//!
//! # Energy and constraints (§4.2)
//! `E(u) = ½ uᵀQu + ω Σ_g w_g ‖B_g u‖`. Forbidden groups (`w_g = ∞`) impose
//! `B_g u = 0`; since jumps are linear on each face this is *equivalent* to
//! equal DOF copies at every vertex of the group's faces, so those copies are
//! merged exactly (forbidden jumps are then identically zero). Groups with
//! `w_g = 0` are unpenalized. Anchored vertices: all copies are fixed (Dirichlet,
//! eliminated). Unanchored: `u ⟂_M̃` the six rigid modes of the exploded mesh.
//!
//! # Normalization
//! With `m_tot` the total mass, `λ₁` the first non-rigid eigenvalue of the
//! continuous `(K, M)` pair (same anchors) and `L = V^{1/3}` (V = mesh volume):
//! `M̂ = M/m_tot` (total mass 1), `Q̂ = Q/(λ₁ m_tot)` (first continuous
//! eigenvalue 1), `B̂_g = B_g / L` (areas measured in units of `L²`). So `ω`
//! is dimensionless: for an M̂-unit displacement the elastic term is `O(1)`
//! and the sparsity term is `ω Σ w_g √(A_g/L²)·rms_jump`. Reported jumps are
//! `‖B̂_g U_i‖ / √(A_g/L²)` = RMS jump over the interface for a mode with
//! `U_iᵀM̂U_i = 1`, i.e. comparable across groups, modes and meshes.
//! Both solvers minimize `½uᵀ(Q̂+δM̂)u + …` with `δ = 1e-8` (see
//! [`problem::DELTA`]).
//!
//! # Solve (§4.3, adapted ICCM)
//! For mode i: `c` = i-th continuous eigenvector (rigid modes skipped)
//! mapped to exploded DOFs. Repeat: solve the convex subproblem with
//! `[U_1 … U_{i−1}, c]ᵀM̂u = [0 … 0, 1]ᵀ` (plus rigid-mode rows when
//! unanchored), `c ← u/‖u‖_M̂`, stop when `‖u − c_old‖_M̂ ≤ ε` (relative,
//! since `‖c_old‖_M̂ = 1`) or after `max_iters`. `U_i = u/‖u‖_M̂`.
//! Before each solve, `c` is M̂-orthogonalized against the previous modes
//! and rigid modes, which leaves the feasible set unchanged but keeps the
//! constraint rows well conditioned.
//!
//! Solvers: [`Solver::Clarabel`] (interior-point conic reference) and
//! [`Solver::Admm`] (Anderson-accelerated group-lasso ADMM with a single
//! sparse Cholesky factor shared across modes and ICCM iterations, warm
//! starts, and an inexact-ICCM tolerance schedule; for ≤ 3000 unknowns the
//! tight confirmation solves are delegated to Clarabel).
//!
//! Large problems (more than [`ModesParams::large_dofs`] unknowns, e.g. a
//! brick wall with ~1300 analysis cells → 16k unknowns) run ADMM only: an
//! interior-point confirmation costs ~35 s there and a tight ADMM tail
//! thousands of iterations. ICCM then stops at
//! `max(eps, ModesParams::eps_large)` with subproblems certified to a tenth
//! of it; on the benchmark wall this reproduces the Level-1 segmentation of
//! the tight schedule exactly (ARI 1.0) at ~1/20 of the cost.
//!
//! # Performance: cell-polynomial reduction
//! The full exploded problem has `3 × #(vertex, cell) pairs` unknowns
//! (~30k for a 20k-tet mesh) and costs minutes. [`Solver::Auto`] therefore
//! solves the full problem (with Clarabel) only below 3000 unknowns and
//! otherwise switches to [`Discretization::CellPolynomial`]`(1)`: a Galerkin
//! subspace in which every analysis cell displaces as the nodal interpolant
//! of an affine field (12 DOFs per cell, containing the 6 rigid modes). Group
//! norms, anchors, forbidden interfaces, rigid-mode and orthogonality
//! constraints are all represented exactly in that subspace (see `reduce`);
//! the unknown count becomes `12 × #cells` independent of the tet count.
//! The known-answer tests pass for both discretizations. Use
//! [`compute_modes_with`] to force a discretization (e.g. `Full` + `Admm`).

mod admm;
mod clarabel_solver;
pub mod dump;
mod level1;
pub mod problem;
mod reduce;

pub use level1::{adjusted_rand_index, segment_from_jumps, segment_level1, segment_level1_balanced, Level1};

use problem::{mdot, Problem};
use std::time::Instant;

/// Free-DOF threshold below which [`Solver::Auto`] uses Clarabel.
pub const AUTO_CLARABEL_MAX_DOFS: usize = 3000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Solver {
    Auto,
    Clarabel,
    Admm,
}

#[derive(Clone, Copy, Debug)]
pub struct ModesParams {
    /// Number of fracture modes.
    pub k: usize,
    /// Sparsity weight (normalized units).
    pub omega: f64,
    /// ICCM tolerance on `‖u − c‖_M̂` (M̂-unit `c`).
    pub eps: f64,
    pub max_iters: usize,
    pub solver: Solver,
    /// Seed for the eigensolver's starting block.
    pub seed: u64,
    /// Problems with more unknowns than this are solved by ADMM alone (no
    /// Clarabel confirmation solves) with ICCM tolerance
    /// `max(eps, eps_large)`; see [`iccm`] for the schedule.
    pub large_dofs: usize,
    pub eps_large: f64,
}

impl Default for ModesParams {
    fn default() -> Self {
        ModesParams {
            k: 10,
            omega: 1e-3,
            eps: 1e-4,
            max_iters: 50,
            solver: Solver::Auto,
            seed: 0x5eed_f2ac,
            large_dofs: 3000,
            eps_large: 1e-3,
        }
    }
}

pub struct ModesInput<'a> {
    pub mesh: &'a frac_fem::TetMesh,
    pub tet_material: &'a [frac_fem::ElasticMaterial],
    /// Analysis cell label per tet, labels `0..n_cells`.
    pub tet_cell: &'a [u32],
    /// `w_g` for cells `(a < b)`; `f64::INFINITY` = forbidden; must be ≥ 0.
    pub group_weight: &'a (dyn Fn(u32, u32) -> f64 + Sync),
    pub anchored_vertices: &'a [u32],
    pub params: ModesParams,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModesOutput {
    /// Sorted `(a < b)` cell pairs that share at least one fault face.
    pub groups: Vec<(u32, u32)>,
    /// Physical interface area per group.
    pub group_area: Vec<f64>,
    /// `jumps[mode][group] = ‖B̂_g U_i‖ / √(A_g/L²)` (RMS jump, normalized units).
    pub jumps: Vec<Vec<f64>>,
    /// Normalized energy `E(U_i)` of each (M̂-unit) mode.
    pub energies: Vec<f64>,
    /// Continuous-mesh eigenvalues (physical, `ω²` in rad²/s²) used for init.
    pub eigenvalues: Vec<f64>,
    /// ICCM iterations per mode.
    pub iterations: Vec<usize>,
    pub converged: Vec<bool>,
    /// Number of unknowns of the solved problem: free exploded DOFs (after
    /// merging forbidden interfaces and removing anchored copies) for the full
    /// discretization, reduced coefficients for the cell-polynomial one.
    pub n_dofs: usize,
    pub solver_used: String,
    /// Wall-clock timings (not deterministic; excluded from comparisons).
    pub timings_ms: Vec<(String, f64)>,
}

impl ModesOutput {
    /// `max_i jumps[i][g]` per group.
    pub fn max_jump(&self) -> Vec<f64> {
        let mut m = vec![0.0f64; self.groups.len()];
        for j in &self.jumps {
            for (g, &v) in j.iter().enumerate() {
                m[g] = m[g].max(v);
            }
        }
        m
    }
}

/// Full problems up to this many unknowns (and all cell-reduced problems) use
/// Clarabel for the tight ICCM confirmation solves when running ADMM (hybrid
/// mode).
pub const HYBRID_CONFIRM_MAX_DOFS: usize = 3000;

/// Relative tolerance range of the inexact (ADMM) inner solves. The tight
/// (confirmation) tolerance is `0.01 ε` clamped to `[1e-8, 1e-4]`.
const ADMM_TOL_MAX: f64 = 1e-3;
#[cfg(test)]
const ADMM_TOL_MIN: f64 = 1e-6;

pub(crate) struct SubResult {
    pub u: Vec<f64>,
    pub iterations: usize,
    pub ok: bool,
}

enum Backend {
    Clarabel,
    Admm(Box<admm::Admm>),
}

/// Discretization of the exploded displacement field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Discretization {
    /// Full cell-exploded P1 space (spec §4.1).
    Full,
    /// Galerkin subspace of the full space: per (super-)cell nodal
    /// interpolant of a vector polynomial of the given degree (1 or 2); see
    /// the `reduce` module docs. Unknowns ≈ 3·s·#cells (s = 4 or 10)
    /// regardless of the tet count.
    CellPolynomial(u8),
}

/// Polynomial degree used by [`Solver::Auto`] for large problems.
pub const AUTO_REDUCED_DEGREE: u8 = 1;

/// Computes `k` fracture modes (spec §4.3) and their per-interface jumps.
///
/// Discretization/solver choice: [`Solver::Clarabel`] and [`Solver::Admm`]
/// solve the full exploded problem. [`Solver::Auto`] solves the full problem
/// with Clarabel when it has fewer than [`AUTO_CLARABEL_MAX_DOFS`] free DOFs;
/// otherwise it switches to the [`Discretization::CellPolynomial`] subspace
/// (degree [`AUTO_REDUCED_DEGREE`]) and solves that with hybrid ADMM (tight
/// ICCM confirmation solves by Clarabel). Use [`compute_modes_with`] to force
/// a discretization.
pub fn compute_modes(input: &ModesInput) -> Result<ModesOutput, String> {
    compute_modes_impl(input, None)
}

/// [`compute_modes`] with an explicit discretization. The solver is taken
/// from `params.solver` (`Auto` = Clarabel for a full problem below
/// [`AUTO_CLARABEL_MAX_DOFS`] unknowns, hybrid ADMM otherwise).
pub fn compute_modes_with(input: &ModesInput, disc: Discretization) -> Result<ModesOutput, String> {
    compute_modes_impl(input, Some(disc))
}

fn compute_modes_impl(input: &ModesInput, disc: Option<Discretization>) -> Result<ModesOutput, String> {
    let mut timings = Vec::new();
    let t_all = Instant::now();
    let p = input.params;
    let (full, info) = problem::build_full(input, &mut timings)?;
    let disc = disc.unwrap_or(match p.solver {
        Solver::Auto if full.n >= AUTO_CLARABEL_MAX_DOFS => Discretization::CellPolynomial(AUTO_REDUCED_DEGREE),
        _ => Discretization::Full,
    });
    let (pb, disc_name) = match disc {
        Discretization::Full => (full, "full".to_string()),
        Discretization::CellPolynomial(d) => {
            let t = Instant::now();
            let r = reduce::reduce(&full, &info, d, p.omega)?;
            timings.push(("reduce".into(), t.elapsed().as_secs_f64() * 1e3));
            (r, format!("cell-p{}", d.clamp(1, 2)))
        }
    };
    let use_clarabel = match p.solver {
        Solver::Clarabel => true,
        Solver::Admm => false,
        // full: Clarabel for small problems; reduced: hybrid ADMM (loose ADMM
        // ICCM iterations + Clarabel confirmation solves) is much faster
        Solver::Auto => disc == Discretization::Full && pb.n < AUTO_CLARABEL_MAX_DOFS,
    };
    let t_f = Instant::now();
    let mut backend = if use_clarabel || pb.active.is_empty() {
        Backend::Clarabel
    } else {
        Backend::Admm(Box::new(admm::Admm::new(&pb)?))
    };
    if let Backend::Admm(_) = backend {
        timings.push(("admm_factor".into(), t_f.elapsed().as_secs_f64() * 1e3));
    }
    let t_iccm = Instant::now();
    let hybrid = disc != Discretization::Full || pb.n <= HYBRID_CONFIRM_MAX_DOFS;
    let res = iccm(&pb, &p, &mut backend, hybrid)?;
    timings.push(("iccm".into(), t_iccm.elapsed().as_secs_f64() * 1e3));
    let mut solver_used = match &backend {
        Backend::Clarabel => "clarabel".to_string(),
        Backend::Admm(a) => format!("admm(refactorizations={}, rho={:.3e})", a.refactorizations, a.rho),
    };
    if disc_name != "full" {
        solver_used = format!("{disc_name}+{solver_used}");
    }
    let l2 = pb.length_scale * pb.length_scale;
    let mut jumps = Vec::new();
    let mut energies = Vec::new();
    for u in &res.modes {
        let gn = pb.group_norms(u);
        jumps.push(
            gn.iter()
                .zip(&pb.group_area)
                .map(|(&n, &a)| if a > 0.0 { n / (a / l2).sqrt() } else { 0.0 })
                .collect(),
        );
        energies.push(pb.objective(u));
    }
    timings.push(("total".into(), t_all.elapsed().as_secs_f64() * 1e3));
    Ok(ModesOutput {
        groups: pb.groups.clone(),
        group_area: pb.group_area.clone(),
        jumps,
        energies,
        eigenvalues: pb.eigenvalues.clone(),
        iterations: res.iterations,
        converged: res.converged,
        n_dofs: pb.n,
        solver_used,
        timings_ms: timings,
    })
}

struct IccmResult {
    modes: Vec<Vec<f64>>,
    iterations: Vec<usize>,
    converged: Vec<bool>,
}

fn orthogonalize(v: &mut [f64], basis: &[Vec<f64>], m: &[f64]) {
    for _ in 0..2 {
        for q in basis {
            let c = mdot(q, v, m);
            for i in 0..v.len() {
                v[i] -= c * q[i];
            }
        }
    }
}

/// ICCM stopping/certification schedule.
struct Schedule {
    /// ICCM tolerance on `‖u − c‖_M̂`.
    eps: f64,
    /// Inner (ADMM) tolerance a subproblem must be solved to for its ICCM
    /// step to certify convergence.
    tol_min: f64,
    /// Certify with Clarabel (hybrid mode) instead of a tight ADMM solve.
    clarabel_confirm: bool,
}

impl Schedule {
    /// `hybrid`: Clarabel confirmation solves are allowed (small/reduced
    /// problems). Problems above `p.large_dofs` unknowns run ADMM only: an
    /// interior-point solve costs ~35 s at 16k unknowns (vs 0.2 s at 1k), and
    /// a tight ADMM tail thousands of iterations; ICCM then stops at
    /// `max(eps, eps_large)` with subproblems certified to a tenth of it,
    /// which leaves the Level-1 segmentation unchanged on the benchmark
    /// masonry wall (ARI 1.0 against the tight reference).
    fn new(p: &ModesParams, n: usize, backend: &Backend, hybrid: bool) -> Schedule {
        let large = matches!(backend, Backend::Admm(_)) && n > p.large_dofs;
        if large {
            let eps = p.eps.max(p.eps_large);
            Schedule { eps, tol_min: (0.1 * eps).clamp(1e-8, ADMM_TOL_MAX), clarabel_confirm: false }
        } else {
            Schedule { eps: p.eps, tol_min: (0.01 * p.eps).clamp(1e-8, 1e-4), clarabel_confirm: hybrid }
        }
    }
}

fn iccm(pb: &Problem, p: &ModesParams, backend: &mut Backend, hybrid: bool) -> Result<IccmResult, String> {
    let n = pb.n;
    let sched = Schedule::new(p, n, backend, hybrid);
    let m = &pb.m;
    // M̂-orthonormal basis of the rigid space (rows are M̂R; recover R)
    let rigid_basis: Vec<Vec<f64>> = pb.rigid_rows.iter().map(|r| (0..n).map(|i| r[i] / m[i]).collect()).collect();
    let mut modes: Vec<Vec<f64>> = Vec::new();
    let mut mode_rows: Vec<Vec<f64>> = Vec::new();
    let mut iterations = Vec::new();
    let mut converged = Vec::new();
    let kk = p.k.min(pb.init.len());
    for i in 0..kk {
        let mut basis = rigid_basis.clone();
        basis.extend(modes.iter().cloned());
        let mut c = pb.init[i].clone();
        orthogonalize(&mut c, &basis, m);
        let mut nc = mdot(&c, &c, m).sqrt();
        if !(nc > 1e-8) {
            // initial vector lies in the span of previous modes: use the
            // first remaining eigenvector that does not
            for j in kk..pb.init.len() {
                c = pb.init[j].clone();
                orthogonalize(&mut c, &basis, m);
                nc = mdot(&c, &c, m).sqrt();
                if nc > 1e-8 {
                    break;
                }
            }
            if !(nc > 1e-8) {
                return Err(format!("mode {i}: no admissible initial vector"));
            }
        }
        c.iter_mut().for_each(|x| *x /= nc);
        if let Backend::Admm(a) = backend {
            a.warm_from(pb, &c);
        }
        let mut persistent: Vec<&[f64]> = pb.rigid_rows.iter().map(|r| r.as_slice()).collect();
        persistent.extend(mode_rows.iter().map(|r| r.as_slice()));
        let mut rhs = vec![0.0; persistent.len() + 1];
        *rhs.last_mut().unwrap() = 1.0;
        let mut u = c.clone();
        let mut its = 0;
        let mut conv = false;
        let mut tol = ADMM_TOL_MAX.max(sched.tol_min);
        let mut tol_sched = tol;
        let mut last_diff = f64::INFINITY;
        let mut confirm = false;
        let mut inner_total = 0usize;
        for it in 0..p.max_iters.max(1) {
            its = it + 1;
            let cur: Vec<f64> = (0..n).map(|q| m[q] * c[q]).collect();
            let sub = match backend {
                Backend::Clarabel => {
                    let mut rows = persistent.clone();
                    rows.push(&cur);
                    clarabel_solver::solve(pb, &rows, &rhs)?
                }
                Backend::Admm(a) => {
                    // inexact ICCM: the inner tolerance follows the outer progress
                    // (never loosening); convergence is only accepted after a
                    // solve to `tol_min`. A failed confirmation does not pin the
                    // schedule to `tol_min`.
                    if confirm {
                        tol = sched.tol_min;
                    } else {
                        tol_sched = (0.1 * last_diff).clamp(sched.tol_min, tol_sched);
                        tol = tol_sched;
                    }
                    if confirm && sched.clarabel_confirm {
                        // ADMM's tail is slow; for small (e.g. reduced) problems the
                        // tight confirmation solve is done by the interior-point
                        // reference solver, then ADMM is re-warmed from it
                        let mut rows = persistent.clone();
                        rows.push(&cur);
                        let tc = Instant::now();
                        let r = clarabel_solver::solve(pb, &rows, &rhs)?;
                        if admm::debug_enabled() {
                            eprintln!("[clarabel] confirm {} iterations ok {} time {:.3}s", r.iterations, r.ok, tc.elapsed().as_secs_f64());
                        }
                        a.warm_from(pb, &r.u);
                        r
                    } else {
                        a.solve(pb, &persistent, &cur, &rhs, tol)?
                    }
                }
            };
            inner_total += sub.iterations;
            let sub_ok = sub.ok;
            u = sub.u;
            let mut d2 = 0.0;
            for q in 0..n {
                let d = u[q] - c[q];
                d2 += d * d * m[q];
            }
            let nu = mdot(&u, &u, m).sqrt();
            if !(nu > 0.0) || !nu.is_finite() {
                return Err(format!("mode {i}: degenerate subproblem solution"));
            }
            c = u.iter().map(|x| x / nu).collect();
            last_diff = d2.sqrt();
            if admm::debug_enabled() {
                eprintln!("[iccm] mode {i} it {it} diff {last_diff:.3e} obj {:.6e}", pb.objective(&u));
            }
            if last_diff <= sched.eps {
                if matches!(backend, Backend::Clarabel) || tol <= sched.tol_min {
                    // only a subproblem solved to tolerance certifies convergence
                    conv = sub_ok;
                    break;
                }
                confirm = true;
            } else {
                confirm = false;
            }
            // keep c exactly admissible (orthogonal to previous modes / rigid)
            orthogonalize(&mut c, &basis, m);
            let nc = mdot(&c, &c, m).sqrt();
            c.iter_mut().for_each(|x| *x /= nc);
        }
        if admm::debug_enabled() {
            eprintln!("[iccm] mode {i}: {its} iterations, {inner_total} inner iterations, converged {conv}");
        }
        let nu = mdot(&u, &u, m).sqrt();
        let ui: Vec<f64> = u.iter().map(|x| x / nu).collect();
        mode_rows.push((0..n).map(|q| m[q] * ui[q]).collect());
        modes.push(ui);
        iterations.push(its);
        converged.push(conv);
    }
    Ok(IccmResult { modes, iterations, converged })
}

#[cfg(test)]
mod tests;
