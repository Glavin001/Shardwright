//! Reference solver: the ICCM subproblem as a conic program for Clarabel.
//!
//! Variables `x = [u (n), t (one per active group)]`:
//!
//! ```text
//! min  ½ uᵀ(Q̂ + δM̂)u + Σ_g ω w_g t_g
//! s.t. C u = d                        (zero cone: orthogonality, rigid modes, c-normalization)
//!      (t_g, B̂_g u) ∈ SOC(1 + r_g)    (one second-order cone per active group)
//! ```
//!
//! Anchors are eliminated (free DOFs only) and forbidden groups are merged
//! into continuous DOFs beforehand, so they need no constraints here.

use crate::problem::{Problem, DELTA};
use crate::SubResult;
use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};
use frac_fem::sparse::CsrMatrix;

fn csc_from_csr(a: &CsrMatrix) -> CscMatrix<f64> {
    let t = a.transpose(); // CSR of A^T == CSC of A
    CscMatrix::new(a.n_rows, a.n_cols, t.row_ptr, t.col_idx, t.vals)
}

pub(crate) fn solve(pb: &Problem, rows: &[&[f64]], rhs: &[f64]) -> Result<SubResult, String> {
    let n = pb.n;
    let ng = pb.active.len();
    let nx = n + ng;
    // P: upper triangle of Q̂ + δM̂
    let mut pt = Vec::with_capacity(pb.q.nnz() / 2 + n);
    for i in 0..n {
        let (c, v) = pb.q.row(i);
        for k in 0..c.len() {
            if c[k] >= i {
                pt.push((i, c[k], v[k]));
            }
        }
        pt.push((i, i, DELTA * pb.m[i]));
    }
    let p = csc_from_csr(&CsrMatrix::from_triplets(nx, nx, pt));
    let mut q = vec![0.0; nx];
    for (k, &l) in pb.lam_act.iter().enumerate() {
        q[n + k] = l;
    }
    // constraints
    let mut at: Vec<(usize, usize, f64)> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut cones: Vec<SupportedConeT<f64>> = Vec::new();
    let mut r = 0usize;
    for (k, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                at.push((r, j, v));
            }
        }
        b.push(rhs[k]);
        r += 1;
    }
    if !rows.is_empty() {
        cones.push(SupportedConeT::ZeroConeT(rows.len()));
    }
    for (k, range) in pb.rows_act.iter().enumerate() {
        // s_0 = t_g
        at.push((r, n + k, -1.0));
        b.push(0.0);
        r += 1;
        for br in range.clone() {
            let (c, v) = pb.b_act.row(br);
            for m in 0..c.len() {
                at.push((r, c[m], -v[m]));
            }
            b.push(0.0);
            r += 1;
        }
        cones.push(SupportedConeT::SecondOrderConeT(1 + range.len()));
    }
    let a = csc_from_csr(&CsrMatrix::from_triplets(r, nx, at));
    let mut settings = DefaultSettings::<f64>::default();
    settings.verbose = false;
    settings.max_iter = 400;
    settings.max_threads = 1;
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings);
    solver.solve();
    let st = solver.solution.status;
    let ok = matches!(st, SolverStatus::Solved | SolverStatus::AlmostSolved);
    let usable = ok || matches!(st, SolverStatus::MaxIterations | SolverStatus::InsufficientProgress);
    let x = &solver.solution.x;
    if !usable || x.iter().take(n).any(|v| !v.is_finite()) {
        return Err(format!("clarabel failed: {st:?}"));
    }
    Ok(SubResult { u: x[..n].to_vec(), iterations: solver.solution.iterations as usize, ok })
}
