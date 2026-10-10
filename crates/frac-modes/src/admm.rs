//! Fast solver: Anderson-accelerated ADMM (group-lasso splitting) for the
//! ICCM subproblem
//!
//! ```text
//! min ½ uᵀ(Q̂+δM̂)u + Σ_g λ_g ‖z_g‖   s.t.  B̂u − z = 0,  C u = d
//! ```
//!
//! Scaled-form ADMM with over-relaxation α:
//!
//! * u-step: `(Q̂ + ρB̂ᵀB̂ + δM̂) u + Cᵀν = ρB̂ᵀ(z − y)`, `C u = d`. The SPD
//!   matrix `A = Q̂ + ρB̂ᵀB̂ + δM̂` is factorized with faer's sparse Cholesky
//!   once (refactorized numerically only when ρ is adapted — the factor is
//!   shared by all modes and ICCM iterations). The few dense equality rows
//!   `C` (≤ 6 rigid + k mode rows) are handled with a small Schur complement
//!   `S = C A⁻¹ Cᵀ`; `A⁻¹Cᵀ` is cached for the persistent rows (rigid modes
//!   and finished modes) and recomputed once per call for the `c` row.
//! * z-step: block soft-thresholding `z_g = max(0, 1 − λ_g/(ρ‖v_g‖)) v_g`.
//! * y-step: `y += h − z` with `h = αB̂u + (1−α)z`.
//!
//! The iteration is run as a fixed-point map on a state vector, either
//! `x = [z; y]` or (for α = 1) the Douglas–Rachford state `t = z + y` (half
//! the length: `z = prox(t)`, `y = t − z`). The map is accelerated with
//! type-II Anderson acceleration (memory 6) with a residual safeguard (in the
//! spirit of Zhang et al. 2019, "Accelerating ADMM for efficient simulation
//! and optimization"): an accelerated iterate is accepted only if its
//! fixed-point residual `‖F(x) − x‖` does not exceed the last accepted one;
//! otherwise the plain ADMM step is taken and the history is cleared.
//!
//! Stopping (Boyd et al. 2011 §3.3): `‖B̂u − z‖ ≤ √m ε_abs + ε_rel max(‖B̂u‖, ‖z‖)`
//! and `ρ‖B̂ᵀ(z⁺ − z)‖ ≤ √n ε_abs + ε_rel ρ‖B̂ᵀy‖`. ρ is adapted with the
//! OSQP residual-balancing rule (only when the suggested change exceeds 5x,
//! to limit refactorizations). `(z, y)` are warm-started across ICCM
//! iterations of the same mode. All buffers are allocated once per call and
//! every reduction has a fixed order, so results are deterministic.

use crate::SubResult;
use crate::problem::{DELTA, Problem};
use frac_fem::dense::{self, Lu};
use frac_fem::sparse::{CsrMatrix, SolveWork, SparseCholesky};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub(crate) struct AdmmSettings {
    pub eps_abs: f64,
    pub eps_rel: f64,
    pub max_iter: usize,
    pub alpha: f64,
    pub adaptive: bool,
    /// Anderson memory (0 disables acceleration).
    pub anderson: usize,
    /// Iterate on the Douglas–Rachford state `t = z + y` (requires α = 1).
    pub dr_state: bool,
}

impl Default for AdmmSettings {
    fn default() -> Self {
        AdmmSettings {
            eps_abs: 1e-9,
            eps_rel: 1e-6,
            max_iter: 20_000,
            alpha: 1.0,
            adaptive: true,
            anderson: 6,
            dr_state: true,
        }
    }
}

thread_local! {
    static SOLVE_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static STEP_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

pub(crate) fn debug_enabled() -> bool {
    std::env::var_os("FRAC_MODES_DEBUG").is_some()
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Unknowns from which the u-step matrix is factorized supernodally.
const SUPERNODAL_MIN_DOFS: usize = 5000;

/// Chunk length of the parallel reductions (fixed, so the summation order
/// never depends on the thread count).
const CHUNK: usize = 4096;

fn dot_seq(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Dot product: sequential partial sums over fixed chunks (computed in
/// parallel for long vectors), added in chunk order — deterministic.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 8 * CHUNK {
        return dot_seq(a, b);
    }
    let parts: Vec<f64> = a
        .par_chunks(CHUNK)
        .zip(b.par_chunks(CHUNK))
        .map(|(x, y)| dot_seq(x, y))
        .collect();
    parts.iter().sum()
}

pub(crate) struct Admm {
    pub settings: AdmmSettings,
    bt: CsrMatrix,
    btb: CsrMatrix,
    base: CsrMatrix, // Q̂ + δM̂ with the union pattern of BᵀB
    chol: SparseCholesky,
    pub rho: f64,
    pub refactorizations: usize,
    persist_w: Vec<Vec<f64>>,
    /// Warm-start `(z, y)` (each of length `rows`).
    z0: Vec<f64>,
    y0: Vec<f64>,
    /// Jump operator and penalties as used by the splitting (group-scaled).
    b: CsrMatrix,
    lam: Vec<f64>,
}

/// Equality-constraint data for one `solve` call.
struct Constraints<'a> {
    rows: Vec<&'a [f64]>,
    n_persist: usize,
    w_cur: Vec<f64>,
    lu: Lu,
    rhs: &'a [f64],
}

/// Reusable buffers of the ADMM map.
struct Work {
    tmp: Vec<f64>,
    r: Vec<f64>,
    bu: Vec<f64>,
    chol: SolveWork,
}

/// Result of one ADMM map evaluation.
struct Step {
    /// Map output `F(x)` (state layout).
    fx: Vec<f64>,
    u: Vec<f64>,
    /// `z` used by the u-step, and the new `(z⁺, y⁺)`.
    z_in: Vec<f64>,
    zn: Vec<f64>,
    yn: Vec<f64>,
    r_prim: f64,
    eps_pri: f64,
    scale_p: f64,
}

impl Step {
    fn new(n: usize, nr: usize, len: usize) -> Step {
        Step {
            fx: vec![0.0; len],
            u: vec![0.0; n],
            z_in: vec![0.0; nr],
            zn: vec![0.0; nr],
            yn: vec![0.0; nr],
            r_prim: 0.0,
            eps_pri: 0.0,
            scale_p: 0.0,
        }
    }
}

/// Group soft-thresholding `out_g = max(0, 1 − λ_g/(ρ‖v_g‖)) v_g`.
fn prox(pb: &Problem, lam: &[f64], rho: f64, v: &[f64], out: &mut [f64]) {
    for (g, range) in pb.rows_act.iter().enumerate() {
        let thr = lam[g] / rho;
        let mut nv = 0.0;
        for i in range.clone() {
            nv += v[i] * v[i];
        }
        let nv = nv.sqrt();
        let f = if nv > thr { 1.0 - thr / nv } else { 0.0 };
        for i in range.clone() {
            out[i] = f * v[i];
        }
    }
}

impl Admm {
    pub fn new(pb: &Problem) -> Result<Admm, String> {
        Self::with_rho(pb, None)
    }

    pub fn with_rho(pb: &Problem, rho0: Option<f64>) -> Result<Admm, String> {
        // per-group equilibration: ‖B̂_g u‖ λ_g = ‖s_g B̂_g u‖ λ_g / s_g, with
        // s_g normalizing the groups' Frobenius norms (a per-group ρ)
        let mut b = pb.b_act.clone();
        let mut lam = pb.lam_act.clone();
        {
            let fro: Vec<f64> = pb
                .rows_act
                .iter()
                .map(|r| {
                    b.vals[b.row_ptr[r.start]..b.row_ptr[r.end]]
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                })
                .collect();
            let mean = fro.iter().sum::<f64>() / fro.len().max(1) as f64;
            for (g, r) in pb.rows_act.iter().enumerate() {
                let sg = if fro[g] > 0.0 {
                    (mean / fro[g]).sqrt()
                } else {
                    1.0
                };
                let (a0, a1) = (b.row_ptr[r.start], b.row_ptr[r.end]);
                b.vals[a0..a1].iter_mut().for_each(|v| *v *= sg);
                lam[g] /= sg;
            }
        }
        let bt = b.transpose();
        let btb = b.ata(None);
        let dm: Vec<f64> = pb.m.iter().map(|&x| DELTA * x).collect();
        // base = Q + δM, padded with the BᵀB pattern (zeros) so A keeps one pattern
        let base = pb.q.add_diagonal(&dm).add(1.0, &btb, 0.0);
        // initial ρ: a fraction of the ratio of the diagonals of Q̂ and B̂ᵀB̂ on
        // DOFs touched by B
        let qd = pb.q.diagonal();
        let bd = btb.diagonal();
        let (mut sq, mut sb) = (0.0, 0.0);
        for i in 0..pb.n {
            if bd[i] > 0.0 {
                sq += qd[i];
                sb += bd[i];
            }
        }
        let rho = rho0.unwrap_or(if sb > 0.0 {
            (0.03 * sq / sb).clamp(1e-6, 1e6)
        } else {
            1.0
        });
        let a = base.add(1.0, &btb, rho);
        // simplicial factors have lean, allocation-free solves (the u-step is
        // solved thousands of times); supernodal ones (dense blocks, half the
        // memory traffic) win for big systems: 14 vs 22 ms per solve at 16k
        // unknowns (masonry wall)
        let chol = if pb.n < SUPERNODAL_MIN_DOFS {
            SparseCholesky::new_simplicial(&a)?
        } else {
            SparseCholesky::new_supernodal(&a)?
        };
        let nr = pb.b_act.n_rows;
        if debug_enabled() {
            let t = std::time::Instant::now();
            let mut v = vec![1.0; pb.n];
            let mut w = SolveWork::default();
            for _ in 0..20 {
                chol.solve_in_place_work(&mut v, &mut w);
            }
            eprintln!(
                "[admm] n {} nnz(A) {} nnz(L) {} solve {:.3} ms",
                pb.n,
                a.nnz(),
                chol.factor_nnz(),
                t.elapsed().as_secs_f64() * 1e3 / 20.0
            );
        }
        let mut settings = AdmmSettings::default();
        if settings.alpha != 1.0 {
            settings.dr_state = false;
        }
        Ok(Admm {
            settings,
            bt,
            btb,
            base,
            chol,
            rho,
            refactorizations: 0,
            persist_w: Vec::new(),
            z0: vec![0.0; nr],
            y0: vec![0.0; nr],
            b,
            lam,
        })
    }

    fn set_rho(&mut self, rho: f64) -> Result<(), String> {
        let a = self.base.add(1.0, &self.btb, rho);
        self.chol.refactor(&a)?;
        self.rho = rho;
        self.persist_w.clear();
        self.refactorizations += 1;
        Ok(())
    }

    /// Initializes `z = B̂u₀`, `y = 0` from a starting displacement.
    pub fn warm_from(&mut self, _pb: &Problem, u0: &[f64]) {
        self.b.matvec(u0, &mut self.z0);
        self.y0.iter_mut().for_each(|v| *v = 0.0);
    }

    fn state_len(&self, nr: usize) -> usize {
        if self.settings.dr_state { nr } else { 2 * nr }
    }

    /// Packs `(z, y)` into the iteration state.
    fn pack(&self, z: &[f64], y: &[f64], x: &mut [f64]) {
        let nr = z.len();
        if self.settings.dr_state {
            for i in 0..nr {
                x[i] = z[i] + y[i];
            }
        } else {
            x[..nr].copy_from_slice(z);
            x[nr..].copy_from_slice(y);
        }
    }

    fn constraints<'a>(
        &mut self,
        persistent: &[&'a [f64]],
        current: &'a [f64],
        rhs: &'a [f64],
    ) -> Result<Constraints<'a>, String> {
        while self.persist_w.len() < persistent.len() {
            let mut w = persistent[self.persist_w.len()].to_vec();
            self.chol.solve_in_place(&mut w);
            self.persist_w.push(w);
        }
        let mut rows: Vec<&[f64]> = persistent.to_vec();
        rows.push(current);
        let mc = rows.len();
        let mut w_cur = current.to_vec();
        self.chol.solve_in_place(&mut w_cur);
        let mut s = vec![0.0; mc * mc];
        for i in 0..mc {
            for j in 0..mc {
                let wj: &[f64] = if j < persistent.len() {
                    &self.persist_w[j]
                } else {
                    &w_cur
                };
                s[i * mc + j] = dot(rows[i], wj);
            }
        }
        for i in 0..mc {
            for j in i + 1..mc {
                let a = 0.5 * (s[i * mc + j] + s[j * mc + i]);
                s[i * mc + j] = a;
                s[j * mc + i] = a;
            }
        }
        let lu =
            Lu::new(s, mc).map_err(|e| format!("ADMM: dependent equality constraints ({e})"))?;
        Ok(Constraints {
            rows,
            n_persist: persistent.len(),
            w_cur,
            lu,
            rhs,
        })
    }

    /// One ADMM map evaluation `F(x)`, written into `out`.
    fn step(
        &self,
        pb: &Problem,
        cs: &Constraints,
        x: &[f64],
        st: &AdmmSettings,
        out: &mut Step,
        w: &mut Work,
    ) {
        let t0 = std::time::Instant::now();
        self.step_inner(pb, cs, x, st, out, w);
        STEP_NS.with(|c| c.set(c.get() + t0.elapsed().as_nanos() as u64));
    }

    fn step_inner(
        &self,
        pb: &Problem,
        cs: &Constraints,
        x: &[f64],
        st: &AdmmSettings,
        out: &mut Step,
        w: &mut Work,
    ) {
        let n = pb.n;
        let nr = pb.b_act.n_rows;
        let rho = self.rho;
        let alpha = st.alpha;
        let y = &mut w.tmp;
        // current (z, y): z in `out.z_in`, y in `w.tmp`
        if st.dr_state {
            prox(pb, &self.lam, rho, x, &mut out.z_in);
            for i in 0..nr {
                y[i] = x[i] - out.z_in[i];
            }
        } else {
            out.z_in.copy_from_slice(&x[..nr]);
            y.copy_from_slice(&x[nr..]);
        }
        let z = &out.z_in;
        // u-step right-hand side ρB̂ᵀ(z − y) (out.yn as scratch)
        for i in 0..nr {
            out.yn[i] = rho * (z[i] - y[i]);
        }
        let r = &mut w.r;
        self.bt.matvec(&out.yn, r);
        let ts = std::time::Instant::now();
        self.chol.solve_in_place_work(r, &mut w.chol);
        SOLVE_NS.with(|c| c.set(c.get() + ts.elapsed().as_nanos() as u64));
        let mc = cs.rows.len();
        let mut cv = vec![0.0; mc];
        for i in 0..mc {
            cv[i] = dot(cs.rows[i], r) - cs.rhs[i];
        }
        let nu = cs.lu.solve(&cv);
        let u = &mut out.u;
        u.copy_from_slice(r);
        for j in 0..mc {
            let wj: &[f64] = if j < cs.n_persist {
                &self.persist_w[j]
            } else {
                &cs.w_cur
            };
            let c = nu[j];
            for k in 0..n {
                u[k] -= c * wj[k];
            }
        }
        let bu = &mut w.bu;
        self.b.matvec(u, bu);
        // z-/y-steps: v = h + y with h = αB̂u + (1−α)z; z⁺ = prox(v); y⁺ = v − z⁺
        let (zn, yn) = (&mut out.zn, &mut out.yn);
        let (mut rp, mut nbu, mut nzn) = (0.0, 0.0, 0.0);
        for (g, range) in pb.rows_act.iter().enumerate() {
            let thr = self.lam[g] / rho;
            let mut nv = 0.0;
            for i in range.clone() {
                let v = alpha * bu[i] + (1.0 - alpha) * z[i] + y[i];
                yn[i] = v;
                nv += v * v;
            }
            let nv = nv.sqrt();
            let f = if nv > thr { 1.0 - thr / nv } else { 0.0 };
            for i in range.clone() {
                let v = yn[i];
                let zi = f * v;
                zn[i] = zi;
                yn[i] = v - zi;
                let d = bu[i] - zi;
                rp += d * d;
                nbu += bu[i] * bu[i];
                nzn += zi * zi;
            }
        }
        if st.dr_state {
            for i in 0..nr {
                out.fx[i] = zn[i] + yn[i];
            }
        } else {
            out.fx[..nr].copy_from_slice(zn);
            out.fx[nr..].copy_from_slice(yn);
        }
        let scale_p = nbu.sqrt().max(nzn.sqrt());
        out.r_prim = rp.sqrt();
        out.eps_pri = (nr as f64).sqrt() * st.eps_abs + st.eps_rel * scale_p;
        out.scale_p = scale_p;
    }

    /// Dual residual `ρ‖B̂ᵀ(z⁺ − z)‖`, its scale `ρ‖B̂ᵀy⁺‖` and tolerance for
    /// the step `s` (evaluated lazily: only when the primal residual is small
    /// or ρ adaptation is due).
    fn dual(&self, s: &Step, st: &AdmmSettings, w: &mut Work) -> (f64, f64, f64) {
        let nr = s.zn.len();
        let n = self.bt.n_rows;
        for i in 0..nr {
            w.tmp[i] = s.zn[i] - s.z_in[i];
        }
        self.bt.matvec(&w.tmp, &mut w.r);
        let rd = self.rho * norm(&w.r);
        self.bt.matvec(&s.yn, &mut w.r);
        let scale_d = self.rho * norm(&w.r);
        (
            rd,
            scale_d,
            (n as f64).sqrt() * st.eps_abs + st.eps_rel * scale_d,
        )
    }

    /// Solves the subproblem with equality rows `persistent ++ [current]`
    /// (`persistent` may only grow between calls), right-hand side `rhs` and
    /// relative tolerance `eps_rel` (floored at the settings' value).
    pub fn solve(
        &mut self,
        pb: &Problem,
        persistent: &[&[f64]],
        current: &[f64],
        rhs: &[f64],
        eps_rel: f64,
    ) -> Result<SubResult, String> {
        let nr = pb.b_act.n_rows;
        let n = pb.n;
        let mut st = self.settings;
        st.eps_rel = eps_rel.max(st.eps_rel);
        st.eps_abs = st.eps_abs.max(1e-3 * eps_rel);
        let t_start = std::time::Instant::now();
        let mut cs = self.constraints(persistent, current, rhs)?;
        let mem = st.anderson;
        let len = self.state_len(nr);
        let mut work = Work {
            tmp: vec![0.0; nr],
            r: vec![0.0; n],
            bu: vec![0.0; nr],
            chol: SolveWork::default(),
        };
        let mut x = vec![0.0; len];
        self.pack(&self.z0, &self.y0, &mut x);
        let mut s = Step::new(n, nr, len);
        let mut s_new = Step::new(n, nr, len);
        self.step(pb, &cs, &x, &st, &mut s, &mut work);
        let fres = |s: &Step, x: &[f64]| -> f64 {
            let mut a = 0.0;
            for i in 0..x.len() {
                let d = s.fx[i] - x[i];
                a += d * d;
            }
            a.sqrt()
        };
        let mut res_good = fres(&s, &x);
        // Anderson history in a ring buffer: dx_p = x_k − x_{k−1},
        // dg_p = g_k − g_{k−1} (g = F(x) − x), with an incrementally updated
        // Gram matrix of the dg's.
        let mut dx: Vec<Vec<f64>> = vec![vec![0.0; len]; mem];
        let mut dg: Vec<Vec<f64>> = vec![vec![0.0; len]; mem];
        let mut gram = vec![0.0; mem * mem];
        let mut hist = 0usize; // number of valid slots
        let mut head = 0usize; // next slot to write
        let mut prev_x = vec![0.0; len];
        let mut prev_g = vec![0.0; len];
        let mut have_prev = false;
        let mut x_new = vec![0.0; len];
        let mut iters = 1;
        let mut ok = false;
        let mut last_adapt = 0usize;
        let mut n_accel = 0usize;
        let mut n_reject = 0usize;
        loop {
            if s.r_prim <= s.eps_pri {
                let (rd, _, eps_dual) = self.dual(&s, &st, &mut work);
                if rd <= eps_dual {
                    ok = true;
                    break;
                }
            }
            if iters >= st.max_iter {
                break;
            }
            // ρ adaptation (resets the acceleration history)
            if st.adaptive && iters >= last_adapt + 50 && self.refactorizations < 60 {
                let (rd, scale_d, _) = self.dual(&s, &st, &mut work);
                let num = s.r_prim / s.scale_p.max(1e-300);
                let den = rd / scale_d.max(1e-300);
                last_adapt = iters;
                if num > 0.0 && den > 0.0 {
                    let ratio = (num / den).sqrt();
                    if !(0.2..=5.0).contains(&ratio) {
                        let new_rho = (self.rho * ratio).clamp(1e-8, 1e8);
                        // restart from the current state; the scaled dual
                        // y = Y/ρ is rescaled
                        let sc = self.rho / new_rho;
                        s.yn.copy_from_slice(&s.z_in); // scratch: z of x
                        let (zc, yc) = (&mut work.tmp, &mut work.bu);
                        if st.dr_state {
                            for i in 0..nr {
                                zc[i] = s.yn[i];
                                yc[i] = (x[i] - zc[i]) * sc;
                            }
                        } else {
                            for i in 0..nr {
                                zc[i] = x[i];
                                yc[i] = x[nr + i] * sc;
                            }
                        }
                        self.set_rho(new_rho)?;
                        self.pack(&work.tmp, &work.bu, &mut x);
                        cs = self.constraints(persistent, current, rhs)?;
                        hist = 0;
                        head = 0;
                        have_prev = false;
                        self.step(pb, &cs, &x, &st, &mut s, &mut work);
                        res_good = fres(&s, &x);
                        iters += 1;
                        continue;
                    }
                }
            }
            // g = F(x) − x is formed on the fly
            if mem > 0 && have_prev {
                let slot = head;
                {
                    let (dxs, dgs) = (&mut dx[slot], &mut dg[slot]);
                    for i in 0..len {
                        let gi = s.fx[i] - x[i];
                        dxs[i] = x[i] - prev_x[i];
                        dgs[i] = gi - prev_g[i];
                    }
                }
                head = (head + 1) % mem;
                hist = (hist + 1).min(mem);
                for q in 0..hist {
                    let qs = (head + mem - hist + q) % mem;
                    let acc = dot(&dg[slot], &dg[qs]);
                    gram[slot * mem + qs] = acc;
                    gram[qs * mem + slot] = acc;
                }
            }
            for i in 0..len {
                prev_x[i] = x[i];
                prev_g[i] = s.fx[i] - x[i];
            }
            have_prev = true;
            x_new.copy_from_slice(&s.fx);
            let mut accelerated = false;
            if hist > 0 {
                let m = hist;
                let slots: Vec<usize> = (0..m).map(|q| (head + mem - m + q) % mem).collect();
                let mut a = vec![0.0; m * m];
                let mut b = vec![0.0; m];
                for p in 0..m {
                    for q in 0..m {
                        a[p * m + q] = gram[slots[p] * mem + slots[q]];
                    }
                    b[p] = dot(&dg[slots[p]], &prev_g);
                }
                let tr: f64 = (0..m).map(|p| a[p * m + p]).sum();
                for p in 0..m {
                    a[p * m + p] += 1e-10 * tr + 1e-300;
                }
                if dense::cholesky_in_place(&mut a, m).is_ok() {
                    dense::cholesky_solve(&a, m, &mut b);
                    // x_new −= Σ_p γ_p (dx_p + dg_p), fused over p (same
                    // per-entry operation order as p sequential sweeps)
                    let hx: Vec<&[f64]> = slots.iter().map(|&q| dx[q].as_slice()).collect();
                    let hg: Vec<&[f64]> = slots.iter().map(|&q| dg[q].as_slice()).collect();
                    for i in 0..len {
                        let mut acc = x_new[i];
                        for p in 0..m {
                            acc -= b[p] * (hx[p][i] + hg[p][i]);
                        }
                        x_new[i] = acc;
                    }
                    accelerated = x_new.iter().all(|v| v.is_finite());
                    if !accelerated {
                        x_new.copy_from_slice(&s.fx);
                    }
                }
            }
            self.step(pb, &cs, &x_new, &st, &mut s_new, &mut work);
            iters += 1;
            let mut r_new = fres(&s_new, &x_new);
            if accelerated {
                if r_new <= res_good {
                    n_accel += 1;
                } else {
                    // safeguard: fall back to the plain ADMM step
                    n_reject += 1;
                    x_new.copy_from_slice(&s.fx);
                    self.step(pb, &cs, &x_new, &st, &mut s_new, &mut work);
                    iters += 1;
                    r_new = fres(&s_new, &x_new);
                    hist = 0;
                    head = 0;
                    have_prev = false;
                }
            }
            res_good = r_new;
            std::mem::swap(&mut x, &mut x_new);
            std::mem::swap(&mut s, &mut s_new);
        }
        // keep the last evaluated map output as the warm start
        let dbg = if debug_enabled() {
            Some(self.dual(&s, &st, &mut work))
        } else {
            None
        };
        self.z0.copy_from_slice(&s.zn);
        self.y0.copy_from_slice(&s.yn);
        if let Some((rd, _, ed)) = dbg {
            eprintln!(
                "[admm] final rp/eps {:.2} rd/eps {:.2} tol {:.1e}",
                s.r_prim / s.eps_pri,
                rd / ed,
                st.eps_rel
            );
            eprintln!(
                "[admm] iters {iters} ok {ok} rho {:.3e} refactors {} accel {n_accel} reject {n_reject} time {:.3}s n {} rows {nr} step {:.3}s solve {:.3}s",
                self.rho,
                self.refactorizations,
                t_start.elapsed().as_secs_f64(),
                pb.n,
                STEP_NS.with(|c| c.replace(0)) as f64 * 1e-9,
                SOLVE_NS.with(|c| c.replace(0)) as f64 * 1e-9,
            );
        }
        Ok(SubResult {
            u: std::mem::take(&mut s.u),
            iterations: iters,
            ok,
        })
    }
}
