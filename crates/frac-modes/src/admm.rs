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
//! The ADMM map `F: (z, y) → (z⁺, y⁺)` is accelerated with type-II Anderson
//! acceleration (memory 6) with a residual safeguard (in the spirit of
//! Zhang et al. 2019, "Accelerating ADMM for efficient simulation and
//! optimization"): an accelerated iterate is accepted only if its fixed-point
//! residual `‖F(x) − x‖` does not exceed the last accepted one; otherwise the
//! plain ADMM step is taken and the history is cleared.
//!
//! Stopping (Boyd et al. 2011 §3.3): `‖B̂u − z‖ ≤ √m ε_abs + ε_rel max(‖B̂u‖, ‖z‖)`
//! and `ρ‖B̂ᵀ(z⁺ − z)‖ ≤ √n ε_abs + ε_rel ρ‖B̂ᵀy‖`. ρ is adapted with the
//! OSQP residual-balancing rule (only when the suggested change exceeds 5x,
//! to limit refactorizations). `(z, y)` are warm-started across ICCM
//! iterations of the same mode. Everything is sequential and deterministic.

use crate::problem::{Problem, DELTA};
use crate::SubResult;
use frac_fem::dense::{self, Lu};
use frac_fem::sparse::{CsrMatrix, SparseCholesky};

#[derive(Clone, Copy, Debug)]
pub(crate) struct AdmmSettings {
    pub eps_abs: f64,
    pub eps_rel: f64,
    pub max_iter: usize,
    pub alpha: f64,
    pub adaptive: bool,
    /// Anderson memory (0 disables acceleration).
    pub anderson: usize,
}

impl Default for AdmmSettings {
    fn default() -> Self {
        AdmmSettings { eps_abs: 1e-9, eps_rel: 1e-6, max_iter: 20_000, alpha: 1.0, adaptive: true, anderson: 6 }
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
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
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
    /// Warm-start state `x = [z; y]` (length 2·rows).
    x: Vec<f64>,
}

/// Equality-constraint data for one `solve` call.
struct Constraints<'a> {
    rows: Vec<&'a [f64]>,
    n_persist: usize,
    w_cur: Vec<f64>,
    lu: Lu,
    rhs: &'a [f64],
}

/// Result of one ADMM map evaluation.
struct Step {
    fx: Vec<f64>,
    u: Vec<f64>,
    r_prim: f64,
    eps_pri: f64,
    scale_p: f64,
}

impl Admm {
    pub fn new(pb: &Problem) -> Result<Admm, String> {
        Self::with_rho(pb, None)
    }

    pub fn with_rho(pb: &Problem, rho0: Option<f64>) -> Result<Admm, String> {
        let bt = pb.b_act.transpose();
        let btb = pb.b_act.ata(None);
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
        let rho = rho0.unwrap_or(if sb > 0.0 { (0.03 * sq / sb).clamp(1e-6, 1e6) } else { 1.0 });
        let a = base.add(1.0, &btb, rho);
        // simplicial factors have lean, allocation-free solves (the u-step is
        // solved thousands of times); supernodal pays off for big systems
        let chol = if pb.n < 20_000 { SparseCholesky::new_simplicial(&a)? } else { SparseCholesky::new(&a)? };
        let nr = pb.b_act.n_rows;
        if debug_enabled() {
            let t = std::time::Instant::now();
            let mut v = vec![1.0; pb.n];
            for _ in 0..20 {
                chol.solve_in_place(&mut v);
            }
            eprintln!(
                "[admm] n {} nnz(A) {} nnz(L) {} solve {:.3} ms",
                pb.n,
                a.nnz(),
                chol.factor_nnz(),
                t.elapsed().as_secs_f64() * 1e3 / 20.0
            );
        }
        Ok(Admm {
            settings: AdmmSettings::default(),
            bt,
            btb,
            base,
            chol,
            rho,
            refactorizations: 0,
            persist_w: Vec::new(),
            x: vec![0.0; 2 * nr],
        })
    }

    fn set_rho(&mut self, rho: f64, x: &mut [f64], nr: usize) -> Result<(), String> {
        let a = self.base.add(1.0, &self.btb, rho);
        self.chol.refactor(&a)?;
        let s = self.rho / rho;
        x[nr..].iter_mut().for_each(|v| *v *= s);
        self.rho = rho;
        self.persist_w.clear();
        self.refactorizations += 1;
        Ok(())
    }

    /// Initializes `z = B̂u₀`, `y = 0` from a starting displacement.
    pub fn warm_from(&mut self, pb: &Problem, u0: &[f64]) {
        let nr = pb.b_act.n_rows;
        pb.b_act.matvec(u0, &mut self.x[..nr]);
        self.x[nr..].iter_mut().for_each(|v| *v = 0.0);
    }

    fn constraints<'a>(
        &mut self,
        persistent: &[&'a [f64]],
        current: &'a [f64],
        rhs: &'a [f64],
    ) -> Result<Constraints<'a>, String> {
        let n = current.len();
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
                let wj: &[f64] = if j < persistent.len() { &self.persist_w[j] } else { &w_cur };
                let mut acc = 0.0;
                for k in 0..n {
                    acc += rows[i][k] * wj[k];
                }
                s[i * mc + j] = acc;
            }
        }
        for i in 0..mc {
            for j in i + 1..mc {
                let a = 0.5 * (s[i * mc + j] + s[j * mc + i]);
                s[i * mc + j] = a;
                s[j * mc + i] = a;
            }
        }
        let lu = Lu::new(s, mc).map_err(|e| format!("ADMM: dependent equality constraints ({e})"))?;
        Ok(Constraints { rows, n_persist: persistent.len(), w_cur, lu, rhs })
    }

    /// One ADMM map evaluation `F(x)` with `x = [z; y]`.
    fn step(&self, pb: &Problem, cs: &Constraints, x: &[f64], st: &AdmmSettings) -> Step {
        let t0 = std::time::Instant::now();
        let r = self.step_inner(pb, cs, x, st);
        STEP_NS.with(|c| c.set(c.get() + t0.elapsed().as_nanos() as u64));
        r
    }

    fn step_inner(&self, pb: &Problem, cs: &Constraints, x: &[f64], st: &AdmmSettings) -> Step {
        let n = pb.n;
        let nr = pb.b_act.n_rows;
        let (z, y) = x.split_at(nr);
        let rho = self.rho;
        let alpha = st.alpha;
        let mut tmp: Vec<f64> = (0..nr).map(|i| z[i] - y[i]).collect();
        let mut r = vec![0.0; n];
        self.bt.matvec(&tmp, &mut r);
        r.iter_mut().for_each(|v| *v *= rho);
        let ts = std::time::Instant::now();
        self.chol.solve_in_place(&mut r);
        SOLVE_NS.with(|c| c.set(c.get() + ts.elapsed().as_nanos() as u64));
        let mc = cs.rows.len();
        let mut cv = vec![0.0; mc];
        for i in 0..mc {
            let mut acc = 0.0;
            for k in 0..n {
                acc += cs.rows[i][k] * r[k];
            }
            cv[i] = acc - cs.rhs[i];
        }
        let nu = cs.lu.solve(&cv);
        let mut u = r;
        for j in 0..mc {
            let wj: &[f64] = if j < cs.n_persist { &self.persist_w[j] } else { &cs.w_cur };
            let c = nu[j];
            for k in 0..n {
                u[k] -= c * wj[k];
            }
        }
        let mut bu = vec![0.0; nr];
        pb.b_act.matvec(&u, &mut bu);
        let mut fx = vec![0.0; 2 * nr];
        {
            let (zn, yn) = fx.split_at_mut(nr);
            for (g, range) in pb.rows_act.iter().enumerate() {
                let thr = pb.lam_act[g] / rho;
                let mut nv = 0.0;
                for i in range.clone() {
                    let h = alpha * bu[i] + (1.0 - alpha) * z[i];
                    tmp[i] = h;
                    let v = h + y[i];
                    nv += v * v;
                }
                let nv = nv.sqrt();
                let f = if nv > thr { 1.0 - thr / nv } else { 0.0 };
                for i in range.clone() {
                    zn[i] = f * (tmp[i] + y[i]);
                }
            }
            for i in 0..nr {
                yn[i] = y[i] + tmp[i] - zn[i];
            }
        }
        let (zn, yn) = fx.split_at(nr);
        let mut rp = 0.0;
        for i in 0..nr {
            let d = bu[i] - zn[i];
            rp += d * d;
        }
        let _ = yn;
        let scale_p = norm(&bu).max(norm(zn));
        Step {
            fx,
            u,
            r_prim: rp.sqrt(),
            eps_pri: (nr as f64).sqrt() * st.eps_abs + st.eps_rel * scale_p,
            scale_p,
        }
    }

    /// Dual residual `ρ‖B̂ᵀ(z⁺ − z)‖`, its scale `ρ‖B̂ᵀy⁺‖` and tolerance for
    /// the step `x → s.fx` (evaluated lazily: only when the primal residual
    /// is small or ρ adaptation is due).
    fn dual(&self, x: &[f64], s: &Step, st: &AdmmSettings) -> (f64, f64, f64) {
        let nr = x.len() / 2;
        let n = self.bt.n_rows;
        let dz: Vec<f64> = (0..nr).map(|i| s.fx[i] - x[i]).collect();
        let mut w = vec![0.0; n];
        self.bt.matvec(&dz, &mut w);
        let rd = self.rho * norm(&w);
        self.bt.matvec(&s.fx[nr..], &mut w);
        let scale_d = self.rho * norm(&w);
        (rd, scale_d, (n as f64).sqrt() * st.eps_abs + st.eps_rel * scale_d)
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
        let mut st = self.settings;
        st.eps_rel = eps_rel.max(st.eps_rel);
        st.eps_abs = st.eps_abs.max(1e-3 * eps_rel);
        let t_start = std::time::Instant::now();
        let mut cs = self.constraints(persistent, current, rhs)?;
        let mem = st.anderson;
        let mut x = std::mem::take(&mut self.x);
        let mut s = self.step(pb, &cs, &x, &st);
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
        let len = x.len();
        let mut dx: Vec<Vec<f64>> = vec![vec![0.0; len]; mem];
        let mut dg: Vec<Vec<f64>> = vec![vec![0.0; len]; mem];
        let mut gram = vec![0.0; mem * mem];
        let mut hist = 0usize; // number of valid slots
        let mut head = 0usize; // next slot to write
        let mut prev_x = vec![0.0; len];
        let mut prev_g = vec![0.0; len];
        let mut have_prev = false;
        let mut g = vec![0.0; len];
        let mut iters = 1;
        let mut ok = false;
        let mut last_adapt = 0usize;
        let mut n_accel = 0usize;
        let mut n_reject = 0usize;
        loop {
            if s.r_prim <= s.eps_pri {
                let (rd, _, eps_dual) = self.dual(&x, &s, &st);
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
                let (rd, scale_d, _) = self.dual(&x, &s, &st);
                let num = s.r_prim / s.scale_p.max(1e-300);
                let den = rd / scale_d.max(1e-300);
                last_adapt = iters;
                if num > 0.0 && den > 0.0 {
                    let ratio = (num / den).sqrt();
                    if !(0.2..=5.0).contains(&ratio) {
                        let new_rho = (self.rho * ratio).clamp(1e-8, 1e8);
                        self.set_rho(new_rho, &mut x, nr)?;
                        cs = self.constraints(persistent, current, rhs)?;
                        hist = 0;
                        head = 0;
                        have_prev = false;
                        s = self.step(pb, &cs, &x, &st);
                        res_good = fres(&s, &x);
                        iters += 1;
                        continue;
                    }
                }
            }
            for i in 0..len {
                g[i] = s.fx[i] - x[i];
            }
            if mem > 0 && have_prev {
                let slot = head;
                for i in 0..len {
                    dx[slot][i] = x[i] - prev_x[i];
                    dg[slot][i] = g[i] - prev_g[i];
                }
                head = (head + 1) % mem;
                hist = (hist + 1).min(mem);
                for q in 0..hist {
                    let qs = (head + mem - hist + q) % mem;
                    let mut acc = 0.0;
                    for i in 0..len {
                        acc += dg[slot][i] * dg[qs][i];
                    }
                    gram[slot * mem + qs] = acc;
                    gram[qs * mem + slot] = acc;
                }
            }
            prev_x.copy_from_slice(&x);
            prev_g.copy_from_slice(&g);
            have_prev = true;
            let mut x_new = s.fx.clone();
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
                    let mut acc = 0.0;
                    for i in 0..len {
                        acc += dg[slots[p]][i] * g[i];
                    }
                    b[p] = acc;
                }
                let tr: f64 = (0..m).map(|p| a[p * m + p]).sum();
                for p in 0..m {
                    a[p * m + p] += 1e-10 * tr + 1e-300;
                }
                if dense::cholesky_in_place(&mut a, m).is_ok() {
                    dense::cholesky_solve(&a, m, &mut b);
                    for p in 0..m {
                        let gp = b[p];
                        let (dxp, dgp) = (&dx[slots[p]], &dg[slots[p]]);
                        for i in 0..len {
                            x_new[i] -= gp * (dxp[i] + dgp[i]);
                        }
                    }
                    accelerated = x_new.iter().all(|v| v.is_finite());
                    if !accelerated {
                        x_new.copy_from_slice(&s.fx);
                    }
                }
            }
            let mut s_new = self.step(pb, &cs, &x_new, &st);
            iters += 1;
            let mut r_new = fres(&s_new, &x_new);
            if accelerated {
                if r_new <= res_good {
                    n_accel += 1;
                } else {
                    // safeguard: fall back to the plain ADMM step
                    n_reject += 1;
                    x_new.copy_from_slice(&s.fx);
                    s_new = self.step(pb, &cs, &x_new, &st);
                    iters += 1;
                    r_new = fres(&s_new, &x_new);
                    hist = 0;
                    head = 0;
                    have_prev = false;
                }
            }
            res_good = r_new;
            x = x_new;
            s = s_new;
        }
        // keep the last evaluated map output as the warm start
        let dbg = if debug_enabled() { Some(self.dual(&x, &s, &st)) } else { None };
        self.x = s.fx;
        if let Some((rd, _, ed)) = dbg {
            eprintln!("[admm] final rp/eps {:.2} rd/eps {:.2} tol {:.1e}", s.r_prim / s.eps_pri, rd / ed, st.eps_rel);
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
        Ok(SubResult { u: s.u, iterations: iters, ok })
    }
}
