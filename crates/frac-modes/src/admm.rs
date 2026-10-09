//! Fast solver: ADMM (group-lasso splitting) for the ICCM subproblem
//!
//! ```text
//! min ½ uᵀ(Q̂+δM̂)u + Σ_g λ_g ‖z_g‖   s.t.  B̂u − z = 0,  C u = d
//! ```
//!
//! Scaled-form ADMM with over-relaxation (α = 1.6):
//!
//! * u-step: `(Q̂ + ρB̂ᵀB̂ + δM̂) u + Cᵀν = ρB̂ᵀ(z − y)`, `C u = d`. The SPD
//!   matrix `A = Q̂ + ρB̂ᵀB̂ + δM̂` is factorized with faer's sparse Cholesky
//!   once (and refactorized numerically only when ρ is adapted — the factor
//!   is shared by all modes and ICCM iterations). The few dense equality rows
//!   `C` (≤ 6 rigid + k mode rows) are handled with a small Schur complement
//!   `S = C A⁻¹ Cᵀ`; `A⁻¹Cᵀ` is cached for the persistent rows (rigid modes
//!   and finished modes) and recomputed once per call for the `c` row.
//! * z-step: block soft-thresholding `z_g = max(0, 1 − λ_g/(ρ‖v_g‖)) v_g`.
//! * y-step: `y += h − z` with `h = αB̂u + (1−α)z`.
//!
//! Stopping (Boyd et al. 2011 §3.3): `‖B̂u − z‖ ≤ √m ε_abs + ε_rel max(‖B̂u‖, ‖z‖)`
//! and `ρ‖B̂ᵀ(z − z_prev)‖ ≤ √n ε_abs + ε_rel ρ‖B̂ᵀy‖`.
//! ρ is adapted with the OSQP residual-balancing rule (only when the
//! suggested change exceeds 5x, to limit refactorizations).
//! `z`, `y` are warm-started across ICCM iterations of the same mode.

use crate::problem::{mdot, Problem, DELTA};
use crate::SubResult;
use frac_fem::dense::Lu;
use frac_fem::sparse::{CsrMatrix, SparseCholesky};

pub(crate) struct AdmmSettings {
    pub eps_abs: f64,
    pub eps_rel: f64,
    pub max_iter: usize,
    pub alpha: f64,
}

impl Default for AdmmSettings {
    fn default() -> Self {
        AdmmSettings { eps_abs: 1e-9, eps_rel: 1e-6, max_iter: 20_000, alpha: 1.6 }
    }
}

pub(crate) struct Admm {
    pub settings: AdmmSettings,
    bt: CsrMatrix,
    btb: CsrMatrix,
    base: CsrMatrix, // Q̂ + δM̂ with the union pattern of BᵀB
    a: CsrMatrix,
    chol: SparseCholesky,
    pub rho: f64,
    pub refactorizations: usize,
    persist_w: Vec<Vec<f64>>,
    pub z: Vec<f64>,
    pub y: Vec<f64>,
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

impl Admm {
    pub fn new(pb: &Problem) -> Result<Admm, String> {
        let bt = pb.b_act.transpose();
        let btb = pb.b_act.ata(None);
        let dm: Vec<f64> = pb.m.iter().map(|&x| DELTA * x).collect();
        // base = Q + δM, padded with the BᵀB pattern (zeros) so A keeps one pattern
        let base = pb.q.add_diagonal(&dm).add(1.0, &btb, 0.0);
        // initial ρ: balance the diagonals of Q̂ and B̂ᵀB̂ on DOFs touched by B
        let qd = pb.q.diagonal();
        let bd = btb.diagonal();
        let (mut sq, mut sb) = (0.0, 0.0);
        for i in 0..pb.n {
            if bd[i] > 0.0 {
                sq += qd[i];
                sb += bd[i];
            }
        }
        let rho = if sb > 0.0 { (0.1 * sq / sb).clamp(1e-6, 1e6) } else { 1.0 };
        let a = base.add(1.0, &btb, rho);
        let chol = SparseCholesky::new(&a)?;
        let nr = pb.b_act.n_rows;
        Ok(Admm {
            settings: AdmmSettings::default(),
            bt,
            btb,
            base,
            a,
            chol,
            rho,
            refactorizations: 0,
            persist_w: Vec::new(),
            z: vec![0.0; nr],
            y: vec![0.0; nr],
        })
    }

    pub fn reset_warm_start(&mut self) {
        self.z.iter_mut().for_each(|v| *v = 0.0);
        self.y.iter_mut().for_each(|v| *v = 0.0);
    }

    fn set_rho(&mut self, rho: f64) -> Result<(), String> {
        let a = self.base.add(1.0, &self.btb, rho);
        self.chol.refactor(&a)?;
        self.a = a;
        let s = self.rho / rho;
        self.y.iter_mut().for_each(|v| *v *= s);
        self.rho = rho;
        self.persist_w.clear();
        self.refactorizations += 1;
        Ok(())
    }

    /// Initializes `z = B̂u₀`, `y = 0` from a starting displacement.
    pub fn warm_from(&mut self, pb: &Problem, u0: &[f64]) {
        pb.b_act.matvec(u0, &mut self.z);
        self.y.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Solves the subproblem with equality rows `persistent ++ [current]`
    /// (`persistent` may only grow between calls) and right-hand side `rhs`.
    pub fn solve(
        &mut self,
        pb: &Problem,
        persistent: &[&[f64]],
        current: &[f64],
        rhs: &[f64],
    ) -> Result<SubResult, String> {
        let n = pb.n;
        let nr = pb.b_act.n_rows;
        let st = AdmmSettings { ..self.settings };
        let mut rows: Vec<&[f64]> = persistent.to_vec();
        rows.push(current);
        let mc = rows.len();
        // Schur complement data
        let mut w_cur: Vec<f64>;
        let mut lu: Lu;
        macro_rules! schur {
            () => {{
                while self.persist_w.len() < persistent.len() {
                    let mut w = persistent[self.persist_w.len()].to_vec();
                    self.chol.solve_in_place(&mut w);
                    self.persist_w.push(w);
                }
                w_cur = current.to_vec();
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
                lu = Lu::new(s, mc).map_err(|e| format!("ADMM: dependent equality constraints ({e})"))?;
            }};
        }
        schur!();
        let mut u = vec![0.0; n];
        let mut bu = vec![0.0; nr];
        let mut r = vec![0.0; n];
        let mut tmp = vec![0.0; nr];
        let mut z_old = vec![0.0; nr];
        let mut iters = 0;
        let mut ok = false;
        let alpha = st.alpha;
        let check_every = 5;
        let mut last_adapt = 0usize;
        for it in 0..st.max_iter {
            iters = it + 1;
            // u-step
            for i in 0..nr {
                tmp[i] = self.z[i] - self.y[i];
            }
            self.bt.matvec(&tmp, &mut r);
            let rho = self.rho;
            r.iter_mut().for_each(|v| *v *= rho);
            self.chol.solve_in_place(&mut r); // r = A^{-1} rhs
            let mut cv = vec![0.0; mc];
            for i in 0..mc {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += rows[i][k] * r[k];
                }
                cv[i] = acc - rhs[i];
            }
            let nu = lu.solve(&cv);
            u.copy_from_slice(&r);
            for j in 0..mc {
                let wj: &[f64] = if j < persistent.len() { &self.persist_w[j] } else { &w_cur };
                let c = nu[j];
                for k in 0..n {
                    u[k] -= c * wj[k];
                }
            }
            // z-step
            pb.b_act.matvec(&u, &mut bu);
            z_old.copy_from_slice(&self.z);
            for (g, range) in pb.rows_act.iter().enumerate() {
                let thr = pb.lam_act[g] / rho;
                let mut nv = 0.0;
                for i in range.clone() {
                    let h = alpha * bu[i] + (1.0 - alpha) * z_old[i];
                    tmp[i] = h;
                    let v = h + self.y[i];
                    nv += v * v;
                }
                let nv = nv.sqrt();
                let f = if nv > thr { 1.0 - thr / nv } else { 0.0 };
                for i in range.clone() {
                    self.z[i] = f * (tmp[i] + self.y[i]);
                }
            }
            for i in 0..nr {
                self.y[i] += tmp[i] - self.z[i];
            }
            if it % check_every == check_every - 1 || it + 1 == st.max_iter {
                let mut rp = 0.0;
                for i in 0..nr {
                    let d = bu[i] - self.z[i];
                    rp += d * d;
                }
                let rp = rp.sqrt();
                for i in 0..nr {
                    tmp[i] = self.z[i] - z_old[i];
                }
                let mut dz = vec![0.0; n];
                self.bt.matvec(&tmp, &mut dz);
                let rd = rho * norm(&dz);
                let mut bty = vec![0.0; n];
                self.bt.matvec(&self.y, &mut bty);
                let bty_n = rho * norm(&bty);
                let scale_p = norm(&bu).max(norm(&self.z));
                let eps_pri = (nr as f64).sqrt() * st.eps_abs + st.eps_rel * scale_p;
                let eps_dual = (n as f64).sqrt() * st.eps_abs + st.eps_rel * bty_n;
                if rp <= eps_pri && rd <= eps_dual {
                    ok = true;
                    break;
                }
                // OSQP-style rho adaptation
                if it >= last_adapt + 50 && self.refactorizations < 60 {
                    let num = rp / scale_p.max(1e-300);
                    let den = rd / bty_n.max(1e-300);
                    if num > 0.0 && den > 0.0 {
                        let ratio = (num / den).sqrt();
                        if !(0.2..=5.0).contains(&ratio) {
                            let new_rho = (self.rho * ratio).clamp(1e-8, 1e8);
                            self.set_rho(new_rho)?;
                            schur!();
                            last_adapt = it;
                        }
                    }
                }
            }
        }
        let _ = mdot;
        Ok(SubResult { u, iterations: iters, ok })
    }
}
