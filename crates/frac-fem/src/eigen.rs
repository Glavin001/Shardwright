//! Sparse generalized symmetric eigensolver for the smallest eigenpairs of
//! `K x = λ M x` with a diagonal (lumped) mass `M`.
//!
//! Method: shift-invert block subspace iteration with Rayleigh–Ritz.
//! `(K + σM)` is factorized once with a sparse Cholesky (σ is a tiny positive
//! shift, `1e-7 · tr(K)/tr(M)`, which makes the matrix SPD even when `K` is
//! singular). Each iteration applies `(K+σM)^{-1} M` to a block of `p`
//! vectors, removes components along user-supplied deflation vectors (e.g.
//! the six rigid-body modes of a free body), M-orthonormalizes the block
//! (modified Gram–Schmidt, twice) and performs a Rayleigh–Ritz projection with
//! `K` (dense Jacobi on the `p x p` projected matrix). Starting vectors come
//! from a seeded `ChaCha8Rng`, so results are bitwise reproducible.
//!
//! Small problems (≤ 240 DOFs) use a dense Jacobi solve directly.

use crate::dense;
use crate::sparse::{CsrMatrix, SparseCholesky};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Debug)]
pub struct EigenOptions {
    /// Number of eigenpairs wanted.
    pub n: usize,
    /// Seed for the starting block.
    pub seed: u64,
    /// Relative residual tolerance `‖Kx − λMx‖ / ‖Kx‖`.
    pub tol: f64,
    pub max_iter: usize,
    /// Block size (default `max(2n, n + 8)`).
    pub block: Option<usize>,
}

impl Default for EigenOptions {
    fn default() -> Self {
        EigenOptions {
            n: 6,
            seed: 0x5eed,
            tol: 1e-7,
            max_iter: 400,
            block: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EigenResult {
    /// Eigenvalues ascending.
    pub values: Vec<f64>,
    /// M-orthonormal eigenvectors.
    pub vectors: Vec<Vec<f64>>,
    pub iterations: usize,
    pub converged: bool,
    pub max_residual: f64,
}

fn mdot(a: &[f64], b: &[f64], m: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * m[i] * b[i];
    }
    s
}

fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for i in 0..y.len() {
        y[i] += a * x[i];
    }
}

fn deflate(v: &mut [f64], r: &[Vec<f64>], m: &[f64]) {
    for _ in 0..2 {
        for q in r {
            let c = mdot(q, v, m);
            axpy(v, -c, q);
        }
    }
}

/// M-orthonormalizes `vs` in place (MGS twice). Columns that collapse are
/// replaced by fresh random deflated vectors.
fn m_orthonormalize(vs: &mut [Vec<f64>], r: &[Vec<f64>], m: &[f64], rng: &mut ChaCha8Rng) {
    let n = m.len();
    for j in 0..vs.len() {
        let mut tries = 0;
        loop {
            let n0 = mdot(&vs[j], &vs[j], m).sqrt();
            for _ in 0..2 {
                for i in 0..j {
                    let (head, tail) = vs.split_at_mut(j);
                    let c = mdot(&head[i], &tail[0], m);
                    axpy(&mut tail[0], -c, &head[i]);
                }
            }
            let nn = mdot(&vs[j], &vs[j], m).sqrt();
            if nn > 1e-10 * n0 && nn > 0.0 && nn.is_finite() {
                let inv = 1.0 / nn;
                vs[j].iter_mut().for_each(|x| *x *= inv);
                break;
            }
            tries += 1;
            if tries > 10 {
                break;
            }
            vs[j] = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
            deflate(&mut vs[j], r, m);
        }
    }
}

/// Smallest eigenpairs of `K x = λ M x` (M diagonal, positive). Vectors in
/// `deflation` (any basis; M-orthonormalized internally) are excluded from
/// the search space.
pub fn smallest_eigenpairs(
    k: &CsrMatrix,
    m: &[f64],
    deflation: &[Vec<f64>],
    opts: &EigenOptions,
) -> Result<EigenResult, String> {
    let ndof = k.n_rows;
    assert_eq!(m.len(), ndof);
    if m.iter().any(|&x| !(x > 0.0)) {
        return Err("mass matrix must have positive diagonal".into());
    }
    let mut rng = ChaCha8Rng::seed_from_u64(opts.seed);
    // orthonormal deflation basis
    let mut r: Vec<Vec<f64>> = deflation.to_vec();
    {
        let mut basis: Vec<Vec<f64>> = Vec::new();
        for mut v in r.drain(..) {
            let n0 = mdot(&v, &v, m).sqrt();
            deflate(&mut v, &basis, m);
            let nn = mdot(&v, &v, m).sqrt();
            if nn > 1e-8 * n0 && nn > 0.0 {
                v.iter_mut().for_each(|x| *x /= nn);
                basis.push(v);
            }
        }
        r = basis;
    }
    let avail = ndof.saturating_sub(r.len());
    let nwant = opts.n.min(avail);
    if nwant == 0 {
        return Ok(EigenResult {
            values: vec![],
            vectors: vec![],
            iterations: 0,
            converged: true,
            max_residual: 0.0,
        });
    }
    if ndof <= 240 {
        return dense_eigenpairs(k, m, &r, nwant);
    }
    let p = opts
        .block
        .unwrap_or((2 * nwant).max(nwant + 8))
        .max(nwant)
        .min(avail);
    let trk: f64 = k.diagonal().iter().sum();
    let trm: f64 = m.iter().sum();
    let sigma = 1e-7 * trk / trm;
    let shifted = k.add_diagonal(&m.iter().map(|&x| sigma * x).collect::<Vec<_>>());
    let chol = SparseCholesky::new(&shifted)?;

    let mut x: Vec<Vec<f64>> = (0..p)
        .map(|_| {
            let mut v: Vec<f64> = (0..ndof).map(|_| rng.gen_range(-1.0..1.0)).collect();
            deflate(&mut v, &r, m);
            v
        })
        .collect();
    m_orthonormalize(&mut x, &r, m, &mut rng);
    let mut theta = vec![0.0; p];
    let mut iterations = 0;
    let mut converged = false;
    let mut max_res = f64::INFINITY;
    for it in 0..opts.max_iter {
        iterations = it + 1;
        // Y = (K + σM)^{-1} M X
        let mut y: Vec<Vec<f64>> = x
            .iter()
            .map(|v| (0..ndof).map(|i| m[i] * v[i]).collect())
            .collect();
        chol.solve_many(&mut y);
        for v in y.iter_mut() {
            deflate(v, &r, m);
        }
        m_orthonormalize(&mut y, &r, m, &mut rng);
        // Rayleigh–Ritz
        let ky: Vec<Vec<f64>> = y.iter().map(|v| k.mul(v)).collect();
        let mut kp = vec![0.0; p * p];
        for a in 0..p {
            for b in a..p {
                let mut s = 0.0;
                for i in 0..ndof {
                    s += y[a][i] * ky[b][i];
                }
                kp[a * p + b] = s;
                kp[b * p + a] = s;
            }
        }
        let (w, v) = dense::sym_eigen(&kp, p);
        let mut nx = vec![vec![0.0; ndof]; p];
        let mut nkx = vec![vec![0.0; ndof]; p];
        for j in 0..p {
            for a in 0..p {
                let c = v[a * p + j];
                if c != 0.0 {
                    axpy(&mut nx[j], c, &y[a]);
                    axpy(&mut nkx[j], c, &ky[a]);
                }
            }
        }
        theta = w;
        x = nx;
        // residuals of wanted pairs
        max_res = 0.0f64;
        for j in 0..nwant {
            let mut rn = 0.0;
            let mut kn = 0.0;
            for i in 0..ndof {
                let ri = nkx[j][i] - theta[j] * m[i] * x[j][i];
                rn += ri * ri;
                kn += nkx[j][i] * nkx[j][i];
            }
            let rel = rn.sqrt() / kn.sqrt().max(1e-300);
            max_res = max_res.max(rel);
        }
        if max_res <= opts.tol {
            converged = true;
            break;
        }
    }
    x.truncate(nwant);
    theta.truncate(nwant);
    Ok(EigenResult {
        values: theta,
        vectors: x,
        iterations,
        converged,
        max_residual: max_res,
    })
}

fn dense_eigenpairs(
    k: &CsrMatrix,
    m: &[f64],
    r: &[Vec<f64>],
    nwant: usize,
) -> Result<EigenResult, String> {
    let n = k.n_rows;
    let kd = k.to_dense();
    let s: Vec<f64> = m.iter().map(|&x| 1.0 / x.sqrt()).collect();
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = s[i] * kd[i * n + j] * s[j];
        }
    }
    // deflate: A' = P A P + big * Y Y^T with Y = M^{1/2} R (orthonormal)
    if !r.is_empty() {
        let ys: Vec<Vec<f64>> = r
            .iter()
            .map(|q| (0..n).map(|i| q[i] * m[i].sqrt()).collect())
            .collect();
        let big = 10.0 * a.iter().map(|x| x.abs()).sum::<f64>() + 1.0;
        // P = I - Y Y^T
        let mut pm = vec![0.0; n * n];
        for i in 0..n {
            pm[i * n + i] = 1.0;
        }
        for y in &ys {
            for i in 0..n {
                for j in 0..n {
                    pm[i * n + j] -= y[i] * y[j];
                }
            }
        }
        let mul = |x: &[f64], y: &[f64]| {
            let mut o = vec![0.0; n * n];
            for i in 0..n {
                for kk in 0..n {
                    let xik = x[i * n + kk];
                    if xik != 0.0 {
                        for j in 0..n {
                            o[i * n + j] += xik * y[kk * n + j];
                        }
                    }
                }
            }
            o
        };
        a = mul(&mul(&pm, &a), &pm);
        for y in &ys {
            for i in 0..n {
                for j in 0..n {
                    a[i * n + j] += big * y[i] * y[j];
                }
            }
        }
    }
    let (w, v) = dense::sym_eigen(&a, n);
    let mut vectors = Vec::new();
    for j in 0..nwant {
        vectors.push((0..n).map(|i| v[i * n + j] * s[i]).collect());
    }
    Ok(EigenResult {
        values: w[..nwant].to_vec(),
        vectors,
        iterations: 1,
        converged: true,
        max_residual: 0.0,
    })
}
