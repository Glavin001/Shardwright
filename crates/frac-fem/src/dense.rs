//! Small dense linear algebra (row-major `Vec<f64>`), used for element
//! matrices, Rayleigh–Ritz projections and Schur complements. Everything is
//! sequential and deterministic.

/// In-place Cholesky `A = L L^T` of a row-major SPD matrix; on success the
/// lower triangle holds `L` (the strict upper triangle is zeroed).
pub fn cholesky_in_place(a: &mut [f64], n: usize) -> Result<(), String> {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if !(d > 0.0) || !d.is_finite() {
            return Err(format!("non-positive pivot {d} at {j}"));
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
        for i in 0..j {
            a[i * n + j] = 0.0;
        }
    }
    Ok(())
}

/// Solves `L L^T x = b` given the Cholesky factor from [`cholesky_in_place`].
pub fn cholesky_solve(l: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= l[k * n + i] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
}

/// LU factorization with partial pivoting (row-major, in place).
pub struct Lu {
    pub n: usize,
    pub a: Vec<f64>,
    pub piv: Vec<usize>,
}

impl Lu {
    pub fn new(mut a: Vec<f64>, n: usize) -> Result<Lu, String> {
        let mut piv: Vec<usize> = (0..n).collect();
        let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
        for k in 0..n {
            let mut p = k;
            let mut best = a[k * n + k].abs();
            for i in k + 1..n {
                let v = a[i * n + k].abs();
                if v > best {
                    best = v;
                    p = i;
                }
            }
            if !(best > 1e-14 * scale) {
                return Err(format!("singular matrix (pivot {best:e} at column {k})"));
            }
            if p != k {
                for j in 0..n {
                    a.swap(k * n + j, p * n + j);
                }
                piv.swap(k, p);
            }
            let d = a[k * n + k];
            for i in k + 1..n {
                let f = a[i * n + k] / d;
                a[i * n + k] = f;
                if f != 0.0 {
                    for j in k + 1..n {
                        a[i * n + j] -= f * a[k * n + j];
                    }
                }
            }
        }
        Ok(Lu { n, a, piv })
    }

    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut x: Vec<f64> = self.piv.iter().map(|&p| b[p]).collect();
        for i in 0..n {
            let mut s = x[i];
            for k in 0..i {
                s -= self.a[i * n + k] * x[k];
            }
            x[i] = s;
        }
        for i in (0..n).rev() {
            let mut s = x[i];
            for k in i + 1..n {
                s -= self.a[i * n + k] * x[k];
            }
            x[i] = s / self.a[i * n + i];
        }
        x
    }
}

/// Inverse of a small square matrix (row-major).
pub fn invert(a: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let lu = Lu::new(a.to_vec(), n)?;
    let mut inv = vec![0.0; n * n];
    let mut e = vec![0.0; n];
    for j in 0..n {
        e.iter_mut().for_each(|v| *v = 0.0);
        e[j] = 1.0;
        let x = lu.solve(&e);
        for i in 0..n {
            inv[i * n + j] = x[i];
        }
    }
    Ok(inv)
}

/// Cyclic Jacobi eigen-decomposition of a symmetric matrix.
/// Returns eigenvalues ascending and eigenvectors as columns of a row-major
/// `n x n` matrix (`v[i*n + j]` = component i of eigenvector j).
pub fn sym_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a.to_vec();
    // symmetrize
    for i in 0..n {
        for j in i + 1..n {
            let m = 0.5 * (a[i * n + j] + a[j * n + i]);
            a[i * n + j] = m;
            a[j * n + i] = m;
        }
    }
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let frob: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
    for _sweep in 0..100 {
        let mut off = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                off += a[i * n + j] * a[i * n + j];
            }
        }
        if off.sqrt() <= 1e-15 * frob {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[i * n + i].total_cmp(&a[j * n + j]).then(i.cmp(&j)));
    let vals: Vec<f64> = order.iter().map(|&i| a[i * n + i]).collect();
    let mut vs = vec![0.0; n * n];
    for (newj, &oldj) in order.iter().enumerate() {
        for i in 0..n {
            vs[i * n + newj] = v[i * n + oldj];
        }
    }
    // deterministic sign: largest-magnitude component positive
    for j in 0..n {
        let mut best = 0;
        for i in 1..n {
            if vs[i * n + j].abs() > vs[best * n + j].abs() {
                best = i;
            }
        }
        if vs[best * n + j] < 0.0 {
            for i in 0..n {
                vs[i * n + j] = -vs[i * n + j];
            }
        }
    }
    (vals, vs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_and_lu() {
        let n = 5;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = 1.0 / (1.0 + i as f64 + j as f64) + if i == j { 2.0 } else { 0.0 };
            }
        }
        let (w, v) = sym_eigen(&a, n);
        for j in 0..n {
            for i in 0..n {
                let mut av = 0.0;
                for k in 0..n {
                    av += a[i * n + k] * v[k * n + j];
                }
                assert!((av - w[j] * v[i * n + j]).abs() < 1e-12);
            }
        }
        let inv = invert(&a, n).unwrap();
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += a[i * n + k] * inv[k * n + j];
                }
                assert!((s - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
        let mut l = a.clone();
        cholesky_in_place(&mut l, n).unwrap();
        let mut b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        cholesky_solve(&l, n, &mut b);
        for i in 0..n {
            let s: f64 = (0..n).map(|k| a[i * n + k] * b[k]).sum();
            assert!((s - (i + 1) as f64).abs() < 1e-12);
        }
    }
}
