//! Linear-elastic constitutive models (Voigt notation).
//!
//! Voigt ordering is `[xx, yy, zz, yz, xz, xy]` with *engineering* shear
//! strains (`γ_yz = 2 ε_yz`), so `σ = C ε` with the 6x6 matrix returned by
//! [`ElasticMaterial::stiffness_voigt`].

use crate::dense;

/// Transversely isotropic material (e.g. wood along the grain).
///
/// `axis` is the longitudinal (grain) direction (need not be normalized).
/// * `e_long`  — Young's modulus along the axis (E_L)
/// * `e_trans` — Young's modulus in the transverse plane (E_T)
/// * `g_long`  — shear modulus for shear in planes containing the axis (G_LT)
/// * `nu_trans` — Poisson ratio in the transverse plane (ν_TT)
/// * `nu_long` — Poisson ratio ν_LT: transverse contraction under longitudinal
///   load, `ε_T = −ν_LT σ_L / E_L`.
#[derive(Clone, Copy, Debug)]
pub struct TransverseIsotropic {
    pub axis: [f64; 3],
    pub e_long: f64,
    pub e_trans: f64,
    pub g_long: f64,
    pub nu_trans: f64,
    pub nu_long: f64,
}

/// Per-tet elastic material. When `transverse` is `Some`, its constants define
/// the stiffness and `youngs`/`poisson` are informational only (e.g. used for
/// scale estimates); `density` is always used for the mass.
#[derive(Clone, Copy, Debug)]
pub struct ElasticMaterial {
    pub youngs: f64,
    pub poisson: f64,
    pub density: f64,
    pub transverse: Option<TransverseIsotropic>,
}

impl ElasticMaterial {
    /// Isotropic material.
    pub fn isotropic(youngs: f64, poisson: f64, density: f64) -> Self {
        ElasticMaterial {
            youngs,
            poisson,
            density,
            transverse: None,
        }
    }

    /// Transversely isotropic material; `youngs`/`poisson` are set to the
    /// longitudinal values for reference.
    pub fn transverse(t: TransverseIsotropic, density: f64) -> Self {
        ElasticMaterial {
            youngs: t.e_long,
            poisson: t.nu_long,
            density,
            transverse: Some(t),
        }
    }

    /// Basic validity check (positive moduli, admissible Poisson ratios,
    /// positive-definite stiffness).
    pub fn validate(&self) -> Result<(), String> {
        if !(self.density > 0.0 && self.density.is_finite()) {
            return Err(format!("density must be positive, got {}", self.density));
        }
        match &self.transverse {
            None => {
                if !(self.youngs > 0.0 && self.youngs.is_finite()) {
                    return Err(format!(
                        "Young's modulus must be positive, got {}",
                        self.youngs
                    ));
                }
                if !(self.poisson > -1.0 && self.poisson < 0.5) {
                    return Err(format!(
                        "Poisson ratio must be in (-1, 0.5), got {}",
                        self.poisson
                    ));
                }
            }
            Some(t) => {
                for (n, v) in [
                    ("e_long", t.e_long),
                    ("e_trans", t.e_trans),
                    ("g_long", t.g_long),
                ] {
                    if !(v > 0.0 && v.is_finite()) {
                        return Err(format!("{n} must be positive, got {v}"));
                    }
                }
                let a = t.axis;
                if !(a[0] * a[0] + a[1] * a[1] + a[2] * a[2] > 0.0) {
                    return Err("transverse axis must be nonzero".into());
                }
            }
        }
        let c = self.stiffness_voigt();
        let mut m = vec![0.0; 36];
        for i in 0..6 {
            for j in 0..6 {
                m[i * 6 + j] = c[i][j];
            }
        }
        if dense::cholesky_in_place(&mut m, 6).is_err() {
            return Err("stiffness matrix is not positive definite".into());
        }
        Ok(())
    }

    /// 6x6 stiffness matrix in global coordinates (Voigt, engineering shear).
    pub fn stiffness_voigt(&self) -> [[f64; 6]; 6] {
        match &self.transverse {
            None => isotropic_c(self.youngs, self.poisson),
            Some(t) => transverse_c(t),
        }
    }

    /// A representative Young's modulus (max of directional moduli).
    pub fn reference_modulus(&self) -> f64 {
        match &self.transverse {
            None => self.youngs,
            Some(t) => t.e_long.max(t.e_trans),
        }
    }
}

/// Isotropic stiffness from Young's modulus and Poisson ratio.
pub fn isotropic_c(e: f64, nu: f64) -> [[f64; 6]; 6] {
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));
    let mut c = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = lambda;
        }
        c[i][i] = lambda + 2.0 * mu;
        c[i + 3][i + 3] = mu;
    }
    c
}

/// Orthonormal frame whose third axis is `axis` (deterministic completion).
pub fn frame_from_axis(axis: [f64; 3]) -> [[f64; 3]; 3] {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let e3 = [axis[0] / n, axis[1] / n, axis[2] / n];
    // pick the coordinate axis least aligned with e3
    let mut k = 0;
    for i in 1..3 {
        if e3[i].abs() < e3[k].abs() {
            k = i;
        }
    }
    let mut h = [0.0; 3];
    h[k] = 1.0;
    let d = dot3(h, e3);
    let mut e1 = [h[0] - d * e3[0], h[1] - d * e3[1], h[2] - d * e3[2]];
    let l = dot3(e1, e1).sqrt();
    for v in &mut e1 {
        *v /= l;
    }
    let e2 = cross3(e3, e1);
    [e1, e2, e3]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

const VOIGT: [[usize; 3]; 3] = [[0, 5, 4], [5, 1, 3], [4, 3, 2]];

/// Rotates a Voigt stiffness defined in the local frame `frame` (rows are the
/// local axes expressed in global coordinates) to global coordinates.
pub fn rotate_stiffness(c_local: &[[f64; 6]; 6], frame: &[[f64; 3]; 3]) -> [[f64; 6]; 6] {
    // R[i][a] = (e_a)_i
    let mut r = [[0.0; 3]; 3];
    for a in 0..3 {
        for i in 0..3 {
            r[i][a] = frame[a][i];
        }
    }
    // tensor form
    let mut t = [0.0f64; 81];
    let idx = |i: usize, j: usize, k: usize, l: usize| ((i * 3 + j) * 3 + k) * 3 + l;
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    t[idx(i, j, k, l)] = c_local[VOIGT[i][j]][VOIGT[k][l]];
                }
            }
        }
    }
    // four successive single-index contractions
    for slot in 0..4 {
        let mut out = [0.0f64; 81];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        let mut s = 0.0;
                        for a in 0..3 {
                            let (ii, jj, kk, ll, rr) = match slot {
                                0 => (a, j, k, l, r[i][a]),
                                1 => (i, a, k, l, r[j][a]),
                                2 => (i, j, a, l, r[k][a]),
                                _ => (i, j, k, a, r[l][a]),
                            };
                            s += rr * t[idx(ii, jj, kk, ll)];
                        }
                        out[idx(i, j, k, l)] = s;
                    }
                }
            }
        }
        t = out;
    }
    let pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];
    let mut c = [[0.0; 6]; 6];
    for (p, &(i, j)) in pairs.iter().enumerate() {
        for (q, &(k, l)) in pairs.iter().enumerate() {
            c[p][q] = t[idx(i, j, k, l)];
        }
    }
    // symmetrize against round-off
    for p in 0..6 {
        for q in p + 1..6 {
            let m = 0.5 * (c[p][q] + c[q][p]);
            c[p][q] = m;
            c[q][p] = m;
        }
    }
    c
}

/// Transversely isotropic stiffness in global coordinates.
pub fn transverse_c(t: &TransverseIsotropic) -> [[f64; 6]; 6] {
    let (el, et, gl, ntt, nlt) = (t.e_long, t.e_trans, t.g_long, t.nu_trans, t.nu_long);
    // local compliance, axis 3 = grain
    let mut s = vec![0.0; 36];
    let set = |s: &mut Vec<f64>, i: usize, j: usize, v: f64| {
        s[i * 6 + j] = v;
        s[j * 6 + i] = v;
    };
    set(&mut s, 0, 0, 1.0 / et);
    set(&mut s, 1, 1, 1.0 / et);
    set(&mut s, 2, 2, 1.0 / el);
    set(&mut s, 0, 1, -ntt / et);
    set(&mut s, 0, 2, -nlt / el);
    set(&mut s, 1, 2, -nlt / el);
    set(&mut s, 3, 3, 1.0 / gl);
    set(&mut s, 4, 4, 1.0 / gl);
    set(&mut s, 5, 5, 2.0 * (1.0 + ntt) / et);
    let inv = dense::invert(&s, 6).expect("singular transversely isotropic compliance");
    let mut cl = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            cl[i][j] = inv[i * 6 + j];
        }
    }
    rotate_stiffness(&cl, &frame_from_axis(t.axis))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transverse_reduces_to_isotropic() {
        let (e, nu) = (2.0e9, 0.3);
        let t = TransverseIsotropic {
            axis: [0.3, -0.5, 0.8],
            e_long: e,
            e_trans: e,
            g_long: e / (2.0 * (1.0 + nu)),
            nu_trans: nu,
            nu_long: nu,
        };
        let a = transverse_c(&t);
        let b = isotropic_c(e, nu);
        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (a[i][j] - b[i][j]).abs() < 1e-6 * e,
                    "{i} {j} {} {}",
                    a[i][j],
                    b[i][j]
                );
            }
        }
    }

    #[test]
    fn transverse_axis_modulus() {
        // Uniaxial stress along the grain axis gives strain sigma/E_L along it.
        let t = TransverseIsotropic {
            axis: [1.0, 1.0, 0.0],
            e_long: 12e9,
            e_trans: 0.8e9,
            g_long: 0.7e9,
            nu_trans: 0.4,
            nu_long: 0.35,
        };
        let c = transverse_c(&t);
        let mut cm = vec![0.0; 36];
        for i in 0..6 {
            for j in 0..6 {
                cm[i * 6 + j] = c[i][j];
            }
        }
        let s = dense::invert(&cm, 6).unwrap();
        // sigma = n n^T with n = (1,1,0)/sqrt2 -> voigt [0.5,0.5,0,0,0,0.5]
        let sig = [0.5, 0.5, 0.0, 0.0, 0.0, 0.5];
        let mut eps = [0.0; 6];
        for i in 0..6 {
            for j in 0..6 {
                eps[i] += s[i * 6 + j] * sig[j];
            }
        }
        // normal strain along n: n^T eps n = 0.5*(exx + eyy) + 0.5*gxy
        let en = 0.5 * (eps[0] + eps[1]) + 0.5 * eps[5];
        assert!((en * 12e9 - 1.0).abs() < 1e-9, "{en}");
        let m = ElasticMaterial::transverse(t, 500.0);
        m.validate().unwrap();
    }
}
