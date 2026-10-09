//! P1 (linear) tetrahedral elements.

use crate::mesh::{det3, sub};

/// Shape-function gradients of a linear tet and its signed volume.
/// Returns `None` for degenerate tets.
pub fn tet_gradients(p: &[[f64; 3]; 4]) -> Option<([[f64; 3]; 4], f64)> {
    let e1 = sub(p[1], p[0]);
    let e2 = sub(p[2], p[0]);
    let e3 = sub(p[3], p[0]);
    let det = det3(e1, e2, e3);
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    // inverse of D = [e1 e2 e3] (columns): rows of D^{-1} are (e2 x e3, e3 x e1, e1 x e2)/det
    let cr = |a: [f64; 3], b: [f64; 3]| {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    };
    let r1 = cr(e2, e3);
    let r2 = cr(e3, e1);
    let r3 = cr(e1, e2);
    let mut g = [[0.0; 3]; 4];
    for k in 0..3 {
        g[1][k] = r1[k] / det;
        g[2][k] = r2[k] / det;
        g[3][k] = r3[k] / det;
        g[0][k] = -(g[1][k] + g[2][k] + g[3][k]);
    }
    Some((g, det / 6.0))
}

/// Strain-displacement matrix (6x12, Voigt engineering strains).
pub fn strain_matrix(g: &[[f64; 3]; 4]) -> [[f64; 12]; 6] {
    let mut b = [[0.0; 12]; 6];
    for a in 0..4 {
        let [gx, gy, gz] = g[a];
        let c = 3 * a;
        b[0][c] = gx;
        b[1][c + 1] = gy;
        b[2][c + 2] = gz;
        b[3][c + 1] = gz;
        b[3][c + 2] = gy;
        b[4][c] = gz;
        b[4][c + 2] = gx;
        b[5][c] = gy;
        b[5][c + 1] = gx;
    }
    b
}

/// 12x12 element stiffness `K = |V| B^T C B` (DOF order: node-major, xyz).
pub fn tet_stiffness(p: &[[f64; 3]; 4], c: &[[f64; 6]; 6]) -> [[f64; 12]; 12] {
    let mut k = [[0.0; 12]; 12];
    let Some((g, vol)) = tet_gradients(p) else {
        return k;
    };
    let vol = vol.abs();
    let b = strain_matrix(&g);
    // CB (6x12)
    let mut cb = [[0.0; 12]; 6];
    for i in 0..6 {
        for j in 0..12 {
            let mut s = 0.0;
            for m in 0..6 {
                s += c[i][m] * b[m][j];
            }
            cb[i][j] = s;
        }
    }
    for i in 0..12 {
        for j in i..12 {
            let mut s = 0.0;
            for m in 0..6 {
                s += b[m][i] * cb[m][j];
            }
            k[i][j] = s * vol;
            k[j][i] = s * vol;
        }
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::isotropic_c;

    #[test]
    fn rigid_modes_in_kernel() {
        let p = [[0.1, 0.0, 0.0], [1.0, 0.2, 0.0], [0.0, 1.0, 0.1], [0.2, 0.1, 1.3]];
        let k = tet_stiffness(&p, &isotropic_c(1.0, 0.3));
        // translation x and rotation about z
        let mut tx = [0.0; 12];
        let mut rz = [0.0; 12];
        for a in 0..4 {
            tx[3 * a] = 1.0;
            rz[3 * a] = -p[a][1];
            rz[3 * a + 1] = p[a][0];
        }
        for v in [tx, rz] {
            for i in 0..12 {
                let s: f64 = (0..12).map(|j| k[i][j] * v[j]).sum();
                assert!(s.abs() < 1e-12, "{s}");
            }
        }
        // uniform strain energy check: u = (x, 0, 0) -> eps_xx = 1, energy = 0.5*C11*V
        let mut u = [0.0; 12];
        for a in 0..4 {
            u[3 * a] = p[a][0];
        }
        let mut e = 0.0;
        for i in 0..12 {
            for j in 0..12 {
                e += 0.5 * u[i] * k[i][j] * u[j];
            }
        }
        let (_, vol) = tet_gradients(&p).unwrap();
        let c11 = isotropic_c(1.0, 0.3)[0][0];
        assert!((e - 0.5 * c11 * vol).abs() < 1e-12);
    }
}
