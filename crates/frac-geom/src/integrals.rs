//! Exact polyhedral mass properties via the divergence theorem.

use crate::polygon::outer;
use glam::{DMat3, DVec3};
use serde::{Deserialize, Serialize};

/// Volume integrals of a closed polyhedron (unit density).
#[derive(Clone, Copy, Debug)]
pub struct VolumeIntegrals {
    pub volume: f64,
    /// ∫ x dV
    pub first: DVec3,
    /// ∫ x xᵀ dV
    pub second: DMat3,
}

impl Default for VolumeIntegrals {
    // NB: glam's `DMat3::default()` is the identity, so spell out zero.
    fn default() -> Self {
        VolumeIntegrals {
            volume: 0.0,
            first: DVec3::ZERO,
            second: DMat3::ZERO,
        }
    }
}

impl VolumeIntegrals {
    pub fn add(&mut self, o: &VolumeIntegrals) {
        self.volume += o.volume;
        self.first += o.first;
        self.second += o.second;
    }
    pub fn scaled(&self, s: f64) -> VolumeIntegrals {
        VolumeIntegrals {
            volume: self.volume * s,
            first: self.first * s,
            second: self.second * s,
        }
    }
    /// Accumulate the signed tetrahedron (r, p0, p1, p2).
    #[inline]
    pub fn add_tet(&mut self, r: DVec3, p0: DVec3, p1: DVec3, p2: DVec3) {
        let v = (p0 - r).dot((p1 - r).cross(p2 - r)) / 6.0;
        if v == 0.0 {
            return;
        }
        self.volume += v;
        let s = r + p0 + p1 + p2;
        self.first += s * (v / 4.0);
        let m = outer(r, r) + outer(p0, p0) + outer(p1, p1) + outer(p2, p2) + outer(s, s);
        self.second += m * (v / 20.0);
    }
    /// Accumulate a closed oriented polygon face (fan triangulated, signed).
    pub fn add_polygon(&mut self, r: DVec3, poly: &[DVec3]) {
        if poly.len() < 3 {
            return;
        }
        let a = poly[0];
        for i in 1..poly.len() - 1 {
            self.add_tet(r, a, poly[i], poly[i + 1]);
        }
    }
    pub fn com(&self) -> DVec3 {
        if self.volume != 0.0 {
            self.first / self.volume
        } else {
            DVec3::ZERO
        }
    }
    pub fn mass_props(&self, density: f64) -> MassProps {
        let c = self.com();
        let cov = (self.second - outer(c, c) * self.volume) * density;
        let tr = cov.x_axis.x + cov.y_axis.y + cov.z_axis.z;
        let inertia = DMat3::from_diagonal(DVec3::splat(tr)) - cov;
        MassProps {
            volume: self.volume,
            mass: self.volume * density,
            com: c,
            inertia,
        }
    }
}

/// Mass properties: inertia tensor about the center of mass.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct MassProps {
    pub volume: f64,
    pub mass: f64,
    pub com: DVec3,
    pub inertia: DMat3,
}

impl Default for MassProps {
    fn default() -> Self {
        MassProps {
            volume: 0.0,
            mass: 0.0,
            com: DVec3::ZERO,
            inertia: DMat3::ZERO,
        }
    }
}

impl MassProps {
    /// Combine bodies with the parallel axis theorem.
    pub fn combine(parts: &[MassProps]) -> MassProps {
        let mass: f64 = parts.iter().map(|p| p.mass).sum();
        let volume: f64 = parts.iter().map(|p| p.volume).sum();
        if mass <= 0.0 {
            return MassProps {
                volume,
                ..Default::default()
            };
        }
        let com = parts.iter().fold(DVec3::ZERO, |a, p| a + p.com * p.mass) / mass;
        let mut inertia = DMat3::ZERO;
        for p in parts {
            let d = p.com - com;
            let shift = DMat3::from_diagonal(DVec3::splat(d.length_squared())) - outer(d, d);
            inertia += p.inertia + shift * p.mass;
        }
        MassProps {
            volume,
            mass,
            com,
            inertia,
        }
    }

    /// Principal moments (ascending) and axes (columns), via Jacobi.
    pub fn principal(&self) -> (DVec3, DMat3) {
        sym_eigen3(&self.inertia)
    }
}

/// Symmetric 3x3 eigen-decomposition (cyclic Jacobi, deterministic).
/// Returns eigenvalues ascending and eigenvectors as matrix columns.
pub fn sym_eigen3(m: &DMat3) -> (DVec3, DMat3) {
    let mut a = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            a[i][j] = m.col(j)[i];
        }
    }
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _sweep in 0..50 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        let scale = a[0][0].abs() + a[1][1].abs() + a[2][2].abs();
        if off <= 1e-300 || off <= scale * 1e-17 {
            break;
        }
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-300 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for k in 0..3 {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let apk = a[p][k];
                let aqk = a[q][k];
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for k in 0..3 {
                let vkp = v[k][p];
                let vkq = v[k][q];
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| {
        a[i][i]
            .partial_cmp(&a[j][j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let vals = DVec3::new(a[idx[0]][idx[0]], a[idx[1]][idx[1]], a[idx[2]][idx[2]]);
    let col = |k: usize| {
        let mut c = DVec3::new(v[0][k], v[1][k], v[2][k]);
        // canonical sign: largest |component| positive
        let am = c.abs();
        let lead = if am.x >= am.y && am.x >= am.z {
            c.x
        } else if am.y >= am.z {
            c.y
        } else {
            c.z
        };
        if lead < 0.0 {
            c = -c;
        }
        c
    };
    let mut c0 = col(idx[0]);
    let c1 = col(idx[1]);
    let mut c2 = col(idx[2]);
    // ensure right-handed
    if c0.cross(c1).dot(c2) < 0.0 {
        c2 = -c2;
    }
    if c0.length_squared() == 0.0 {
        c0 = DVec3::X;
    }
    (vals, DMat3::from_cols(c0, c1, c2))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_tris(lo: DVec3, hi: DVec3) -> Vec<[DVec3; 4]> {
        let p = |x: usize, y: usize, z: usize| {
            DVec3::new(
                if x == 0 { lo.x } else { hi.x },
                if y == 0 { lo.y } else { hi.y },
                if z == 0 { lo.z } else { hi.z },
            )
        };
        vec![
            [p(0, 0, 0), p(0, 1, 0), p(1, 1, 0), p(1, 0, 0)],
            [p(0, 0, 1), p(1, 0, 1), p(1, 1, 1), p(0, 1, 1)],
            [p(0, 0, 0), p(1, 0, 0), p(1, 0, 1), p(0, 0, 1)],
            [p(0, 1, 0), p(0, 1, 1), p(1, 1, 1), p(1, 1, 0)],
            [p(0, 0, 0), p(0, 0, 1), p(0, 1, 1), p(0, 1, 0)],
            [p(1, 0, 0), p(1, 1, 0), p(1, 1, 1), p(1, 0, 1)],
        ]
    }

    #[test]
    fn box_mass_props() {
        let lo = DVec3::new(1.0, 2.0, 3.0);
        let hi = DVec3::new(3.0, 3.0, 6.0);
        let mut vi = VolumeIntegrals::default();
        let r = DVec3::new(0.3, -1.0, 2.0);
        for f in box_tris(lo, hi) {
            vi.add_polygon(r, &f);
        }
        let mp = vi.mass_props(2.0);
        assert!((mp.volume - 6.0).abs() < 1e-12);
        assert!((mp.com - DVec3::new(2.0, 2.5, 4.5)).length() < 1e-12);
        // Ixx = m/12 (b^2 + c^2) with sides a=2 (x), b=1 (y), c=3 (z), m = 12
        let m = 12.0;
        assert!((mp.inertia.x_axis.x - m / 12.0 * (1.0 + 9.0)).abs() < 1e-10);
        assert!((mp.inertia.y_axis.y - m / 12.0 * (4.0 + 9.0)).abs() < 1e-10);
        assert!((mp.inertia.z_axis.z - m / 12.0 * (4.0 + 1.0)).abs() < 1e-10);
        assert!(mp.inertia.x_axis.y.abs() < 1e-10);
    }

    #[test]
    fn eigen() {
        let m = DMat3::from_cols(
            DVec3::new(2., 1., 0.),
            DVec3::new(1., 2., 0.),
            DVec3::new(0., 0., 5.),
        );
        let (vals, vecs) = sym_eigen3(&m);
        assert!((vals - DVec3::new(1., 3., 5.)).length() < 1e-12);
        let v0 = vecs.col(0);
        assert!((m * v0 - v0 * vals.x).length() < 1e-12);
    }
}
