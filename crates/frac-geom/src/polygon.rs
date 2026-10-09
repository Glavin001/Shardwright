//! Planar polygon integrals (area, centroid, second moments) for polygons
//! with holes, plus Newell normals and 2D helpers.

use glam::{DMat3, DVec2, DVec3};

/// Newell area vector of a closed loop: `0.5 * sum(p_i x p_{i+1})`.
/// Its length is the loop area; its direction is the right-hand normal.
pub fn newell(loop_: &[DVec3]) -> DVec3 {
    let n = loop_.len();
    if n < 3 {
        return DVec3::ZERO;
    }
    let r = loop_[0];
    let mut a = DVec3::ZERO;
    for i in 1..n - 1 {
        a += (loop_[i] - r).cross(loop_[i + 1] - r);
    }
    a * 0.5
}

/// Integrals of a planar region: `area`, `first = ∫x dA`,
/// `second = ∫ x xᵀ dA` (about the origin).
#[derive(Clone, Copy, Debug)]
pub struct AreaIntegrals {
    pub area: f64,
    pub area_vec: DVec3,
    pub first: DVec3,
    pub second: DMat3,
}

impl Default for AreaIntegrals {
    // NB: glam's `DMat3::default()` is the identity, so spell out zero.
    fn default() -> Self {
        AreaIntegrals { area: 0.0, area_vec: DVec3::ZERO, first: DVec3::ZERO, second: DMat3::ZERO }
    }
}

impl AreaIntegrals {
    pub fn add(&mut self, o: &AreaIntegrals) {
        self.area += o.area;
        self.area_vec += o.area_vec;
        self.first += o.first;
        self.second += o.second;
    }
    pub fn centroid(&self) -> DVec3 {
        if self.area > 0.0 { self.first / self.area } else { DVec3::ZERO }
    }
    /// Second moment about the centroid: `∫ (x-c)(x-c)ᵀ dA`.
    pub fn central_second(&self) -> DMat3 {
        let c = self.centroid();
        self.second - outer(c, c) * self.area
    }
}

#[inline]
pub fn outer(a: DVec3, b: DVec3) -> DMat3 {
    DMat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// Integrals over a planar region bounded by `loops` (outer CCW about
/// `normal`, holes CW), all in the same plane. `normal` must be unit length.
/// Uses a signed fan from a reference point, which is exact for arbitrary
/// simple polygons with holes.
pub fn area_integrals(loops: &[Vec<DVec3>], normal: DVec3) -> AreaIntegrals {
    let mut out = AreaIntegrals::default();
    let r = match loops.iter().find(|l| !l.is_empty()) {
        Some(l) => l[0],
        None => return out,
    };
    for l in loops {
        let n = l.len();
        for i in 0..n {
            let p1 = l[i];
            let p2 = l[(i + 1) % n];
            let av = (p1 - r).cross(p2 - r) * 0.5;
            let a = av.dot(normal);
            if a == 0.0 {
                continue;
            }
            out.area += a;
            out.area_vec += av;
            let s = r + p1 + p2;
            out.first += s * (a / 3.0);
            let m = outer(r, r) + outer(p1, p1) + outer(p2, p2) + outer(s, s);
            out.second += m * (a / 12.0);
        }
    }
    out
}

/// Triangle integrals (unsigned, oriented area vector).
pub fn triangle_integrals(p0: DVec3, p1: DVec3, p2: DVec3) -> AreaIntegrals {
    let av = (p1 - p0).cross(p2 - p0) * 0.5;
    let a = av.length();
    let s = p0 + p1 + p2;
    let m = outer(p0, p0) + outer(p1, p1) + outer(p2, p2) + outer(s, s);
    AreaIntegrals { area: a, area_vec: av, first: s * (a / 3.0), second: m * (a / 12.0) }
}

/// An orthonormal in-plane basis (u, v) for a unit normal n, such that
/// (u, v, n) is right-handed.
pub fn plane_basis(n: DVec3) -> (DVec3, DVec3) {
    let a = if n.x.abs() < 0.6 {
        DVec3::X
    } else if n.y.abs() < 0.6 {
        DVec3::Y
    } else {
        DVec3::Z
    };
    let u = a.cross(n).normalize();
    let v = n.cross(u);
    (u, v)
}

/// Signed area of a 2D polygon (CCW positive).
pub fn signed_area_2d(p: &[DVec2]) -> f64 {
    let n = p.len();
    let mut a = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        a += p[i].x * p[j].y - p[j].x * p[i].y;
    }
    0.5 * a
}

/// Ramer–Douglas–Peucker simplification of a closed 3D loop.
pub fn simplify_loop(pts: &[DVec3], tol: f64) -> Vec<DVec3> {
    let n = pts.len();
    if n <= 4 {
        return pts.to_vec();
    }
    // split at the two farthest points (deterministic: first index wins)
    let mut i0 = 0;
    let mut i1 = 0;
    let mut best = -1.0;
    for i in 0..n {
        let d = (pts[i] - pts[0]).length_squared();
        if d > best {
            best = d;
            i1 = i;
        }
    }
    best = -1.0;
    for i in 0..n {
        let d = (pts[i] - pts[i1]).length_squared();
        if d > best {
            best = d;
            i0 = i;
        }
    }
    let (a, b) = if i0 < i1 { (i0, i1) } else { (i1, i0) };
    let mut keep = vec![false; n];
    keep[a] = true;
    keep[b] = true;
    let chain1: Vec<usize> = (a..=b).collect();
    let chain2: Vec<usize> = (b..n).chain(0..=a).collect();
    for ch in [chain1, chain2] {
        rdp(pts, &ch, tol, &mut keep);
    }
    (0..n).filter(|&i| keep[i]).map(|i| pts[i]).collect()
}

fn rdp(pts: &[DVec3], idx: &[usize], tol: f64, keep: &mut [bool]) {
    if idx.len() < 3 {
        return;
    }
    let a = pts[idx[0]];
    let b = pts[*idx.last().unwrap()];
    let ab = b - a;
    let l2 = ab.length_squared();
    let mut best = -1.0;
    let mut bi = 0;
    for (k, &i) in idx.iter().enumerate().take(idx.len() - 1).skip(1) {
        let p = pts[i];
        let d = if l2 > 0.0 {
            let t = ((p - a).dot(ab) / l2).clamp(0.0, 1.0);
            (a + ab * t - p).length()
        } else {
            (p - a).length()
        };
        if d > best {
            best = d;
            bi = k;
        }
    }
    if best > tol {
        keep[idx[bi]] = true;
        rdp(pts, &idx[..=bi], tol, keep);
        rdp(pts, &idx[bi..], tol, keep);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_with_hole() {
        let outer_l = vec![
            DVec3::new(0., 0., 1.),
            DVec3::new(2., 0., 1.),
            DVec3::new(2., 2., 1.),
            DVec3::new(0., 2., 1.),
        ];
        let hole = vec![
            DVec3::new(0.5, 0.5, 1.),
            DVec3::new(0.5, 1.0, 1.),
            DVec3::new(1.0, 1.0, 1.),
            DVec3::new(1.0, 0.5, 1.),
        ];
        let ai = area_integrals(&[outer_l.clone(), hole], DVec3::Z);
        assert!((ai.area - 3.75).abs() < 1e-14);
        let full = area_integrals(&[outer_l], DVec3::Z);
        assert!((full.area - 4.0).abs() < 1e-14);
        // Ixx about centroid of 2x2 square = b h^3/12 = 2*8/12
        let c = full.central_second();
        assert!((c.y_axis.y - 16.0 / 12.0).abs() < 1e-12);
        assert!((full.centroid() - DVec3::new(1., 1., 1.)).length() < 1e-14);
    }
}
