//! Exact geometric predicates on `f64` points.
//!
//! Convention: `orient3d(a, b, c, d) > 0` iff `d` lies on the side of the
//! plane through `a, b, c` that the right-hand normal `(b-a)x(c-a)` points to.
//! A tetrahedron `(a, b, c, d)` is *positively oriented* iff
//! `orient3d(a, b, c, d) > 0`.

use crate::exact::{det2, det3, Expansion, Field, F64E};
use crate::exact_sign;

pub type P3 = [f64; 3];
pub type P2 = [f64; 2];

#[inline]
fn d3<F: Field>(p: &P3, q: &P3) -> [F; 3] {
    [
        F::from_f64(p[0]).sub(&F::from_f64(q[0])),
        F::from_f64(p[1]).sub(&F::from_f64(q[1])),
        F::from_f64(p[2]).sub(&F::from_f64(q[2])),
    ]
}

pub fn orient3d_val<F: Field>(a: &P3, b: &P3, c: &P3, d: &P3) -> F {
    let m = [d3::<F>(b, a), d3::<F>(c, a), d3::<F>(d, a)];
    det3(&m)
}

/// Exact sign of the orientation determinant.
pub fn orient3d(a: &P3, b: &P3, c: &P3, d: &P3) -> i8 {
    exact_sign!(|F| orient3d_val::<F>(a, b, c, d))
}

pub fn orient2d_val<F: Field>(a: &P2, b: &P2, c: &P2) -> F {
    let bx = F::from_f64(b[0]).sub(&F::from_f64(a[0]));
    let by = F::from_f64(b[1]).sub(&F::from_f64(a[1]));
    let cx = F::from_f64(c[0]).sub(&F::from_f64(a[0]));
    let cy = F::from_f64(c[1]).sub(&F::from_f64(a[1]));
    det2(&bx, &by, &cx, &cy)
}

/// `> 0` iff `a, b, c` are counter-clockwise.
pub fn orient2d(a: &P2, b: &P2, c: &P2) -> i8 {
    exact_sign!(|F| orient2d_val::<F>(a, b, c))
}

/// det4 of rows `[p - e, l]` for p in a..d, with the last column given.
fn lifted_det<F: Field>(rows: &[[F; 3]; 4], last: &[F; 4]) -> F {
    // Cofactor expansion along the last column (column index 3).
    let mut acc = F::zero();
    for i in 0..4 {
        let mut m: [[F; 3]; 3] = [
            [F::zero(), F::zero(), F::zero()],
            [F::zero(), F::zero(), F::zero()],
            [F::zero(), F::zero(), F::zero()],
        ];
        let mut r = 0;
        for (k, row) in rows.iter().enumerate() {
            if k == i {
                continue;
            }
            m[r] = row.clone();
            r += 1;
        }
        let minor = det3(&m);
        // sign (-1)^(i+3)
        let term = last[i].mul(&minor);
        acc = if (i + 3) % 2 == 0 { acc.add(&term) } else { acc.sub(&term) };
    }
    acc
}

fn insphere_val<F: Field>(a: &P3, b: &P3, c: &P3, d: &P3, e: &P3) -> F {
    let rows = [d3::<F>(a, e), d3::<F>(b, e), d3::<F>(c, e), d3::<F>(d, e)];
    let lift = |r: &[F; 3]| r[0].mul(&r[0]).add(&r[1].mul(&r[1])).add(&r[2].mul(&r[2]));
    let last = [lift(&rows[0]), lift(&rows[1]), lift(&rows[2]), lift(&rows[3])];
    lifted_det(&rows, &last).neg()
}

/// Exact insphere sign: for a positively oriented tet `(a,b,c,d)`, returns
/// `> 0` iff `e` is strictly inside its circumsphere.
pub fn insphere(a: &P3, b: &P3, c: &P3, d: &P3, e: &P3) -> i8 {
    exact_sign!(|F| insphere_val::<F>(a, b, c, d, e))
}

/// Insphere with Simulation of Simplicity: the lifted coordinate of point
/// with rank `r` is perturbed by `eps^r` (smaller rank dominates). Never
/// returns 0 unless the configuration is degenerate beyond perturbation
/// (e.g., coincident points).
pub fn insphere_sos(pts: [&P3; 5], ranks: [u64; 5]) -> i8 {
    let s = insphere(pts[0], pts[1], pts[2], pts[3], pts[4]);
    if s != 0 {
        return s;
    }
    // Order the five points by rank (ascending => most dominant first).
    let mut order = [0usize, 1, 2, 3, 4];
    order.sort_by_key(|&i| ranks[i]);
    let e = pts[4];
    let rows: [[Expansion; 3]; 4] = [d3(pts[0], e), d3(pts[1], e), d3(pts[2], e), d3(pts[3], e)];
    for &q in order.iter() {
        let one = Expansion::from_f64(1.0);
        let zero = Expansion::zero();
        let last: [Expansion; 4] = if q == 4 {
            let m = one.neg();
            [m.clone(), m.clone(), m.clone(), m]
        } else {
            let mut l = [zero.clone(), zero.clone(), zero.clone(), zero];
            l[q] = one;
            l
        };
        let coef = lifted_det(&rows, &last).neg();
        let s = coef.sign();
        if s != 0 {
            return s;
        }
    }
    0
}

/// Exact segment/triangle-plane classification helpers.
pub fn orient3d_f64(a: &P3, b: &P3, c: &P3, d: &P3) -> f64 {
    orient3d_val::<f64>(a, b, c, d)
}

#[allow(dead_code)]
fn _assert_f64e(_: F64E) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_convention() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let d = [0.0, 0.0, 1.0];
        assert_eq!(orient3d(&a, &b, &c, &d), 1);
        assert_eq!(orient3d(&a, &c, &b, &d), -1);
        let inside = [0.2, 0.2, 0.2];
        let outside = [2.0, 2.0, 2.0];
        assert_eq!(insphere(&a, &b, &c, &d, &inside), 1);
        assert_eq!(insphere(&a, &b, &c, &d, &outside), -1);
        // cospherical: the point (1,1,1) is on the sphere through the unit tet corners
        let on = [1.0, 1.0, 0.0];
        assert_eq!(insphere(&a, &b, &c, &d, &on), 0);
        let s = insphere_sos([&a, &b, &c, &d, &on], [0, 1, 2, 3, 4]);
        assert_ne!(s, 0);
    }

    #[test]
    fn near_degenerate_orient() {
        let a = [0.1, 0.1, 0.1];
        let b = [0.3, 0.7, 0.1];
        let c = [0.9, 0.2, 0.1];
        let d = [0.5, 0.5, f64::from_bits(0.1f64.to_bits() + 1)];
        // abc is clockwise seen from +z, so a point just above is negative
        assert_eq!(orient3d(&a, &b, &c, &d), -1);
        let d0 = [0.5, 0.5, 0.1];
        assert_eq!(orient3d(&a, &b, &c, &d0), 0);
    }
}
