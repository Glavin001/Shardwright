//! Plane systems with exact evaluation and a globally consistent symbolic
//! perturbation (Simulation of Simplicity).
//!
//! Every plane is an exact affine function `f_p(x)` of an `f64` point. All
//! combinatorial decisions of the clipping kernel are signs of polynomials in
//! the values `f_p(v)` at *input mesh vertices* `v`, evaluated exactly with a
//! floating-point filter. Zero results are resolved by perturbing each plane
//! offset by a sparse combination of infinitesimals `ε_s` (smaller rank `s`
//! dominates). For Voronoi planes `f_ij = D_i - D_j` with
//! `D_i = |x - s_i|^2 + ε_i`, which keeps every bisector incidence exact
//! *and* coincides with the lifted-weight perturbation used by the Delaunay
//! insphere SoS, so the cell complex and the clipping decisions describe the
//! same perturbed configuration.

use frac_geom::exact::{Expansion, Field, det2, det3};
use frac_geom::exact_sign;
use smallvec::SmallVec;

pub type PlaneId = u64;
pub type P3 = [f64; 3];

/// Canonical key of a line (intersection of two planes). For Voronoi
/// triple lines `{a<b<c}` the key is `(id(a,b), id(a,c))`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LineKey(pub PlaneId, pub PlaneId);

#[derive(Clone, Debug)]
pub enum PlaneSystem {
    Voronoi(VoronoiPlanes),
    Boxes(BoxPlanes),
}

/// Bisector planes of a point set: plane `(i, j)`, `i < j`, is
/// `f(x) = |x - s_i|^2 - |x - s_j|^2` (negative on `i`'s side).
#[derive(Clone, Debug)]
pub struct VoronoiPlanes {
    pub seeds: Vec<P3>,
}

/// Axis-aligned planes `f(x) = x[axis] - offset`.
#[derive(Clone, Debug, Default)]
pub struct BoxPlanes {
    pub planes: Vec<(u8, f64)>,
}

#[inline]
pub fn vor_id(i: u32, j: u32) -> PlaneId {
    let (a, b) = if i < j { (i, j) } else { (j, i) };
    ((a as u64) << 32) | b as u64
}
#[inline]
pub fn vor_pair(p: PlaneId) -> (u32, u32) {
    ((p >> 32) as u32, (p & 0xffff_ffff) as u32)
}

/// Sparse perturbation: list of (symbol rank, coefficient).
pub type Pert = SmallVec<[(u64, i8); 2]>;

impl PlaneSystem {
    #[inline]
    pub fn eval<F: Field>(&self, p: PlaneId, x: &P3) -> F {
        match self {
            PlaneSystem::Voronoi(v) => {
                let (i, j) = vor_pair(p);
                let si = &v.seeds[i as usize];
                let sj = &v.seeds[j as usize];
                let mut acc = F::zero();
                for k in 0..3 {
                    let xk = F::from_f64(x[k]);
                    let a = xk.sub(&F::from_f64(si[k]));
                    let b = xk.sub(&F::from_f64(sj[k]));
                    acc = acc.add(&a.mul(&a)).sub(&b.mul(&b));
                }
                acc
            }
            PlaneSystem::Boxes(b) => {
                let (ax, off) = b.planes[p as usize];
                F::from_f64(x[ax as usize]).sub(&F::from_f64(off))
            }
        }
    }

    /// Approximate gradient direction (plane normal, unnormalized).
    pub fn normal(&self, p: PlaneId) -> [f64; 3] {
        match self {
            PlaneSystem::Voronoi(v) => {
                let (i, j) = vor_pair(p);
                let si = v.seeds[i as usize];
                let sj = v.seeds[j as usize];
                [
                    2.0 * (sj[0] - si[0]),
                    2.0 * (sj[1] - si[1]),
                    2.0 * (sj[2] - si[2]),
                ]
            }
            PlaneSystem::Boxes(b) => {
                let mut n = [0.0; 3];
                n[b.planes[p as usize].0 as usize] = 1.0;
                n
            }
        }
    }

    #[inline]
    pub fn pert(&self, p: PlaneId) -> Pert {
        let mut v = Pert::new();
        match self {
            PlaneSystem::Voronoi(_) => {
                let (i, j) = vor_pair(p);
                v.push((i as u64, 1));
                v.push((j as u64, -1));
            }
            PlaneSystem::Boxes(_) => v.push((p, 1)),
        }
        v
    }

    /// Canonical line through planes `p` and `q`.
    pub fn line(&self, p: PlaneId, q: PlaneId) -> LineKey {
        match self {
            PlaneSystem::Voronoi(_) => {
                let (a, b) = vor_pair(p);
                let (c, d) = vor_pair(q);
                let mut s: SmallVec<[u32; 4]> = SmallVec::from_slice(&[a, b, c, d]);
                s.sort_unstable();
                s.dedup();
                if s.len() == 3 {
                    LineKey(vor_id(s[0], s[1]), vor_id(s[0], s[2]))
                } else if p < q {
                    LineKey(p, q)
                } else {
                    LineKey(q, p)
                }
            }
            PlaneSystem::Boxes(_) => {
                if p < q {
                    LineKey(p, q)
                } else {
                    LineKey(q, p)
                }
            }
        }
    }

    /// Does plane `r` contain line `l` (exactly, by construction)?
    pub fn line_contains(&self, l: LineKey, r: PlaneId) -> bool {
        if r == l.0 || r == l.1 {
            return true;
        }
        match self {
            PlaneSystem::Voronoi(_) => {
                let (a, b) = vor_pair(l.0);
                let (c, d) = vor_pair(l.1);
                // triple line iff they share the first seed
                if a != c {
                    return false;
                }
                let (x, y) = vor_pair(r);
                let t = [a, b, d];
                t.contains(&x) && t.contains(&y)
            }
            PlaneSystem::Boxes(_) => false,
        }
    }

    // ------------------------------------------------------------------
    // Predicates. All return -1/+1 (0 only for configurations that are
    // degenerate beyond perturbation, e.g. zero-area triangles).
    // ------------------------------------------------------------------

    /// Sign of `f_p(a)` (perturbed).
    pub fn side_vertex(&self, a: &P3, p: PlaneId) -> i8 {
        let s = exact_sign!(|F| self.eval::<F>(p, a));
        if s != 0 {
            return s;
        }
        lead_sign(&self.pert(p))
    }

    /// Sign of `f_q` at the point `EP(ab, p)` where segment `ab` crosses `p`.
    pub fn side_edge_point(&self, a: &P3, b: &P3, p: PlaneId, q: PlaneId) -> i8 {
        let s1 = self.d2_sign(a, b, p, q);
        let s2 = exact_sign!(|F| self.eval::<F>(p, a).sub(&self.eval::<F>(p, b)));
        s1 * s2
    }

    /// Perturbed sign of `D_pq(a,b) = f_p(a) f_q(b) - f_p(b) f_q(a)`.
    pub fn d2_sign(&self, a: &P3, b: &P3, p: PlaneId, q: PlaneId) -> i8 {
        let s = exact_sign!(|F| {
            let fpa = self.eval::<F>(p, a);
            let fpb = self.eval::<F>(p, b);
            let fqa = self.eval::<F>(q, a);
            let fqb = self.eval::<F>(q, b);
            det2(&fpa, &fpb, &fqa, &fqb)
        });
        if s != 0 {
            return s;
        }
        // D + δ_q (f_p(a) - f_p(b)) + δ_p (f_q(b) - f_q(a))
        let fpa: Expansion = self.eval(p, a);
        let fpb: Expansion = self.eval(p, b);
        let fqa: Expansion = self.eval(q, a);
        let fqb: Expansion = self.eval(q, b);
        let cq = fpa.sub(&fpb);
        let cp = fqb.sub(&fqa);
        sos_resolve(&[(self.pert(q), cq), (self.pert(p), cp)])
    }

    /// Sign of `f_r` at `TL(abc, p, q)`, the point of triangle plane `abc`
    /// on planes `p` and `q`.
    pub fn side_tri_line(&self, t: [&P3; 3], p: PlaneId, q: PlaneId, r: PlaneId) -> i8 {
        let den = exact_sign!(|F| {
            let one = F::one();
            let m = [
                [one.clone(), one.clone(), one],
                [
                    self.eval::<F>(p, t[0]),
                    self.eval::<F>(p, t[1]),
                    self.eval::<F>(p, t[2]),
                ],
                [
                    self.eval::<F>(q, t[0]),
                    self.eval::<F>(q, t[1]),
                    self.eval::<F>(q, t[2]),
                ],
            ];
            det3(&m)
        });
        if den == 0 {
            return 0;
        }
        let num = exact_sign!(|F| {
            let m = [
                [
                    self.eval::<F>(p, t[0]),
                    self.eval::<F>(p, t[1]),
                    self.eval::<F>(p, t[2]),
                ],
                [
                    self.eval::<F>(q, t[0]),
                    self.eval::<F>(q, t[1]),
                    self.eval::<F>(q, t[2]),
                ],
                [
                    self.eval::<F>(r, t[0]),
                    self.eval::<F>(r, t[1]),
                    self.eval::<F>(r, t[2]),
                ],
            ];
            det3(&m)
        });
        if num != 0 {
            return num * den;
        }
        let rows: [[Expansion; 3]; 3] = [
            [self.eval(p, t[0]), self.eval(p, t[1]), self.eval(p, t[2])],
            [self.eval(q, t[0]), self.eval(q, t[1]), self.eval(q, t[2])],
            [self.eval(r, t[0]), self.eval(r, t[1]), self.eval(r, t[2])],
        ];
        let ones = [
            Expansion::from_f64(1.0),
            Expansion::from_f64(1.0),
            Expansion::from_f64(1.0),
        ];
        let mut terms = Vec::with_capacity(3);
        for (k, plane) in [p, q, r].into_iter().enumerate() {
            let mut m = rows.clone();
            m[k] = ones.clone();
            terms.push((self.pert(plane), det3(&m)));
        }
        sos_resolve(&terms) * den
    }

    /// Exact comparison of `f_r` at `TL(t1, p, q)` and `TL(t2, p, q)`
    /// (both on the same line). Ties (impossible for distinct pierce points
    /// of an embedded surface) return `Equal`.
    pub fn cmp_along_line(
        &self,
        t1: [&P3; 3],
        t2: [&P3; 3],
        p: PlaneId,
        q: PlaneId,
        r: PlaneId,
    ) -> std::cmp::Ordering {
        fn nd<F: Field>(
            ps: &PlaneSystem,
            t: [&P3; 3],
            p: PlaneId,
            q: PlaneId,
            r: PlaneId,
        ) -> (F, F) {
            let rp = [
                ps.eval::<F>(p, t[0]),
                ps.eval::<F>(p, t[1]),
                ps.eval::<F>(p, t[2]),
            ];
            let rq = [
                ps.eval::<F>(q, t[0]),
                ps.eval::<F>(q, t[1]),
                ps.eval::<F>(q, t[2]),
            ];
            let rr = [
                ps.eval::<F>(r, t[0]),
                ps.eval::<F>(r, t[1]),
                ps.eval::<F>(r, t[2]),
            ];
            let one = F::one();
            let n = det3(&[rp.clone(), rq.clone(), rr]);
            let d = det3(&[[one.clone(), one.clone(), one], rp, rq]);
            (n, d)
        }
        let s = exact_sign!(|F| {
            let (n1, d1) = nd::<F>(self, t1, p, q, r);
            let (n2, d2) = nd::<F>(self, t2, p, q, r);
            n1.mul(&d2).sub(&n2.mul(&d1))
        });
        let sd = exact_sign!(|F| {
            let (_, d1) = nd::<F>(self, t1, p, q, r);
            let (_, d2) = nd::<F>(self, t2, p, q, r);
            d1.mul(&d2)
        });
        let mut s = s;
        if s == 0 {
            // Symbolic tie-break: first-order terms of the perturbed values.
            // f_r(T_i) = (N_i + δ_p C_p,i + δ_q C_q,i + δ_r D_i) / D_i, and δ_r
            // cancels in the difference.
            let cof = |t: [&P3; 3]| -> (Expansion, Expansion, Expansion) {
                let rp: [Expansion; 3] =
                    [self.eval(p, t[0]), self.eval(p, t[1]), self.eval(p, t[2])];
                let rq: [Expansion; 3] =
                    [self.eval(q, t[0]), self.eval(q, t[1]), self.eval(q, t[2])];
                let rr: [Expansion; 3] =
                    [self.eval(r, t[0]), self.eval(r, t[1]), self.eval(r, t[2])];
                let one = [
                    Expansion::from_f64(1.0),
                    Expansion::from_f64(1.0),
                    Expansion::from_f64(1.0),
                ];
                let cp = det3(&[one.clone(), rq.clone(), rr.clone()]);
                let cq = det3(&[rp.clone(), one.clone(), rr]);
                let d = det3(&[one, rp, rq]);
                (cp, cq, d)
            };
            let (cp1, cq1, d1) = cof(t1);
            let (cp2, cq2, d2) = cof(t2);
            let vp = cp1.mul(&d2).sub(&cp2.mul(&d1));
            let vq = cq1.mul(&d2).sub(&cq2.mul(&d1));
            let terms = [(self.pert(p), vp), (self.pert(q), vq)];
            // sos_resolve falls back to +1 when everything vanishes; detect that.
            let all_zero = terms.iter().all(|(_, v)| v.sign() == 0);
            s = if all_zero { 0 } else { sos_resolve(&terms) };
        }
        match s * sd {
            x if x < 0 => std::cmp::Ordering::Less,
            x if x > 0 => std::cmp::Ordering::Greater,
            _ => std::cmp::Ordering::Equal,
        }
    }

    /// Does line `(p ∩ q)` pierce the interior of triangle `abc`?
    pub fn pierces(&self, t: [&P3; 3], p: PlaneId, q: PlaneId) -> bool {
        let s0 = self.d2_sign(t[1], t[2], p, q);
        let s1 = self.d2_sign(t[2], t[0], p, q);
        let s2 = self.d2_sign(t[0], t[1], p, q);
        s0 != 0 && s0 == s1 && s1 == s2
    }

    // ------------------------------------------------------------------
    // Canonical coordinates of derived points.
    // ------------------------------------------------------------------

    /// Point where segment `ab` (a, b in canonical order) crosses plane `p`.
    pub fn edge_point(&self, a: &P3, b: &P3, p: PlaneId) -> P3 {
        let fa = frac_geom::exact_value!(|F| self.eval::<F>(p, a));
        let fb = frac_geom::exact_value!(|F| self.eval::<F>(p, b));
        let (o, d, t) = if fa.abs() <= fb.abs() {
            (a, b, fa / (fa - fb))
        } else {
            (b, a, fb / (fb - fa))
        };
        if !t.is_finite() {
            return *a;
        }
        let mut x = [
            o[0] + (d[0] - o[0]) * t,
            o[1] + (d[1] - o[1]) * t,
            o[2] + (d[2] - o[2]) * t,
        ];
        self.snap(p, &mut x);
        x
    }

    /// Put a point that lies on plane `p` exactly on it where the plane is
    /// representable (axis-aligned box planes): every vertex of that plane
    /// then has the same coordinate bit for bit, so collinearity and
    /// coplanarity on it survive rounding.
    #[inline]
    pub fn snap(&self, p: PlaneId, x: &mut P3) {
        if let PlaneSystem::Boxes(b) = self {
            let (ax, off) = b.planes[p as usize];
            x[ax as usize] = off;
        }
    }

    /// Point of triangle `t` on line `(p, q)` via exact barycentric minors.
    pub fn tri_line_point(&self, t: [&P3; 3], p: PlaneId, q: PlaneId) -> P3 {
        let minor = |x: &P3, y: &P3| {
            frac_geom::exact_value!(|F| {
                det2(
                    &self.eval::<F>(p, x),
                    &self.eval::<F>(p, y),
                    &self.eval::<F>(q, x),
                    &self.eval::<F>(q, y),
                )
            })
        };
        let la = minor(t[1], t[2]);
        let lb = minor(t[2], t[0]);
        let lc = minor(t[0], t[1]);
        let s = la + lb + lc;
        if s == 0.0 || !s.is_finite() {
            return [
                (t[0][0] + t[1][0] + t[2][0]) / 3.0,
                (t[0][1] + t[1][1] + t[2][1]) / 3.0,
                (t[0][2] + t[1][2] + t[2][2]) / 3.0,
            ];
        }
        let (wa, wb, wc) = (la / s, lb / s, lc / s);
        let mut x = [
            wa * t[0][0] + wb * t[1][0] + wc * t[2][0],
            wa * t[0][1] + wb * t[1][1] + wc * t[2][1],
            wa * t[0][2] + wb * t[1][2] + wc * t[2][2],
        ];
        self.snap(p, &mut x);
        self.snap(q, &mut x);
        x
    }
}

/// Sign of the dominant perturbation term of a single plane.
fn lead_sign(p: &Pert) -> i8 {
    let mut best: Option<(u64, i8)> = None;
    for &(s, c) in p {
        if best.map(|b| s < b.0).unwrap_or(true) {
            best = Some((s, c));
        }
    }
    best.map(|b| b.1).unwrap_or(1)
}

/// Resolve the sign of `Σ_rows Σ_s coef(row, s) ε_s · value(row)`:
/// group by symbol, evaluate in rank order, return the first nonzero sign.
fn sos_resolve(terms: &[(Pert, Expansion)]) -> i8 {
    let mut syms: SmallVec<[u64; 8]> = SmallVec::new();
    for (p, _) in terms {
        for &(s, _) in p {
            syms.push(s);
        }
    }
    syms.sort_unstable();
    syms.dedup();
    for s in syms {
        let mut acc = Expansion::zero();
        for (p, v) in terms {
            for &(sym, c) in p {
                if sym == s {
                    acc = if c > 0 { acc.add(v) } else { acc.sub(v) };
                }
            }
        }
        let sg = acc.sign();
        if sg != 0 {
            return sg;
        }
    }
    // Fully degenerate: deterministic fallback.
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voronoi_lines_and_incidence() {
        let ps = PlaneSystem::Voronoi(VoronoiPlanes {
            seeds: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        });
        let l1 = ps.line(vor_id(1, 2), vor_id(1, 0));
        let l2 = ps.line(vor_id(0, 2), vor_id(2, 1));
        assert_eq!(l1, l2);
        assert_eq!(l1, LineKey(vor_id(0, 1), vor_id(0, 2)));
        assert!(ps.line_contains(l1, vor_id(1, 2)));
        assert!(!ps.line_contains(l1, vor_id(1, 3)));
    }

    #[test]
    fn sos_consistency_vertex_on_plane() {
        // A vertex exactly on a box plane gets a consistent nonzero side.
        let ps = PlaneSystem::Boxes(BoxPlanes {
            planes: vec![(0, 0.5), (1, 0.5)],
        });
        assert_eq!(ps.side_vertex(&[0.5, 0.2, 0.0], 0), 1);
        // edge from x=0 to x=1 crosses plane 0 at (0.5, y) with y=0.5 exactly on plane 1
        let a = [0.0, 0.5, 0.0];
        let b = [1.0, 0.5, 0.0];
        let s = ps.side_edge_point(&a, &b, 0, 1);
        assert_ne!(s, 0);
        // the same decision from the other ordering must agree
        assert_eq!(s, ps.side_edge_point(&b, &a, 0, 1));
    }

    #[test]
    fn tri_line_point_on_planes() {
        let ps = PlaneSystem::Boxes(BoxPlanes {
            planes: vec![(0, 0.25), (1, 0.3)],
        });
        let t = [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 2.0]];
        let x = ps.tri_line_point([&t[0], &t[1], &t[2]], 0, 1);
        assert!((x[0] - 0.25).abs() < 1e-15 && (x[1] - 0.3).abs() < 1e-15);
        assert!((x[2] - (0.25 + 0.6)).abs() < 1e-14);
        assert!(ps.pierces([&t[0], &t[1], &t[2]], 0, 1));
        // side of TL vs a third plane
        let ps2 = PlaneSystem::Boxes(BoxPlanes {
            planes: vec![(0, 0.25), (1, 0.3), (2, 0.5)],
        });
        assert_eq!(ps2.side_tri_line([&t[0], &t[1], &t[2]], 0, 1, 2), 1);
    }
}
