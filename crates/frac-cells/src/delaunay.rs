//! Incremental 3D Delaunay tetrahedralization (Bowyer–Watson) with exact
//! predicates and Simulation of Simplicity on the lifted coordinate
//! (rank = point index), consistent with the plane perturbation in
//! [`crate::planes`].

use frac_geom::predicates::{P3, insphere_sos, orient3d};

pub const NONE: u32 = u32::MAX;

#[derive(Clone, Debug)]
pub struct Delaunay {
    pub points: Vec<P3>,
    /// Positively oriented tets (`orient3d > 0`).
    pub tets: Vec<[u32; 4]>,
    /// `neigh[t][i]` is the tet sharing the face opposite vertex `i`.
    pub neigh: Vec<[u32; 4]>,
    pub alive: Vec<bool>,
    /// Indices of the 4 super-tetrahedron points (always the first four).
    pub n_super: usize,
}

impl Delaunay {
    /// Triangulate `pts` (inserted in the given order after 4 super points
    /// placed far around `center` at distance `radius * 1e3`).
    /// Point indices in the result are offset by 4.
    pub fn build(pts: &[P3], center: P3, radius: f64) -> Result<Delaunay, String> {
        let r = radius.max(1e-6) * 1.0e3;
        let c = center;
        let mut points: Vec<P3> = vec![
            [c[0] + r, c[1] + r, c[2] + r],
            [c[0] + r, c[1] - r, c[2] - r],
            [c[0] - r, c[1] + r, c[2] - r],
            [c[0] - r, c[1] - r, c[2] + r],
        ];
        // orient so super tet is positive
        if orient3d(&points[0], &points[1], &points[2], &points[3]) < 0 {
            points.swap(2, 3);
        }
        points.extend_from_slice(pts);
        let mut d = Delaunay {
            points,
            tets: vec![[0, 1, 2, 3]],
            neigh: vec![[NONE; 4]],
            alive: vec![true],
            n_super: 4,
        };
        let mut last = 0u32;
        for i in 4..d.points.len() as u32 {
            last = d.insert(i, last)?;
        }
        d.compact();
        Ok(d)
    }

    fn orient_with(&self, t: u32, slot: usize, p: u32) -> i8 {
        let mut v = self.tets[t as usize];
        v[slot] = p;
        orient3d(
            &self.points[v[0] as usize],
            &self.points[v[1] as usize],
            &self.points[v[2] as usize],
            &self.points[v[3] as usize],
        )
    }

    fn in_conflict(&self, t: u32, p: u32) -> bool {
        let v = self.tets[t as usize];
        let pts = [
            &self.points[v[0] as usize],
            &self.points[v[1] as usize],
            &self.points[v[2] as usize],
            &self.points[v[3] as usize],
            &self.points[p as usize],
        ];
        insphere_sos(
            pts,
            [v[0] as u64, v[1] as u64, v[2] as u64, v[3] as u64, p as u64],
        ) > 0
    }

    fn locate(&self, p: u32, start: u32) -> u32 {
        let mut t = if self.alive[start as usize] {
            start
        } else {
            self.alive.iter().rposition(|&a| a).unwrap() as u32
        };
        let mut steps = 0usize;
        // deterministic pseudo-random face order to avoid cycling
        let mut rot = 0usize;
        'walk: loop {
            steps += 1;
            if steps > 4 * self.tets.len() + 100 {
                break;
            }
            rot = rot.wrapping_mul(1103515245).wrapping_add(12345);
            let off = (rot >> 16) % 4;
            for k in 0..4 {
                let i = (k + off) % 4;
                if self.orient_with(t, i, p) < 0 {
                    let n = self.neigh[t as usize][i];
                    if n == NONE {
                        break 'walk;
                    }
                    t = n;
                    continue 'walk;
                }
            }
            return t;
        }
        // fallback: linear scan for a containing tet
        for (ti, &a) in self.alive.iter().enumerate() {
            if a && (0..4).all(|i| self.orient_with(ti as u32, i, p) >= 0) {
                return ti as u32;
            }
        }
        t
    }

    fn insert(&mut self, p: u32, hint: u32) -> Result<u32, String> {
        let t0 = self.locate(p, hint);
        // find a conflicting tet near t0
        let mut seed = NONE;
        if self.in_conflict(t0, p) {
            seed = t0;
        } else {
            let mut stack = vec![t0];
            let mut seen = std::collections::BTreeSet::new();
            seen.insert(t0);
            while let Some(t) = stack.pop() {
                if self.in_conflict(t, p) {
                    seed = t;
                    break;
                }
                if seen.len() > 64 {
                    break;
                }
                for &n in &self.neigh[t as usize] {
                    if n != NONE && seen.insert(n) {
                        stack.push(n);
                    }
                }
            }
            if seed == NONE {
                for (ti, &a) in self.alive.iter().enumerate() {
                    if a && self.in_conflict(ti as u32, p) {
                        seed = ti as u32;
                        break;
                    }
                }
            }
        }
        if seed == NONE {
            return Err(format!(
                "delaunay: no conflict tet for point {p} (duplicate point?)"
            ));
        }
        // cavity BFS
        let mut cavity = vec![seed];
        let mut in_cav: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        in_cav.insert(seed);
        let mut k = 0;
        while k < cavity.len() {
            let t = cavity[k];
            k += 1;
            for &n in &self.neigh[t as usize] {
                if n != NONE && !in_cav.contains(&n) && self.in_conflict(n, p) {
                    in_cav.insert(n);
                    cavity.push(n);
                }
            }
        }
        // boundary faces -> new tets
        let mut new_tets: Vec<u32> = Vec::new();
        // map from sorted face-with-p edge key (a,b) -> (tet, slot)
        let mut pending: Vec<((u32, u32), u32, usize)> = Vec::new();
        for &t in &cavity {
            for i in 0..4 {
                let n = self.neigh[t as usize][i];
                if n != NONE && in_cav.contains(&n) {
                    continue;
                }
                let mut v = self.tets[t as usize];
                v[i] = p;
                let o = orient3d(
                    &self.points[v[0] as usize],
                    &self.points[v[1] as usize],
                    &self.points[v[2] as usize],
                    &self.points[v[3] as usize],
                );
                if o <= 0 {
                    return Err(format!("delaunay: non-star-shaped cavity at point {p}"));
                }
                let nt = self.tets.len() as u32;
                self.tets.push(v);
                let mut nn = [NONE; 4];
                nn[i] = n;
                self.neigh.push(nn);
                self.alive.push(true);
                if n != NONE {
                    // fix back pointer of outer tet
                    let back = self.neigh[n as usize].iter().position(|&x| x == t).unwrap();
                    self.neigh[n as usize][back] = nt;
                }
                new_tets.push(nt);
                // faces containing p: opposite each slot j != i
                for j in 0..4 {
                    if j == i {
                        continue;
                    }
                    let mut e: [u32; 2] = [0; 2];
                    let mut c = 0;
                    for (s, &vv) in v.iter().enumerate() {
                        if s != i && s != j {
                            e[c] = vv;
                            c += 1;
                        }
                    }
                    let key = (e[0].min(e[1]), e[0].max(e[1]));
                    pending.push((key, nt, j));
                }
            }
        }
        pending.sort_unstable();
        let mut idx = 0;
        while idx < pending.len() {
            if idx + 1 < pending.len() && pending[idx].0 == pending[idx + 1].0 {
                let (_, ta, sa) = pending[idx];
                let (_, tb, sb) = pending[idx + 1];
                self.neigh[ta as usize][sa] = tb;
                self.neigh[tb as usize][sb] = ta;
                idx += 2;
            } else {
                return Err(format!("delaunay: unmatched cavity face at point {p}"));
            }
        }
        for &t in &cavity {
            self.alive[t as usize] = false;
        }
        Ok(*new_tets.last().unwrap())
    }

    fn compact(&mut self) {
        let mut map = vec![NONE; self.tets.len()];
        let mut tets = Vec::new();
        for (i, &a) in self.alive.iter().enumerate() {
            if a {
                map[i] = tets.len() as u32;
                tets.push(self.tets[i]);
            }
        }
        let mut neigh = Vec::with_capacity(tets.len());
        for (i, &a) in self.alive.iter().enumerate() {
            if a {
                let n = self.neigh[i];
                neigh.push([0, 1, 2, 3].map(|k| {
                    if n[k] == NONE {
                        NONE
                    } else {
                        map[n[k] as usize]
                    }
                }));
            }
        }
        self.alive = vec![true; tets.len()];
        self.tets = tets;
        self.neigh = neigh;
    }

    /// Structural validation: orientation and neighbor symmetry.
    pub fn validate(&self) -> Result<(), String> {
        for (t, v) in self.tets.iter().enumerate() {
            let o = orient3d(
                &self.points[v[0] as usize],
                &self.points[v[1] as usize],
                &self.points[v[2] as usize],
                &self.points[v[3] as usize],
            );
            if o <= 0 {
                return Err(format!("tet {t} not positive"));
            }
            for i in 0..4 {
                let n = self.neigh[t][i];
                if n == NONE {
                    continue;
                }
                if !self.neigh[n as usize].contains(&(t as u32)) {
                    return Err(format!("asymmetric neighbor {t} {n}"));
                }
            }
        }
        Ok(())
    }

    /// Exact circumcenter (rounded) of tet `t`.
    pub fn circumcenter(&self, t: usize) -> P3 {
        let v = self.tets[t];
        circumcenter(
            &self.points[v[0] as usize],
            &self.points[v[1] as usize],
            &self.points[v[2] as usize],
            &self.points[v[3] as usize],
        )
    }

    /// For each vertex, one incident tet.
    pub fn vertex_tet(&self) -> Vec<u32> {
        let mut vt = vec![NONE; self.points.len()];
        for (t, v) in self.tets.iter().enumerate() {
            for &x in v {
                if vt[x as usize] == NONE {
                    vt[x as usize] = t as u32;
                }
            }
        }
        vt
    }

    /// All tets incident to vertex `v` (sorted).
    pub fn star(&self, v: u32, vt: &[u32]) -> Vec<u32> {
        let s = vt[v as usize];
        if s == NONE {
            return Vec::new();
        }
        let mut out = vec![s];
        let mut seen = std::collections::BTreeSet::new();
        seen.insert(s);
        let mut k = 0;
        while k < out.len() {
            let t = out[k];
            k += 1;
            let tv = self.tets[t as usize];
            for i in 0..4 {
                if tv[i] == v {
                    continue; // face opposite v does not contain v
                }
                let n = self.neigh[t as usize][i];
                if n != NONE && seen.insert(n) {
                    out.push(n);
                }
            }
        }
        out.sort_unstable();
        out
    }
}

/// Circumcenter of a tetrahedron computed from exact numerators/denominator.
pub fn circumcenter(a: &P3, b: &P3, c: &P3, d: &P3) -> P3 {
    use frac_geom::exact::{Expansion, det3};
    let sub = |p: &P3, q: &P3| -> [Expansion; 3] {
        [
            Expansion::from_f64(p[0]).sub(&Expansion::from_f64(q[0])),
            Expansion::from_f64(p[1]).sub(&Expansion::from_f64(q[1])),
            Expansion::from_f64(p[2]).sub(&Expansion::from_f64(q[2])),
        ]
    };
    let u = sub(b, a);
    let v = sub(c, a);
    let w = sub(d, a);
    let l2 = |x: &[Expansion; 3]| x[0].mul(&x[0]).add(&x[1].mul(&x[1])).add(&x[2].mul(&x[2]));
    let (lu, lv, lw) = (l2(&u), l2(&v), l2(&w));
    // Solve [u;v;w] x = 0.5 [lu; lv; lw] by Cramer.
    let half = Expansion::from_f64(0.5);
    let rhs = [lu.mul(&half), lv.mul(&half), lw.mul(&half)];
    let m = [u.clone(), v.clone(), w.clone()];
    let den = det3(&m).approx();
    let mut x = [0.0; 3];
    for k in 0..3 {
        let mut mk = m.clone();
        for r in 0..3 {
            mk[r][k] = rhs[r].clone();
        }
        x[k] = det3(&mk).approx() / den;
    }
    [a[0] + x[0], a[1] + x[1], a[2] + x[2]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn random_points() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
        let pts: Vec<P3> = (0..2000)
            .map(|_| {
                [
                    rng.gen_range(0.0..1.0),
                    rng.gen_range(0.0..1.0),
                    rng.gen_range(0.0..1.0),
                ]
            })
            .collect();
        let d = Delaunay::build(&pts, [0.5; 3], 1.0).unwrap();
        d.validate().unwrap();
        // empty circumsphere property spot-check
        for t in (0..d.tets.len()).step_by(97) {
            let v = d.tets[t];
            for p in (4..d.points.len()).step_by(13) {
                if v.contains(&(p as u32)) {
                    continue;
                }
                let s = frac_geom::predicates::insphere(
                    &d.points[v[0] as usize],
                    &d.points[v[1] as usize],
                    &d.points[v[2] as usize],
                    &d.points[v[3] as usize],
                    &d.points[p],
                );
                assert!(s <= 0);
            }
        }
    }

    #[test]
    fn grid_points_degenerate() {
        // cospherical/coplanar everywhere: exercises SoS
        let mut pts = Vec::new();
        for i in 0..6 {
            for j in 0..6 {
                for k in 0..6 {
                    pts.push([i as f64, j as f64, k as f64]);
                }
            }
        }
        let d = Delaunay::build(&pts, [2.5; 3], 5.0).unwrap();
        d.validate().unwrap();
        // volume of tets inside the grid hull equals 125
        let mut vol = 0.0;
        for v in &d.tets {
            if v.iter().all(|&x| x >= 4) {
                let p = |i: u32| glam::DVec3::from_array(d.points[i as usize]);
                vol += (p(v[1]) - p(v[0])).dot((p(v[2]) - p(v[0])).cross(p(v[3]) - p(v[0]))) / 6.0;
            }
        }
        assert!((vol - 125.0).abs() < 1e-9, "vol {vol}");
    }
}
