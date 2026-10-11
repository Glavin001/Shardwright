//! Tetrahedral mesh container and helpers.

/// Tetrahedral mesh. All tets are positively oriented:
/// `det[b-a, c-a, d-a] > 0`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TetMesh {
    pub verts: Vec<[f64; 3]>,
    pub tets: Vec<[u32; 4]>,
}

#[inline]
pub(crate) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(crate) fn det3(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

/// Signed volume of a tet given its corner positions.
#[inline]
pub fn tet_signed_volume(p: &[[f64; 3]; 4]) -> f64 {
    det3(sub(p[1], p[0]), sub(p[2], p[0]), sub(p[3], p[0])) / 6.0
}

/// Local faces of a tet, each oriented outward for a positively oriented tet.
pub const TET_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

impl TetMesh {
    pub fn tet_points(&self, t: usize) -> [[f64; 3]; 4] {
        let tt = self.tets[t];
        [
            self.verts[tt[0] as usize],
            self.verts[tt[1] as usize],
            self.verts[tt[2] as usize],
            self.verts[tt[3] as usize],
        ]
    }

    pub fn tet_volume(&self, t: usize) -> f64 {
        tet_signed_volume(&self.tet_points(t))
    }

    pub fn tet_centroid(&self, t: usize) -> [f64; 3] {
        let p = self.tet_points(t);
        let mut c = [0.0; 3];
        for q in &p {
            for k in 0..3 {
                c[k] += 0.25 * q[k];
            }
        }
        c
    }

    /// Total (signed) volume.
    pub fn volume(&self) -> f64 {
        (0..self.tets.len()).map(|t| self.tet_volume(t)).sum()
    }

    /// Flips negatively oriented tets (swaps two vertices) in place.
    pub fn fix_orientation(&mut self) {
        for t in 0..self.tets.len() {
            if self.tet_volume(t) < 0.0 {
                self.tets[t].swap(2, 3);
            }
        }
    }

    /// Smallest signed tet volume (useful for validity checks).
    pub fn min_volume(&self) -> f64 {
        (0..self.tets.len())
            .map(|t| self.tet_volume(t))
            .fold(f64::INFINITY, f64::min)
    }

    /// All tet faces as `(sorted vertex triple, tet, local face)`, sorted.
    pub fn sorted_faces(&self) -> Vec<([u32; 3], u32, u8)> {
        let mut f = Vec::with_capacity(self.tets.len() * 4);
        for (t, tt) in self.tets.iter().enumerate() {
            for (lf, lv) in TET_FACES.iter().enumerate() {
                let mut k = [tt[lv[0]], tt[lv[1]], tt[lv[2]]];
                k.sort_unstable();
                f.push((k, t as u32, lf as u8));
            }
        }
        f.sort_unstable();
        f
    }

    /// Boundary faces (faces referenced by exactly one tet), oriented
    /// outward, in deterministic order.
    pub fn boundary_faces(&self) -> Vec<[u32; 3]> {
        let f = self.sorted_faces();
        let mut out = Vec::new();
        let mut i = 0;
        while i < f.len() {
            let mut j = i + 1;
            while j < f.len() && f[j].0 == f[i].0 {
                j += 1;
            }
            if j - i == 1 {
                let (_, t, lf) = f[i];
                let tt = self.tets[t as usize];
                let lv = TET_FACES[lf as usize];
                out.push([tt[lv[0]], tt[lv[1]], tt[lv[2]]]);
            }
            i = j;
        }
        out
    }

    /// Removes vertices not referenced by any tet (keeps relative order).
    pub fn remove_unreferenced(&mut self) {
        let mut used = vec![false; self.verts.len()];
        for t in &self.tets {
            for &v in t {
                used[v as usize] = true;
            }
        }
        let mut remap = vec![u32::MAX; self.verts.len()];
        let mut nv = Vec::new();
        for (i, &u) in used.iter().enumerate() {
            if u {
                remap[i] = nv.len() as u32;
                nv.push(self.verts[i]);
            }
        }
        for t in &mut self.tets {
            for v in t.iter_mut() {
                *v = remap[*v as usize];
            }
        }
        self.verts = nv;
    }

    /// Number of connected components (tets connected through shared faces)
    /// and the component id per tet (ids in order of first tet).
    pub fn tet_components(&self) -> (usize, Vec<u32>) {
        let n = self.tets.len();
        let mut parent: Vec<u32> = (0..n as u32).collect();
        fn find(p: &mut [u32], mut x: u32) -> u32 {
            while p[x as usize] != x {
                p[x as usize] = p[p[x as usize] as usize];
                x = p[x as usize];
            }
            x
        }
        let f = self.sorted_faces();
        for w in f.windows(2) {
            if w[0].0 == w[1].0 {
                let a = find(&mut parent, w[0].1);
                let b = find(&mut parent, w[1].1);
                if a != b {
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    parent[hi as usize] = lo;
                }
            }
        }
        let mut label = vec![u32::MAX; n];
        let mut ids = vec![0u32; n];
        let mut count = 0u32;
        for t in 0..n {
            let r = find(&mut parent, t as u32) as usize;
            if label[r] == u32::MAX {
                label[r] = count;
                count += 1;
            }
            ids[t] = label[r];
        }
        (count as usize, ids)
    }

    /// Bounding box `(min, max)`.
    pub fn bounds(&self) -> ([f64; 3], [f64; 3]) {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for v in &self.verts {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
        (lo, hi)
    }

    /// Mean edge length over all tet edges (each tet edge counted per tet).
    pub fn mean_edge_length(&self) -> f64 {
        if self.tets.is_empty() {
            return 0.0;
        }
        let mut s = 0.0;
        for t in 0..self.tets.len() {
            let p = self.tet_points(t);
            for a in 0..4 {
                for b in a + 1..4 {
                    let d = sub(p[a], p[b]);
                    s += (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                }
            }
        }
        s / (6.0 * self.tets.len() as f64)
    }
}
