//! Indexed triangle meshes, topology validation and exact self-intersection
//! testing.

use crate::aabb::Aabb;
use crate::bvh::Bvh;
use crate::integrals::VolumeIntegrals;
use crate::predicates::{orient2d, orient3d, P3};
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct TriMesh {
    pub verts: Vec<DVec3>,
    pub tris: Vec<[u32; 3]>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TopologyReport {
    pub boundary_edges: usize,
    pub nonmanifold_edges: usize,
    pub inconsistent_edges: usize,
    pub nonmanifold_vertices: usize,
    pub degenerate_tris: usize,
    pub unreferenced_vertices: usize,
}

impl TopologyReport {
    pub fn is_closed_manifold(&self) -> bool {
        self.boundary_edges == 0
            && self.nonmanifold_edges == 0
            && self.inconsistent_edges == 0
            && self.nonmanifold_vertices == 0
            && self.degenerate_tris == 0
    }
}

#[inline]
pub fn p3(v: DVec3) -> P3 {
    [v.x, v.y, v.z]
}

impl TriMesh {
    pub fn new(verts: Vec<DVec3>, tris: Vec<[u32; 3]>) -> Self {
        TriMesh { verts, tris }
    }
    pub fn aabb(&self) -> Aabb {
        Aabb::from_points(self.verts.iter())
    }
    pub fn tri_aabb(&self, t: usize) -> Aabb {
        let [a, b, c] = self.tris[t];
        Aabb::from_points([&self.verts[a as usize], &self.verts[b as usize], &self.verts[c as usize]])
    }
    pub fn tri_aabbs(&self) -> Vec<Aabb> {
        (0..self.tris.len()).map(|t| self.tri_aabb(t)).collect()
    }
    pub fn tri_points(&self, t: usize) -> [DVec3; 3] {
        let [a, b, c] = self.tris[t];
        [self.verts[a as usize], self.verts[b as usize], self.verts[c as usize]]
    }
    pub fn volume_integrals(&self) -> VolumeIntegrals {
        let mut vi = VolumeIntegrals::default();
        let r = if self.verts.is_empty() { DVec3::ZERO } else { self.aabb().center() };
        for t in 0..self.tris.len() {
            let [a, b, c] = self.tri_points(t);
            vi.add_tet(r, a, b, c);
        }
        vi
    }
    pub fn signed_volume(&self) -> f64 {
        self.volume_integrals().volume
    }
    pub fn area(&self) -> f64 {
        (0..self.tris.len())
            .map(|t| {
                let [a, b, c] = self.tri_points(t);
                0.5 * (b - a).cross(c - a).length()
            })
            .sum()
    }

    /// Exact degeneracy test: a triangle is degenerate when its vertices are
    /// collinear (or coincide).
    pub fn is_degenerate(&self, t: usize) -> bool {
        let [a, b, c] = self.tris[t];
        if a == b || b == c || a == c {
            return true;
        }
        let pa = self.verts[a as usize];
        let pb = self.verts[b as usize];
        let pc = self.verts[c as usize];
        let n = (pb - pa).cross(pc - pa);
        if n != DVec3::ZERO {
            return false;
        }
        // exact collinearity check via three projections
        let pr = |v: DVec3, i: usize, j: usize| [v[i], v[j]];
        for (i, j) in [(0, 1), (1, 2), (0, 2)] {
            if orient2d(&pr(pa, i, j), &pr(pb, i, j), &pr(pc, i, j)) != 0 {
                return false;
            }
        }
        true
    }

    /// Directed edge -> list of triangles using it.
    pub fn edge_map(&self) -> BTreeMap<(u32, u32), Vec<u32>> {
        let mut m: BTreeMap<(u32, u32), Vec<u32>> = BTreeMap::new();
        for (t, tri) in self.tris.iter().enumerate() {
            for k in 0..3 {
                m.entry((tri[k], tri[(k + 1) % 3])).or_default().push(t as u32);
            }
        }
        m
    }

    pub fn topology(&self) -> TopologyReport {
        // Allocation-light (hot in validation): sorted directed-edge list
        // instead of a map of per-edge vectors, CSR vertex->triangle lists.
        let mut rep = TopologyReport::default();
        let mut edges: Vec<(u32, u32)> = Vec::with_capacity(3 * self.tris.len());
        for tri in &self.tris {
            for k in 0..3 {
                edges.push((tri[k], tri[(k + 1) % 3]));
            }
        }
        edges.sort_unstable();
        let count = |e: (u32, u32)| -> usize {
            let lo = edges.partition_point(|x| *x < e);
            let hi = edges.partition_point(|x| *x <= e);
            hi - lo
        };
        let mut i = 0;
        while i < edges.len() {
            let (a, b) = edges[i];
            let mut j = i;
            while j < edges.len() && edges[j] == (a, b) {
                j += 1;
            }
            let c = j - i;
            let rev = count((b, a));
            if c > 1 {
                rep.inconsistent_edges += 1;
            }
            if a < b || rev == 0 {
                if rev == 0 {
                    rep.boundary_edges += 1;
                } else if c + rev > 2 {
                    rep.nonmanifold_edges += 1;
                }
            }
            i = j;
        }
        rep.degenerate_tris = (0..self.tris.len()).filter(|&t| self.is_degenerate(t)).count();
        // vertex manifoldness: the triangles around each vertex form one fan
        let nv = self.verts.len();
        let mut start = vec![0u32; nv + 1];
        for tri in &self.tris {
            for &v in tri {
                start[v as usize + 1] += 1;
            }
        }
        for v in 0..nv {
            start[v + 1] += start[v];
        }
        let mut fill = start.clone();
        let mut vt = vec![0u32; start[nv] as usize];
        for (t, tri) in self.tris.iter().enumerate() {
            for &v in tri {
                vt[fill[v as usize] as usize] = t as u32;
                fill[v as usize] += 1;
            }
        }
        fn find(p: &mut [usize], x: usize) -> usize {
            let mut r = x;
            while p[r] != r {
                r = p[r];
            }
            let mut y = x;
            while p[y] != r {
                let n = p[y];
                p[y] = r;
                y = n;
            }
            r
        }
        let mut parent: Vec<usize> = Vec::new();
        let mut others: Vec<(u32, usize)> = Vec::new();
        for v in 0..nv {
            let ts = &vt[start[v] as usize..start[v + 1] as usize];
            if ts.is_empty() {
                rep.unreferenced_vertices += 1;
                continue;
            }
            // union-find over triangles sharing an edge incident to v
            parent.clear();
            parent.extend(0..ts.len());
            others.clear();
            for (i, &t) in ts.iter().enumerate() {
                for &w in &self.tris[t as usize] {
                    if w as usize != v {
                        others.push((w, i));
                    }
                }
            }
            others.sort_unstable();
            let mut k = 0;
            while k < others.len() {
                let mut l = k + 1;
                while l < others.len() && others[l].0 == others[k].0 {
                    let a = find(&mut parent, others[k].1);
                    let b = find(&mut parent, others[l].1);
                    if a != b {
                        parent[a] = b;
                    }
                    l += 1;
                }
                k = l;
            }
            let roots = (0..ts.len()).filter(|&i| find(&mut parent, i) == i).count();
            if roots > 1 {
                rep.nonmanifold_vertices += 1;
            }
        }
        rep
    }

    /// Number of intersecting triangle pairs (exact). Adjacent triangles
    /// (sharing a vertex or an edge) are only reported when they overlap
    /// beyond their shared feature.
    pub fn self_intersections(&self, limit: usize) -> Vec<(u32, u32)> {
        let boxes = self.tri_aabbs();
        let bvh = Bvh::build(&boxes);
        let mut out = Vec::new();
        bvh.self_pairs(&boxes, |i, j| {
            if out.len() >= limit {
                return;
            }
            if tris_intersect(self, i as usize, j as usize) {
                out.push((i, j));
            }
        });
        out.sort_unstable();
        out
    }

    /// [`self_intersections`](Self::self_intersections) restricted to the
    /// candidate pairs `keep` accepts (tested before the exact predicate).
    pub fn self_intersections_where(&self, limit: usize, keep: impl Fn(u32, u32) -> bool) -> Vec<(u32, u32)> {
        let boxes = self.tri_aabbs();
        let bvh = Bvh::build(&boxes);
        let mut out = Vec::new();
        bvh.self_pairs(&boxes, |i, j| {
            if out.len() >= limit || !keep(i, j) {
                return;
            }
            if tris_intersect(self, i as usize, j as usize) {
                out.push((i, j));
            }
        });
        out.sort_unstable();
        out
    }

    /// Merge vertices with bit-identical coordinates; returns new mesh.
    pub fn weld_exact(&self) -> TriMesh {
        // ids by first appearance, so a hash map gives the same mesh
        let mut map: std::collections::HashMap<[u64; 3], u32> = std::collections::HashMap::with_capacity(self.verts.len());
        let mut remap = vec![0u32; self.verts.len()];
        let mut verts = Vec::new();
        for (i, v) in self.verts.iter().enumerate() {
            let k = [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()];
            let id = *map.entry(k).or_insert_with(|| {
                verts.push(*v);
                (verts.len() - 1) as u32
            });
            remap[i] = id;
        }
        let tris = self
            .tris
            .iter()
            .map(|t| [remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]])
            .filter(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2])
            .collect();
        TriMesh { verts, tris }
    }

    /// Merge vertices closer than `eps` (grid hashing, deterministic).
    pub fn weld(&self, eps: f64) -> TriMesh {
        if eps <= 0.0 {
            return self.weld_exact();
        }
        let key = |v: DVec3| [(v.x / eps).floor() as i64, (v.y / eps).floor() as i64, (v.z / eps).floor() as i64];
        let mut grid: BTreeMap<[i64; 3], Vec<u32>> = BTreeMap::new();
        let mut remap = vec![u32::MAX; self.verts.len()];
        let mut verts: Vec<DVec3> = Vec::new();
        for (i, &v) in self.verts.iter().enumerate() {
            let k = key(v);
            let mut found = None;
            'outer: for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(c) = grid.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) {
                            for &nid in c {
                                if (verts[nid as usize] - v).length() <= eps {
                                    found = Some(nid);
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
            }
            let id = match found {
                Some(id) => id,
                None => {
                    verts.push(v);
                    let id = (verts.len() - 1) as u32;
                    grid.entry(k).or_default().push(id);
                    id
                }
            };
            remap[i] = id;
        }
        let tris = self
            .tris
            .iter()
            .map(|t| [remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]])
            .filter(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2])
            .collect();
        TriMesh { verts, tris }
    }

    /// Remove unreferenced vertices.
    pub fn compact(&self) -> TriMesh {
        let mut used = vec![u32::MAX; self.verts.len()];
        let mut verts = Vec::new();
        let mut tris = Vec::with_capacity(self.tris.len());
        for t in &self.tris {
            let mut nt = [0u32; 3];
            for k in 0..3 {
                let v = t[k] as usize;
                if used[v] == u32::MAX {
                    used[v] = verts.len() as u32;
                    verts.push(self.verts[v]);
                }
                nt[k] = used[v];
            }
            tris.push(nt);
        }
        TriMesh { verts, tris }
    }

    pub fn flipped(&self) -> TriMesh {
        TriMesh { verts: self.verts.clone(), tris: self.tris.iter().map(|t| [t[0], t[2], t[1]]).collect() }
    }

    pub fn transformed(&self, f: impl Fn(DVec3) -> DVec3) -> TriMesh {
        TriMesh { verts: self.verts.iter().map(|&v| f(v)).collect(), tris: self.tris.clone() }
    }

    /// Append another mesh.
    pub fn append(&mut self, o: &TriMesh) {
        let off = self.verts.len() as u32;
        self.verts.extend_from_slice(&o.verts);
        self.tris.extend(o.tris.iter().map(|t| [t[0] + off, t[1] + off, t[2] + off]));
    }

    /// Connected components by shared edges; returns per-triangle labels.
    pub fn components(&self) -> (usize, Vec<u32>) {
        let n = self.tris.len();
        let mut parent: Vec<u32> = (0..n as u32).collect();
        fn find(p: &mut [u32], x: u32) -> u32 {
            let mut r = x;
            while p[r as usize] != r {
                r = p[r as usize];
            }
            let mut y = x;
            while p[y as usize] != r {
                let nx = p[y as usize];
                p[y as usize] = r;
                y = nx;
            }
            r
        }
        let mut em: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        for (t, tri) in self.tris.iter().enumerate() {
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                let key = (a.min(b), a.max(b));
                if let Some(&o) = em.get(&key) {
                    let ra = find(&mut parent, o);
                    let rb = find(&mut parent, t as u32);
                    if ra != rb {
                        parent[ra.max(rb) as usize] = ra.min(rb);
                    }
                } else {
                    em.insert(key, t as u32);
                }
            }
        }
        let mut label = vec![u32::MAX; n];
        let mut map: BTreeMap<u32, u32> = BTreeMap::new();
        for t in 0..n {
            let r = find(&mut parent, t as u32);
            let next = map.len() as u32;
            let l = *map.entry(r).or_insert(next);
            label[t] = l;
        }
        (map.len(), label)
    }

    /// Extract the sub-mesh of the given triangles (compacted).
    pub fn submesh(&self, tris: &[u32]) -> TriMesh {
        TriMesh { verts: self.verts.clone(), tris: tris.iter().map(|&t| self.tris[t as usize]).collect() }.compact()
    }
}

/// Exact segment–triangle intersection test (closed: touching counts).
pub fn segment_triangle(p: &P3, q: &P3, a: &P3, b: &P3, c: &P3) -> bool {
    let s1 = orient3d(a, b, c, p);
    let s2 = orient3d(a, b, c, q);
    if s1 * s2 > 0 {
        return false;
    }
    if s1 == 0 && s2 == 0 {
        return coplanar_segment_triangle(p, q, a, b, c);
    }
    let o1 = orient3d(p, q, a, b);
    let o2 = orient3d(p, q, b, c);
    let o3 = orient3d(p, q, c, a);
    let pos = (o1 >= 0) && (o2 >= 0) && (o3 >= 0);
    let neg = (o1 <= 0) && (o2 <= 0) && (o3 <= 0);
    pos || neg
}

fn dominant_axes(a: &P3, b: &P3, c: &P3) -> (usize, usize) {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    let an = [n[0].abs(), n[1].abs(), n[2].abs()];
    if an[0] >= an[1] && an[0] >= an[2] {
        (1, 2)
    } else if an[1] >= an[2] {
        (2, 0)
    } else {
        (0, 1)
    }
}

fn pr(p: &P3, ax: (usize, usize)) -> [f64; 2] {
    [p[ax.0], p[ax.1]]
}

fn seg_seg_2d(p: &[f64; 2], q: &[f64; 2], a: &[f64; 2], b: &[f64; 2]) -> bool {
    let d1 = orient2d(p, q, a);
    let d2 = orient2d(p, q, b);
    let d3 = orient2d(a, b, p);
    let d4 = orient2d(a, b, q);
    if d1 * d2 < 0 && d3 * d4 < 0 {
        return true;
    }
    let on = |x: &[f64; 2], y: &[f64; 2], z: &[f64; 2]| {
        z[0] >= x[0].min(y[0]) && z[0] <= x[0].max(y[0]) && z[1] >= x[1].min(y[1]) && z[1] <= x[1].max(y[1])
    };
    (d1 == 0 && on(p, q, a)) || (d2 == 0 && on(p, q, b)) || (d3 == 0 && on(a, b, p)) || (d4 == 0 && on(a, b, q))
}

fn point_in_tri_2d(p: &[f64; 2], a: &[f64; 2], b: &[f64; 2], c: &[f64; 2]) -> bool {
    let o = orient2d(a, b, c);
    let s1 = orient2d(a, b, p) * o;
    let s2 = orient2d(b, c, p) * o;
    let s3 = orient2d(c, a, p) * o;
    s1 >= 0 && s2 >= 0 && s3 >= 0
}

fn coplanar_segment_triangle(p: &P3, q: &P3, a: &P3, b: &P3, c: &P3) -> bool {
    let ax = dominant_axes(a, b, c);
    let (p2, q2, a2, b2, c2) = (pr(p, ax), pr(q, ax), pr(a, ax), pr(b, ax), pr(c, ax));
    point_in_tri_2d(&p2, &a2, &b2, &c2)
        || point_in_tri_2d(&q2, &a2, &b2, &c2)
        || seg_seg_2d(&p2, &q2, &a2, &b2)
        || seg_seg_2d(&p2, &q2, &b2, &c2)
        || seg_seg_2d(&p2, &q2, &c2, &a2)
}

/// Do two mesh triangles intersect beyond their shared topology?
pub fn tris_intersect(m: &TriMesh, i: usize, j: usize) -> bool {
    let ti = m.tris[i];
    let tj = m.tris[j];
    // shared vertices without allocating (hot: called per candidate pair)
    let mut shared = [0u32; 3];
    let mut ns = 0usize;
    for &v in &ti {
        if tj.contains(&v) {
            shared[ns] = v;
            ns += 1;
        }
    }
    let others = |t: [u32; 3], s: u32| -> [u32; 2] {
        let mut o = [0u32; 2];
        let mut k = 0;
        for &v in &t {
            if v != s && k < 2 {
                o[k] = v;
                k += 1;
            }
        }
        o
    };
    let pt = |v: u32| p3(m.verts[v as usize]);
    match ns {
        0 => {
            let a = [pt(ti[0]), pt(ti[1]), pt(ti[2])];
            let b = [pt(tj[0]), pt(tj[1]), pt(tj[2])];
            // exact plane-side rejection: one triangle strictly on one side
            // of the other's plane cannot meet it
            let strictly_one_side = |p: &[P3; 3], q: &[P3; 3]| {
                let o = [orient3d(&p[0], &p[1], &p[2], &q[0]), orient3d(&p[0], &p[1], &p[2], &q[1]), orient3d(&p[0], &p[1], &p[2], &q[2])];
                o[0] != 0 && o[0] == o[1] && o[1] == o[2]
            };
            if strictly_one_side(&b, &a) || strictly_one_side(&a, &b) {
                return false;
            }
            for k in 0..3 {
                if segment_triangle(&a[k], &a[(k + 1) % 3], &b[0], &b[1], &b[2]) {
                    return true;
                }
                if segment_triangle(&b[k], &b[(k + 1) % 3], &a[0], &a[1], &a[2]) {
                    return true;
                }
            }
            false
        }
        1 => {
            let s = shared[0];
            let (oi, oj) = (others(ti, s), others(tj, s));
            let (ps, pi0, pi1, pj0, pj1) = (pt(s), pt(oi[0]), pt(oi[1]), pt(oj[0]), pt(oj[1]));
            // exact rejection: the rest of one triangle strictly on one side
            // of the other's plane leaves only the shared vertex in common
            let (o0, o1) = (orient3d(&ps, &pi0, &pi1, &pj0), orient3d(&ps, &pi0, &pi1, &pj1));
            if o0 != 0 && o0 == o1 {
                return false;
            }
            let (q0, q1) = (orient3d(&ps, &pj0, &pj1, &pi0), orient3d(&ps, &pj0, &pj1, &pi1));
            if q0 != 0 && q0 == q1 {
                return false;
            }
            // Opposite edges against the other triangle
            if segment_triangle(&pi0, &pi1, &ps, &pj0, &pj1) || segment_triangle(&pj0, &pj1, &ps, &pi0, &pi1) {
                return true;
            }
            // Coplanar overlap around the shared vertex
            if orient3d(&ps, &pi0, &pi1, &pj0) == 0 && orient3d(&ps, &pi0, &pi1, &pj1) == 0 {
                let ax = dominant_axes(&ps, &pi0, &pi1);
                let (s2, a0, a1, b0, b1) = (pr(&ps, ax), pr(&pi0, ax), pr(&pi1, ax), pr(&pj0, ax), pr(&pj1, ax));
                // a small step from the shared vertex into triangle j inside triangle i?
                let inside = |x: &[f64; 2]| point_in_tri_2d(x, &s2, &a0, &a1) && orient2d(&s2, x, &a0) != 0 && orient2d(&s2, x, &a1) != 0;
                let mid = [(b0[0] + b1[0]) * 0.5, (b0[1] + b1[1]) * 0.5];
                let mid2 = [(a0[0] + a1[0]) * 0.5, (a0[1] + a1[1]) * 0.5];
                let inside2 = |x: &[f64; 2]| point_in_tri_2d(x, &s2, &b0, &b1) && orient2d(&s2, x, &b0) != 0 && orient2d(&s2, x, &b1) != 0;
                if inside(&mid) || inside2(&mid2) {
                    return true;
                }
            }
            false
        }
        2 => {
            let oi = ti.iter().copied().find(|v| !tj.contains(v)).unwrap();
            let oj = tj.iter().copied().find(|v| !ti.contains(v)).unwrap();
            let (u, v) = (pt(shared[0]), pt(shared[1]));
            let (a, b) = (pt(oi), pt(oj));
            if orient3d(&u, &v, &a, &b) != 0 {
                return false;
            }
            // coplanar: fold-over iff a and b on the same side of line uv
            let ax = dominant_axes(&u, &v, &a);
            let (u2, v2, a2, b2) = (pr(&u, ax), pr(&v, ax), pr(&a, ax), pr(&b, ax));
            orient2d(&u2, &v2, &a2) == orient2d(&u2, &v2, &b2)
        }
        _ => true, // duplicate triangle
    }
}

/// Axis-aligned box mesh (outward oriented), useful for tests and bounds.
pub fn box_mesh(lo: DVec3, hi: DVec3) -> TriMesh {
    let v = (0..8)
        .map(|i| {
            DVec3::new(
                if i & 1 == 0 { lo.x } else { hi.x },
                if i & 2 == 0 { lo.y } else { hi.y },
                if i & 4 == 0 { lo.z } else { hi.z },
            )
        })
        .collect();
    let quads = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
    let mut tris = Vec::new();
    for q in quads {
        tris.push([q[0], q[1], q[2]]);
        tris.push([q[0], q[2], q[3]]);
    }
    TriMesh { verts: v, tris }
}

/// UV sphere-like icosphere (outward oriented), subdivided `level` times.
pub fn icosphere(center: DVec3, radius: f64, level: u32) -> TriMesh {
    let t = (1.0 + 5f64.sqrt()) / 2.0;
    let mut verts: Vec<DVec3> = [
        [-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0],
        [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t],
        [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0],
    ]
    .iter()
    .map(|p| DVec3::from_array(*p).normalize())
    .collect();
    let mut tris: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    for _ in 0..level {
        let mut mid: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        let mut nt = Vec::with_capacity(tris.len() * 4);
        let mut m = |a: u32, b: u32, verts: &mut Vec<DVec3>| -> u32 {
            let k = (a.min(b), a.max(b));
            *mid.entry(k).or_insert_with(|| {
                verts.push(((verts[a as usize] + verts[b as usize]) * 0.5).normalize());
                (verts.len() - 1) as u32
            })
        };
        for &[a, b, c] in &tris {
            let ab = m(a, b, &mut verts);
            let bc = m(b, c, &mut verts);
            let ca = m(c, a, &mut verts);
            nt.extend_from_slice(&[[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        tris = nt;
    }
    TriMesh { verts: verts.into_iter().map(|v| center + v * radius).collect(), tris }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_is_closed() {
        let m = box_mesh(DVec3::ZERO, DVec3::ONE);
        assert!(m.topology().is_closed_manifold());
        assert!((m.signed_volume() - 1.0).abs() < 1e-14);
        assert!(m.self_intersections(10).is_empty());
        let s = icosphere(DVec3::ZERO, 1.0, 2);
        assert!(s.topology().is_closed_manifold());
        assert!(s.signed_volume() > 4.0);
        assert!(s.self_intersections(10).is_empty());
    }

    #[test]
    fn detects_intersection() {
        let mut m = box_mesh(DVec3::ZERO, DVec3::ONE);
        m.append(&box_mesh(DVec3::splat(0.5), DVec3::splat(1.5)));
        assert!(!m.self_intersections(10).is_empty());
    }
}
