//! Convex hulls (exact incremental construction) and convex polytopes with
//! plane clipping, used for collision shapes and validation.

use crate::integrals::VolumeIntegrals;
use crate::mesh::{p3, TriMesh};
use crate::predicates::orient3d;
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Exact 3D convex hull of a point set (incremental, exact orientation).
/// Returns a closed outward-oriented triangle mesh over a subset of the input
/// points, or `None` when the points are coplanar/degenerate.
pub fn convex_hull(points: &[DVec3]) -> Option<TriMesh> {
    hull_impl(points, false)
}

/// Floating-point Quickhull with a relative visibility tolerance (points
/// within `1e-12·scale²·|n|` of a face plane count as on it). Much faster on
/// inputs with many coplanar points; intended for collision shapes, where
/// exactness is not required.
pub fn convex_hull_fast(points: &[DVec3]) -> Option<TriMesh> {
    hull_impl(points, true)
}

fn hull_impl(points: &[DVec3], fast: bool) -> Option<TriMesh> {
    let n = points.len();
    if n < 4 {
        return None;
    }
    let p = |i: usize| p3(points[i]);
    // initial simplex: extreme x, farthest from it, farthest from line, farthest from plane
    let mut i0 = 0;
    for i in 1..n {
        if (points[i].x, points[i].y, points[i].z) < (points[i0].x, points[i0].y, points[i0].z) {
            i0 = i;
        }
    }
    let mut i1 = i0;
    let mut best = 0.0;
    for i in 0..n {
        let d = (points[i] - points[i0]).length_squared();
        if d > best {
            best = d;
            i1 = i;
        }
    }
    if i1 == i0 {
        return None;
    }
    let mut i2 = usize::MAX;
    best = 0.0;
    let dir = (points[i1] - points[i0]).normalize();
    for i in 0..n {
        let v = points[i] - points[i0];
        let d = (v - dir * v.dot(dir)).length_squared();
        if d > best {
            best = d;
            i2 = i;
        }
    }
    if i2 == usize::MAX {
        return None;
    }
    let mut i3 = usize::MAX;
    best = 0.0;
    let nrm = (points[i1] - points[i0]).cross(points[i2] - points[i0]);
    for i in 0..n {
        let d = (points[i] - points[i0]).dot(nrm).abs();
        if d > best && orient3d(&p(i0), &p(i1), &p(i2), &p(i)) != 0 {
            best = d;
            i3 = i;
        }
    }
    if i3 == usize::MAX {
        // try exact search
        for i in 0..n {
            if orient3d(&p(i0), &p(i1), &p(i2), &p(i)) != 0 {
                i3 = i;
                break;
            }
        }
        if i3 == usize::MAX {
            return None;
        }
    }
    let (a, b, c, d) = if orient3d(&p(i0), &p(i1), &p(i2), &p(i3)) > 0 { (i0, i2, i1, i3) } else { (i0, i1, i2, i3) };
    // Quickhull with conflict (outside) sets; visibility by exact orient3d.
    struct Face {
        v: [usize; 3],
        /// Unnormalized normal `(y-x)×(z-x)` and its length (cached).
        n: DVec3,
        l: f64,
        outside: Vec<usize>,
        alive: bool,
    }
    let mk = |v: [usize; 3]| -> Face {
        let (x, y, z) = (points[v[0]], points[v[1]], points[v[2]]);
        let n = (y - x).cross(z - x);
        Face { v, n, l: n.length(), outside: Vec::new(), alive: true }
    };
    let mut faces: Vec<Face> = [[a, b, c], [a, d, b], [b, d, c], [c, d, a]].iter().map(|&v| mk(v)).collect();
    let mut edge_face: std::collections::HashMap<(usize, usize), usize, EdgeHash> = std::collections::HashMap::with_capacity_and_hasher(4 * n.min(4096), EdgeHash);
    // visited stamps per face (faces are only appended)
    let mut stamp: Vec<u32> = Vec::new();
    let mut round: u32 = 0;
    for (fi, f) in faces.iter().enumerate() {
        for k in 0..3 {
            edge_face.insert((f.v[k], f.v[(k + 1) % 3]), fi);
        }
    }
    let scale = crate::aabb::Aabb::from_points(points.iter()).diagonal().max(1e-300);
    let tol = 1e-11 * scale;
    let sees = |f: &Face, q: usize| {
        if fast {
            f.l > 0.0 && f.n.dot(points[q] - points[f.v[0]]) > tol * f.l
        } else {
            let f = &f.v;
            orient3d(&p(f[0]), &p(f[1]), &p(f[2]), &p(q)) > 0
        }
    };
    let dist = |f: &Face, q: usize| -> f64 { f.n.dot(points[q] - points[f.v[0]]) };
    let used = [a, b, c, d];
    for q in 0..n {
        if used.contains(&q) {
            continue;
        }
        for f in faces.iter_mut() {
            if sees(f, q) {
                f.outside.push(q);
                break;
            }
        }
    }
    let mut queue: std::collections::VecDeque<usize> = (0..4).collect();
    while let Some(fi) = queue.pop_front() {
        if !faces[fi].alive || faces[fi].outside.is_empty() {
            continue;
        }
        // farthest outside point (ties: smallest index)
        let ff = &faces[fi];
        let apex = *ff.outside.iter().max_by(|&&x, &&y| dist(ff, x).partial_cmp(&dist(ff, y)).unwrap().then(y.cmp(&x))).unwrap();
        // visible region (connected) by BFS over edge neighbors
        let mut visible = vec![fi];
        round += 1;
        stamp.resize(faces.len(), 0);
        stamp[fi] = round;
        let mut k = 0;
        while k < visible.len() {
            let f = visible[k];
            k += 1;
            let v = faces[f].v;
            for e in 0..3 {
                if let Some(&g) = edge_face.get(&(v[(e + 1) % 3], v[e])) {
                    if faces[g].alive && stamp[g] != round && sees(&faces[g], apex) {
                        stamp[g] = round;
                        visible.push(g);
                    }
                }
            }
        }
        // horizon edges, in the orientation of the visible faces
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        for &f in &visible {
            let v = faces[f].v;
            for e in 0..3 {
                let (u, w) = (v[e], v[(e + 1) % 3]);
                let nb = edge_face.get(&(w, u)).copied();
                if nb.map(|g| stamp[g] != round).unwrap_or(true) {
                    horizon.push((u, w));
                }
            }
        }
        horizon.sort_unstable();
        let mut orphans: Vec<usize> = Vec::new();
        for &f in &visible {
            faces[f].alive = false;
            let v = faces[f].v;
            for e in 0..3 {
                if edge_face.get(&(v[e], v[(e + 1) % 3])) == Some(&f) {
                    edge_face.remove(&(v[e], v[(e + 1) % 3]));
                }
            }
            orphans.extend(std::mem::take(&mut faces[f].outside));
        }
        orphans.sort_unstable();
        orphans.dedup();
        let first_new = faces.len();
        for (u, w) in horizon {
            let fi2 = faces.len();
            faces.push(mk([u, w, apex]));
            for (x, y) in [(u, w), (w, apex), (apex, u)] {
                edge_face.insert((x, y), fi2);
            }
        }
        for q in orphans {
            if q == apex {
                continue;
            }
            for f2 in first_new..faces.len() {
                if sees(&faces[f2], q) {
                    faces[f2].outside.push(q);
                    break;
                }
            }
        }
        for f2 in first_new..faces.len() {
            if !faces[f2].outside.is_empty() {
                queue.push_back(f2);
            }
        }
    }
    let faces: Vec<Option<[usize; 3]>> = faces.into_iter().map(|f| if f.alive { Some(f.v) } else { None }).collect();
    let tris_idx: Vec<[usize; 3]> = faces.into_iter().flatten().collect();
    let mut map: BTreeMap<usize, u32> = BTreeMap::new();
    for t in &tris_idx {
        for &v in t {
            let k = map.len() as u32;
            map.entry(v).or_insert(k);
        }
    }
    // re-number in ascending original index for determinism
    let mut keys: Vec<usize> = map.keys().copied().collect();
    keys.sort_unstable();
    let remap: BTreeMap<usize, u32> = keys.iter().enumerate().map(|(i, &k)| (k, i as u32)).collect();
    let verts = keys.iter().map(|&k| points[k]).collect();
    let tris = tris_idx.iter().map(|t| [remap[&t[0]], remap[&t[1]], remap[&t[2]]]).collect();
    Some(TriMesh { verts, tris })
}

/// Cheap deterministic hasher for vertex-index edge keys (lookups only;
/// map iteration order is never used).
#[derive(Clone, Copy, Default)]
struct EdgeHash;
impl std::hash::BuildHasher for EdgeHash {
    type Hasher = EdgeHasher;
    fn build_hasher(&self) -> EdgeHasher {
        EdgeHasher(0)
    }
}
struct EdgeHasher(u64);
impl std::hash::Hasher for EdgeHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
        }
    }
    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

/// Plane `n·x <= d` (n unit length) bounding a convex polytope.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct HalfSpace {
    pub n: DVec3,
    pub d: f64,
}

impl HalfSpace {
    #[inline]
    pub fn dist(&self, x: DVec3) -> f64 {
        self.n.dot(x) - self.d
    }
}

/// A convex polytope stored as a set of planar faces (each a CCW polygon
/// w.r.t. its outward normal). Floating-point; used for collision hulls.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ConvexPolytope {
    pub faces: Vec<(HalfSpace, Vec<DVec3>)>,
}

impl ConvexPolytope {
    pub fn from_box(lo: DVec3, hi: DVec3) -> Self {
        let m = crate::mesh::box_mesh(lo, hi);
        Self::from_hull_mesh(&m)
    }

    /// From a closed convex triangle mesh (e.g. `convex_hull` output);
    /// coplanar triangles are merged into polygon faces.
    pub fn from_hull_mesh(m: &TriMesh) -> Self {
        let mut faces = Vec::new();
        // group by plane (tolerance-based merge of coplanar adjacent tris)
        let mut done = vec![false; m.tris.len()];
        let scale = m.aabb().diagonal().max(1e-300);
        let em = m.edge_map();
        for t in 0..m.tris.len() {
            if done[t] {
                continue;
            }
            let [a, b, c] = m.tri_points(t);
            let nv = (b - a).cross(c - a);
            if nv.length_squared() == 0.0 {
                done[t] = true;
                continue;
            }
            let n = nv.normalize();
            let d = n.dot(a);
            // flood fill coplanar neighbors
            let mut group = vec![t];
            done[t] = true;
            let mut k = 0;
            while k < group.len() {
                let g = group[k];
                k += 1;
                let tri = m.tris[g];
                for e in 0..3 {
                    let (u, v) = (tri[e], tri[(e + 1) % 3]);
                    if let Some(ns) = em.get(&(v, u)) {
                        for &o in ns {
                            let o = o as usize;
                            if done[o] {
                                continue;
                            }
                            let pts = m.tri_points(o);
                            if pts.iter().all(|p| (n.dot(*p) - d).abs() <= 1e-12 * scale) {
                                done[o] = true;
                                group.push(o);
                            }
                        }
                    }
                }
            }
            // boundary loop of the group
            let mut dir: BTreeMap<u32, u32> = BTreeMap::new();
            let mut set = std::collections::BTreeSet::new();
            for &g in &group {
                let tri = m.tris[g];
                for e in 0..3 {
                    set.insert((tri[e], tri[(e + 1) % 3]));
                }
            }
            for &(u, v) in &set {
                if !set.contains(&(v, u)) {
                    dir.insert(u, v);
                }
            }
            if let Some((&start, _)) = dir.iter().next() {
                let mut poly = vec![m.verts[start as usize]];
                let mut cur = dir[&start];
                let mut guard = 0;
                while cur != start && guard < dir.len() + 2 {
                    poly.push(m.verts[cur as usize]);
                    cur = match dir.get(&cur) {
                        Some(&x) => x,
                        None => break,
                    };
                    guard += 1;
                }
                faces.push((HalfSpace { n, d }, poly));
            }
        }
        ConvexPolytope { faces }
    }

    pub fn from_points(points: &[DVec3]) -> Option<Self> {
        convex_hull_fast(points).map(|m| Self::from_hull_mesh(&m))
    }

    pub fn vertices(&self) -> Vec<DVec3> {
        let mut v: Vec<DVec3> = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for (_, f) in &self.faces {
            for p in f {
                if seen.insert([p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]) {
                    v.push(*p);
                }
            }
        }
        v
    }

    pub fn is_empty(&self) -> bool {
        self.faces.len() < 4
    }

    pub fn volume_integrals(&self) -> VolumeIntegrals {
        let mut vi = VolumeIntegrals::default();
        let r = self.faces.first().and_then(|f| f.1.first().copied()).unwrap_or(DVec3::ZERO);
        for (_, f) in &self.faces {
            vi.add_polygon(r, f);
        }
        vi
    }

    pub fn volume(&self) -> f64 {
        self.volume_integrals().volume.max(0.0)
    }

    pub fn scale(&self) -> f64 {
        let vs = self.vertices();
        crate::aabb::Aabb::from_points(vs.iter()).diagonal()
    }

    /// Keep the part with `h.dist(x) <= 0`.
    pub fn clip(&self, h: &HalfSpace) -> ConvexPolytope {
        let eps = 1e-12 * self.scale().max(1e-300);
        let mut out = Vec::new();
        let mut cut_pts: Vec<DVec3> = Vec::new();
        for (hs, poly) in &self.faces {
            let n = poly.len();
            let mut res = Vec::with_capacity(n + 2);
            for i in 0..n {
                let a = poly[i];
                let b = poly[(i + 1) % n];
                let da = h.dist(a);
                let db = h.dist(b);
                let ain = da <= eps;
                let bin = db <= eps;
                if ain {
                    res.push(a);
                    if da.abs() <= eps {
                        cut_pts.push(a);
                    }
                }
                if ain != bin && da.abs() > eps && db.abs() > eps {
                    let t = da / (da - db);
                    let x = a + (b - a) * t;
                    res.push(x);
                    cut_pts.push(x);
                }
            }
            if res.len() >= 3 {
                out.push((*hs, res));
            }
        }
        // cap polygon
        if cut_pts.len() >= 3 {
            let c = cut_pts.iter().fold(DVec3::ZERO, |a, &p| a + p) / cut_pts.len() as f64;
            let (u, v) = crate::polygon::plane_basis(h.n);
            let mut pts: Vec<(f64, DVec3)> = cut_pts.iter().map(|&p| (libm::atan2((p - c).dot(v), (p - c).dot(u)), p)).collect();
            pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let mut cap: Vec<DVec3> = Vec::new();
            for (_, p) in pts {
                if cap.last().map(|q: &DVec3| (*q - p).length() > eps * 10.0).unwrap_or(true) {
                    cap.push(p);
                }
            }
            while cap.len() > 1 && (cap[0] - *cap.last().unwrap()).length() <= eps * 10.0 {
                cap.pop();
            }
            if cap.len() >= 3 {
                out.push((*h, cap));
            }
        }
        let res = ConvexPolytope { faces: out };
        if res.faces.len() < 4 { ConvexPolytope::default() } else { res }
    }

    pub fn clip_all(&self, hs: &[HalfSpace]) -> ConvexPolytope {
        let mut p = self.clone();
        for h in hs {
            if p.is_empty() {
                break;
            }
            p = p.clip(h);
        }
        p
    }

    pub fn halfspaces(&self) -> Vec<HalfSpace> {
        self.faces.iter().map(|f| f.0).collect()
    }

    /// Shrink by moving every face plane inward by `margin`.
    pub fn shrunk(&self, margin: f64) -> ConvexPolytope {
        if margin <= 0.0 {
            return self.clone();
        }
        let vs = self.vertices();
        let bb = crate::aabb::Aabb::from_points(vs.iter()).expanded(1.0);
        let base = ConvexPolytope::from_box(bb.min, bb.max);
        let hs: Vec<HalfSpace> = self.faces.iter().map(|(h, _)| HalfSpace { n: h.n, d: h.d - margin }).collect();
        base.clip_all(&hs)
    }

    /// Intersection volume with another polytope.
    pub fn intersection_volume(&self, o: &ConvexPolytope) -> f64 {
        if self.is_empty() || o.is_empty() {
            return 0.0;
        }
        self.clip_all(&o.halfspaces()).volume()
    }

    /// Triangle mesh (fan per face, welded).
    pub fn to_mesh(&self) -> TriMesh {
        let mut m = TriMesh::default();
        for (_, f) in &self.faces {
            let base = m.verts.len() as u32;
            m.verts.extend_from_slice(f);
            for i in 1..f.len() as u32 - 1 {
                m.tris.push([base, base + i, base + i + 1]);
            }
        }
        m.weld_exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hull_of_cube_points() {
        let mut pts = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    pts.push(DVec3::new(i as f64, j as f64, k as f64) * 0.5);
                }
            }
        }
        let h = convex_hull(&pts).unwrap();
        assert!(h.topology().is_closed_manifold());
        assert!((h.signed_volume() - 1.0).abs() < 1e-12);
        let poly = ConvexPolytope::from_hull_mesh(&h);
        assert_eq!(poly.faces.len(), 6);
        assert!((poly.volume() - 1.0).abs() < 1e-12);
        let c = poly.clip(&HalfSpace { n: DVec3::X, d: 0.25 });
        assert!((c.volume() - 0.25).abs() < 1e-12);
        let s = poly.shrunk(0.1);
        assert!((s.volume() - 0.8f64.powi(3)).abs() < 1e-10);
        let other = ConvexPolytope::from_box(DVec3::splat(0.5), DVec3::splat(1.5));
        assert!((poly.intersection_volume(&other) - 0.125).abs() < 1e-12);
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    #[test]
    #[ignore]
    fn hull_speed() {
        let m = crate::mesh::icosphere(DVec3::ZERO, 1.0, 4);
        let t = std::time::Instant::now();
        let h = convex_hull(&m.verts).unwrap();
        eprintln!("{} pts -> {} tris in {:?}", m.verts.len(), h.tris.len(), t.elapsed());
    }
}
