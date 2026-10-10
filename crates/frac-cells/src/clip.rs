//! Exact clipping of a closed, oriented, manifold solid by a convex cell
//! complex.
//!
//! Output vertices are *symbolic* ([`VKey`]): an input vertex, a mesh edge
//! crossing a plane, a mesh triangle pierced by a complex line, or a complex
//! vertex. Coordinates are a pure function of the key, and every topological
//! decision is an exact (filtered + SoS) predicate from [`crate::planes`].
//! Consequently each interface patch is computed **once** and is bit-identical
//! for both adjacent cells, and every cell boundary is watertight by
//! construction.

use crate::complex::Complex;
use crate::delaunay::NONE;
use crate::planes::{LineKey, PlaneId, P3};
use crate::tri2d;
use frac_geom::bvh::Bvh;
use frac_geom::{Aabb, DVec3, TriMesh};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VKey {
    /// Input mesh vertex.
    Orig(u32),
    /// Mesh edge `(a < b)` crossing a plane.
    Ep(u32, u32, PlaneId),
    /// Mesh triangle pierced by a canonical line.
    Tl(u32, LineKey),
    /// Complex vertex.
    Cv(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tag {
    Mesh(u32, u32),
    Plane(PlaneId),
}

/// An exterior polygon (part of the solid surface inside one complex cell).
#[derive(Clone, Debug)]
pub struct ExtPolyOut {
    pub verts: Vec<u32>,
    pub cell: u32,
    pub src_tri: u32,
    /// Triangulation without zero-area triangles (see [`triangulate_poly`]).
    pub tris: Vec<[u32; 3]>,
}

/// Triangulate a planar polygon given by vertex ids, keeping collinear
/// (T-junction) vertices and avoiding zero-area triangles when possible.
pub fn triangulate_poly(verts: &[DVec3], poly: &[u32]) -> Vec<[u32; 3]> {
    if poly.len() == 3 {
        return vec![[poly[0], poly[1], poly[2]]];
    }
    let pts: Vec<DVec3> = poly.iter().map(|&v| verts[v as usize]).collect();
    let n = frac_geom::polygon::newell(&pts);
    if n.length_squared() == 0.0 {
        return (1..poly.len() - 1).map(|k| [poly[0], poly[k], poly[k + 1]]).collect();
    }
    let (u, v) = frac_geom::polygon::plane_basis(n.normalize());
    let p2: Vec<[f64; 2]> = pts.iter().map(|p| [p.dot(u), p.dot(v)]).collect();
    tri2d::triangulate(&p2, &[(0..poly.len()).collect()]).into_iter().map(|t| [poly[t[0]], poly[t[1]], poly[t[2]]]).collect()
}

/// One connected planar interface patch (outer loop + holes).
#[derive(Clone, Debug)]
pub struct PatchOut {
    pub loops: Vec<Vec<u32>>,
    pub tris: Vec<[u32; 3]>,
    pub normal: DVec3,
    /// (negative side, positive side) complex cell.
    pub cells: [u32; 2],
    pub face: u32,
    pub area: f64,
}

#[derive(Clone, Debug, Default)]
pub struct ClipOutput {
    pub verts: Vec<DVec3>,
    pub keys: Vec<VKey>,
    pub ext: Vec<ExtPolyOut>,
    pub patches: Vec<PatchOut>,
    pub warnings: Vec<String>,
}

pub struct Clipper<'a> {
    pub mesh: &'a TriMesh,
    pub cx: &'a Complex,
    pts: Vec<P3>,
    bvh: Bvh,
    bbox: Aabb,
    /// (cell, plane) -> neighbor cells across that plane.
    nbr: Vec<BTreeMap<PlaneId, Vec<u32>>>,
    /// Facet subdivision edge segments per cell: plane -> [(q, ends)].
    subdiv: Vec<BTreeMap<PlaneId, Vec<(PlaneId, [(PlaneId, i8); 2])>>>,
}

impl<'a> Clipper<'a> {
    pub fn new(mesh: &'a TriMesh, cx: &'a Complex) -> Self {
        let pts: Vec<P3> = mesh.verts.iter().map(|v| [v.x, v.y, v.z]).collect();
        let boxes = mesh.tri_aabbs();
        let bvh = Bvh::build(&boxes);
        let bbox = mesh.aabb();
        let mut nbr: Vec<BTreeMap<PlaneId, Vec<u32>>> = vec![BTreeMap::new(); cx.cells.len()];
        for f in &cx.faces {
            let [a, b] = f.cells;
            if a != NONE && b != NONE {
                nbr[a as usize].entry(f.plane).or_default().push(b);
                nbr[b as usize].entry(f.plane).or_default().push(a);
            }
        }
        for m in nbr.iter_mut() {
            for v in m.values_mut() {
                v.sort_unstable();
                v.dedup();
            }
        }
        let subdiv = cx.cells.iter().map(|c| c.facet_subdiv.iter().cloned().collect()).collect();
        Clipper { mesh, cx, pts, bvh, bbox, nbr, subdiv }
    }

    #[inline]
    fn tri(&self, t: u32) -> [&P3; 3] {
        let [a, b, c] = self.mesh.tris[t as usize];
        [&self.pts[a as usize], &self.pts[b as usize], &self.pts[c as usize]]
    }

    /// Side of a symbolic vertex (lying on triangle `t`) w.r.t. plane `q`.
    fn vsign(&self, k: &VKey, t: u32, q: PlaneId) -> i8 {
        let ps = &self.cx.planes;
        match *k {
            VKey::Orig(v) => ps.side_vertex(&self.pts[v as usize], q),
            VKey::Ep(a, b, p) => {
                if p == q {
                    0
                } else {
                    ps.side_edge_point(&self.pts[a as usize], &self.pts[b as usize], p, q)
                }
            }
            VKey::Tl(tt, l) => {
                debug_assert_eq!(tt, t);
                if ps.line_contains(l, q) {
                    0
                } else {
                    ps.side_tri_line(self.tri(tt), l.0, l.1, q)
                }
            }
            VKey::Cv(_) => unreachable!("complex vertex in surface polygon"),
        }
    }

    fn cut_key(&self, t: u32, tag: Tag, q: PlaneId) -> VKey {
        match tag {
            Tag::Mesh(a, b) => VKey::Ep(a.min(b), a.max(b), q),
            Tag::Plane(p) => VKey::Tl(t, self.cx.planes.line(p, q)),
        }
    }

    /// Sutherland–Hodgman clip of a tagged polygon on triangle `t` by the
    /// half-space `sgn * f_q <= 0`.
    fn clip_poly(&self, t: u32, poly: &[(VKey, Tag)], q: PlaneId, sgn: i8) -> Vec<(VKey, Tag)> {
        let n = poly.len();
        let s: Vec<i8> = poly.iter().map(|(k, _)| sgn * self.vsign(k, t, q)).collect();
        if s.iter().all(|&x| x <= 0) {
            return poly.to_vec();
        }
        if s.iter().all(|&x| x >= 0) {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(n + 2);
        for i in 0..n {
            let j = (i + 1) % n;
            let (ki, tagi) = poly[i];
            let (si, sj) = (s[i], s[j]);
            if si <= 0 {
                let tag = if si == 0 && sj > 0 { Tag::Plane(q) } else { tagi };
                out.push((ki, tag));
            }
            if (si < 0 && sj > 0) || (si > 0 && sj < 0) {
                let x = self.cut_key(t, tagi, q);
                if si < 0 {
                    // exiting: edge from x runs along q to the re-entry point
                    out.push((x, Tag::Plane(q)));
                } else {
                    out.push((x, tagi));
                }
            }
        }
        if out.len() < 3 { Vec::new() } else { out }
    }

    /// Clip triangle `t` by complex cell `c`; returns the tagged polygon.
    fn clip_tri_cell(&self, t: u32, c: u32) -> Vec<(VKey, Tag)> {
        let [a, b, cc] = self.mesh.tris[t as usize];
        let mut poly = vec![(VKey::Orig(a), Tag::Mesh(a, b)), (VKey::Orig(b), Tag::Mesh(b, cc)), (VKey::Orig(cc), Tag::Mesh(cc, a))];
        for &(q, sgn) in &self.cx.cells[c as usize].halfspaces {
            poly = self.clip_poly(t, &poly, q, sgn);
            if poly.is_empty() {
                return poly;
            }
        }
        // T-junction splits on subdivided facets (box complexes)
        if !self.subdiv[c as usize].is_empty() {
            let mut out: Vec<(VKey, Tag)> = Vec::with_capacity(poly.len() + 4);
            let n = poly.len();
            for i in 0..n {
                let (ki, tagi) = poly[i];
                out.push((ki, tagi));
                if let Tag::Plane(p) = tagi {
                    if let Some(qs) = self.subdiv[c as usize].get(&p) {
                        let kj = poly[(i + 1) % n].0;
                        let mut ins: Vec<(PlaneId, VKey)> = Vec::new();
                        for &(q, ends) in qs {
                            let si = self.vsign(&ki, t, q);
                            let sj = self.vsign(&kj, t, q);
                            if si * sj >= 0 {
                                continue;
                            }
                            let l = self.cx.planes.line(p, q);
                            let tri = self.tri(t);
                            let within = ends.iter().all(|&(r, sg)| sg * self.cx.planes.side_tri_line(tri, l.0, l.1, r) < 0);
                            if within && !ins.iter().any(|x| x.0 == q) {
                                ins.push((q, VKey::Tl(t, l)));
                            }
                        }
                        // exact order along ki -> kj: A before B iff B lies on
                        // kj's side of A's plane
                        if ins.len() > 1 {
                            let mut sorted: Vec<(PlaneId, VKey)> = Vec::with_capacity(ins.len());
                            for it in ins {
                                let pos = sorted
                                    .iter()
                                    .position(|&(qa, _)| self.vsign(&it.1, t, qa) != self.vsign(&kj, t, qa))
                                    .unwrap_or(sorted.len());
                                sorted.insert(pos, it);
                            }
                            ins = sorted;
                        }
                        for (_, k) in ins {
                            out.push((k, tagi));
                        }
                    }
                }
            }
            poly = out;
        }
        poly
    }

    /// Canonical coordinates of a symbolic vertex.
    pub fn key_point(&self, k: &VKey) -> P3 {
        let ps = &self.cx.planes;
        match *k {
            VKey::Orig(v) => self.pts[v as usize],
            VKey::Ep(a, b, p) => ps.edge_point(&self.pts[a as usize], &self.pts[b as usize], p),
            VKey::Tl(t, l) => ps.tri_line_point(self.tri(t), l.0, l.1),
            VKey::Cv(v) => self.cx.verts[v as usize],
        }
    }

    /// Exact point location of an input vertex: a cell containing it.
    fn locate_vertex(&self, v: u32, sites: &SiteGrid) -> Option<u32> {
        let p = self.pts[v as usize];
        let inside = |c: u32| {
            self.cx.cells[c as usize].halfspaces.iter().all(|&(q, sgn)| sgn * self.cx.planes.side_vertex(&p, q) <= 0)
        };
        let start = sites.nearest(p)?;
        if inside(start) {
            return Some(start);
        }
        // BFS over neighbors by proximity
        let mut seen = BTreeSet::new();
        seen.insert(start);
        let mut frontier = vec![start];
        for _ in 0..64 {
            let mut next = Vec::new();
            for &c in &frontier {
                for ns in self.nbr[c as usize].values() {
                    for &nb in ns {
                        if seen.insert(nb) {
                            if inside(nb) {
                                return Some(nb);
                            }
                            next.push(nb);
                        }
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        (0..self.cx.cells.len() as u32).find(|&c| inside(c))
    }

    /// Crossings of complex edge `e` with the surface, ordered from `v[0]`.
    fn edge_crossings(&self, e: u32) -> Vec<VKey> {
        let ce = &self.cx.edges[e as usize];
        let a = self.cx.verts[ce.v[0] as usize];
        let b = self.cx.verts[ce.v[1] as usize];
        let mut bb = Aabb::from_points([&DVec3::from_array(a), &DVec3::from_array(b)]);
        if !bb.overlaps(&self.bbox) {
            return Vec::new();
        }
        let pad = 1e-7 * self.bbox.diagonal();
        bb = bb.expanded(pad);
        let ps = &self.cx.planes;
        let mut out: Vec<u32> = Vec::new();
        self.bvh.query(&bb, |t| {
            let tri = self.tri(t);
            if !ps.pierces(tri, ce.line.0, ce.line.1) {
                return;
            }
            for &(q, sgn) in &ce.end {
                if sgn * ps.side_tri_line(tri, ce.line.0, ce.line.1, q) >= 0 {
                    return;
                }
            }
            out.push(t);
        });
        // exact order along the edge: by -sign0 * f_{end0} (distance from v[0])
        let (r, s0) = ce.end[0];
        out.sort_by(|&t1, &t2| {
            let o = ps.cmp_along_line(self.tri(t1), self.tri(t2), ce.line.0, ce.line.1, r);
            let o = if s0 > 0 { o.reverse() } else { o };
            o.then(t1.cmp(&t2))
        });
        out.into_iter().map(|t| VKey::Tl(t, ce.line)).collect()
    }

    /// Run the full clipping. `active[c]` selects which complex cells are
    /// real (others are treated as outside the region of interest).
    pub fn run(&self) -> Result<ClipOutput, String> {
        let cx = self.cx;
        let mut warnings = Vec::new();
        let ncell = cx.cells.len();
        // ---- 1. crossings on complex edges and in/out state of vertices
        let crossings: Vec<Vec<VKey>> = (0..cx.edges.len() as u32).into_par_iter().map(|e| self.edge_crossings(e)).collect();
        let nv = cx.verts.len();
        let mut adj: Vec<Vec<(u32, u32)>> = vec![Vec::new(); nv];
        for (ei, e) in cx.edges.iter().enumerate() {
            adj[e.v[0] as usize].push((e.v[1], ei as u32));
            adj[e.v[1] as usize].push((e.v[0], ei as u32));
        }
        let mut state: Vec<i8> = vec![-1; nv]; // -1 unknown, 0 out, 1 in
        let margin = 1e-6 * self.bbox.diagonal().max(1e-12);
        let outer = self.bbox.expanded(margin);
        let mut queue = std::collections::VecDeque::new();
        for v in 0..nv {
            if !outer.contains(DVec3::from_array(cx.verts[v])) {
                state[v] = 0;
                queue.push_back(v as u32);
            }
        }
        if queue.is_empty() && nv > 0 {
            return Err("clip: no complex vertex outside the solid bounds".into());
        }
        let mut conflicts = 0usize;
        while let Some(v) = queue.pop_front() {
            for &(w, e) in &adj[v as usize] {
                let flip = (crossings[e as usize].len() % 2) as i8;
                let sw = state[v as usize] ^ flip;
                if state[w as usize] < 0 {
                    state[w as usize] = sw;
                    queue.push_back(w);
                } else if state[w as usize] != sw {
                    conflicts += 1;
                }
            }
        }
        if conflicts > 0 {
            return Err(format!("clip: inconsistent inside/outside parity on {conflicts} complex edges (solid not closed?)"));
        }
        if state.iter().any(|&s| s < 0) {
            warnings.push("clip: unreachable complex vertices".into());
        }

        // ---- 2. exterior polygons by per-triangle BFS over cells
        let sites = SiteGrid::new(&cx.sites);
        let per_tri: Vec<Result<Vec<(u32, Vec<VKey>)>, String>> = (0..self.mesh.tris.len() as u32)
            .into_par_iter()
            .map(|t| {
                let v0 = self.mesh.tris[t as usize][0];
                let start = self.locate_vertex(v0, &sites).ok_or_else(|| format!("clip: vertex {v0} not in any cell"))?;
                let mut res = Vec::new();
                let mut seen = BTreeSet::new();
                let mut stack = vec![start];
                seen.insert(start);
                while let Some(c) = stack.pop() {
                    let poly = self.clip_tri_cell(t, c);
                    if poly.is_empty() {
                        continue;
                    }
                    for &(_, tag) in &poly {
                        if let Tag::Plane(q) = tag {
                            if let Some(ns) = self.nbr[c as usize].get(&q) {
                                for &nb in ns {
                                    if seen.insert(nb) {
                                        stack.push(nb);
                                    }
                                }
                            }
                        }
                    }
                    res.push((c, poly.into_iter().map(|x| x.0).collect()));
                }
                res.sort_by_key(|x| x.0);
                Ok(res)
            })
            .collect();
        let mut ext_raw: Vec<(u32, u32, Vec<VKey>)> = Vec::new(); // (cell, tri, keys)
        for (t, r) in per_tri.into_iter().enumerate() {
            for (c, keys) in r? {
                ext_raw.push((c, t as u32, keys));
            }
        }

        // ---- 3. face regions
        let face_ids: Vec<u32> = (0..cx.faces.len() as u32).collect();
        let regions: Vec<Result<Vec<Vec<VKey>>, String>> =
            face_ids.par_iter().map(|&f| self.face_region(f, &crossings, &state)).collect();
        let mut face_loops: Vec<(u32, Vec<Vec<VKey>>)> = Vec::new();
        for (f, r) in regions.into_iter().enumerate() {
            let loops = r?;
            if loops.is_empty() {
                continue;
            }
            let fc = &cx.faces[f];
            if fc.cells[0] == NONE || fc.cells[1] == NONE {
                return Err(format!("clip: solid reaches outside the complex (face {f})"));
            }
            face_loops.push((f as u32, loops));
        }

        // ---- 4. global vertex table
        let mut all: Vec<VKey> = Vec::new();
        for (_, _, k) in &ext_raw {
            all.extend_from_slice(k);
        }
        for (_, ls) in &face_loops {
            for l in ls {
                all.extend_from_slice(l);
            }
        }
        all.par_sort_unstable();
        all.dedup();
        let coords: Vec<P3> = all.par_iter().map(|k| self.key_point(k)).collect();
        // Weld coincident points. Symbolically distinct vertices can coincide
        // geometrically in exactly degenerate inputs (e.g. a mesh edge through
        // a complex line); their rounded coordinates then differ by ~1 ulp.
        // Points closer than 1e-11 of the model scale are merged
        // (deterministic union-find, smallest key index wins).
        let eps = 1e-11 * self.bbox.diagonal().max(1e-300);
        let cell = |p: &P3| [(p[0] / eps).floor() as i64, (p[1] / eps).floor() as i64, (p[2] / eps).floor() as i64];
        let mut grid: BTreeMap<[i64; 3], Vec<u32>> = BTreeMap::new();
        for (i, p) in coords.iter().enumerate() {
            grid.entry(cell(p)).or_default().push(i as u32);
        }
        let mut parent: Vec<u32> = (0..all.len() as u32).collect();
        fn find(p: &mut [u32], x: u32) -> u32 {
            let mut r = x;
            while p[r as usize] != r {
                r = p[r as usize];
            }
            let mut y = x;
            while p[y as usize] != r {
                let n = p[y as usize];
                p[y as usize] = r;
                y = n;
            }
            r
        }
        for (i, p) in coords.iter().enumerate() {
            let c = cell(p);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(v) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                            for &j in v {
                                if (j as usize) <= i {
                                    continue;
                                }
                                let q = coords[j as usize];
                                if dist2(*p, q) <= eps * eps {
                                    let (ra, rb) = (find(&mut parent, i as u32), find(&mut parent, j));
                                    if ra != rb {
                                        parent[ra.max(rb) as usize] = ra.min(rb);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut remap = vec![0u32; all.len()];
        let mut verts: Vec<DVec3> = Vec::new();
        let mut keys: Vec<VKey> = Vec::new();
        let mut root_id: BTreeMap<u32, u32> = BTreeMap::new();
        for i in 0..all.len() {
            let r = find(&mut parent, i as u32);
            let id = *root_id.entry(r).or_insert_with(|| {
                verts.push(DVec3::from_array(coords[r as usize]));
                keys.push(all[r as usize]);
                (verts.len() - 1) as u32
            });
            remap[i] = id;
        }
        let index = |k: &VKey| -> u32 { remap[all.binary_search(k).unwrap()] };
        let clean = |l: &[VKey]| -> Vec<u32> {
            let mut v: Vec<u32> = l.iter().map(index).collect();
            v.dedup();
            while v.len() > 1 && v[0] == *v.last().unwrap() {
                v.pop();
            }
            v
        };
        let mut ext = Vec::with_capacity(ext_raw.len());
        for (c, t, k) in &ext_raw {
            let v = clean(k);
            if v.len() >= 3 {
                let tris = triangulate_poly(&verts, &v);
                ext.push(ExtPolyOut { verts: v, cell: *c, src_tri: *t, tris });
            }
        }
        // ---- 5. patches: group loops into outer+holes, triangulate
        let patch_lists: Vec<Vec<PatchOut>> = face_loops
            .par_iter()
            .map(|(f, ls)| {
                let fc = &cx.faces[*f as usize];
                let n = DVec3::from_array(cx.planes.normal(fc.plane)).normalize();
                let loops: Vec<Vec<u32>> = ls.iter().map(|l| clean(l)).filter(|l| l.len() >= 3).collect();
                build_patches(&verts, &loops, n, fc.cells, *f)
            })
            .collect();
        let patches: Vec<PatchOut> = patch_lists.into_iter().flatten().collect();
        let _ = ncell;
        Ok(ClipOutput { verts, keys, ext, patches, warnings })
    }

    /// Region of face `f` inside the solid, as closed loops (CCW about the
    /// plane gradient, holes CW).
    fn face_region(&self, f: u32, crossings: &[Vec<VKey>], state: &[i8]) -> Result<Vec<Vec<VKey>>, String> {
        let cx = self.cx;
        let fc = &cx.faces[f as usize];
        let p = fc.plane;
        let ps = &cx.planes;
        // constraining half-spaces
        let mut hs: Vec<(PlaneId, i8)> = Vec::new();
        for &c in &fc.cells {
            if c != NONE {
                for &(q, s) in &cx.cells[c as usize].halfspaces {
                    if q != p {
                        hs.push((q, s));
                    }
                }
            }
        }
        hs.sort_unstable();
        hs.dedup();
        // bounding box of the face polygon
        let mut fb = Aabb::EMPTY;
        for &v in &fc.loop_verts {
            fb.grow(DVec3::from_array(cx.verts[v as usize]));
        }
        let mut segs: Vec<(VKey, VKey)> = Vec::new();
        if fb.overlaps(&self.bbox) {
            let pad = 1e-7 * self.bbox.diagonal();
            let q = fb.expanded(pad);
            let mut cand = Vec::new();
            self.bvh.query(&q, |t| cand.push(t));
            cand.sort_unstable();
            for t in cand {
                let [a, b, c] = self.mesh.tris[t as usize];
                let s = [
                    ps.side_vertex(&self.pts[a as usize], p),
                    ps.side_vertex(&self.pts[b as usize], p),
                    ps.side_vertex(&self.pts[c as usize], p),
                ];
                if s[0] == s[1] && s[1] == s[2] {
                    continue;
                }
                let vs = [a, b, c];
                let lone = if s[0] != s[1] && s[0] != s[2] {
                    0
                } else if s[1] != s[0] && s[1] != s[2] {
                    1
                } else {
                    2
                };
                let v0 = vs[lone];
                let v1 = vs[(lone + 1) % 3];
                let v2 = vs[(lone + 2) % 3];
                let e01 = VKey::Ep(v0.min(v1), v0.max(v1), p);
                let e20 = VKey::Ep(v0.min(v2), v0.max(v2), p);
                let (mut sa, mut sb) = if s[lone] > 0 { (e01, e20) } else { (e20, e01) };
                // clip the segment by the face half-spaces
                let mut alive = true;
                for &(q, sg) in &hs {
                    let ia = sg * self.vsign(&sa, t, q);
                    let ib = sg * self.vsign(&sb, t, q);
                    if ia <= 0 && ib <= 0 {
                        continue;
                    }
                    if ia >= 0 && ib >= 0 {
                        alive = false;
                        break;
                    }
                    let x = VKey::Tl(t, ps.line(p, q));
                    if ia > 0 {
                        sa = x;
                    } else {
                        sb = x;
                    }
                }
                if alive && sa != sb {
                    segs.push((sa, sb));
                }
            }
        }
        // edge portions along the face boundary
        let nl = fc.loop_verts.len();
        for k in 0..nl {
            let va = fc.loop_verts[k];
            let vb = fc.loop_verts[(k + 1) % nl];
            let e = fc.loop_edges[k];
            let ce = &cx.edges[e as usize];
            let mut cr = crossings[e as usize].clone();
            if ce.v[0] != va {
                cr.reverse();
            }
            let mut st = state[va as usize];
            if st < 0 {
                continue;
            }
            let mut cur = VKey::Cv(va);
            for x in cr.iter() {
                if st == 1 {
                    segs.push((cur, *x));
                }
                st ^= 1;
                cur = *x;
            }
            if st != state[vb as usize] {
                return Err(format!("clip: parity mismatch on complex edge {e}"));
            }
            if st == 1 {
                segs.push((cur, VKey::Cv(vb)));
            }
        }
        if segs.is_empty() {
            return Ok(Vec::new());
        }
        // chain segments into loops
        let mut next: BTreeMap<VKey, Vec<usize>> = BTreeMap::new();
        for (i, (a, _)) in segs.iter().enumerate() {
            next.entry(*a).or_default().push(i);
        }
        let mut used = vec![false; segs.len()];
        let mut loops = Vec::new();
        for s0 in 0..segs.len() {
            if used[s0] {
                continue;
            }
            let mut lp = vec![segs[s0].0];
            used[s0] = true;
            let start = segs[s0].0;
            let mut cur = segs[s0].1;
            let mut guard = 0;
            while cur != start {
                lp.push(cur);
                let cand = match next.get(&cur) {
                    Some(c) => c,
                    None => {
                        if std::env::var("FRAC_DEBUG").is_ok() {
                            eprintln!("open loop face {f} plane {} cells {:?} at {:?}", p, fc.cells, cur);
                            for (a, b) in &segs {
                                eprintln!("  seg {:?} {:?} -> {:?} {:?}", a, self.key_point(a), b, self.key_point(b));
                            }
                            for (k, &v) in fc.loop_verts.iter().enumerate() {
                                eprintln!("  cv {} {:?} state {} edge {} cr {:?}", v, cx.verts[v as usize], state[v as usize], fc.loop_edges[k], crossings[fc.loop_edges[k] as usize]);
                            }
                        }
                        return Err(format!("clip: open interface loop on face {f}"));
                    }
                };
                let nx = cand.iter().copied().find(|&i| !used[i]).ok_or_else(|| format!("clip: dangling interface loop on face {f}"))?;
                used[nx] = true;
                cur = segs[nx].1;
                guard += 1;
                if guard > segs.len() + 1 {
                    return Err(format!("clip: runaway loop on face {f}"));
                }
            }
            if lp.len() >= 3 {
                loops.push(lp);
            }
        }
        Ok(loops)
    }
}

#[inline]
fn dist2(a: P3, b: P3) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// Group loops into patches (outer CCW + contained CW holes) and triangulate.
fn build_patches(verts: &[DVec3], loops: &[Vec<u32>], n: DVec3, cells: [u32; 2], face: u32) -> Vec<PatchOut> {
    let (u, v) = frac_geom::polygon::plane_basis(n);
    let mut local: BTreeMap<u32, usize> = BTreeMap::new();
    let mut pts2: Vec<[f64; 2]> = Vec::new();
    let mut ids: Vec<u32> = Vec::new();
    for l in loops {
        for &x in l {
            local.entry(x).or_insert_with(|| {
                let p = verts[x as usize];
                pts2.push([p.dot(u), p.dot(v)]);
                ids.push(x);
                pts2.len() - 1
            });
        }
    }
    let ll: Vec<Vec<usize>> = loops.iter().map(|l| l.iter().map(|x| local[x]).collect()).collect();
    let areas: Vec<f64> = ll.iter().map(|l| tri2d::signed_area(&pts2, l)).collect();
    // A loop is a hole only when it is clearly negative and contained in an
    // outer loop; degenerate (zero-area) loops from symbolic perturbation
    // stand alone as their own (zero-area) patches.
    let degen: Vec<bool> = ll.iter().map(|l| tri2d::is_degenerate(&pts2, l)).collect();
    let outers: Vec<usize> = (0..ll.len()).filter(|&i| areas[i] > 0.0 && !degen[i]).collect();
    let mut holes_of: Vec<Vec<usize>> = vec![Vec::new(); ll.len()];
    let mut standalone: Vec<usize> = (0..ll.len()).filter(|&i| degen[i]).collect();
    for i in 0..ll.len() {
        if degen[i] || areas[i] > 0.0 {
            continue;
        }
        // smallest containing outer loop
        let p = pts2[ll[i][0]];
        let mut best: Option<usize> = None;
        for &o in &outers {
            if point_in_loop(&pts2, &ll[o], p) && best.map(|b| areas[o] < areas[b]).unwrap_or(true) {
                best = Some(o);
            }
        }
        match best {
            Some(o) => holes_of[o].push(i),
            None => standalone.push(i),
        }
    }
    let mut out = Vec::new();
    let mut groups: Vec<usize> = outers.clone();
    groups.extend(standalone.iter().copied());
    groups.sort_unstable();
    for &o in &groups {
        let mut pl: Vec<Vec<usize>> = vec![ll[o].clone()];
        for &h in &holes_of[o] {
            pl.push(ll[h].clone());
        }
        let tris = tri2d::triangulate(&pts2, &pl);
        let area: f64 = pl.iter().map(|l| tri2d::signed_area(&pts2, l)).sum();
        out.push(PatchOut {
            loops: pl.iter().map(|l| l.iter().map(|&i| ids[i]).collect()).collect(),
            tris: tris.iter().map(|t| [ids[t[0]], ids[t[1]], ids[t[2]]]).collect(),
            normal: n,
            cells,
            face,
            area,
        });
    }
    out
}

fn point_in_loop(pts: &[[f64; 2]], l: &[usize], p: [f64; 2]) -> bool {
    let mut inside = false;
    let n = l.len();
    for k in 0..n {
        let a = pts[l[k]];
        let b = pts[l[(k + 1) % n]];
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
            if p[0] < x {
                inside = !inside;
            }
        }
    }
    inside
}

/// Uniform grid over cell sites for approximate nearest-site queries.
pub struct SiteGrid {
    sites: Vec<P3>,
    lo: P3,
    inv: f64,
    dims: [usize; 3],
    buckets: Vec<Vec<u32>>,
}

impl SiteGrid {
    pub fn new(sites: &[P3]) -> Self {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for s in sites {
            for k in 0..3 {
                lo[k] = lo[k].min(s[k]);
                hi[k] = hi[k].max(s[k]);
            }
        }
        let n = sites.len().max(1);
        let ext = [(hi[0] - lo[0]).max(1e-12), (hi[1] - lo[1]).max(1e-12), (hi[2] - lo[2]).max(1e-12)];
        let vol = ext[0] * ext[1] * ext[2];
        let cell = (vol / n as f64).cbrt().max(1e-12);
        let dims = [0, 1, 2].map(|k| ((ext[k] / cell).ceil() as usize).clamp(1, 256));
        let inv = 1.0 / cell;
        let mut buckets = vec![Vec::new(); dims[0] * dims[1] * dims[2]];
        let g = SiteGrid { sites: sites.to_vec(), lo, inv, dims, buckets: Vec::new() };
        for (i, s) in sites.iter().enumerate() {
            let b = g.bucket(s);
            buckets[b].push(i as u32);
        }
        SiteGrid { buckets, ..g }
    }
    fn coord(&self, p: &P3) -> [usize; 3] {
        [0, 1, 2].map(|k| (((p[k] - self.lo[k]) * self.inv).floor().max(0.0) as usize).min(self.dims[k] - 1))
    }
    fn bucket(&self, p: &P3) -> usize {
        let c = self.coord(p);
        (c[2] * self.dims[1] + c[1]) * self.dims[0] + c[0]
    }
    pub fn nearest(&self, p: P3) -> Option<u32> {
        if self.sites.is_empty() {
            return None;
        }
        let c = self.coord(&p);
        let mut best: Option<(f64, u32)> = None;
        let maxr = self.dims[0].max(self.dims[1]).max(self.dims[2]);
        for r in 0..=maxr {
            for z in c[2].saturating_sub(r)..=(c[2] + r).min(self.dims[2] - 1) {
                for y in c[1].saturating_sub(r)..=(c[1] + r).min(self.dims[1] - 1) {
                    for x in c[0].saturating_sub(r)..=(c[0] + r).min(self.dims[0] - 1) {
                        let on_shell = x.abs_diff(c[0]) == r || y.abs_diff(c[1]) == r || z.abs_diff(c[2]) == r;
                        if !on_shell {
                            continue;
                        }
                        for &i in &self.buckets[(z * self.dims[1] + y) * self.dims[0] + x] {
                            let d = dist2(self.sites[i as usize], p);
                            if best.map(|b| d < b.0 || (d == b.0 && i < b.1)).unwrap_or(true) {
                                best = Some((d, i));
                            }
                        }
                    }
                }
            }
            if let Some((d, _)) = best {
                // ring r fully searched: any point in ring r+1 is at least r/inv away
                let reach = r as f64 / self.inv;
                if reach * reach >= d {
                    break;
                }
            }
        }
        best.map(|b| b.1)
    }
}
