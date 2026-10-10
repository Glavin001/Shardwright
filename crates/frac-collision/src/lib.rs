//! Stage 8: collision hulls.
//!
//! * Method B (single convex cell): the cell's exact convex hull.
//! * Method A (built-in collision-aware decomposition in the spirit of
//!   CoACD): non-convex cells are split by planes minimizing the
//!   collision-aware concavity; pieces are then merged greedily, bottom-up
//!   across hierarchy levels, while the merged hull's concavity (maximum
//!   distance of hull-surface samples lying *outside* the fragment to the
//!   fragment surface) stays below `concavity * diameter`, and until the
//!   hull budget is met.
//! * Non-overlap: hulls of neighboring fragments are clipped by the planes
//!   of their (planar) interfaces, or by a separating plane through the
//!   overlap otherwise, then shrunk by the margin.

use frac_core::*;
use frac_geom::hull::{ConvexPolytope, HalfSpace};
use frac_geom::inside::MeshQuery;
use frac_geom::{Aabb, DVec3, TriMesh};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering as AO};

static T_HULL: AtomicU64 = AtomicU64::new(0);
static T_CONC: AtomicU64 = AtomicU64::new(0);
static T_MESH: AtomicU64 = AtomicU64::new(0);
static N_HULL: AtomicU64 = AtomicU64::new(0);
static N_CONC: AtomicU64 = AtomicU64::new(0);
fn tadd(c: &AtomicU64, t: std::time::Instant) {
    c.fetch_add(t.elapsed().as_nanos() as u64, AO::Relaxed);
}

#[derive(Clone, Debug)]
pub struct CollisionParams {
    pub concavity: f64,
    pub max_hulls: usize,
    pub margin: f64,
    pub min_rigid_size: f64,
    /// Levels that receive hulls (empty = all).
    pub levels: Vec<u8>,
    pub max_split_depth: u32,
}

/// Closed boundary mesh of a set of cells (exterior polygons + patches to
/// cells outside the set), using the component vertex table.
pub fn cells_boundary_mesh(asset: &Asset, cells: &[CellId]) -> TriMesh {
    let set: BTreeSet<CellId> = cells.iter().copied().collect();
    let mut comps: BTreeSet<u32> = BTreeSet::new();
    for c in cells {
        comps.insert(asset.cells[c.idx()].component.0);
    }
    let mut out = TriMesh::default();
    for ci in comps {
        let comp = &asset.components[ci as usize];
        let g = &comp.geometry;
        let mut tris = Vec::new();
        for e in &g.ext_polys {
            if set.contains(&e.cell) {
                tris.extend(e.tris.iter().copied());
            }
        }
        for p in &g.patches {
            let (a, b) = (set.contains(&p.cells.0), set.contains(&p.cells.1));
            if a && !b {
                tris.extend(p.tris.iter().copied());
            } else if b && !a {
                tris.extend(p.tris.iter().map(|t| [t[0], t[2], t[1]]));
            }
        }
        let m = TriMesh { verts: g.verts.clone(), tris }.compact();
        out.append(&m);
    }
    out
}

/// Points of a set of cells (all boundary polygon vertices).
fn cell_points(asset: &Asset, cell: CellId) -> Vec<DVec3> {
    let comp = &asset.components[asset.cells[cell.idx()].component.idx()];
    let g = &comp.geometry;
    let mut ids: BTreeSet<u32> = BTreeSet::new();
    for e in &g.ext_polys {
        if e.cell == cell {
            ids.extend(e.verts.iter().copied());
        }
    }
    for p in &g.patches {
        if p.cells.0 == cell || p.cells.1 == cell {
            for l in &p.loops {
                ids.extend(l.iter().copied());
            }
        }
    }
    ids.into_iter().map(|i| g.verts[i as usize]).collect()
}

/// Polygons of a cell (outward oriented) for plane splitting.
fn cell_polys(asset: &Asset, cell: CellId) -> Vec<Vec<DVec3>> {
    let comp = &asset.components[asset.cells[cell.idx()].component.idx()];
    let g = &comp.geometry;
    let mut out = Vec::new();
    for e in &g.ext_polys {
        if e.cell == cell {
            out.push(e.verts.iter().map(|&v| g.verts[v as usize]).collect());
        }
    }
    for p in &g.patches {
        for t in &p.tris {
            if p.cells.0 == cell {
                out.push(t.iter().map(|&v| g.verts[v as usize]).collect());
            } else if p.cells.1 == cell {
                out.push([t[0], t[2], t[1]].iter().map(|&v| g.verts[v as usize]).collect());
            }
        }
    }
    out
}

/// Fragment-surface oracle for concavity measurement.
struct Surface<'a> {
    q: MeshQuery<'a>,
    /// Sampling resolution (the concavity threshold).
    res: f64,
}

impl<'a> Surface<'a> {
    /// Collision-aware concavity of a hull: max distance from hull surface
    /// samples outside the solid to the solid's surface.
    fn concavity(&self, h: &ConvexPolytope) -> f64 {
        self.concavity_capped(h, f64::INFINITY)
    }

    /// Like `concavity` but stops as soon as the value exceeds `cap`.
    fn concavity_capped(&self, h: &ConvexPolytope, cap: f64) -> f64 {
        let t0 = std::time::Instant::now();
        N_CONC.fetch_add(1, AO::Relaxed);
        let r = self.concavity_inner(h, cap);
        tadd(&T_CONC, t0);
        r
    }

    fn concavity_inner(&self, h: &ConvexPolytope, cap: f64) -> f64 {
        let mut worst: f64 = 0.0;
        for (_, f) in &h.faces {
            let n = f.len();
            let c = f.iter().fold(DVec3::ZERO, |a, &p| a + p) / n as f64;
            let mut samples = vec![c];
            // edge midpoints / sub-centroids only on faces large enough to
            // hide concavity beyond the threshold resolution
            let big = f.iter().any(|q| (*q - c).length() > 2.0 * self.res);
            if big {
                for k in 0..n {
                    let a = f[k];
                    let b = f[(k + 1) % n];
                    samples.push((a + b) * 0.5);
                    samples.push((a + b + c) / 3.0);
                }
            }
            for s in samples {
                // only points farther than the current worst can matter;
                // the (costlier) inside test runs just for those
                let Some((_, d2, _)) = self.q.closest_point(s) else { continue };
                let d = d2.sqrt();
                if d > worst && !self.q.contains(s) {
                    {
                        worst = d;
                        if worst > cap {
                            return worst;
                        }
                    }
                }
            }
        }
        worst
    }
}

fn polys_volume(polys: &[Vec<DVec3>]) -> f64 {
    let mut vi = frac_geom::VolumeIntegrals::default();
    let r = polys.first().and_then(|p| p.first()).copied().unwrap_or(DVec3::ZERO);
    for p in polys {
        vi.add_polygon(r, p);
    }
    vi.volume
}

fn hull_of(points: &[DVec3]) -> Option<ConvexPolytope> {
    let t0 = std::time::Instant::now();
    N_HULL.fetch_add(1, AO::Relaxed);
    let r = ConvexPolytope::from_points(points).filter(|p| !p.is_empty());
    tadd(&T_HULL, t0);
    r
}

/// Recursively split a non-convex point cloud (cell polygons) by planes.
fn split_cell(polys: &[Vec<DVec3>], surf: &Surface, thresh: f64, depth: u32) -> Vec<ConvexPolytope> {
    let pts: Vec<DVec3> = polys.iter().flatten().copied().collect();
    let Some(h) = hull_of(&pts) else { return Vec::new() };
    // convex cells (the common case) need no concavity evaluation
    let hv = h.volume();
    if depth == 0 || hv - polys_volume(polys) <= 1e-9 * hv || surf.concavity_capped(&h, thresh) <= thresh {
        return vec![h];
    }
    // candidate planes: principal axes and coordinate axes at 3 offsets
    let bb = Aabb::from_points(pts.iter());
    let c = pts.iter().fold(DVec3::ZERO, |a, &p| a + p) / pts.len() as f64;
    let cov = pts.iter().fold(glam::DMat3::ZERO, |a, &p| a + frac_geom::polygon::outer(p - c, p - c));
    let (_, ax) = frac_geom::integrals::sym_eigen3(&cov);
    let dirs = [ax.col(2), ax.col(1), ax.col(0)];
    let mut cands: Vec<(DVec3, f64)> = Vec::new();
    for d in dirs {
        let (lo, hi) = pts.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| (lo.min(p.dot(d)), hi.max(p.dot(d))));
        if hi - lo < 1e-9 * bb.diagonal() {
            continue;
        }
        for t in [0.35, 0.5, 0.65] {
            cands.push((d, lo + (hi - lo) * t));
        }
    }
    // score candidates in parallel; deterministic pick (lowest score, then index)
    let scored: Vec<Option<(f64, Vec<Vec<DVec3>>, Vec<Vec<DVec3>>)>> = cands
        .par_iter()
        .map(|&(d, off)| {
            let (l, r) = split_polys(polys, d, off);
            let hl = hull_of(&l.iter().flatten().copied().collect::<Vec<_>>())?;
            let hr = hull_of(&r.iter().flatten().copied().collect::<Vec<_>>())?;
            let score = surf.concavity(&hl).max(surf.concavity(&hr));
            Some((score, l, r))
        })
        .collect();
    let mut best: Option<(f64, Vec<Vec<DVec3>>, Vec<Vec<DVec3>>)> = None;
    for c in scored.into_iter().flatten() {
        if best.as_ref().map(|b| c.0 < b.0).unwrap_or(true) {
            best = Some(c);
        }
    }
    match best {
        Some((_, l, r)) => {
            let mut parts = split_cell(&l, surf, thresh, depth - 1);
            parts.extend(split_cell(&r, surf, thresh, depth - 1));
            if parts.is_empty() { vec![h] } else { parts }
        }
        None => vec![h],
    }
}

/// Split polygons by plane `d·x = off` (keeps both sides; cut points shared).
fn split_polys(polys: &[Vec<DVec3>], d: DVec3, off: f64) -> (Vec<Vec<DVec3>>, Vec<Vec<DVec3>>) {
    let mut l = Vec::new();
    let mut r = Vec::new();
    for p in polys {
        let n = p.len();
        let mut a = Vec::new();
        let mut b = Vec::new();
        for k in 0..n {
            let x = p[k];
            let y = p[(k + 1) % n];
            let sx = x.dot(d) - off;
            let sy = y.dot(d) - off;
            if sx <= 0.0 {
                a.push(x);
            }
            if sx >= 0.0 {
                b.push(x);
            }
            if (sx < 0.0 && sy > 0.0) || (sx > 0.0 && sy < 0.0) {
                let t = sx / (sx - sy);
                let q = x + (y - x) * t;
                a.push(q);
                b.push(q);
            }
        }
        if a.len() >= 3 {
            l.push(a);
        }
        if b.len() >= 3 {
            r.push(b);
        }
    }
    (l, r)
}

#[derive(Clone)]
struct Piece {
    hull: ConvexPolytope,
    pts: Vec<DVec3>,
    cells: Vec<CellId>,
}

/// Greedy merging of pieces down to the budget, with cached pair costs
/// (only pairs touching the merged piece are re-evaluated).
fn merge_pieces(pieces: Vec<Piece>, surf: &Surface, thresh: f64, budget: usize, cell_adj: &BTreeSet<(CellId, CellId)>) -> Vec<Piece> {
    let mut alive: BTreeMap<usize, Piece> = pieces.into_iter().enumerate().collect();
    let mut next_id = alive.len();
    let bbox = |p: &Piece| Aabb::from_points(p.pts.iter());
    let diag = alive.values().fold(Aabb::EMPTY, |a, p| a.union(&bbox(p))).diagonal().max(1e-12);
    // cost key: (concavity, excess) ; value: merged hull
    let adjacent = |a: &Piece, b: &Piece| -> bool {
        // same cell (split pieces) or cells sharing an interface
        a.cells.iter().any(|ca| b.cells.iter().any(|cb| ca == cb || cell_adj.contains(&(*ca.min(cb), *ca.max(cb)))))
    };
    let eval = |a: &Piece, b: &Piece| -> Option<(f64, f64, ConvexPolytope)> {
        if !bbox(a).expanded(1e-6 * diag).overlaps(&bbox(b)) || !adjacent(a, b) {
            return None;
        }
        let mut pts = a.pts.clone();
        pts.extend_from_slice(&b.pts);
        let h = hull_of(&pts)?;
        let hv = h.volume();
        let excess = hv - a.hull.volume() - b.hull.volume();
        // 0 = acceptable merge, 1 = too concave (only used to meet the budget)
        let conc = if excess <= 1e-9 * hv { 0.0 } else if surf.concavity_capped(&h, thresh) <= thresh { 0.0 } else { 1.0 };
        Some((conc, excess, h))
    };
    let mut cache: BTreeMap<(usize, usize), (f64, f64, ConvexPolytope)> = BTreeMap::new();
    let ids: Vec<usize> = alive.keys().copied().collect();
    let pairs: Vec<(usize, usize)> = ids.iter().enumerate().flat_map(|(k, &i)| ids[k + 1..].iter().map(move |&j| (i, j))).collect();
    let evals: Vec<Option<(f64, f64, ConvexPolytope)>> = pairs.par_iter().map(|&(i, j)| eval(&alive[&i], &alive[&j])).collect();
    for (pr, e) in pairs.into_iter().zip(evals) {
        if let Some(e) = e {
            cache.insert(pr, e);
        }
    }
    while alive.len() > 1 {
        let best = cache
            .iter()
            .min_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap().then(a.1 .1.partial_cmp(&b.1 .1).unwrap()).then(a.0.cmp(b.0)))
            .map(|(k, v)| (*k, v.0));
        let Some(((i, j), conc)) = best else { break };
        if conc > 0.5 && alive.len() <= budget {
            break;
        }
        let (_, _, h) = cache.remove(&(i, j)).unwrap();
        let pi = alive.remove(&i).unwrap();
        let pj = alive.remove(&j).unwrap();
        cache.retain(|k, _| k.0 != i && k.1 != i && k.0 != j && k.1 != j);
        let mut pts = pi.pts;
        pts.extend(pj.pts);
        let mut cells = pi.cells;
        cells.extend(pj.cells);
        let merged = Piece { pts: h.vertices(), hull: h, cells };
        let _ = pts;
        let nid = next_id;
        next_id += 1;
        let others: Vec<usize> = alive.keys().copied().collect();
        let evals: Vec<Option<(f64, f64, ConvexPolytope)>> = others.par_iter().map(|&o| eval(&alive[&o], &merged)).collect();
        for (o, e) in others.into_iter().zip(evals) {
            if let Some(e) = e {
                cache.insert((o, nid), e);
            }
        }
        alive.insert(nid, merged);
    }
    alive.into_values().collect()
}

/// Compute hulls for all fragments. Returns hulls and per-fragment ranges.
pub fn build_hulls(asset: &Asset, p: &CollisionParams) -> (Vec<Hull>, Vec<std::ops::Range<u32>>) {
    let h = &asset.hierarchy;
    let nl = h.levels as usize;
    let want_level = |l: usize| p.levels.is_empty() || p.levels.contains(&(l as u8));
    let mut pieces_of: Vec<Option<Vec<Piece>>> = vec![None; h.fragments.len()];
    // bottom-up: finest level first
    for level in (0..nl).rev() {
        let tl = std::time::Instant::now();
        let r = h.level_ranges[level].clone();
        let results: Vec<(u32, Vec<Piece>)> = (r.start..r.end)
            .into_par_iter()
            .map(|fi| {
                let f = &h.fragments[fi as usize];
                let cells = asset.fragment_cells(f).to_vec();
                let tm = std::time::Instant::now();
                let mesh = cells_boundary_mesh(asset, &cells);
                tadd(&T_MESH, tm);
                let diam = f.mass.volume.cbrt().max(Aabb::from_points(mesh.verts.iter()).diagonal() * 0.5);
                let thresh = p.concavity * diam;
                let surf = Surface { q: MeshQuery::new(&mesh), res: thresh };
                let pieces: Vec<Piece> = if level == nl - 1 || f.children.is_empty() {
                    // leaf level: per-cell pieces (split if non-convex)
                    let mut v = Vec::new();
                    for &c in &cells {
                        let polys = cell_polys(asset, c);
                        for hl in split_cell(&polys, &surf, thresh, p.max_split_depth) {
                            let pts = hl.vertices();
                            v.push(Piece { hull: hl, pts, cells: vec![c] });
                        }
                    }
                    v
                } else {
                    let mut v = Vec::new();
                    for ch in f.children.clone() {
                        if let Some(ps) = &pieces_of_snapshot(&pieces_of, ch) {
                            v.extend(ps.iter().cloned());
                        }
                    }
                    v
                };
                let budget = if f.particle_candidate { 1 } else { p.max_hulls.max(1) };
                let merged = if f.particle_candidate {
                    let pts: Vec<DVec3> = cells.iter().flat_map(|&c| cell_points(asset, c)).collect();
                    hull_of(&pts).map(|hl| vec![Piece { pts: hl.vertices(), hull: hl, cells: cells.clone() }]).unwrap_or_default()
                } else {
                    let set: BTreeSet<CellId> = cells.iter().copied().collect();
                    let mut cell_adj = BTreeSet::new();
                    for it in &asset.interfaces {
                        if let CellOrWorld::Cell(cb) = it.cells.1 {
                            if set.contains(&it.cells.0) && set.contains(&cb) {
                                cell_adj.insert((it.cells.0.min(cb), it.cells.0.max(cb)));
                            }
                        }
                    }
                    let mut merged = merge_pieces(pieces, &surf, thresh, budget, &cell_adj);
                    // disconnected leftovers above budget: force merges by proximity
                    if merged.len() > budget {
                        let all: BTreeSet<(CellId, CellId)> = cells.iter().flat_map(|&a| cells.iter().map(move |&b| (a.min(b), a.max(b)))).collect();
                        merged = merge_pieces(merged, &surf, thresh, budget, &all);
                    }
                    merged
                };
                (fi, merged)
            })
            .collect();
        for (fi, ps) in results {
            pieces_of[fi as usize] = Some(ps);
        }
        if std::env::var("FRAC_PROFILE").is_ok() {
            eprintln!(
                "collision level {level}: {:?} (cumulative cpu: hull {:.1}s x{}, concavity {:.1}s x{}, mesh {:.1}s)",
                tl.elapsed(),
                T_HULL.load(AO::Relaxed) as f64 * 1e-9,
                N_HULL.load(AO::Relaxed),
                T_CONC.load(AO::Relaxed) as f64 * 1e-9,
                N_CONC.load(AO::Relaxed),
                T_MESH.load(AO::Relaxed) as f64 * 1e-9
            );
        }
    }
    // non-overlap enforcement per level, then margin shrink
    let mut hull_polys: Vec<Vec<ConvexPolytope>> = pieces_of.iter().map(|o| o.as_ref().map(|v| v.iter().map(|p| p.hull.clone()).collect()).unwrap_or_default()).collect();
    for level in 0..nl {
        if !want_level(level) {
            continue;
        }
        // planes per fragment from bonds at this level
        let mut clips: BTreeMap<u32, Vec<HalfSpace>> = BTreeMap::new();
        for b in asset.level_bonds(level as u8) {
            let FragmentOrWorld::Fragment(fb) = b.b else { continue };
            let fa = b.a;
            if b.planarity >= 0.9999 {
                let n = b.normal;
                let d = n.dot(b.centroid);
                clips.entry(fa.0).or_default().push(HalfSpace { n, d });
                clips.entry(fb.0).or_default().push(HalfSpace { n: -n, d: -d });
            }
        }
        let r = h.level_ranges[level].clone();
        for fi in r.clone() {
            if let Some(hs) = clips.get(&fi) {
                hull_polys[fi as usize] = hull_polys[fi as usize].iter().map(|hp| hp.clip_all(hs)).filter(|hp| !hp.is_empty()).collect();
            }
        }
        // non-planar neighbors: separate overlapping hull pairs
        for b in asset.level_bonds(level as u8) {
            let FragmentOrWorld::Fragment(fb) = b.b else { continue };
            if b.planarity >= 0.9999 {
                continue;
            }
            let (ia, ib) = (b.a.idx(), fb.idx());
            for x in 0..hull_polys[ia].len() {
                for y in 0..hull_polys[ib].len() {
                    let ov = hull_polys[ia][x].clip_all(&hull_polys[ib][y].halfspaces());
                    if ov.is_empty() || ov.volume() <= 0.0 {
                        continue;
                    }
                    let c = ov.volume_integrals().com();
                    let ca = hull_polys[ia][x].volume_integrals().com();
                    let cb = hull_polys[ib][y].volume_integrals().com();
                    let mut n = (cb - ca).normalize_or_zero();
                    if n == DVec3::ZERO {
                        n = b.normal;
                    }
                    let d = n.dot(c);
                    hull_polys[ia][x] = hull_polys[ia][x].clip(&HalfSpace { n, d });
                    hull_polys[ib][y] = hull_polys[ib][y].clip(&HalfSpace { n: -n, d: -d });
                }
            }
            hull_polys[ia].retain(|hp| !hp.is_empty());
            hull_polys[ib].retain(|hp| !hp.is_empty());
        }
    }
    // output
    let mut hulls = Vec::new();
    let mut ranges = vec![0..0; h.fragments.len()];
    for (fi, hp) in hull_polys.iter().enumerate() {
        let level = h.fragments[fi].level as usize;
        if !want_level(level) {
            continue;
        }
        let start = hulls.len() as u32;
        for poly in hp {
            let s = poly.shrunk(p.margin);
            let s = if s.is_empty() { poly.clone() } else { s };
            hulls.push(to_hull(FragmentId(fi as u32), &s));
        }
        ranges[fi] = start..hulls.len() as u32;
    }
    (hulls, ranges)
}

fn pieces_of_snapshot(v: &[Option<Vec<Piece>>], i: u32) -> Option<Vec<Piece>> {
    v[i as usize].clone()
}

pub fn to_hull(f: FragmentId, p: &ConvexPolytope) -> Hull {
    let mut verts: Vec<DVec3> = Vec::new();
    let mut index: BTreeMap<[u64; 3], u32> = BTreeMap::new();
    let mut faces = Vec::new();
    for (_, poly) in &p.faces {
        let mut face = Vec::new();
        for &q in poly {
            let k = [q.x.to_bits(), q.y.to_bits(), q.z.to_bits()];
            let id = *index.entry(k).or_insert_with(|| {
                verts.push(q);
                (verts.len() - 1) as u32
            });
            if face.last() != Some(&id) {
                face.push(id);
            }
        }
        if face.len() >= 3 {
            faces.push(face);
        }
    }
    Hull { fragment: f, vertices: verts, faces }
}

/// Polytope of a stored hull.
pub fn hull_polytope(h: &Hull) -> ConvexPolytope {
    ConvexPolytope::from_points(&h.vertices).unwrap_or_default()
}
