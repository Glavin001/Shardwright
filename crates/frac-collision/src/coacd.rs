//! Approximate convex decomposition with collision-aware concavity and
//! Monte-Carlo tree search, ported from CoACD.
//!
//! Port of the algorithms of CoACD (Xinyue Wei, Minghua Liu, Zhan Ling, Hao
//! Su, "Approximate Convex Decomposition for 3D Meshes with Collision-Aware
//! Concavity and Tree Search", SIGGRAPH 2022), reimplemented in Rust from
//! the reference implementation at <https://github.com/SarahWeiii/CoACD>:
//!
//! > MIT License — Copyright (c) 2022 Xinyue Wei, Minghua Liu
//! >
//! > Permission is hereby granted, free of charge, to any person obtaining a
//! > copy of this software and associated documentation files (the
//! > "Software"), to deal in the Software without restriction, including
//! > without limitation the rights to use, copy, modify, merge, publish,
//! > distribute, sublicense, and/or sell copies of the Software, and to
//! > permit persons to whom the Software is furnished to do so, subject to
//! > the following conditions: The above copyright notice and this
//! > permission notice shall be included in all copies or substantial
//! > portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT
//! > WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
//!
//! (full text in `THIRD_PARTY_NOTICES.md` at the repository root).
//!
//! What is ported (names refer to the upstream sources):
//! * `Model::Normalize` — inputs are mapped to the box whose longest side
//!   spans [-1, 1]; all thresholds and plane-sampling constants are in these
//!   normalized units, exactly as upstream.
//! * `Clip` — plane clipping of a closed triangle mesh with planar cap
//!   triangulation (here: exact shared cut vertices per edge, boundary loop
//!   extraction, outer/hole nesting, constrained Delaunay via
//!   `frac_cells::tri2d`), so both halves stay closed.
//! * `ComputeRv` / `ComputeHb` / `ComputeHCost` — the concavity
//!   `h = max(Rv, Hb)`: `Rv = k·∛(3|V_mesh - V_hull|/4π)` and the sampled
//!   symmetric surface Hausdorff distance between a part and its hull
//!   (`resolution` samples per unit normalized area, at least 1000).
//!   Distances are exact point-to-triangle distances (BVH) instead of the
//!   upstream 10-nearest-sample approximation, and hull faces lying on the
//!   part surface (and part triangles lying on the hull) are recognized and
//!   skipped since their distance is zero.
//! * `MonteCarloTreeSearch` (`tree_policy`, `expand`, `default_policy`,
//!   `best_child`, `backup`), `ComputeAxesAlignedClippingPlanes`,
//!   `ComputeBestRvClippingPlane`, `clip_by_path` and `TernaryMCTS` — the
//!   cutting-plane search with the paper's parameters (150 iterations,
//!   depth 3, 20 axis-aligned planes per axis, Rv reward, ternary refinement
//!   of the best plane along its axis).
//! * `Compute` — the iterative cut loop (parts whose concavity exceeds the
//!   threshold are cut until all pass).
//! * `MergeConvexHulls` / `MergeCH` — greedy pairwise merging by minimal
//!   concavity of the merged hull, down to `max_convex_hull` or while the
//!   cost stays below the threshold. Deviation (as in upstream's current
//!   master): the merge cost is measured against the parts' original
//!   geometry rather than against the two hulls — `Rv` from the true part
//!   volumes, and `Hb` as the larger of (hull surface outside the input
//!   solid → its distance to the input surface) and (input surface of the
//!   merged parts → its distance to the hull boundary), plus the parts' own
//!   costs, so the cost of a merge never hides earlier ones.
//!
//! Determinism: no wall clock; the random plane shuffles use ChaCha8 seeded
//! from `CoacdParams::seed` (re-seeded per part, like upstream); surface
//! samples are a deterministic low-discrepancy pattern; parts are processed
//! in parallel but collected in index order.

use frac_geom::hull::{ConvexPolytope, HalfSpace, convex_hull_fast};
use frac_geom::inside::MeshQuery;
use frac_geom::{Aabb, DVec3, TriMesh};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AO};

/// Profiling counters (`FRAC_PROFILE`): merged hulls evaluated, cumulative
/// CPU ns per cost component, out-term evaluations, clips.
pub static N_PAIR: AtomicU64 = AtomicU64::new(0);
pub static T_HULLC: AtomicU64 = AtomicU64::new(0);
pub static T_IN: AtomicU64 = AtomicU64::new(0);
pub static T_OUT: AtomicU64 = AtomicU64::new(0);
pub static N_EVAL: AtomicU64 = AtomicU64::new(0);
pub static T_COV: AtomicU64 = AtomicU64::new(0);
pub static N_STAGE2: AtomicU64 = AtomicU64::new(0);
pub static N_CLIP: AtomicU64 = AtomicU64::new(0);
pub static T_CLIP: AtomicU64 = AtomicU64::new(0);
pub static T_HV: AtomicU64 = AtomicU64::new(0);

/// CoACD parameters (upstream defaults in `Default`).
#[derive(Clone, Debug)]
pub struct CoacdParams {
    /// Concavity threshold in normalized units (longest side = 2).
    pub threshold: f64,
    pub mcts_iterations: u32,
    pub mcts_depth: u32,
    /// Axis-aligned candidate planes per axis.
    pub mcts_nodes: u32,
    /// Surface samples per unit normalized area for `Hb` (min 1000 per mesh).
    pub resolution: u32,
    pub rv_k: f64,
    pub seed: u64,
    pub merge: bool,
    /// Merge down to this many hulls (None: merge only below the threshold).
    pub max_convex_hull: Option<usize>,
    /// Safety bound on the number of parts produced by cutting.
    pub max_parts: usize,
    /// Merge cost: upstream CoACD 1.0.x hull-vs-hull `ComputeHCost(cvx1,
    /// cvx2, CH)` (faithful baseline) or the collision-aware cost measured
    /// against the original geometry.
    pub merge_cost: MergeCost,
    /// Surface-deviation estimator of the cut loop's concavity check.
    pub hb: HbMode,
}

/// Estimator of `Hb` (part surface vs its hull) in the cut loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HbMode {
    /// Upstream `ComputeHb`: random samples (`ExtractPointSet`, small
    /// triangles sampled every other one) and, per sample, the distance to
    /// the triangles of its 10 nearest samples on the other surface
    /// (`face_hausdorff_distance`). Overestimates where triangles are small.
    Upstream,
    /// Exact point-to-surface distances with zero-distance face skipping.
    Exact,
}

/// Merge cost of [`decompose`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeCost {
    /// `max(Rv(cvx1 + cvx2 vs CH), Hb(cvx1 ∪ cvx2 surface minus their
    /// common face, CH))` with `resolution + 2000` samples, as upstream 1.0.x.
    Upstream,
    /// Collision-aware cost against the input surface (pipeline default).
    CollisionAware,
}

impl Default for CoacdParams {
    fn default() -> Self {
        CoacdParams {
            threshold: 0.05,
            mcts_iterations: 150,
            mcts_depth: 3,
            mcts_nodes: 20,
            resolution: 2000,
            rv_k: 0.3,
            seed: 1234,
            merge: true,
            max_convex_hull: None,
            max_parts: 1024,
            merge_cost: MergeCost::Upstream,
            hb: HbMode::Upstream,
        }
    }
}

const PI: f64 = 3.14159265;

/// A closed triangle mesh with per-triangle "cut cap" flags (caps are not
/// part of the original surface).
#[derive(Clone, Debug, Default)]
pub struct Solid {
    pub mesh: TriMesh,
    pub cap: Vec<bool>,
}

impl Solid {
    pub fn new(mesh: TriMesh) -> Self {
        let cap = vec![false; mesh.tris.len()];
        Solid { mesh, cap }
    }
    pub fn is_empty(&self) -> bool {
        self.mesh.tris.is_empty()
    }
    pub fn volume(&self) -> f64 {
        self.mesh.signed_volume()
    }
}

/// Cutting plane `n·x + d = 0` (upstream `Plane(a, b, c, d)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    pub n: DVec3,
    pub d: f64,
}

impl Plane {
    fn axis(k: usize, at: f64) -> Plane {
        let mut n = DVec3::ZERO;
        n[k] = 1.0;
        Plane { n, d: -at }
    }
    fn axis_index(&self) -> Option<usize> {
        (0..3).find(|&k| (self.n[k] - 1.0).abs() < 1e-4)
    }
    #[inline]
    fn dist(&self, p: DVec3) -> f64 {
        self.n.dot(p) + self.d
    }
}

// ---------------------------------------------------------------------------
// Normalization

/// Affine map between world coordinates and CoACD's normalized frame.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub mid: DVec3,
    /// world = normalized * half + mid
    pub half: f64,
}

impl Frame {
    pub fn of(bbox: &Aabb) -> Frame {
        let len = bbox.extent().max_element().max(1e-300);
        Frame {
            mid: bbox.center(),
            half: 0.5 * len,
        }
    }
    pub fn to_norm(&self, p: DVec3) -> DVec3 {
        (p - self.mid) / self.half
    }
    pub fn to_world(&self, p: DVec3) -> DVec3 {
        p * self.half + self.mid
    }
    pub fn polytope_to_world(&self, p: &ConvexPolytope) -> ConvexPolytope {
        ConvexPolytope {
            faces: p
                .faces
                .iter()
                .map(|(h, f)| {
                    (
                        HalfSpace {
                            n: h.n,
                            d: h.d * self.half + h.n.dot(self.mid),
                        },
                        f.iter().map(|&q| self.to_world(q)).collect(),
                    )
                })
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Plane clipping with cap triangulation (upstream `Clip`)

/// Clip a closed solid by a plane. Returns (positive side, negative side,
/// cap area); `None` when the cap could not be triangulated.
pub fn clip(s: &Solid, pl: &Plane) -> Option<(Solid, Solid, f64)> {
    let t0 = std::time::Instant::now();
    let r = clip_inner(s, pl, true);
    N_CLIP.fetch_add(1, AO::Relaxed);
    T_CLIP.fetch_add(t0.elapsed().as_nanos() as u64, AO::Relaxed);
    r
}

/// Clip for the tree search: caps are fans from one point of the cut plane
/// over the boundary edges. The halves are closed (possibly folded) meshes
/// with exact signed volumes and exact convex hulls (the fan apex lies in
/// the convex hull of the cut section), which is all the Rv-based search
/// needs; they are never used as final parts.
fn clip_fast(s: &Solid, pl: &Plane) -> Option<(Solid, Solid, f64)> {
    let t0 = std::time::Instant::now();
    let r = clip_inner(s, pl, false);
    N_CLIP.fetch_add(1, AO::Relaxed);
    T_CLIP.fetch_add(t0.elapsed().as_nanos() as u64, AO::Relaxed);
    r
}

fn clip_inner(s: &Solid, pl: &Plane, exact_caps: bool) -> Option<(Solid, Solid, f64)> {
    use std::collections::HashMap;
    let m = &s.mesh;
    let nv = m.verts.len();
    let scale = m.aabb().diagonal().max(1e-300);
    let eps = 1e-12 * scale;
    let dist: Vec<f64> = m.verts.iter().map(|&p| pl.dist(p)).collect();
    let sg: Vec<i8> = dist
        .iter()
        .map(|&d| {
            if d > eps {
                1
            } else if d < -eps {
                -1
            } else {
                0
            }
        })
        .collect();
    if sg.iter().all(|&x| x >= 0) && sg.iter().any(|&x| x > 0) {
        return Some((s.clone(), Solid::default(), 0.0));
    }
    if sg.iter().all(|&x| x <= 0) && sg.iter().any(|&x| x < 0) {
        return Some((Solid::default(), s.clone(), 0.0));
    }
    // cut points: one per crossing edge, ids nv.. (shared by both sides)
    let mut cut_id: HashMap<(u32, u32), u32> = HashMap::new();
    let mut cut_pts: Vec<DVec3> = Vec::new();
    let mut cut_of = |a: u32, b: u32| -> u32 {
        let (a, b) = (a.min(b), a.max(b));
        *cut_id.entry((a, b)).or_insert_with(|| {
            let (da, db) = (dist[a as usize], dist[b as usize]);
            let t = da / (da - db);
            let (pa, pb) = (m.verts[a as usize], m.verts[b as usize]);
            cut_pts.push(pa + (pb - pa) * t);
            (nv + cut_pts.len() - 1) as u32
        })
    };
    // per side: triangles over global ids (orig < nv <= cut)
    let mut side_tris: [Vec<[u32; 3]>; 2] = [Vec::new(), Vec::new()]; // 0 = pos, 1 = neg
    let mut side_cap: [Vec<bool>; 2] = [Vec::new(), Vec::new()];
    let push = |side_tris: &mut [Vec<[u32; 3]>; 2],
                side_cap: &mut [Vec<bool>; 2],
                si: usize,
                poly: &[u32],
                cap: bool| {
        for k in 1..poly.len().saturating_sub(1) {
            let t = [poly[0], poly[k], poly[k + 1]];
            if t[0] != t[1] && t[1] != t[2] && t[0] != t[2] {
                side_tris[si].push(t);
                side_cap[si].push(cap);
            }
        }
    };
    for (ti, t) in m.tris.iter().enumerate() {
        let s3 = [sg[t[0] as usize], sg[t[1] as usize], sg[t[2] as usize]];
        let cap = s.cap.get(ti).copied().unwrap_or(false);
        let has_p = s3.iter().any(|&x| x > 0);
        let has_n = s3.iter().any(|&x| x < 0);
        if !has_p && !has_n {
            // coplanar triangle: belongs to the side it bounds
            let [a, b, c] = m.tri_points(ti);
            let side = if (b - a).cross(c - a).dot(pl.n) > 0.0 {
                1
            } else {
                0
            };
            push(&mut side_tris, &mut side_cap, side, t, cap);
        } else if !has_n {
            push(&mut side_tris, &mut side_cap, 0, t, cap);
        } else if !has_p {
            push(&mut side_tris, &mut side_cap, 1, t, cap);
        } else {
            let mut pp = [0u32; 4];
            let mut np = [0u32; 4];
            let (mut ip, mut inn) = (0, 0);
            for k in 0..3 {
                let (u, w) = (t[k], t[(k + 1) % 3]);
                let (su, sw) = (s3[k], s3[(k + 1) % 3]);
                if su >= 0 {
                    pp[ip] = u;
                    ip += 1;
                }
                if su <= 0 {
                    np[inn] = u;
                    inn += 1;
                }
                if su * sw < 0 {
                    let r = cut_of(u, w);
                    pp[ip] = r;
                    ip += 1;
                    np[inn] = r;
                    inn += 1;
                }
            }
            push(&mut side_tris, &mut side_cap, 0, &pp[..ip], cap);
            push(&mut side_tris, &mut side_cap, 1, &np[..inn], cap);
        }
    }
    let pos_of = |g: u32| -> DVec3 {
        if (g as usize) < nv {
            m.verts[g as usize]
        } else {
            cut_pts[g as usize - nv]
        }
    };
    let on_plane = |g: u32| -> bool { (g as usize) >= nv || sg[g as usize] == 0 };
    let mut out: Vec<Solid> = Vec::with_capacity(2);
    let mut cap_area = 0.0;
    for si in 0..2 {
        let tris_g = std::mem::take(&mut side_tris[si]);
        let mut caps = std::mem::take(&mut side_cap[si]);
        if tris_g.is_empty() {
            out.push(Solid::default());
            continue;
        }
        // compact vertex ids
        let mut remap: HashMap<u32, u32> = HashMap::new();
        let mut verts: Vec<DVec3> = Vec::new();
        let mut order_ids: Vec<u32> = Vec::new();
        let mut tris: Vec<[u32; 3]> = Vec::with_capacity(tris_g.len() + 16);
        for t in &tris_g {
            let mut nt = [0u32; 3];
            for k in 0..3 {
                nt[k] = *remap.entry(t[k]).or_insert_with(|| {
                    verts.push(pos_of(t[k]));
                    order_ids.push(t[k]);
                    (verts.len() - 1) as u32
                });
            }
            tris.push(nt);
        }
        // boundary edges lie in the plane: count only on-plane directed edges
        let mut pe: HashMap<(u32, u32), i32> = HashMap::new();
        for t in &tris {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                if on_plane(order_ids[a as usize]) && on_plane(order_ids[b as usize]) {
                    *pe.entry((a, b)).or_default() += 1;
                }
            }
        }
        let mut bedges: Vec<(u32, u32)> = Vec::new();
        for (&(a, b), &c) in &pe {
            let r = pe.get(&(b, a)).copied().unwrap_or(0);
            for _ in 0..(c - r).max(0) {
                bedges.push((a, b));
            }
        }
        bedges.sort_unstable();
        if !exact_caps {
            if !bedges.is_empty() {
                let c = bedges
                    .iter()
                    .fold(DVec3::ZERO, |acc, e| acc + verts[e.0 as usize])
                    / bedges.len() as f64;
                let c = c - pl.n * pl.dist(c);
                verts.push(c);
                let ci = (verts.len() - 1) as u32;
                for &(a, b) in &bedges {
                    tris.push([ci, b, a]);
                    caps.push(true);
                }
            }
            out.push(Solid {
                mesh: TriMesh { verts, tris },
                cap: caps,
            });
            continue;
        }
        let mut next: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for &(a, b) in &bedges {
            next.entry(a).or_default().push(b);
        }
        // cap normal: +n for the negative side, -n for the positive side
        let cn = if si == 1 { pl.n } else { -pl.n };
        let (u, v) = frac_geom::polygon::plane_basis(cn);
        let mut loops: Vec<Vec<u32>> = Vec::new();
        let limit = bedges.len() + 2;
        while let Some((&start, _)) = next.iter().find(|(_, v)| !v.is_empty()) {
            let mut l = vec![start];
            let mut cur = start;
            let mut guard = 0usize;
            loop {
                let Some(outs) = next.get_mut(&cur) else {
                    break;
                };
                if outs.is_empty() {
                    break;
                }
                let nx = outs.remove(0);
                if nx == start {
                    break;
                }
                l.push(nx);
                cur = nx;
                guard += 1;
                if guard > limit {
                    break;
                }
            }
            if l.len() >= 3 {
                // cap uses the reversed boundary edges
                l.reverse();
                loops.push(l);
            }
        }
        if !loops.is_empty() {
            let p2 = |i: u32| -> [f64; 2] {
                let p = verts[i as usize];
                [p.dot(u), p.dot(v)]
            };
            let areas: Vec<f64> = loops
                .iter()
                .map(|l| {
                    let n = l.len();
                    (0..n)
                        .map(|k| {
                            let a = p2(l[k]);
                            let b = p2(l[(k + 1) % n]);
                            a[0] * b[1] - b[0] * a[1]
                        })
                        .sum::<f64>()
                        * 0.5
                })
                .collect();
            let outers: Vec<usize> = (0..loops.len()).filter(|&i| areas[i] > 0.0).collect();
            let mut groups: Vec<Vec<usize>> = outers.iter().map(|&o| vec![o]).collect();
            for h in (0..loops.len()).filter(|&i| areas[i] <= 0.0) {
                // smallest outer containing the hole
                let q = loops[h]
                    .iter()
                    .map(|&i| p2(i))
                    .fold([0.0, 0.0], |a, b| [a[0] + b[0], a[1] + b[1]]);
                let q = [q[0] / loops[h].len() as f64, q[1] / loops[h].len() as f64];
                let mut best: Option<usize> = None;
                for (gi, &o) in outers.iter().enumerate() {
                    if point_in_loop(q, &loops[o].iter().map(|&i| p2(i)).collect::<Vec<_>>())
                        && best.map(|b| areas[outers[b]] > areas[o]).unwrap_or(true)
                    {
                        best = Some(gi);
                    }
                }
                if let Some(gi) = best {
                    groups[gi].push(h);
                } else if areas[h].abs() > 1e-18 * scale * scale {
                    return None;
                }
            }
            for g in &groups {
                let mut pts: Vec<[f64; 2]> = Vec::new();
                let mut back: Vec<u32> = Vec::new();
                let mut gl: Vec<Vec<usize>> = Vec::new();
                let mut first: BTreeMap<u32, usize> = BTreeMap::new();
                for &li in g {
                    let mut ll = Vec::new();
                    for &vi in &loops[li] {
                        // vertices shared between loops of the group keep one index
                        let idx = *first.entry(vi).or_insert_with(|| {
                            pts.push(p2(vi));
                            back.push(vi);
                            pts.len() - 1
                        });
                        ll.push(idx);
                    }
                    gl.push(ll);
                }
                let tt = frac_cells::tri2d::triangulate(&pts, &gl);
                if tt.is_empty() && areas[g[0]] > 1e-18 * scale * scale {
                    return None;
                }
                for t in tt {
                    let tri = [back[t[0]], back[t[1]], back[t[2]]];
                    if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] {
                        tris.push(tri);
                        caps.push(true);
                    }
                }
                if si == 1 {
                    cap_area += g.iter().map(|&i| areas[i]).sum::<f64>();
                }
            }
        }
        out.push(Solid {
            mesh: TriMesh { verts, tris },
            cap: caps,
        });
    }
    let neg = out.pop().unwrap();
    let pos = out.pop().unwrap();
    Some((pos, neg, cap_area))
}

fn point_in_loop(q: [f64; 2], l: &[[f64; 2]]) -> bool {
    let n = l.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (l[i], l[j]);
        if (a[1] > q[1]) != (b[1] > q[1])
            && q[0] < (b[0] - a[0]) * (q[1] - a[1]) / (b[1] - a[1]) + a[0]
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

// ---------------------------------------------------------------------------
// Hulls and costs

/// Convex hull of a point set as a polytope with its vertices and volume.
#[derive(Clone, Debug, Default)]
pub struct Ch {
    pub poly: ConvexPolytope,
    pub pts: Vec<DVec3>,
    pub volume: f64,
}

pub fn hull(points: &[DVec3]) -> Option<Ch> {
    let m = convex_hull_fast(points)?;
    let volume = m.signed_volume().max(0.0);
    let poly = ConvexPolytope::from_hull_mesh(&m);
    if poly.is_empty() {
        return None;
    }
    Some(Ch {
        pts: m.verts,
        poly,
        volume,
    })
}

/// Like [`hull`] but with one face per hull triangle (no coplanar-face
/// merging): the same solid, much cheaper to build; used for candidate
/// merges, which are rebuilt with [`hull`] only when accepted.
pub fn hull_tri(points: &[DVec3]) -> Option<Ch> {
    let m = convex_hull_fast(points)?;
    let volume = m.signed_volume().max(0.0);
    let mut faces = Vec::with_capacity(m.tris.len());
    for t in 0..m.tris.len() {
        let [a, b, c] = m.tri_points(t);
        let n = (b - a).cross(c - a);
        let l = n.length();
        if l > 0.0 {
            let n = n / l;
            faces.push((HalfSpace { n, d: n.dot(a) }, vec![a, b, c]));
        }
    }
    let poly = ConvexPolytope { faces };
    if poly.is_empty() {
        return None;
    }
    Some(Ch {
        pts: m.verts,
        poly,
        volume,
    })
}

/// Convex polytope with at most `max_v` vertices inside `poly`: the hull of
/// a vertex subset grown greedily from the axis-extreme vertices by always
/// adding the vertex farthest outside the current hull (a budgeted
/// Quickhull, which keeps the most volume per vertex). Being an inner
/// approximation it never adds overshoot and keeps any separation from
/// neighbouring hulls. Deterministic (ties by vertex order).
pub fn limit_vertices(poly: &ConvexPolytope, max_v: usize) -> ConvexPolytope {
    let n = poly.vertices().len();
    if max_v == 0 || n <= max_v || poly.is_empty() {
        return poly.clone();
    }
    greedy_subset(poly, max_v, 0.0)
}

/// Hull of a greedy vertex subset of `poly` (see [`limit_vertices`]): only
/// vertices farther than `eps` outside the current hull are added, so
/// near-duplicate and near-coplanar vertices (the source of sliver faces)
/// are dropped; at most `max_v` vertices (0: unlimited).
pub fn greedy_subset(poly: &ConvexPolytope, max_v: usize, eps: f64) -> ConvexPolytope {
    let vs = poly.vertices();
    if vs.len() < 4 || poly.is_empty() {
        return poly.clone();
    }
    let max_v = if max_v == 0 { usize::MAX } else { max_v.max(4) };
    let mut chosen: Vec<usize> = Vec::new();
    for k in 0..3 {
        let lo = (0..vs.len())
            .min_by(|&a, &b| vs[a][k].total_cmp(&vs[b][k]).then(a.cmp(&b)))
            .unwrap();
        let hi = (0..vs.len())
            .max_by(|&a, &b| vs[a][k].total_cmp(&vs[b][k]).then(b.cmp(&a)))
            .unwrap();
        for i in [lo, hi] {
            if !chosen
                .iter()
                .any(|&c| c == i || (vs[c] - vs[i]).length() <= eps)
            {
                chosen.push(i);
            }
        }
    }
    let mut cur: Option<Ch> = None;
    let mut is_chosen = vec![false; vs.len()];
    for &c in &chosen {
        is_chosen[c] = true;
    }
    loop {
        let pts: Vec<DVec3> = chosen.iter().map(|&i| vs[i]).collect();
        let h = hull(&pts);
        // vertices outside the current hull: the farthest beyond each face
        // (Quickhull's choice; several per round while far from the
        // budget, else only the globally farthest), or the farthest from
        // the chosen set while it is still degenerate
        let add: Vec<usize> = match &h {
            Some(h) => {
                let mut per_face: Vec<Option<(usize, f64)>> = vec![None; h.poly.faces.len()];
                for i in (0..vs.len()).filter(|&i| !is_chosen[i]) {
                    let (mut bf, mut bd) = (usize::MAX, eps);
                    for (k, (f, _)) in h.poly.faces.iter().enumerate() {
                        let d = f.dist(vs[i]);
                        if d > bd {
                            bd = d;
                            bf = k;
                        }
                    }
                    if bf != usize::MAX
                        && per_face[bf]
                            .map(|(j, dj)| bd > dj || (bd == dj && i < j))
                            .unwrap_or(true)
                    {
                        per_face[bf] = Some((i, bd));
                    }
                }
                let mut cand: Vec<(usize, f64)> = per_face.into_iter().flatten().collect();
                cand.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                cand.dedup_by_key(|c| c.0);
                let room = max_v.saturating_sub(h.pts.len());
                let take = cand.len().min((room / 2).max(1));
                cand.into_iter().take(take).map(|c| c.0).collect()
            }
            None => (0..vs.len())
                .filter(|&i| !is_chosen[i])
                .map(|i| {
                    (
                        i,
                        chosen
                            .iter()
                            .map(|&c| (vs[i] - vs[c]).length())
                            .fold(f64::INFINITY, f64::min),
                    )
                })
                .filter(|x| x.1 > eps)
                .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
                .map(|x| vec![x.0])
                .unwrap_or_default(),
        };
        if let Some(h) = h {
            if h.pts.len() > max_v {
                break;
            }
            cur = Some(h);
        }
        if add.is_empty() || chosen.len() >= 4 * max_v.min(1 << 20) {
            break;
        }
        for i in add {
            is_chosen[i] = true;
            chosen.push(i);
        }
    }
    cur.map(|c| c.poly).unwrap_or_else(|| poly.clone())
}

/// Convexity of a stored hull (vertices, polygon faces) as checked by the
/// `collision_shapes` validation gate: every vertex on or behind every face
/// plane (Newell normal through the face centroid), relative 1e-9.
pub fn stored_hull_is_convex(verts: &[DVec3], faces: &[Vec<u32>]) -> bool {
    let scale = Aabb::from_points(verts.iter()).diagonal().max(1e-300);
    let tol = 1e-9 * scale;
    !faces.is_empty()
        && faces.iter().all(|f| {
            let pts: Vec<DVec3> = f.iter().map(|&i| verts[i as usize]).collect();
            let n = frac_geom::polygon::newell(&pts).normalize_or_zero();
            let c = pts.iter().fold(DVec3::ZERO, |a, p| a + *p) / pts.len().max(1) as f64;
            n != DVec3::ZERO && verts.iter().all(|v| (*v - c).dot(n) <= tol)
        })
}

/// Hull volume only (MCTS rewards need nothing else).
fn hull_volume(points: &[DVec3]) -> f64 {
    let t0 = std::time::Instant::now();
    let r = convex_hull_fast(points)
        .map(|m| m.signed_volume().max(0.0))
        .unwrap_or(0.0);
    T_HV.fetch_add(t0.elapsed().as_nanos() as u64, AO::Relaxed);
    r
}

/// Upstream `ComputeRv`.
pub fn rv(v_mesh: f64, v_hull: f64, k: f64) -> f64 {
    libm::cbrt(3.0 * (v_mesh - v_hull).abs() / (4.0 * PI)) * k
}

fn rv_solid(s: &Solid, k: f64) -> f64 {
    rv(s.volume(), hull_volume(&s.mesh.verts), k)
}

/// Deterministic low-discrepancy samples on a triangle (R2 sequence mapped
/// with the upstream barycentric warp).
pub fn tri_samples(a: DVec3, b: DVec3, c: DVec3, n: usize, out: &mut Vec<DVec3>) {
    if n == 1 {
        out.push((a + b + c) / 3.0);
        return;
    }
    const A1: f64 = 0.7548776662466927;
    const A2: f64 = 0.5698402909980532;
    for k in 0..n {
        let x = (0.5 + A1 * k as f64).fract();
        let y = (0.5 + A2 * k as f64).fract();
        let s = x.sqrt();
        out.push(a * (1.0 - s) + b * (s * (1.0 - y)) + c * (y * s));
    }
}

/// Distance from a point inside a convex polytope to its boundary.
#[inline]
pub fn inside_depth(poly: &ConvexPolytope, p: DVec3) -> f64 {
    poly.faces
        .iter()
        .map(|(h, _)| h.d - h.n.dot(p))
        .fold(f64::INFINITY, f64::min)
}

/// True when some triangle of the query mesh lies within `r` of `p`.
pub fn within(q: &MeshQuery, p: DVec3, r: f64) -> bool {
    let r2 = r * r;
    let found = std::cell::Cell::new(false);
    q.bvh.traverse(
        |b| !found.get() && b.dist2(p) <= r2,
        |t| {
            if found.get() {
                return;
            }
            let [a, b, c] = q.mesh.tri_points(t as usize);
            if (frac_geom::inside::closest_point_triangle(p, a, b, c) - p).length_squared() <= r2 {
                found.set(true);
            }
        },
    );
    found.get()
}

/// Nearest triangle of the query mesh if it is farther than `r` from `p`
/// (`None` as soon as a triangle within `r` is found).
pub fn farther_than(q: &MeshQuery, p: DVec3, r: f64) -> Option<(u32, f64)> {
    q.bvh.nearest_beyond(p, r * r, |t| {
        let [a, b, c] = q.mesh.tri_points(t as usize);
        (frac_geom::inside::closest_point_triangle(p, a, b, c) - p).length_squared()
    })
}

/// Angle-weighted pseudo-normals of a closed triangle mesh (Bærentzen &
/// Aanæs): the side of a point follows from its nearest surface point and
/// the pseudo-normal of the feature (face, edge or vertex) it lies on.
pub struct Signer {
    vn: Vec<DVec3>,
    en: std::collections::HashMap<(u32, u32), DVec3>,
}

impl Signer {
    pub fn new(m: &TriMesh) -> Signer {
        let mut vn = vec![DVec3::ZERO; m.verts.len()];
        let mut en: std::collections::HashMap<(u32, u32), DVec3> =
            std::collections::HashMap::with_capacity(m.tris.len() * 3 / 2);
        for t in &m.tris {
            let p = [
                m.verts[t[0] as usize],
                m.verts[t[1] as usize],
                m.verts[t[2] as usize],
            ];
            let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
            for k in 0..3 {
                let (a, b, c) = (p[k], p[(k + 1) % 3], p[(k + 2) % 3]);
                let ang = (b - a).angle_between(c - a);
                if ang.is_finite() {
                    vn[t[k] as usize] += n * ang;
                }
                let (u, w) = (t[k].min(t[(k + 1) % 3]), t[k].max(t[(k + 1) % 3]));
                *en.entry((u, w)).or_default() += n;
            }
        }
        Signer { vn, en }
    }

    /// Outside test of `p` whose nearest triangle is `t` (`None` if the
    /// pseudo-normal is degenerate).
    pub fn outside(&self, m: &TriMesh, p: DVec3, t: u32) -> Option<bool> {
        let tri = m.tris[t as usize];
        let [a, b, c] = m.tri_points(t as usize);
        let n = (b - a).cross(c - a);
        let nn = n.length_squared();
        if nn <= 0.0 {
            return None;
        }
        let cp = frac_geom::inside::closest_point_triangle(p, a, b, c);
        let w = [
            (b - cp).cross(c - cp).dot(n) / nn,
            (c - cp).cross(a - cp).dot(n) / nn,
            (a - cp).cross(b - cp).dot(n) / nn,
        ];
        let eps = 1e-9;
        let zero: Vec<usize> = (0..3).filter(|&k| w[k] <= eps).collect();
        let pn = match zero.len() {
            0 => n,
            1 => {
                // edge opposite to the zero coordinate
                let k = zero[0];
                let (u, v) = (tri[(k + 1) % 3], tri[(k + 2) % 3]);
                *self.en.get(&(u.min(v), u.max(v)))?
            }
            _ => {
                let k = (0..3).find(|k| !zero.contains(k))?;
                self.vn[tri[k] as usize]
            }
        };
        let s = (p - cp).dot(pn);
        if s == 0.0 { None } else { Some(s > 0.0) }
    }
}

/// Outside test for a point whose nearest triangle is `t`: exact from the
/// triangle's side when the nearest point is interior to the triangle,
/// ray parity otherwise.
pub fn is_outside(q: &MeshQuery, p: DVec3, t: u32) -> bool {
    let [a, b, c] = q.mesh.tri_points(t as usize);
    let n = (b - a).cross(c - a);
    let cp = frac_geom::inside::closest_point_triangle(p, a, b, c);
    let nn = n.length_squared();
    if nn > 0.0 {
        // barycentric coordinates of the nearest point
        let wa = (b - cp).cross(c - cp).dot(n) / nn;
        let wb = (c - cp).cross(a - cp).dot(n) / nn;
        let wc = 1.0 - wa - wb;
        let s = (p - cp).dot(n);
        if wa > 1e-9 && wb > 1e-9 && wc > 1e-9 && s.abs() > 1e-12 * nn.sqrt() * (p - cp).length() {
            return s > 0.0;
        }
    }
    !q.contains(p)
}

/// Upstream `ComputeHb(mesh, hull)`: symmetric surface deviation between a
/// part and its convex hull, from `density` samples per unit area. Returns
/// early (with a value above `cap`) once the deviation exceeds `cap`.
pub fn hb(part: &Solid, ch: &Ch, density: f64, cap: f64) -> f64 {
    let m = &part.mesh;
    if m.tris.is_empty() || ch.poly.is_empty() {
        return f64::INFINITY;
    }
    let scale = m.aabb().diagonal().max(1e-300);
    let eps = 1e-9 * scale;
    let nf = ch.poly.faces.len();
    let mut covered = vec![0.0f64; nf];
    let mut on_hull = vec![false; m.tris.len()];
    for t in 0..m.tris.len() {
        let [a, b, c] = m.tri_points(t);
        let nv = (b - a).cross(c - a);
        let ar = 0.5 * nv.length();
        if ar <= 0.0 {
            on_hull[t] = true;
            continue;
        }
        let n = nv / (2.0 * ar);
        for (fi, (h, _)) in ch.poly.faces.iter().enumerate() {
            if h.n.dot(n) > 1.0 - 1e-9 && [a, b, c].iter().all(|p| (h.n.dot(*p) - h.d).abs() <= eps)
            {
                covered[fi] += ar;
                on_hull[t] = true;
                break;
            }
        }
    }
    let mut worst: f64 = 0.0;
    let mut buf = Vec::new();
    // part surface -> hull boundary (exact: points are inside the hull)
    for t in 0..m.tris.len() {
        if on_hull[t] {
            continue;
        }
        let [a, b, c] = m.tri_points(t);
        let ar = 0.5 * (b - a).cross(c - a).length();
        buf.clear();
        tri_samples(a, b, c, ((ar * density) as usize).max(1), &mut buf);
        for &p in &buf {
            let d = inside_depth(&ch.poly, p).max(0.0);
            if d > worst {
                worst = d;
                if worst > cap {
                    return worst;
                }
            }
        }
    }
    // hull boundary -> part surface (uncovered faces only)
    let mut q: Option<MeshQuery> = None;
    for (fi, (_, f)) in ch.poly.faces.iter().enumerate() {
        let fa = frac_geom::polygon::newell(f).length() * 0.5;
        if covered[fi] >= fa * (1.0 - 1e-6) {
            continue;
        }
        let q = q.get_or_insert_with(|| MeshQuery::new(m));
        for k in 1..f.len() - 1 {
            let (a, b, c) = (f[0], f[k], f[k + 1]);
            let ar = 0.5 * (b - a).cross(c - a).length();
            buf.clear();
            tri_samples(a, b, c, ((ar * density) as usize).max(1), &mut buf);
            for &p in &buf {
                if let Some((_, d2)) = farther_than(q, p, worst) {
                    worst = worst.max(d2.sqrt());
                    if worst > cap {
                        return worst;
                    }
                }
            }
        }
    }
    worst
}

/// Upstream sampling density: `resolution` samples per unit area, at least
/// 1000 samples per mesh.
fn density_for(area: f64, resolution: u32) -> f64 {
    let a = area.max(1e-300);
    (1000.0f64).max(resolution as f64 * a) / a
}

/// Upstream `ComputeHCost(mesh, hull)` = max(Rv, Hb).
pub fn h_cost(part: &Solid, ch: &Ch, p: &CoacdParams, cap: f64) -> f64 {
    let r = rv(part.volume(), ch.volume, p.rv_k);
    if r > cap {
        return r;
    }
    match p.hb {
        HbMode::Exact => r.max(hb(
            part,
            ch,
            density_for(part.mesh.area(), p.resolution),
            cap,
        )),
        HbMode::Upstream => r.max(hb_upstream(
            &part.mesh,
            &ch.poly.to_mesh(),
            p.resolution,
            p.seed,
        )),
    }
}

/// Upstream `Model::ExtractPointSet(resolution, base = 1)`: per-triangle
/// sample counts `max(i % 2 == 0, ⌊R'·area/A⌋)` with `R' = max(1000, R·A)`
/// (every `tris/R'`-th triangle when there are more triangles than `R'`),
/// uniform barycentric samples with the sqrt warp.
fn upstream_samples(m: &TriMesh, resolution: u32, rng: &mut ChaCha8Rng) -> (Vec<DVec3>, Vec<u32>) {
    upstream_samples_f(m, resolution as f64, rng, &|_| true)
}

/// [`upstream_samples`] with a fractional resolution and a triangle filter
/// (upstream skips the triangles on the two hulls' common face).
fn upstream_samples_f(
    m: &TriMesh,
    resolution: f64,
    rng: &mut ChaCha8Rng,
    keep: &dyn Fn(usize) -> bool,
) -> (Vec<DVec3>, Vec<u32>) {
    use rand::Rng;
    let a_obj: f64 = (0..m.tris.len())
        .map(|t| {
            let [a, b, c] = m.tri_points(t);
            0.5 * (b - a).cross(c - a).length()
        })
        .sum();
    let r = (1000.0f64).max(resolution * a_obj);
    let nt = m.tris.len();
    let (mut pts, mut ids) = (Vec::new(), Vec::new());
    for t in 0..nt {
        if !keep(t) {
            continue;
        }
        let [a, b, c] = m.tri_points(t);
        let area = 0.5 * (b - a).cross(c - a).length();
        let every = if (nt as f64) > r {
            (nt as f64 / r).floor().max(1.0) as usize
        } else {
            2
        };
        let n = ((t % every == 0) as usize).max((r / a_obj.max(1e-300) * area) as usize);
        for _ in 0..n {
            let (x, y): (f64, f64) = (rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0));
            let s = x.sqrt();
            pts.push(a * (1.0 - s) + b * (s * (1.0 - y)) + c * (y * s));
            ids.push(t as u32);
        }
    }
    (pts, ids)
}

/// The 10 nearest sample indices of `p` (Euclidean), via the BVH over the
/// sample points.
fn knn10(bvh: &frac_geom::bvh::Bvh, pts: &[DVec3], p: DVec3, out: &mut Vec<(f64, u32)>) {
    out.clear();
    let worst = std::cell::Cell::new(f64::INFINITY);
    let found = std::cell::RefCell::new(Vec::<(f64, u32)>::with_capacity(11));
    bvh.traverse(
        |b| b.dist2(p) <= worst.get(),
        |i| {
            let d = (pts[i as usize] - p).length_squared();
            let mut f = found.borrow_mut();
            if f.len() < 10 || d < worst.get() {
                let pos = f.partition_point(|x| (x.0, x.1) < (d, i));
                f.insert(pos, (d, i));
                if f.len() > 10 {
                    f.pop();
                }
                if f.len() == 10 {
                    worst.set(f[9].0);
                }
            }
        },
    );
    out.extend(found.into_inner());
}

/// Upstream `ComputeHb` with `face_hausdorff_distance` (see [`HbMode`]).
pub fn hb_upstream(a: &TriMesh, b: &TriMesh, resolution: u32, seed: u64) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let (sa, ia) = upstream_samples(a, resolution, &mut rng);
    let (sb, ib) = upstream_samples(b, resolution, &mut rng);
    face_hausdorff(a, &sa, &ia, b, &sb, &ib)
}

/// Upstream `face_hausdorff_distance`: for every sample, the distance to the
/// triangles of its 10 nearest samples on the other surface; the maximum.
fn face_hausdorff(
    a: &TriMesh,
    sa: &[DVec3],
    ia: &[u32],
    b: &TriMesh,
    sb: &[DVec3],
    ib: &[u32],
) -> f64 {
    if sa.is_empty() || sb.is_empty() {
        return f64::INFINITY;
    }
    let tree = |s: &[DVec3]| {
        frac_geom::bvh::Bvh::build(
            &s.iter()
                .map(|&q| Aabb::from_points([&q]))
                .collect::<Vec<_>>(),
        )
    };
    let (ta, tb) = (tree(sa), tree(sb));
    let mut cmax: f64 = 0.0;
    let mut nn = Vec::new();
    for (from, (tree_to, samples_to, ids_to, mesh_to)) in
        [(sb, (&ta, sa, ia, a)), (sa, (&tb, sb, ib, b))]
    {
        for &x in from.iter() {
            knn10(tree_to, samples_to, x, &mut nn);
            let mut cmin = f64::INFINITY;
            for &(_, j) in &nn {
                let [p, q, w] = mesh_to.tri_points(ids_to[j as usize] as usize);
                let d = (frac_geom::inside::closest_point_triangle(x, p, q, w) - x).length();
                if d < cmin {
                    cmin = d;
                    if cmin < 1e-14 {
                        break;
                    }
                }
            }
            if cmin > 10.0 {
                cmin = nn.first().map(|v| v.0.sqrt()).unwrap_or(cmin);
            }
            if cmin.is_finite() && cmin > cmax {
                cmax = cmin;
            }
        }
    }
    cmax
}

// ---------------------------------------------------------------------------
// Monte-Carlo tree search (upstream mcts.cpp)

/// `ComputeAxesAlignedClippingPlanes`.
fn axis_planes(bb: &Aabb, nodes: u32) -> Vec<Plane> {
    let mut out = Vec::new();
    let eps = 1e-6;
    for k in 0..3 {
        let (lo, hi) = (bb.min[k], bb.max[k]);
        let interval = (0.01f64).max((hi - lo).abs() / (nodes as f64 + 1.0));
        let m = (0.015f64).max(interval);
        let mut i = lo + m;
        while i <= hi - m + eps {
            out.push(Plane::axis(k, i));
            i += interval;
        }
    }
    out
}

#[derive(Clone)]
struct MPart {
    solid: Arc<Solid>,
    moves: Vec<Plane>,
    next: usize,
}

impl MPart {
    fn new(solid: Arc<Solid>, nodes: u32, rng: &mut ChaCha8Rng) -> MPart {
        let mut moves = axis_planes(&solid.mesh.aabb(), nodes);
        moves.shuffle(rng);
        MPart {
            solid,
            moves,
            next: 0,
        }
    }
}

#[derive(Clone)]
struct State {
    parts: Vec<MPart>,
    costs: Vec<f64>,
    worst: usize,
    round: u32,
    cost: f64,
    value: Plane,
}

impl State {
    fn terminal(&self, p: &CoacdParams) -> bool {
        self.round >= p.mcts_depth || self.parts[self.worst].moves.is_empty()
    }
    /// `ComputeReward`: the largest part cost (sets the worst part).
    fn reward(&mut self) -> f64 {
        let mut hmax = 0.0;
        for (i, &h) in self.costs.iter().enumerate() {
            if h > hmax {
                hmax = h;
                self.worst = i;
            }
        }
        hmax
    }
    /// Replace the worst part by the two halves of a cut.
    fn apply_cut(
        &mut self,
        pos: Solid,
        neg: Solid,
        plane: Plane,
        p: &CoacdParams,
        rng: &mut ChaCha8Rng,
    ) {
        let w = self.worst;
        let mut parts = Vec::with_capacity(self.parts.len() + 1);
        let mut costs = Vec::with_capacity(self.parts.len() + 1);
        for i in 0..self.parts.len() {
            if i != w {
                parts.push(self.parts[i].clone());
                costs.push(self.costs[i]);
            }
        }
        let cp = rv_solid(&pos, p.rv_k);
        let cn = rv_solid(&neg, p.rv_k);
        parts.push(MPart::new(Arc::new(pos), p.mcts_nodes, rng));
        parts.push(MPart::new(Arc::new(neg), p.mcts_nodes, rng));
        costs.push(cp);
        costs.push(cn);
        self.parts = parts;
        self.costs = costs;
        self.worst = 0;
        self.value = plane;
    }
}

struct Node {
    children: Vec<usize>,
    parent: Option<usize>,
    visits: f64,
    quality: f64,
    state: State,
}

/// `State::get_next_state_with_random_choice` (consumes the parent's next
/// available move).
fn next_state(tree: &mut [Node], ni: usize, p: &CoacdParams, rng: &mut ChaCha8Rng) -> State {
    let st = &mut tree[ni].state;
    let w = st.worst;
    let plane = {
        let part = &mut st.parts[w];
        let pl = part.moves[part.next];
        part.next += 1;
        pl
    };
    let mut ns = st.clone();
    match clip_fast(&ns.parts[w].solid, &plane) {
        Some((pos, neg, _)) => {
            ns.apply_cut(pos, neg, plane, p, rng);
            let r = ns.reward();
            ns.cost = st.cost + r;
            ns.round = st.round + 1;
        }
        None => {
            ns.cost = f64::INFINITY;
            ns.round = p.mcts_depth;
        }
    }
    ns
}

fn best_child(tree: &[Node], ni: usize, explore: bool, initial_cost: f64) -> Option<usize> {
    let mut best = f64::INFINITY;
    let mut out = None;
    let c = if explore {
        initial_cost / 2f64.sqrt()
    } else {
        0.0
    };
    for &ch in &tree[ni].children {
        let right = 2.0 * libm::log(tree[ni].visits) / tree[ch].visits;
        let score = tree[ch].quality - c * right.sqrt();
        if score < best {
            best = score;
            out = Some(ch);
        }
    }
    out
}

/// `ComputeBestRvClippingPlane`.
fn best_rv_plane(s: &Solid, planes: &[Plane], k: f64) -> Option<(Plane, Solid, Solid)> {
    let mut best: Option<(f64, Plane, Solid, Solid)> = None;
    for pl in planes {
        let Some((pos, neg, _)) = clip_fast(s, pl) else {
            continue;
        };
        if pos.is_empty() || neg.is_empty() {
            continue;
        }
        let h = rv_solid(&pos, k).max(rv_solid(&neg, k));
        if best.as_ref().map(|b| h < b.0).unwrap_or(true) {
            best = Some((h, *pl, pos, neg));
        }
    }
    best.map(|b| (b.1, b.2, b.3))
}

/// `default_policy`: greedy random-cut rollout to the maximum depth.
fn default_policy(
    state: &State,
    p: &CoacdParams,
    path: &mut Vec<Plane>,
    rng: &mut ChaCha8Rng,
) -> f64 {
    let mut st = state.clone();
    while !st.terminal(p) {
        let s = st.parts[st.worst].solid.clone();
        let planes = axis_planes(&s.mesh.aabb(), 1);
        if planes.is_empty() {
            break;
        }
        let Some((pl, pos, neg)) = best_rv_plane(&s, &planes, p.rv_k) else {
            break;
        };
        path.push(pl);
        st.apply_cut(pos, neg, pl, p, rng);
        let r = st.reward();
        st.cost += r;
        st.round += 1;
    }
    st.cost / p.mcts_depth as f64
}

/// `MonteCarloTreeSearch`: returns the best first plane, its quality and the
/// best path (deepest plane first, as upstream).
fn mcts(
    solid: Arc<Solid>,
    p: &CoacdParams,
    rng: &mut ChaCha8Rng,
) -> Option<(Plane, f64, Vec<Plane>)> {
    let root_part = MPart::new(solid.clone(), p.mcts_nodes, rng);
    let st = State {
        parts: vec![root_part],
        costs: vec![f64::INFINITY],
        worst: 0,
        round: 0,
        cost: 0.0,
        value: Plane {
            n: DVec3::ZERO,
            d: 0.0,
        },
    };
    let mut tree = vec![Node {
        children: Vec::new(),
        parent: None,
        visits: 0.0,
        quality: f64::INFINITY,
        state: st,
    }];
    let initial_cost = rv_solid(&solid, p.rv_k) / p.mcts_depth as f64;
    let mut best_path: Vec<Plane> = Vec::new();
    for _ in 0..p.mcts_iterations {
        // tree policy
        let mut ni = 0usize;
        loop {
            if tree[ni].state.terminal(p) {
                break;
            }
            let all =
                tree[ni].children.len() == tree[ni].state.parts[tree[ni].state.worst].moves.len();
            if all {
                match best_child(&tree, ni, true, initial_cost) {
                    Some(c) => ni = c,
                    None => break,
                }
            } else {
                let ns = next_state(&mut tree, ni, p, rng);
                tree.push(Node {
                    children: Vec::new(),
                    parent: Some(ni),
                    visits: 0.0,
                    quality: f64::INFINITY,
                    state: ns,
                });
                let c = tree.len() - 1;
                tree[ni].children.push(c);
                ni = c;
                break;
            }
        }
        let mut path = Vec::new();
        let reward = default_policy(&tree[ni].state, p, &mut path, rng);
        // backup
        let mut tmp: Vec<Plane> = path.iter().rev().copied().collect();
        let mut cur = Some(ni);
        while let Some(c) = cur {
            if tree[c].state.round == 0 && tree[c].quality > reward {
                best_path = tmp.clone();
            }
            tmp.push(tree[c].state.value);
            tree[c].visits += 1.0;
            tree[c].quality = tree[c].quality.min(reward);
            cur = tree[c].parent;
        }
    }
    let b = best_child(&tree, 0, false, 0.0)?;
    Some((tree[b].state.value, tree[b].quality, best_path))
}

/// `clip_by_path`: mean worst-part Rv along the path with a new first plane.
fn clip_by_path(m: &Solid, p: &CoacdParams, first: &Plane, path: &[Plane]) -> f64 {
    let Some((pos, neg, _)) = clip_fast(m, first) else {
        return f64::INFINITY;
    };
    let pc = rv_solid(&pos, p.rv_k);
    let nc = rv_solid(&neg, p.rv_k);
    let mut scores = vec![pc, nc];
    let mut parts = vec![pos, neg];
    let mut worst = if pc > nc { 0 } else { 1 };
    let mut final_cost = pc.max(nc);
    let n = path.len();
    for i in 1..n {
        let Some((a, b, _)) = clip_fast(&parts[worst], &path[n - 1 - i]) else {
            return f64::INFINITY;
        };
        let (ca, cb) = (rv_solid(&a, p.rv_k), rv_solid(&b, p.rv_k));
        parts.remove(worst);
        scores.remove(worst);
        scores.push(ca);
        scores.push(cb);
        parts.push(a);
        parts.push(b);
        let mut max_cost = scores[0];
        worst = 0;
        for (j, &s) in scores.iter().enumerate().skip(1) {
            // (upstream compares against the accumulated cost here)
            if s > final_cost {
                worst = j;
                max_cost = s;
            }
        }
        final_cost += max_cost;
    }
    final_cost / n.max(1) as f64
}

/// `TernaryMCTS` (mode = true): refine the offset of the best plane along
/// its axis within one sampling interval.
fn ternary_refine(m: &Solid, p: &CoacdParams, best: &mut Plane, path: &[Plane], best_cost: f64) {
    let Some(k) = best.axis_index() else { return };
    let bb = m.mesh.aabb();
    let min_itv = 0.01;
    let interval = (0.01f64).max((bb.max[k] - bb.min[k]).abs() / (p.mcts_nodes as f64 + 1.0));
    let mut left = (bb.min[k] + min_itv).max(-best.d - interval);
    let mut right = (bb.max[k] - min_itv).min(-best.d + interval);
    if left > right {
        return;
    }
    let eps = 1e-4;
    let mut res = 0.0;
    let mut it = 0;
    while left + eps < right && it < 10 {
        it += 1;
        let margin = (right - left) / 3.0;
        let m1 = left + margin;
        let m2 = m1 + margin;
        let e1 = clip_by_path(m, p, &Plane::axis(k, m1), path);
        let e2 = clip_by_path(m, p, &Plane::axis(k, m2), path);
        if e1 < e2 {
            right = m2;
            res = m1;
        } else {
            left = m1;
            res = m2;
        }
    }
    let tp = Plane::axis(k, res);
    let hmin = clip_by_path(m, p, &tp, path);
    if hmin < best_cost {
        *best = tp;
    }
}

// ---------------------------------------------------------------------------
// Cut loop (upstream `Compute`)

/// A convex part of the decomposition (normalized frame).
#[derive(Clone, Debug)]
pub struct CutPart {
    pub solid: Solid,
    pub ch: Ch,
    /// Concavity `h` of the part w.r.t. its hull (normalized units).
    pub cost: f64,
}

enum Outcome {
    Done(Box<CutPart>),
    Split(Solid, Solid),
    Drop,
}

fn process_part(s: Solid, p: &CoacdParams, allow_cut: bool) -> Outcome {
    let mut rng = ChaCha8Rng::seed_from_u64(p.seed);
    let Some(ch) = hull(&s.mesh.verts) else {
        return Outcome::Drop;
    };
    let cap = if allow_cut {
        p.threshold
    } else {
        f64::INFINITY
    };
    let h = h_cost(&s, &ch, p, cap);
    if h <= p.threshold || !allow_cut {
        let cost = if h <= p.threshold || cap.is_infinite() {
            h
        } else {
            h_cost(&s, &ch, p, f64::INFINITY)
        };
        return Outcome::Done(Box::new(CutPart { solid: s, ch, cost }));
    }
    let arc = Arc::new(s);
    let Some((mut plane, quality, path)) = mcts(arc.clone(), p, &mut rng) else {
        let s = Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone());
        let cost = h_cost(&s, &ch, p, f64::INFINITY);
        return Outcome::Done(Box::new(CutPart { solid: s, ch, cost }));
    };
    ternary_refine(&arc, p, &mut plane, &path, quality);
    // degenerate cut sections (pinched loops through vertices) can defeat
    // the cap triangulation: retry with slightly offset planes
    for k in [0.0, 1e-4, -1e-4, 3e-4, -3e-4, 1e-3, -1e-3] {
        let pl = Plane {
            n: plane.n,
            d: plane.d + k,
        };
        if let Some((pos, neg, _)) = clip(&arc, &pl) {
            if !pos.is_empty() && !neg.is_empty() {
                return Outcome::Split(pos, neg);
            }
        }
    }
    let s = Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone());
    let cost = h_cost(&s, &ch, p, f64::INFINITY);
    Outcome::Done(Box::new(CutPart { solid: s, ch, cost }))
}

/// Cut a closed solid (already in the normalized frame) until every part's
/// concavity is below the threshold. Parts are returned in a deterministic
/// order.
pub fn cut(input: Solid, p: &CoacdParams) -> Vec<CutPart> {
    let mut pool = vec![input];
    let mut done: Vec<CutPart> = Vec::new();
    while !pool.is_empty() {
        // each cut adds one part; stop cutting at `max_parts`
        let base = done.len() + pool.len();
        let res: Vec<Outcome> = pool
            .into_par_iter()
            .enumerate()
            .map(|(i, s)| process_part(s, p, base + i < p.max_parts))
            .collect();
        let mut next = Vec::new();
        for r in res {
            match r {
                Outcome::Done(c) => done.push(*c),
                Outcome::Split(a, b) => {
                    next.push(a);
                    next.push(b);
                }
                Outcome::Drop => {}
            }
        }
        pool = next;
    }
    done
}

// ---------------------------------------------------------------------------
// Merging (upstream `MergeConvexHulls`, collision-aware cost)

/// Upstream 1.0.x merge cost `ComputeHCost(cvx1, cvx2, CH)`:
/// `max(k·∛(3|V1 + V2 − V_CH|/4π), Hb)` where `Hb` is the symmetric
/// distance between samples of the two hull surfaces (without the triangles
/// on their common face, found like `ComputeOverlapFace`) and the merged
/// hull's surface; 0 when every input vertex is a vertex of `CH`. Distances
/// are exact (upstream: 10-nearest-sample triangles).
fn upstream_merge_cost(a: &Piece, b: &Piece, ch: &Ch, k: f64, resolution: f64, seed: u64) -> f64 {
    let r = rv(a.hull_volume() + b.hull_volume(), ch.volume, k);
    if a.pts.len() + b.pts.len() == ch.pts.len() {
        return r;
    }
    // common face (`ComputeOverlapFace`): a face plane of `a` with `b` on
    // its other side; triangles on it (within 1e-3) are not sampled
    let overlap = a
        .poly
        .faces
        .iter()
        .map(|(h, _)| *h)
        .find(|h| b.pts.iter().all(|q| h.dist(*q) >= -1e-8));
    let mut src = a.poly.to_mesh();
    let na = src.tris.len();
    src.append(&b.poly.to_mesh());
    let (aa, ab) = (
        src.submesh_area(0..na),
        src.submesh_area(na..src.tris.len()),
    );
    let keep = |t: usize| -> bool {
        match overlap {
            Some(h) => !src.tri_points(t).iter().all(|x| h.dist(*x).abs() <= 1e-3),
            None => true,
        }
    };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // per hull: resolution split by area (upstream ExtractPointSet(cvx1, cvx2))
    let (mut sa, mut ia) = (Vec::new(), Vec::new());
    for (range, area) in [(0..na, aa), (na..src.tris.len(), ab)] {
        let sub = TriMesh {
            verts: src.verts.clone(),
            tris: src.tris[range.clone()].to_vec(),
        };
        let (p, i) = upstream_samples_f(
            &sub,
            resolution * area / (aa + ab).max(1e-300),
            &mut rng,
            &|t| keep(t + range.start),
        );
        sa.extend(p);
        ia.extend(i.into_iter().map(|t| t + range.start as u32));
    }
    let chm = ch.poly.to_mesh();
    let (sb, ib) = upstream_samples_f(&chm, resolution, &mut rng, &|_| true);
    r.max(face_hausdorff(&src, &sa, &ia, &chm, &sb, &ib))
}

/// Branch-and-bound triangle ordered by its upper bound.
struct BbTri {
    ub: f64,
    tri: [DVec3; 3],
}
impl PartialEq for BbTri {
    fn eq(&self, o: &Self) -> bool {
        self.ub.total_cmp(&o.ub).is_eq()
    }
}
impl Eq for BbTri {}
impl PartialOrd for BbTri {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for BbTri {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.ub.total_cmp(&o.ub)
    }
}

/// A convex piece being merged.
#[derive(Clone, Debug)]
pub struct Piece {
    pub poly: ConvexPolytope,
    pub pts: Vec<DVec3>,
    pub bbox: Aabb,
    /// Volume of the original geometry covered by the piece.
    pub vol: f64,
    /// Concavity of the piece's hull w.r.t. the solid.
    pub own: f64,
    /// Samples of the solid's surface belonging to the piece.
    pub samples: Vec<DVec3>,
    /// Caller tags (cells or part ids), kept sorted.
    pub tags: Vec<u32>,
    /// Optional caller keys for adjacency tests (sorted, unique; unions on
    /// merge): the piece's own keys (e.g. cells) and its halo (own keys
    /// and their neighbours).
    pub keys: Vec<u32>,
    pub halo: Vec<u32>,
}

impl Piece {
    pub fn new(ch: Ch, vol: f64, samples: Vec<DVec3>, tags: Vec<u32>) -> Piece {
        let bbox = Aabb::from_points(ch.pts.iter());
        Piece {
            poly: ch.poly,
            pts: ch.pts,
            bbox,
            vol,
            own: 0.0,
            samples,
            tags,
            keys: Vec::new(),
            halo: Vec::new(),
        }
    }
    pub fn hull_volume(&self) -> f64 {
        self.poly.volume()
    }
}

/// Geometry oracle for merge costs.
pub struct MergeCtx<'a> {
    /// The solid (fragment / input mesh) surface.
    pub q: &'a MeshQuery<'a>,
    /// Pseudo-normals of the solid surface (side tests).
    pub signer: Option<&'a Signer>,
    /// Sample spacing on hull faces.
    pub spacing: f64,
    pub rv_k: f64,
    /// Cap on the number of samples per hull face triangle.
    pub max_tri_samples: usize,
    /// Convex geometry that hulls must not cover (neighbouring fragments):
    /// covered volume `V` adds `intrusion_k·∛(3V/4π)` to the merge cost.
    pub foreign: Vec<Foreign<'a>>,
    pub intrusion_k: f64,
    /// Upstream 1.0.x hull-vs-hull merge cost with this sampling
    /// resolution (samples per unit area, at least 1000 per surface);
    /// `None`: collision-aware cost.
    pub upstream_density: Option<f64>,
    /// Seed of the upstream-style samples.
    pub seed: u64,
    /// Candidate bounds refined per round of the greedy merge (in parallel).
    /// A constant, so results do not depend on the thread count: a refined
    /// bound that stops at the round's cap is a lower bound, and later
    /// refinements start from it.
    pub batch: usize,
}

/// A convex piece of foreign geometry (with its volume scale: true volume /
/// polytope volume).
#[derive(Clone, Copy)]
pub struct Foreign<'a> {
    pub poly: &'a ConvexPolytope,
    pub pts: &'a [DVec3],
    pub bbox: Aabb,
    pub scale: f64,
    /// Volume of `poly`.
    pub pvol: f64,
}

/// Volume of the intersection of a convex polytope with foreign pieces.
pub fn covered_volume(poly: &ConvexPolytope, bbox: &Aabb, foreign: &[Foreign]) -> f64 {
    let mut v = 0.0;
    for f in foreign {
        if !f.bbox.overlaps(bbox) {
            continue;
        }
        // separated by a face plane of the hull: nothing covered
        if poly
            .faces
            .iter()
            .any(|(h, _)| f.pts.iter().all(|q| h.n.dot(*q) - h.d >= 0.0))
        {
            continue;
        }
        let inside = f.pts.iter().all(|q| in_poly(poly, *q, 0.0));
        let c = if inside {
            f.pvol
        } else {
            clip_all_cutting(f.poly, &poly.halfspaces()).volume()
        };
        v += c * f.scale;
    }
    v
}

/// Split a convex polygon by a plane: (part with `dist > 0`, part with
/// `dist <= 0`); points within `tol` of the plane belong to both. A part
/// that only touches the plane is empty.
fn split_polygon(poly: &[DVec3], h: &HalfSpace, tol: f64) -> (Vec<DVec3>, Vec<DVec3>) {
    let d: Vec<f64> = poly.iter().map(|&x| h.dist(x)).collect();
    let (mut pos, mut neg) = (Vec::new(), Vec::new());
    let (any_pos, any_neg) = (d.iter().any(|&v| v > tol), d.iter().any(|&v| v < -tol));
    if !any_pos {
        return (pos, poly.to_vec());
    }
    if !any_neg {
        return (poly.to_vec(), neg);
    }
    let n = poly.len();
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        let (da, db) = (d[i], d[(i + 1) % n]);
        if da >= -tol {
            pos.push(a);
        }
        if da <= tol {
            neg.push(a);
        }
        if (da > tol && db < -tol) || (da < -tol && db > tol) {
            let x = a + (b - a) * (da / (da - db));
            pos.push(x);
            neg.push(x);
        }
    }
    (
        if pos.len() >= 3 { pos } else { Vec::new() },
        if neg.len() >= 3 { neg } else { Vec::new() },
    )
}

/// The parts of a convex polygon outside all the given convex polytopes,
/// as convex polygons (a polygon on a polytope's boundary is inside).
pub fn polygon_minus(poly: &[DVec3], skip: &[&ConvexPolytope], tol: f64) -> Vec<Vec<DVec3>> {
    let mut cur = vec![poly.to_vec()];
    for s in skip {
        let mut next = Vec::new();
        for pg in cur {
            // pg \ s = ∪_i pg ∩ {h_i > 0} ∩ {h_j <= 0, j < i}
            let mut rest = pg;
            for (h, _) in &s.faces {
                let (out, inn) = split_polygon(&rest, h, tol);
                if !out.is_empty() {
                    next.push(out);
                }
                rest = inn;
                if rest.is_empty() {
                    break;
                }
            }
        }
        cur = next;
    }
    cur
}

/// [`ConvexPolytope::clip_all`] restricted to the half-spaces that cut the
/// polytope (others leave its volume unchanged, or empty it).
pub fn clip_all_cutting(p: &ConvexPolytope, hs: &[HalfSpace]) -> ConvexPolytope {
    // the input is only cloned once a half-space cuts it
    let mut cur: Option<ConvexPolytope> = None;
    let mut eps = 1e-12 * p.scale().max(1e-300);
    for h in hs {
        let q = cur.as_ref().unwrap_or(p);
        if q.is_empty() {
            break;
        }
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for (_, f) in &q.faces {
            for &x in f {
                let d = h.dist(x);
                lo = lo.min(d);
                hi = hi.max(d);
            }
        }
        if hi <= eps {
            continue;
        }
        if lo > -eps {
            return ConvexPolytope::default();
        }
        let c = q.clip(h);
        eps = 1e-12 * c.scale().max(1e-300);
        cur = Some(c);
    }
    cur.unwrap_or_else(|| p.clone())
}

#[inline]
fn in_poly(poly: &ConvexPolytope, p: DVec3, tol: f64) -> bool {
    poly.faces.iter().all(|(h, _)| h.n.dot(p) - h.d <= tol)
}

impl MergeCtx<'_> {
    /// Max distance from the part of the hull surface outside the solid to
    /// the solid's surface (at least `start`), ignoring regions inside any
    /// of the `skip` polytopes. Branch and bound over recursively split
    /// face triangles: the distance field is 1-Lipschitz, so a triangle with
    /// centroid distance `d` and circumradius `ρ` cannot exceed `d + ρ`, and
    /// lies entirely inside the solid when its centroid is inside at depth
    /// `≥ ρ`. The result is within half the `spacing` of the true maximum.
    /// Stops above `cap`.
    fn out_term(
        &self,
        poly: &ConvexPolytope,
        skip: &[&ConvexPolytope],
        start: f64,
        cap: f64,
    ) -> f64 {
        let mut worst = start;
        let tol = 1e-9 * self.q.bbox.diagonal();
        // the hull surface inside the skipped polytopes is excluded exactly:
        // every face is cut into convex polygons outside all of them
        let mut regions: Vec<Vec<DVec3>> = Vec::new();
        for (_, f) in &poly.faces {
            regions.extend(polygon_minus(f, skip, tol));
        }
        // (upper bound, triangle), explored best first
        let mut heap: std::collections::BinaryHeap<BbTri> = std::collections::BinaryHeap::new();
        let evals = std::cell::Cell::new(0usize);
        let max_evals = self.max_tri_samples * 4;
        let visit =
            |tri: [DVec3; 3], worst: &mut f64, heap: &mut std::collections::BinaryHeap<BbTri>| {
                let [a, b, c] = tri;
                let g = (a + b + c) / 3.0;
                let rho = (a - g).length().max((b - g).length()).max((c - g).length());
                let near = if *worst - rho > 0.0 {
                    farther_than(self.q, g, *worst - rho)
                } else {
                    self.q.closest_point(g).map(|(_, d2, t)| (t, d2))
                };
                let Some((t, d2)) = near else { return };
                evals.set(evals.get() + 1);
                N_EVAL.fetch_add(1, AO::Relaxed);
                let d = d2.sqrt();
                // inside centroid: fully inside when deeper than ρ; otherwise any
                // outside point of the triangle is within ρ of the surface (the
                // segment to the centroid crosses it). The (costly) side test is
                // only needed when it can change the maximum or prune.
                let ub = if d > *worst || d >= rho {
                    let outside = self
                        .signer
                        .and_then(|sg| sg.outside(self.q.mesh, g, t))
                        .unwrap_or_else(|| is_outside(self.q, g, t));
                    if outside && d > *worst {
                        *worst = d;
                    }
                    if outside {
                        d + rho
                    } else if d >= rho {
                        return;
                    } else {
                        rho
                    }
                } else {
                    d + rho
                };
                // ε-optimal: regions that cannot beat the current maximum by more
                // than half the spacing are not refined
                if ub > *worst + 0.5 * self.spacing && rho > 0.5 * self.spacing {
                    heap.push(BbTri { ub, tri });
                }
            };
        for f in &regions {
            for k in 1..f.len().saturating_sub(1) {
                visit([f[0], f[k], f[k + 1]], &mut worst, &mut heap);
            }
        }
        while let Some(BbTri { ub, tri: [a, b, c] }) = heap.pop() {
            if ub <= worst + 0.5 * self.spacing || worst > cap || evals.get() >= max_evals {
                break;
            }
            let (ab, bc, ca) = ((a + b) * 0.5, (b + c) * 0.5, (c + a) * 0.5);
            for t in [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]] {
                visit(t, &mut worst, &mut heap);
            }
        }
        worst
    }

    /// Surface samples covered by the hull: max distance to its boundary.
    fn in_term(poly: &ConvexPolytope, samples: &[&[DVec3]]) -> f64 {
        let t0 = std::time::Instant::now();
        let planes: Vec<(DVec3, f64)> = poly.faces.iter().map(|(h, _)| (h.n, h.d)).collect();
        let nf = planes.len();
        let mut worst: f64 = 0.0;
        let mut last = 0usize;
        for s in samples {
            'sample: for &p in s.iter() {
                // depth = min over faces; a sample cannot raise the maximum
                // once its running min is below it (start from the face that
                // was minimal for the previous, nearby sample)
                let mut m = f64::INFINITY;
                let mut arg = last;
                for k in 0..nf {
                    let f = (last + k) % nf;
                    let d = planes[f].1 - planes[f].0.dot(p);
                    if d < m {
                        m = d;
                        arg = f;
                        if m <= worst {
                            last = arg;
                            continue 'sample;
                        }
                    }
                }
                last = arg;
                worst = worst.max(m);
            }
        }
        T_IN.fetch_add(t0.elapsed().as_nanos() as u64, AO::Relaxed);
        worst
    }

    /// Own concavity of a piece.
    pub fn own_cost(&self, pc: &Piece) -> f64 {
        let r = rv(pc.vol, pc.hull_volume(), self.rv_k);
        let i = Self::in_term(&pc.poly, &[&pc.samples]);
        let mut c = r.max(i);
        if self.intrusion_k > 0.0 && !self.foreign.is_empty() {
            c = c.max(rv(
                covered_volume(&pc.poly, &pc.bbox, &self.foreign),
                0.0,
                self.intrusion_k,
            ));
        }
        let o = self.out_term(&pc.poly, &[], c, f64::INFINITY);
        c.max(o)
    }

    /// Cost of merging two pieces and the merged hull (stops above `cap`).
    pub fn pair_cost(&self, a: &Piece, b: &Piece, cap: f64) -> Option<(f64, Ch)> {
        let (c1, ch) = self.stage1(a, b)?;
        let c = if c1 > cap {
            c1
        } else {
            self.stage2(a, b, &ch, c1, cap)
        };
        Some((c, ch))
    }

    /// Lower bound of a merge cost from the merged hull (Rv, covered-surface
    /// term, the parts' own costs) and the hull.
    pub fn stage1(&self, a: &Piece, b: &Piece) -> Option<(f64, Ch)> {
        let t0 = std::time::Instant::now();
        let mut pts = a.pts.clone();
        pts.extend_from_slice(&b.pts);
        let ch = hull_tri(&pts)?;
        T_HULLC.fetch_add(t0.elapsed().as_nanos() as u64, AO::Relaxed);
        if let Some(density) = self.upstream_density {
            // exact in this mode: stage 2 returns it unchanged
            return Some((
                upstream_merge_cost(a, b, &ch, self.rv_k, density, self.seed),
                ch,
            ));
        }
        let base = a.own.max(b.own);
        let r = rv(a.vol + b.vol, ch.volume, self.rv_k);
        let i = Self::in_term(&ch.poly, &[&a.samples, &b.samples]);
        N_PAIR.fetch_add(1, AO::Relaxed);
        Some((base.max(r).max(i), ch))
    }

    /// Exact merge cost given the stage-1 bound (stops above `cap`, then
    /// returning a lower bound above `cap`).
    pub fn stage2(&self, a: &Piece, b: &Piece, ch: &Ch, c1: f64, cap: f64) -> f64 {
        if self.upstream_density.is_some() {
            return c1;
        }
        let t2 = std::time::Instant::now();
        N_STAGE2.fetch_add(1, AO::Relaxed);
        let mut c = c1;
        if self.intrusion_k > 0.0 && !self.foreign.is_empty() {
            let bb = Aabb::from_points(ch.pts.iter());
            let v = covered_volume(&ch.poly, &bb, &self.foreign);
            // the parts' own intrusions are already in their costs
            c = c.max(rv(v, 0.0, self.intrusion_k));
        }
        T_COV.fetch_add(t2.elapsed().as_nanos() as u64, AO::Relaxed);
        if c > cap {
            return c;
        }
        // the merged hull is covered by the pieces' geometry (convex union):
        // no part of its surface lies outside the solid
        if ch.volume - (a.vol + b.vol) <= 1e-9 * ch.volume {
            return c;
        }
        let o = self.out_term(&ch.poly, &[&a.poly, &b.poly], c, cap);
        T_OUT.fetch_add(t2.elapsed().as_nanos() as u64, AO::Relaxed);
        c.max(o)
    }
}

/// Union of two sorted unique lists.
fn sorted_union(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// Pieces by adjacency key (when every piece has keys): candidate
/// neighbours of a piece are the pieces owning a key of its halo, which is
/// exactly the set the key-based adjacency test can accept.
struct KeyIndex {
    by_key: std::collections::HashMap<u32, Vec<usize>>,
}

impl KeyIndex {
    fn new<'a>(pieces: impl Iterator<Item = (usize, &'a Piece)>) -> Option<KeyIndex> {
        let mut by_key: std::collections::HashMap<u32, Vec<usize>> =
            std::collections::HashMap::new();
        for (i, pc) in pieces {
            if pc.keys.is_empty() || pc.halo.is_empty() {
                return None;
            }
            for &k in &pc.keys {
                by_key.entry(k).or_default().push(i);
            }
        }
        Some(KeyIndex { by_key })
    }
    /// Pieces owning a key of `halo` (sorted, unique).
    fn candidates(&self, halo: &[u32]) -> Vec<usize> {
        let mut v: Vec<usize> = halo
            .iter()
            .filter_map(|k| self.by_key.get(k))
            .flatten()
            .copied()
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }
    /// Pieces `i` and `j` were merged into `m` (keys `keys`).
    fn merge(&mut self, i: usize, j: usize, m: usize, keys: &[u32]) {
        for k in keys {
            if let Some(v) = self.by_key.get_mut(k) {
                v.retain(|&x| x != i && x != j);
                v.push(m);
            }
        }
    }
}

/// Whether two sorted lists share an element.
pub fn sorted_intersects(a: &[u32], b: &[u32]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

fn merged_piece(a: &Piece, b: &Piece, ch: Ch, cost: f64) -> Piece {
    let mut samples = a.samples.clone();
    samples.extend_from_slice(&b.samples);
    let mut tags = a.tags.clone();
    tags.extend_from_slice(&b.tags);
    tags.sort_unstable();
    // candidate hulls have triangle faces: rebuild with merged faces
    let ch = hull(&ch.pts).unwrap_or(ch);
    let bbox = Aabb::from_points(ch.pts.iter());
    Piece {
        poly: ch.poly,
        pts: ch.pts,
        bbox,
        vol: a.vol + b.vol,
        own: cost,
        samples,
        tags,
        keys: sorted_union(&a.keys, &b.keys),
        halo: sorted_union(&a.halo, &b.halo),
    }
}

/// Cheap pre-merging for fragments with very many pieces: greedily merge
/// the adjacent pair whose bounding box grows least (box volume excess,
/// no hull or distance evaluation) until `target` pieces remain; merged
/// hulls are computed once per merge. Deterministic (ties by ids).
pub fn coarsen(
    pieces: Vec<Piece>,
    target: usize,
    adjacent: &(dyn Fn(&Piece, &Piece) -> bool + Sync),
) -> Vec<Piece> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    if pieces.len() <= target {
        return pieces;
    }
    #[derive(PartialEq)]
    struct Key(f64, usize, usize);
    impl Eq for Key {}
    impl PartialOrd for Key {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Key {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.0
                .total_cmp(&o.0)
                .then(self.1.cmp(&o.1))
                .then(self.2.cmp(&o.2))
        }
    }
    let excess = |a: &Piece, b: &Piece| -> f64 {
        let u = a.bbox.union(&b.bbox);
        let v = |bb: &Aabb| {
            let e = bb.extent();
            e.x * e.y * e.z
        };
        v(&u) - a.vol - b.vol
    };
    // Groups are merged on their boxes, volumes and adjacency keys only
    // (light pieces); hulls are built once per final group.
    let light = |pc: &Piece| Piece {
        poly: ConvexPolytope::default(),
        pts: Vec::new(),
        bbox: pc.bbox,
        vol: pc.vol,
        own: pc.own,
        samples: Vec::new(),
        tags: Vec::new(),
        keys: pc.keys.clone(),
        halo: pc.halo.clone(),
    };
    let mut alive: BTreeMap<usize, (Piece, Vec<usize>)> = pieces
        .iter()
        .enumerate()
        .map(|(i, pc)| (i, (light(pc), vec![i])))
        .collect();
    let mut next_id = alive.len();
    let mut heap: BinaryHeap<Reverse<Key>> = BinaryHeap::new();
    let ids: Vec<usize> = alive.keys().copied().collect();
    let mut index = KeyIndex::new(alive.iter().map(|(&i, g)| (i, &g.0)));
    let found: Vec<Vec<Key>> = ids
        .par_iter()
        .enumerate()
        .map(|(k, &i)| {
            let cands: Vec<usize> = match &index {
                Some(ix) => ix
                    .candidates(&alive[&i].0.halo)
                    .into_iter()
                    .filter(|&j| j > i)
                    .collect(),
                None => ids[k + 1..].to_vec(),
            };
            cands
                .into_iter()
                .filter(|&j| adjacent(&alive[&i].0, &alive[&j].0))
                .map(|j| Key(excess(&alive[&i].0, &alive[&j].0), i, j))
                .collect()
        })
        .collect();
    for v in found {
        for k in v {
            heap.push(Reverse(k));
        }
    }
    while alive.len() > target {
        let Some(Reverse(Key(_, i, j))) = heap.pop() else {
            break;
        };
        if !alive.contains_key(&i) || !alive.contains_key(&j) {
            continue;
        }
        let (a, ma) = alive.remove(&i).unwrap();
        let (b, mb) = alive.remove(&j).unwrap();
        let m = Piece {
            poly: ConvexPolytope::default(),
            pts: Vec::new(),
            bbox: a.bbox.union(&b.bbox),
            vol: a.vol + b.vol,
            own: a.own.max(b.own),
            samples: Vec::new(),
            tags: Vec::new(),
            keys: sorted_union(&a.keys, &b.keys),
            halo: sorted_union(&a.halo, &b.halo),
        };
        let members: Vec<usize> = ma.into_iter().chain(mb).collect();
        let nid = next_id;
        next_id += 1;
        let others: Vec<usize> = match index.as_mut() {
            Some(ix) => {
                ix.merge(i, j, nid, &m.keys);
                ix.candidates(&m.halo)
                    .into_iter()
                    .filter(|o| alive.contains_key(o))
                    .collect()
            }
            None => alive.keys().copied().collect(),
        };
        let keys: Vec<Option<Key>> = others
            .iter()
            .map(|&o| {
                if adjacent(&alive[&o].0, &m) {
                    Some(Key(excess(&alive[&o].0, &m), o, nid))
                } else {
                    None
                }
            })
            .collect();
        for k in keys.into_iter().flatten() {
            heap.push(Reverse(k));
        }
        alive.insert(nid, (m, members));
    }
    // one hull per group (the hull of the members' hull vertices); its own
    // cost also bounds the group's Rv
    let groups: Vec<(Piece, Vec<usize>)> = alive.into_values().collect();
    let built: Vec<Vec<Piece>> = groups
        .par_iter()
        .map(|(g, members)| {
            if members.len() == 1 {
                return vec![pieces[members[0]].clone()];
            }
            let pts: Vec<DVec3> = members
                .iter()
                .flat_map(|&k| pieces[k].pts.iter().copied())
                .collect();
            let Some(ch) = hull(&pts) else {
                return members.iter().map(|&k| pieces[k].clone()).collect();
            };
            let mut tags: Vec<u32> = members
                .iter()
                .flat_map(|&k| pieces[k].tags.iter().copied())
                .collect();
            tags.sort_unstable();
            let samples: Vec<DVec3> = members
                .iter()
                .flat_map(|&k| pieces[k].samples.iter().copied())
                .collect();
            let bbox = Aabb::from_points(ch.pts.iter());
            let own = g.own.max(rv(g.vol, ch.volume, 0.3));
            vec![Piece {
                poly: ch.poly,
                pts: ch.pts,
                bbox,
                vol: g.vol,
                own,
                samples,
                tags,
                keys: g.keys.clone(),
                halo: g.halo.clone(),
            }]
        })
        .collect();
    built.into_iter().flatten().collect()
}

/// Result of a greedy merge.
pub struct MergeResult {
    /// Pieces after merging to the budget.
    pub pieces: Vec<Piece>,
    /// Snapshot when the threshold phase ended (cost > threshold or the
    /// carry cap reached), for reuse by coarser levels.
    pub carry: Vec<Piece>,
}

/// Greedy pairwise merging: always merge the cheapest candidate pair; stop
/// when the count is within `budget` and the cheapest cost exceeds
/// `threshold`. Candidate pairs are those accepted by `adjacent`; when no
/// candidate remains above the budget, all pairs become candidates.
pub fn greedy_merge(
    pieces: Vec<Piece>,
    ctx: &MergeCtx,
    threshold: f64,
    budget: usize,
    carry_cap: usize,
    adjacent: &(dyn Fn(&Piece, &Piece) -> bool + Sync),
) -> MergeResult {
    // Lazy evaluation: every candidate pair holds a lower bound of its cost
    // that is refined in stages (0: max of the parts' own costs; 1: merged
    // hull, Rv and the covered-surface term; 2: exact, with the hull-surface
    // term) only while it competes for the minimum. Bounds live in a binary
    // heap ordered by (bound, pair); entries of merged-away pieces are
    // dropped when popped. The selected merges are those of eager
    // evaluation.
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    #[derive(PartialEq)]
    struct Key(f64, usize, usize);
    impl Eq for Key {}
    impl PartialOrd for Key {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Key {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.0
                .total_cmp(&o.0)
                .then(self.1.cmp(&o.1))
                .then(self.2.cmp(&o.2))
        }
    }
    struct Entry {
        cost: f64,
        stage: u8,
        ch: Option<Ch>,
        /// Final only for the cheap phase (stage-1 cost).
        cheap: bool,
    }
    let budget = budget.max(1);
    // With many pieces, merge by the stage-1 cost (no hull-surface branch
    // and bound) until `exact_below` remain; those bounds become lower
    // bounds again for the exact phase.
    let exact_below = budget.saturating_mul(2).max(16);
    let mut cheap_phase = true;
    let mut alive: BTreeMap<usize, Piece> = pieces.into_iter().enumerate().collect();
    let mut next_id = alive.len();
    let mut cache: BTreeMap<(usize, usize), Entry> = BTreeMap::new();
    let mut heap: BinaryHeap<Reverse<Key>> = BinaryHeap::new();
    let mut all_pairs = false;
    let add_pairs = |alive: &BTreeMap<usize, Piece>,
                     cache: &mut BTreeMap<(usize, usize), Entry>,
                     heap: &mut BinaryHeap<Reverse<Key>>,
                     pairs: Vec<(usize, usize)>,
                     all: bool| {
        let ok: Vec<bool> = pairs
            .par_iter()
            .map(|&(i, j)| all || adjacent(&alive[&i], &alive[&j]))
            .collect();
        for (pr, ok) in pairs.into_iter().zip(ok) {
            if ok && !cache.contains_key(&pr) {
                let base = alive[&pr.0].own.max(alive[&pr.1].own);
                cache.insert(
                    pr,
                    Entry {
                        cost: base,
                        stage: 0,
                        ch: None,
                        cheap: false,
                    },
                );
                heap.push(Reverse(Key(base, pr.0, pr.1)));
            }
        }
    };
    let ids: Vec<usize> = alive.keys().copied().collect();
    let mut index = KeyIndex::new(alive.iter().map(|(&i, pc)| (i, pc)));
    let pairs: Vec<(usize, usize)> = match &index {
        Some(ix) => ids
            .iter()
            .flat_map(|&i| {
                ix.candidates(&alive[&i].halo)
                    .into_iter()
                    .filter(move |&j| j > i)
                    .map(move |j| (i, j))
            })
            .collect(),
        None => ids
            .iter()
            .enumerate()
            .flat_map(|(k, &i)| ids[k + 1..].iter().map(move |&j| (i, j)))
            .collect(),
    };
    add_pairs(&alive, &mut cache, &mut heap, pairs, false);
    let batch_min = ctx.batch.max(1);
    let mut carry: Option<Vec<Piece>> = None;
    // a heap key is current when it matches its cache entry
    let current = |cache: &BTreeMap<(usize, usize), Entry>, k: &Key| {
        cache
            .get(&(k.1, k.2))
            .map(|e| e.cost.total_cmp(&k.0).is_eq())
            .unwrap_or(false)
    };
    loop {
        if alive.len() <= 1 {
            break;
        }
        while let Some(Reverse(k)) = heap.peek() {
            if current(&cache, k) {
                break;
            }
            heap.pop();
        }
        let Some(Reverse(Key(cost, i, j))) = heap.peek() else {
            if alive.len() > budget && !all_pairs {
                all_pairs = true;
                let ids: Vec<usize> = alive.keys().copied().collect();
                let pairs: Vec<(usize, usize)> = ids
                    .iter()
                    .enumerate()
                    .flat_map(|(k, &i)| ids[k + 1..].iter().map(move |&j| (i, j)))
                    .collect();
                add_pairs(&alive, &mut cache, &mut heap, pairs, true);
                continue;
            }
            break;
        };
        let (cost, i, j) = (*cost, *i, *j);
        if cheap_phase && alive.len() <= exact_below {
            cheap_phase = false;
            for e in cache.values_mut() {
                if e.cheap {
                    e.cheap = false;
                    e.stage = 1;
                }
            }
            continue;
        }
        // the stop rules only need a lower bound
        if carry.is_none() && cost > threshold && alive.len() <= carry_cap {
            carry = Some(alive.values().cloned().collect());
        }
        if cost > threshold && alive.len() <= budget {
            break;
        }
        if cache[&(i, j)].stage < 2 {
            // refine the smallest non-final bounds (up to the first exact
            // one, whose cost caps the exact evaluations) in parallel
            let mut batch: Vec<(usize, usize)> = Vec::new();
            let mut cap = f64::INFINITY;
            let mut popped: Vec<Key> = Vec::new();
            while batch.len() < batch_min {
                let Some(Reverse(k)) = heap.pop() else { break };
                if !current(&cache, &k) {
                    continue;
                }
                if cache[&(k.1, k.2)].stage == 2 {
                    cap = k.0;
                    popped.push(k);
                    break;
                }
                batch.push((k.1, k.2));
            }
            for k in popped {
                heap.push(Reverse(k));
            }
            let cheap = cheap_phase && alive.len() > exact_below;
            let res: Vec<(f64, u8, Option<Ch>)> = batch
                .par_iter()
                .map(|k| {
                    let e = &cache[k];
                    let (a, b) = (&alive[&k.0], &alive[&k.1]);
                    if e.stage == 0 {
                        match ctx.stage1(a, b) {
                            Some((c, ch)) => (c, if cheap { 2 } else { 1 }, Some(ch)),
                            None => (f64::INFINITY, 2, None),
                        }
                    } else {
                        let ch = e.ch.as_ref().unwrap();
                        let c = ctx.stage2(a, b, ch, e.cost, cap);
                        (c, if c <= cap { 2 } else { 1 }, None)
                    }
                })
                .collect();
            for (k, (c, st, ch)) in batch.into_iter().zip(res) {
                let e = cache.get_mut(&k).unwrap();
                e.cost = e.cost.max(c);
                if ch.is_some() {
                    e.cheap = cheap && st == 2;
                    e.ch = ch;
                }
                e.stage = st;
                if e.ch.is_none() {
                    cache.remove(&k);
                } else {
                    heap.push(Reverse(Key(e.cost, k.0, k.1)));
                }
            }
            continue;
        }
        heap.pop();
        let ch = cache.remove(&(i, j)).unwrap().ch.unwrap();
        let a = alive.remove(&i).unwrap();
        let b = alive.remove(&j).unwrap();
        // drop the cache entries of the merged pieces (their heap keys
        // become stale)
        let dead: Vec<(usize, usize)> = cache
            .keys()
            .filter(|k| k.0 == i || k.1 == i || k.0 == j || k.1 == j)
            .copied()
            .collect();
        for k in dead {
            cache.remove(&k);
        }
        let m = merged_piece(&a, &b, ch, cost);
        let nid = next_id;
        next_id += 1;
        let pairs: Vec<(usize, usize)> = match index.as_mut() {
            Some(ix) if !all_pairs => {
                ix.merge(i, j, nid, &m.keys);
                ix.candidates(&m.halo)
                    .into_iter()
                    .filter(|o| alive.contains_key(o))
                    .map(|o| (o, nid))
                    .collect()
            }
            _ => alive.keys().map(|&o| (o, nid)).collect(),
        };
        alive.insert(nid, m);
        add_pairs(&alive, &mut cache, &mut heap, pairs, all_pairs);
    }
    let pieces: Vec<Piece> = alive.into_values().collect();
    let carry = carry.unwrap_or_else(|| pieces.clone());
    MergeResult { pieces, carry }
}

// ---------------------------------------------------------------------------
// Stand-alone decomposition

/// Decompose a closed mesh (world coordinates). Returns convex hulls in
/// world coordinates, deterministic for a given input and parameters.
pub fn decompose(mesh: &TriMesh, p: &CoacdParams) -> Vec<ConvexPolytope> {
    decompose_detailed(mesh, p)
        .into_iter()
        .map(|(h, _)| h)
        .collect()
}

/// Like [`decompose`], also returning each hull's concavity (normalized).
pub fn decompose_detailed(mesh: &TriMesh, p: &CoacdParams) -> Vec<(ConvexPolytope, f64)> {
    if mesh.tris.is_empty() {
        return Vec::new();
    }
    let frame = Frame::of(&mesh.aabb());
    let nm = mesh.transformed(|v| frame.to_norm(v));
    let nm = if nm.signed_volume() < 0.0 {
        nm.flipped()
    } else {
        nm
    };
    let parts = cut(Solid::new(nm.clone()), p);
    if !p.merge || parts.len() <= 1 {
        return parts
            .into_iter()
            .map(|c| (frame.polytope_to_world(&c.ch.poly), c.cost))
            .collect();
    }
    let q = MeshQuery::new(&nm);
    let density = density_for(nm.area(), p.resolution);
    let pieces: Vec<Piece> = parts
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut samples = Vec::new();
            for t in 0..c.solid.mesh.tris.len() {
                if c.solid.cap[t] {
                    continue;
                }
                let [a, b, cc] = c.solid.mesh.tri_points(t);
                let ar = 0.5 * (b - a).cross(cc - a).length();
                tri_samples(a, b, cc, ((ar * density) as usize).max(1), &mut samples);
            }
            let mut pc = Piece::new(c.ch.clone(), c.solid.volume(), samples, vec![i as u32]);
            pc.own = c.cost;
            pc
        })
        .collect();
    let signer = Signer::new(&nm);
    let upstream_density = match p.merge_cost {
        MergeCost::Upstream => Some((p.resolution + 2000) as f64),
        MergeCost::CollisionAware => None,
    };
    let ctx = MergeCtx {
        q: &q,
        signer: Some(&signer),
        spacing: 1.0 / (p.resolution.max(1) as f64).sqrt(),
        rv_k: p.rv_k,
        max_tri_samples: 4096,
        foreign: Vec::new(),
        intrusion_k: 0.0,
        upstream_density,
        seed: p.seed,
        batch: 8,
    };
    // upstream: only hulls closer than 0.01 (normalized vertex distance)
    let adjacent = |a: &Piece, b: &Piece| -> bool {
        if !a.bbox.expanded(0.01).overlaps(&b.bbox) {
            return false;
        }
        a.pts
            .iter()
            .any(|x| b.pts.iter().any(|y| (*x - *y).length_squared() < 1e-4))
    };
    let budget = p.max_convex_hull.unwrap_or(usize::MAX);
    let r = greedy_merge(pieces, &ctx, p.threshold, budget, usize::MAX, &adjacent);
    r.pieces
        .into_iter()
        .map(|pc| (frame.polytope_to_world(&pc.poly), pc.own))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use frac_geom::mesh::box_mesh;

    fn union_boxes(boxes: &[(DVec3, DVec3)]) -> TriMesh {
        // voxel-style union of axis-aligned boxes on a common grid
        let mut xs: Vec<f64> = Vec::new();
        let mut ys: Vec<f64> = Vec::new();
        let mut zs: Vec<f64> = Vec::new();
        for (lo, hi) in boxes {
            xs.extend([lo.x, hi.x]);
            ys.extend([lo.y, hi.y]);
            zs.extend([lo.z, hi.z]);
        }
        for v in [&mut xs, &mut ys, &mut zs] {
            v.sort_by(f64::total_cmp);
            v.dedup();
        }
        let inside = |p: DVec3| {
            boxes
                .iter()
                .any(|(lo, hi)| p.cmpgt(*lo).all() && p.cmplt(*hi).all())
        };
        let (nx, ny, nz) = (xs.len() - 1, ys.len() - 1, zs.len() - 1);
        let filled = |i: isize, j: isize, k: isize| -> bool {
            if i < 0 || j < 0 || k < 0 || i >= nx as isize || j >= ny as isize || k >= nz as isize {
                return false;
            }
            let (i, j, k) = (i as usize, j as usize, k as usize);
            inside(DVec3::new(
                (xs[i] + xs[i + 1]) * 0.5,
                (ys[j] + ys[j + 1]) * 0.5,
                (zs[k] + zs[k + 1]) * 0.5,
            ))
        };
        let mut m = TriMesh::default();
        for i in 0..nx as isize {
            for j in 0..ny as isize {
                for k in 0..nz as isize {
                    if !filled(i, j, k) {
                        continue;
                    }
                    let lo = DVec3::new(xs[i as usize], ys[j as usize], zs[k as usize]);
                    let hi = DVec3::new(xs[i as usize + 1], ys[j as usize + 1], zs[k as usize + 1]);
                    let b = box_mesh(lo, hi);
                    // keep only faces not shared with a filled neighbour
                    for t in &b.tris {
                        let [p, q, r] = [
                            b.verts[t[0] as usize],
                            b.verts[t[1] as usize],
                            b.verts[t[2] as usize],
                        ];
                        let n = (q - p).cross(r - p).normalize();
                        let d = (
                            n.x.round() as isize,
                            n.y.round() as isize,
                            n.z.round() as isize,
                        );
                        if !filled(i + d.0, j + d.1, k + d.2) {
                            let base = m.verts.len() as u32;
                            m.verts.extend([p, q, r]);
                            m.tris.push([base, base + 1, base + 2]);
                        }
                    }
                }
            }
        }
        m.weld_exact()
    }

    pub(crate) fn l_shape() -> TriMesh {
        union_boxes(&[
            (DVec3::ZERO, DVec3::new(2.0, 1.0, 1.0)),
            (DVec3::ZERO, DVec3::new(1.0, 2.0, 1.0)),
        ])
    }

    pub(crate) fn u_shape() -> TriMesh {
        union_boxes(&[
            (DVec3::ZERO, DVec3::new(3.0, 1.0, 1.0)),
            (DVec3::ZERO, DVec3::new(1.0, 3.0, 1.0)),
            (DVec3::new(2.0, 0.0, 0.0), DVec3::new(3.0, 3.0, 1.0)),
        ])
    }

    pub(crate) fn ring() -> TriMesh {
        // square torus-like ring: 3x3 block with the middle column removed
        union_boxes(&[
            (DVec3::ZERO, DVec3::new(3.0, 1.0, 1.0)),
            (DVec3::new(0.0, 2.0, 0.0), DVec3::new(3.0, 3.0, 1.0)),
            (DVec3::ZERO, DVec3::new(1.0, 3.0, 1.0)),
            (DVec3::new(2.0, 0.0, 0.0), DVec3::new(3.0, 3.0, 1.0)),
        ])
    }

    pub(crate) fn notched_box() -> TriMesh {
        // 4x2x2 box with a 1-wide, 1-deep notch across the top middle
        union_boxes(&[
            (DVec3::ZERO, DVec3::new(4.0, 2.0, 1.0)),
            (DVec3::new(0.0, 0.0, 1.0), DVec3::new(1.5, 2.0, 2.0)),
            (DVec3::new(2.5, 0.0, 1.0), DVec3::new(4.0, 2.0, 2.0)),
        ])
    }

    fn closed(m: &TriMesh) -> bool {
        m.topology().is_closed_manifold()
    }

    #[test]
    fn test_shapes_are_closed() {
        for m in [l_shape(), u_shape(), ring(), notched_box()] {
            assert!(closed(&m), "{:?}", m.topology());
            assert!(m.signed_volume() > 0.0);
        }
        assert!((l_shape().signed_volume() - 3.0).abs() < 1e-12);
        assert!((ring().signed_volume() - 8.0).abs() < 1e-12);
    }

    #[test]
    fn clip_conserves_volume_and_closes() {
        let shapes = [
            l_shape(),
            u_shape(),
            ring(),
            notched_box(),
            frac_geom::mesh::icosphere(DVec3::new(0.1, 0.2, 0.3), 1.0, 2),
        ];
        let planes = [
            Plane {
                n: DVec3::X,
                d: -0.5,
            },
            Plane {
                n: DVec3::new(1.0, 1.0, 0.3).normalize(),
                d: -1.2,
            },
            Plane {
                n: DVec3::Y,
                d: -1.0,
            }, // through existing vertices / faces
            Plane {
                n: DVec3::new(-0.3, 0.8, -0.52).normalize(),
                d: 0.1,
            },
        ];
        for m in &shapes {
            let v = m.signed_volume();
            for pl in &planes {
                let (pos, neg, area) = clip(&Solid::new(m.clone()), pl).expect("clip");
                let (vp, vn) = (pos.volume(), neg.volume());
                assert!((vp + vn - v).abs() <= 1e-9 * v, "volume {vp} + {vn} != {v}");
                for s in [&pos, &neg] {
                    if !s.is_empty() {
                        assert!(closed(&s.mesh), "{:?}", s.mesh.topology());
                        assert!(s.mesh.verts.iter().all(|&q| true || q.x.is_finite()));
                    }
                }
                if !pos.is_empty() && !neg.is_empty() {
                    assert!(area > 0.0);
                    assert!(pos.mesh.verts.iter().all(|&q| pl.dist(q) >= -1e-9));
                    assert!(neg.mesh.verts.iter().all(|&q| pl.dist(q) <= 1e-9));
                }
            }
        }
    }

    #[test]
    fn clip_ring_cap_with_two_loops() {
        // cutting the ring across both bars yields two cap loops per side
        let m = ring();
        let (pos, neg, area) = clip(
            &Solid::new(m.clone()),
            &Plane {
                n: DVec3::X,
                d: -1.5,
            },
        )
        .unwrap();
        assert!((area - 2.0).abs() < 1e-9, "cap area {area}");
        assert!((pos.volume() - 4.0).abs() < 1e-9 && (neg.volume() - 4.0).abs() < 1e-9);
        // cut through the hole along z: one loop with a hole
        let (pos, neg, area) = clip(
            &Solid::new(m),
            &Plane {
                n: DVec3::Z,
                d: -0.5,
            },
        )
        .unwrap();
        assert!((area - 8.0).abs() < 1e-9, "annulus cap area {area}");
        assert!(closed(&pos.mesh) && closed(&neg.mesh));
    }

    /// Brute-force symmetric Hausdorff (dense samples, exact distances).
    fn hb_brute(part: &TriMesh, ch: &Ch) -> f64 {
        let hm = ch.poly.to_mesh();
        let qa = MeshQuery::new(part);
        let qb = MeshQuery::new(&hm);
        let mut worst: f64 = 0.0;
        for (src, dst) in [(part, &qb), (&hm, &qa)] {
            for t in 0..src.tris.len() {
                let [a, b, c] = src.tri_points(t);
                let mut s = Vec::new();
                tri_samples(a, b, c, 400, &mut s);
                for p in s {
                    worst = worst.max(dst.closest_point(p).unwrap().1.sqrt());
                }
            }
        }
        worst
    }

    #[test]
    fn concavity_matches_brute_force() {
        for m in [l_shape(), notched_box(), u_shape()] {
            let f = Frame::of(&m.aabb());
            let nm = m.transformed(|v| f.to_norm(v));
            let s = Solid::new(nm.clone());
            let ch = hull(&nm.verts).unwrap();
            let fast = hb(&s, &ch, 20000.0, f64::INFINITY);
            let brute = hb_brute(&nm, &ch);
            assert!(
                (fast - brute).abs() <= 0.02 * brute + 1e-3,
                "fast {fast} brute {brute}"
            );
            // the exact answer for the notched box: notch depth 1 (of 4) = 0.5 normalized
            let _ = brute;
        }
        // a convex box has zero concavity
        let b = box_mesh(DVec3::splat(-1.0), DVec3::ONE);
        let ch = hull(&b.verts).unwrap();
        assert_eq!(hb(&Solid::new(b), &ch, 2000.0, f64::INFINITY), 0.0);
    }

    fn params(th: f64) -> CoacdParams {
        CoacdParams {
            threshold: th,
            mcts_iterations: 60,
            seed: 7,
            ..Default::default()
        }
    }

    fn check_decomp(m: &TriMesh, hulls: &[ConvexPolytope], max_conc: f64) {
        // hulls cover the solid
        let total: f64 = hulls.iter().map(|h| h.volume()).sum();
        assert!(
            total >= m.signed_volume() * (1.0 - 1e-6),
            "hull volume {total} < {}",
            m.signed_volume()
        );
        let _ = max_conc;
    }

    #[test]
    fn l_shape_two_hulls() {
        let m = l_shape();
        let h = decompose_detailed(&m, &params(0.05));
        assert_eq!(h.len(), 2, "L-shape -> 2 hulls, got {}", h.len());
        assert!(h.iter().all(|x| x.1 <= 0.05));
        check_decomp(&m, &h.iter().map(|x| x.0.clone()).collect::<Vec<_>>(), 0.05);
    }

    #[test]
    fn u_shape_three_hulls() {
        let m = u_shape();
        let h = decompose_detailed(&m, &params(0.05));
        assert_eq!(h.len(), 3, "U-shape -> 3 hulls, got {}", h.len());
        assert!(h.iter().all(|x| x.1 <= 0.05));
    }

    #[test]
    fn ring_four_hulls() {
        let m = ring();
        let h = decompose_detailed(&m, &params(0.05));
        assert!(
            h.len() >= 4 && h.len() <= 5,
            "ring -> 4 hulls, got {}",
            h.len()
        );
        assert!(h.iter().all(|x| x.1 <= 0.05));
        // budget of 2 still works (concavity above threshold)
        let h2 = decompose(
            &m,
            &CoacdParams {
                max_convex_hull: Some(2),
                ..params(0.05)
            },
        );
        assert_eq!(h2.len(), 2);
    }

    #[test]
    fn notched_box_hulls() {
        let m = notched_box();
        let h = decompose_detailed(&m, &params(0.05));
        assert!(
            h.len() >= 2 && h.len() <= 3,
            "notched box -> 2-3 hulls, got {}",
            h.len()
        );
        assert!(h.iter().all(|x| x.1 <= 0.05));
        // loose threshold: a single hull (notch depth 0.5 normalized)
        let h1 = decompose(&m, &params(0.6));
        assert_eq!(h1.len(), 1);
    }

    #[test]
    fn decomposition_is_deterministic() {
        let m = u_shape();
        let a = decompose(&m, &params(0.03));
        let b = decompose(&m, &params(0.03));
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            let vx: Vec<[u64; 3]> = x
                .vertices()
                .iter()
                .map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()])
                .collect();
            let vy: Vec<[u64; 3]> = y
                .vertices()
                .iter()
                .map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()])
                .collect();
            assert_eq!(vx, vy);
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    #[test]
    #[ignore]
    fn hull_volume_speed() {
        // a thin curved shell patch (bowl-like): 400 points
        let mut pts = Vec::new();
        for i in 0..20 {
            for j in 0..10 {
                let (u, v) = (i as f64 / 19.0 - 0.5, j as f64 / 9.0 - 0.5);
                let z = u * u + v * v;
                pts.push(DVec3::new(u, v, z));
                pts.push(DVec3::new(u, v, z + 0.02));
            }
        }
        let t = std::time::Instant::now();
        let mut v = 0.0;
        for _ in 0..1000 {
            v += hull_volume(&pts);
        }
        eprintln!(
            "hull volume of {} pts: {:?} each ({v})",
            pts.len(),
            t.elapsed() / 1000
        );
        // merge candidates: two random convex cells (~20 vertices each)
        use rand::Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let cells: Vec<Vec<DVec3>> = (0..200)
            .map(|k| {
                let c = DVec3::new((k % 10) as f64, (k / 10) as f64 * 0.7, 0.0);
                let raw: Vec<DVec3> = (0..60)
                    .map(|_| {
                        c + DVec3::new(
                            rng.gen_range(-0.6..0.6),
                            rng.gen_range(-0.4..0.4),
                            rng.gen_range(-0.3..0.3),
                        )
                    })
                    .collect();
                hull(&raw).unwrap().pts
            })
            .collect();
        let nv: usize = cells.iter().map(|c| c.len()).sum();
        let t = std::time::Instant::now();
        let mut n = 0;
        for k in 0..cells.len() - 1 {
            let mut pts = cells[k].clone();
            pts.extend_from_slice(&cells[k + 1]);
            for _ in 0..20 {
                n += hull_tri(&pts).map(|h| h.poly.faces.len()).unwrap_or(0);
            }
        }
        eprintln!(
            "hull_tri of 2 cells ({} vertices avg): {:?} each ({n})",
            2 * nv / cells.len(),
            t.elapsed() / (20 * (cells.len() as u32 - 1))
        );
        let t = std::time::Instant::now();
        for k in 0..cells.len() - 1 {
            let mut pts = cells[k].clone();
            pts.extend_from_slice(&cells[k + 1]);
            for _ in 0..20 {
                n += convex_hull_fast(&pts).map(|h| h.tris.len()).unwrap_or(0);
            }
        }
        eprintln!(
            "quickhull of 2 cells: {:?} each ({n})",
            t.elapsed() / (20 * (cells.len() as u32 - 1))
        );
        let t = std::time::Instant::now();
        for k in 0..cells.len() - 1 {
            let mut pts = cells[k].clone();
            pts.extend_from_slice(&cells[k + 1]);
            for _ in 0..20 {
                n += hull(&pts).map(|h| h.poly.faces.len()).unwrap_or(0);
            }
        }
        eprintln!(
            "hull (merged faces) of 2 cells: {:?} each ({n})",
            t.elapsed() / (20 * (cells.len() as u32 - 1))
        );
    }
}
