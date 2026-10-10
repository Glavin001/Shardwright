//! Stage 8: collision hulls.
//!
//! The decomposition follows CoACD (Wei et al., SIGGRAPH 2022; see
//! [`coacd`] for the port and its MIT notice), adapted to the cell complex:
//!
//! * Atoms: every leaf cell is an exact convex piece when convex (the common
//!   case: Voronoi cells of convex solids); non-convex cells are cut by the
//!   ported CoACD tree search ([`coacd::cut`]) until their collision-aware
//!   concavity is below the threshold.
//! * Merging, bottom-up across hierarchy levels: for each fragment, pieces
//!   (atoms at the leaf level, the children's carried pieces above) are
//!   merged greedily by minimal concavity of the merged hull
//!   ([`coacd::greedy_merge`], CoACD's `MergeConvexHulls` with a
//!   collision-aware cost measured against the fragment surface) until the
//!   hull budget is met and the cheapest merge exceeds the threshold
//!   (`concavity` × fragment diameter). The pieces left when the threshold
//!   phase ends are carried to the parent level.
//! * Non-overlap (not in CoACD): every overlapping pair of hulls of
//!   neighbouring fragments at a level is separated by the plane (among the
//!   bond / interface normals and the faces bounding the overlap, at the best
//!   offset) that removes the least of the fragments' own geometry; hulls are
//!   finally shrunk by the margin.

pub mod coacd;

use coacd::{CoacdParams, Frame, MergeCtx, Piece};
use frac_core::*;
use frac_geom::hull::{ConvexPolytope, HalfSpace};
use frac_geom::inside::MeshQuery;
use frac_geom::{Aabb, DVec3, TriMesh};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering as AO};

static T_ATOMS: AtomicU64 = AtomicU64::new(0);
static T_MERGE: AtomicU64 = AtomicU64::new(0);
static T_MESH: AtomicU64 = AtomicU64::new(0);
fn tadd(c: &AtomicU64, t: std::time::Instant) {
    c.fetch_add(t.elapsed().as_nanos() as u64, AO::Relaxed);
}

/// Effort of the CoACD cutting-plane search for non-convex cells.
#[derive(Clone, Debug)]
pub struct SearchEffort {
    pub mcts_iterations: u32,
    pub mcts_depth: u32,
    pub mcts_nodes: u32,
    /// `Hb` samples per unit normalized area.
    pub resolution: u32,
}

impl Default for SearchEffort {
    /// Upstream CoACD defaults.
    fn default() -> Self {
        SearchEffort { mcts_iterations: 150, mcts_depth: 3, mcts_nodes: 20, resolution: 2000 }
    }
}

#[derive(Clone, Debug)]
pub struct CollisionParams {
    /// Concavity threshold as a fraction of the fragment diameter.
    pub concavity: f64,
    /// If set, overrides `concavity`: threshold in CoACD's normalized units
    /// (fraction of half the longest bounding-box side).
    pub coacd_threshold: Option<f64>,
    /// Hull budget per fragment (`usize::MAX`: unlimited).
    pub max_hulls: usize,
    pub margin: f64,
    pub min_rigid_size: f64,
    /// Levels that receive hulls (empty = all).
    pub levels: Vec<u8>,
    pub search: SearchEffort,
    pub seed: u64,
    /// Max pieces carried from a fragment to its parent level.
    pub carry_cap: usize,
    /// Weight of the neighbour-intrusion term of the merge cost (volume of
    /// a merged hull inside bonded neighbours, which non-overlap
    /// enforcement would later remove), like CoACD's `rv_k`.
    pub intrusion_k: f64,
}

impl Default for CollisionParams {
    fn default() -> Self {
        CollisionParams {
            concavity: 0.02,
            coacd_threshold: None,
            max_hulls: 8,
            margin: 0.0005,
            min_rigid_size: 0.01,
            levels: Vec::new(),
            search: SearchEffort::default(),
            seed: 1,
            carry_cap: 48,
            intrusion_k: std::env::var("FRAC_DBG_INTR").ok().and_then(|v| v.parse().ok()).unwrap_or(0.6),
        }
    }
}

impl CollisionParams {
    /// World-space concavity threshold of a solid with the given volume and
    /// bounding box.
    fn threshold(&self, volume: f64, bbox: &Aabb) -> f64 {
        match self.coacd_threshold {
            Some(t) => t * 0.5 * bbox.extent().max_element(),
            None => self.concavity * volume.max(0.0).cbrt().max(bbox.diagonal() * 0.5),
        }
    }
}

/// Closed boundary mesh of a set of cells (exterior polygons + patches to
/// cells outside the set), using the component vertex table.
pub fn cells_boundary_mesh(asset: &Asset, cells: &[CellId]) -> TriMesh {
    boundary_mesh_cells(asset, cells).0
}

/// Like [`cells_boundary_mesh`], also returning the cell of each triangle.
fn boundary_mesh_cells(asset: &Asset, cells: &[CellId]) -> (TriMesh, Vec<CellId>) {
    let set: BTreeSet<CellId> = cells.iter().copied().collect();
    let mut comps: BTreeSet<u32> = BTreeSet::new();
    for c in cells {
        comps.insert(asset.cells[c.idx()].component.0);
    }
    let mut out = TriMesh::default();
    let mut owner = Vec::new();
    for ci in comps {
        let comp = &asset.components[ci as usize];
        let g = &comp.geometry;
        let mut tris = Vec::new();
        for e in &g.ext_polys {
            if set.contains(&e.cell) {
                tris.extend(e.tris.iter().copied());
                owner.extend(std::iter::repeat_n(e.cell, e.tris.len()));
            }
        }
        for p in &g.patches {
            let (a, b) = (set.contains(&p.cells.0), set.contains(&p.cells.1));
            if a && !b {
                tris.extend(p.tris.iter().copied());
                owner.extend(std::iter::repeat_n(p.cells.0, p.tris.len()));
            } else if b && !a {
                tris.extend(p.tris.iter().map(|t| [t[0], t[2], t[1]]));
                owner.extend(std::iter::repeat_n(p.cells.1, p.tris.len()));
            }
        }
        let m = TriMesh { verts: g.verts.clone(), tris }.compact();
        out.append(&m);
    }
    (out, owner)
}

/// Points of a cell (all boundary polygon vertices).
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

/// A convex atom: a convex leaf cell or a CoACD part of a non-convex cell.
#[derive(Clone, Debug)]
struct Atom {
    cell: CellId,
    poly: ConvexPolytope,
    pts: Vec<DVec3>,
    vol: f64,
    /// Concavity of the atom's hull w.r.t. its geometry (0 for exact cells).
    own: f64,
}

/// Atoms of a cell.
fn cell_atoms(asset: &Asset, cell: CellId, p: &CollisionParams) -> Vec<Atom> {
    let mesh = cells_boundary_mesh(asset, &[cell]);
    if mesh.tris.is_empty() {
        return Vec::new();
    }
    let pts = cell_points(asset, cell);
    let Some(ch) = coacd::hull(&pts) else { return Vec::new() };
    let vol = mesh.signed_volume();
    if ch.volume - vol <= 1e-9 * ch.volume {
        return vec![Atom { cell, poly: ch.poly, pts: ch.pts, vol, own: 0.0 }];
    }
    let bbox = mesh.aabb();
    let frame = Frame::of(&bbox);
    let thresh = p.threshold(vol, &bbox);
    let cp = CoacdParams {
        threshold: thresh / frame.half,
        mcts_iterations: p.search.mcts_iterations,
        mcts_depth: p.search.mcts_depth,
        mcts_nodes: p.search.mcts_nodes,
        resolution: p.search.resolution,
        seed: frac_core::determinism::sub_seed(p.seed, cell.0 as u64),
        merge: false,
        max_convex_hull: None,
        max_parts: (2 * p.max_hulls).clamp(2, 64),
        ..Default::default()
    };
    let nm = mesh.transformed(|v| frame.to_norm(v));
    let parts = coacd::cut(coacd::Solid::new(nm), &cp);
    let h3 = frame.half.powi(3);
    let atoms: Vec<Atom> = parts
        .into_iter()
        .map(|c| {
            let poly = frame.polytope_to_world(&c.ch.poly);
            let pts = c.ch.pts.iter().map(|&q| frame.to_world(q)).collect();
            Atom { cell, poly, pts, vol: c.solid.volume() * h3, own: c.cost * frame.half }
        })
        .collect();
    if atoms.is_empty() {
        return vec![Atom { cell, poly: ch.poly, pts: ch.pts, vol, own: 0.0 }];
    }
    atoms
}

/// Fragment geometry for merging at one level.
struct FragGeom {
    mesh: TriMesh,
    owner: Vec<CellId>,
    thresh: f64,
    spacing: f64,
}

fn frag_geom(asset: &Asset, f: &Fragment, p: &CollisionParams) -> FragGeom {
    let cells = asset.fragment_cells(f);
    let (mesh, owner) = boundary_mesh_cells(asset, cells);
    let bbox = mesh.aabb();
    let thresh = p.threshold(f.mass.volume, &bbox);
    let area = mesh.area();
    // sample spacing: half the threshold, at most ~6000 surface samples
    let spacing = (0.5 * thresh).max((area / 6000.0).sqrt()).max(1e-12);
    FragGeom { mesh, owner, thresh, spacing }
}

/// Distribute fragment-surface samples over pieces (by cell; split cells by
/// the nearest piece hull).
fn assign_samples(g: &FragGeom, pieces: &mut [Piece], atoms: &[Atom]) {
    let mut by_cell: BTreeMap<CellId, Vec<usize>> = BTreeMap::new();
    for (i, pc) in pieces.iter_mut().enumerate() {
        pc.samples.clear();
        let mut cs: Vec<CellId> = pc.tags.iter().map(|&a| atoms[a as usize].cell).collect();
        cs.dedup();
        for c in cs {
            by_cell.entry(c).or_default().push(i);
        }
    }
    let mut buf = Vec::new();
    for t in 0..g.mesh.tris.len() {
        let Some(cands) = by_cell.get(&g.owner[t]) else { continue };
        let [a, b, c] = g.mesh.tri_points(t);
        let ar = 0.5 * (b - a).cross(c - a).length();
        let n = ((ar / (g.spacing * g.spacing)).ceil() as usize).clamp(1, 64);
        buf.clear();
        coacd::tri_samples(a, b, c, n, &mut buf);
        for &x in &buf {
            let k = if cands.len() == 1 {
                cands[0]
            } else {
                // piece whose hull is closest (deepest inside)
                *cands.iter().max_by(|&&i, &&j| coacd::inside_depth(&pieces[i].poly, x).total_cmp(&coacd::inside_depth(&pieces[j].poly, x)).then(j.cmp(&i))).unwrap()
            };
            pieces[k].samples.push(x);
        }
    }
}

fn atom_piece(a: &Atom, id: u32) -> Piece {
    let ch = coacd::Ch { poly: a.poly.clone(), pts: a.pts.clone(), volume: a.poly.volume() };
    let mut pc = Piece::new(ch, a.vol, Vec::new(), vec![id]);
    pc.own = a.own;
    pc
}

/// Per-fragment merge output: hull pieces and the pieces carried upward.
struct FragOut {
    hulls: Vec<Piece>,
    carry: Vec<Piece>,
}

/// Compute hulls for all fragments. Returns hulls and per-fragment ranges.
pub fn build_hulls(asset: &Asset, p: &CollisionParams) -> (Vec<Hull>, Vec<std::ops::Range<u32>>) {
    let h = &asset.hierarchy;
    let nl = h.levels as usize;
    let want_level = |l: usize| p.levels.is_empty() || p.levels.contains(&(l as u8));
    let profile = std::env::var("FRAC_PROFILE").is_ok();
    // ---- atoms (convex cells, CoACD parts of non-convex cells)
    let ta = std::time::Instant::now();
    let per_cell: Vec<Vec<Atom>> = (0..asset.cells.len() as u32).into_par_iter().map(|c| cell_atoms(asset, CellId(c), p)).collect();
    let mut atoms: Vec<Atom> = Vec::new();
    let mut atoms_of_cell: Vec<std::ops::Range<u32>> = Vec::with_capacity(per_cell.len());
    for v in per_cell {
        let s = atoms.len() as u32;
        atoms.extend(v);
        atoms_of_cell.push(s..atoms.len() as u32);
    }
    tadd(&T_ATOMS, ta);
    if profile {
        eprintln!(
            "collision atoms: {} from {} cells in {:?} (cpu: {} clips {:.1}s, hull volumes {:.1}s)",
            atoms.len(),
            asset.cells.len(),
            ta.elapsed(),
            coacd::N_CLIP.load(AO::Relaxed),
            coacd::T_CLIP.load(AO::Relaxed) as f64 * 1e-9,
            coacd::T_HV.load(AO::Relaxed) as f64 * 1e-9
        );
    }
    // cell adjacency (shared interfaces)
    let mut cell_nbrs: Vec<Vec<CellId>> = vec![Vec::new(); asset.cells.len()];
    for it in &asset.interfaces {
        if let CellOrWorld::Cell(cb) = it.cells.1 {
            cell_nbrs[it.cells.0.idx()].push(cb);
            cell_nbrs[cb.idx()].push(it.cells.0);
        }
    }
    for v in cell_nbrs.iter_mut() {
        v.sort_unstable();
        v.dedup();
    }
    // ---- bottom-up merging
    let mut outs: Vec<Option<FragOut>> = (0..h.fragments.len()).map(|_| None).collect();
    for level in (0..nl).rev() {
        let tl = std::time::Instant::now();
        let r = h.level_ranges[level].clone();
        // bonded neighbours at this level (hulls must not overlap theirs)
        let mut nbr_frags: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
        for b in asset.level_bonds(level as u8) {
            if let FragmentOrWorld::Fragment(fb) = b.b {
                nbr_frags.entry(b.a.0).or_default().insert(fb.0);
                nbr_frags.entry(fb.0).or_default().insert(b.a.0);
            }
        }
        let results: Vec<(u32, FragOut)> = (r.start..r.end)
            .into_par_iter()
            .map(|fi| {
                let f = &h.fragments[fi as usize];
                let cells = asset.fragment_cells(f);
                let tm = std::time::Instant::now();
                let g = frag_geom(asset, f, p);
                tadd(&T_MESH, tm);
                if f.particle_candidate {
                    let pts: Vec<DVec3> = cells.iter().flat_map(|&c| cell_points(asset, c)).collect();
                    let tags: Vec<u32> = cells.iter().flat_map(|&c| atoms_of_cell[c.idx()].clone()).collect();
                    let hull = coacd::hull(&pts).map(|ch| vec![Piece::new(ch, f.mass.volume, Vec::new(), tags)]).unwrap_or_default();
                    return (fi, FragOut { carry: hull.clone(), hulls: hull });
                }
                let mut pieces: Vec<Piece> = if level == nl - 1 || f.children.is_empty() {
                    cells.iter().flat_map(|&c| atoms_of_cell[c.idx()].clone()).map(|a| atom_piece(&atoms[a as usize], a)).collect()
                } else {
                    f.children.clone().flat_map(|ch| outs[ch as usize].as_ref().map(|o| o.carry.clone()).unwrap_or_default()).collect()
                };
                let tmg = std::time::Instant::now();
                assign_samples(&g, &mut pieces, &atoms);
                let q = MeshQuery::new(&g.mesh);
                let foreign: Vec<coacd::Foreign> = nbr_frags
                    .get(&fi)
                    .map(|ns| {
                        ns.iter()
                            .flat_map(|&nf| asset.fragment_cells(&h.fragments[nf as usize]).iter().flat_map(|&c| atoms_of_cell[c.idx()].clone()))
                            .map(|a| {
                                let at = &atoms[a as usize];
                                let pv = at.poly.volume();
                                coacd::Foreign { poly: &at.poly, pts: &at.pts, bbox: Aabb::from_points(at.pts.iter()), scale: if pv > 0.0 { at.vol / pv } else { 1.0 } }
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let signer = coacd::Signer::new(&g.mesh);
                let ctx = MergeCtx { q: &q, signer: Some(&signer), spacing: g.spacing, rv_k: 0.3, max_tri_samples: 256, foreign, intrusion_k: p.intrusion_k };
                // Pieces keep their own costs from the finer level (atoms: from
                // the cut search): a fragment's surface is a subset of its
                // children's, so those costs bound the costs at this level.
                let atom_cells = |pc: &Piece| -> Vec<CellId> {
                    let mut v: Vec<CellId> = pc.tags.iter().map(|&a| atoms[a as usize].cell).collect();
                    v.dedup();
                    v
                };
                let adjacent = |a: &Piece, b: &Piece| -> bool {
                    if !a.bbox.expanded(1e-6 * g.thresh.max(1e-12)).overlaps(&b.bbox) {
                        return false;
                    }
                    let (ca, cb) = (atom_cells(a), atom_cells(b));
                    ca.iter().any(|x| cb.iter().any(|y| x == y || cell_nbrs[x.idx()].binary_search(y).is_ok()))
                };
                let budget = p.max_hulls.max(1);
                let res = coacd::greedy_merge(pieces, &ctx, g.thresh, budget, p.carry_cap.max(budget), &adjacent);
                tadd(&T_MERGE, tmg);
                (fi, FragOut { hulls: res.pieces, carry: res.carry })
            })
            .collect();
        if profile {
            let (nh, nc) = results.iter().fold((0, 0), |a, (_, o)| (a.0 + o.hulls.len(), a.1 + o.carry.len()));
            let maxv = results.iter().flat_map(|(_, o)| o.carry.iter().map(|p| p.pts.len())).max().unwrap_or(0);
            eprintln!("  level {level}: {} fragments -> {nh} hulls, {nc} carried (max {maxv} hull vertices)", results.len());
        }
        for (fi, o) in results {
            outs[fi as usize] = Some(o);
        }
        if profile {
            eprintln!(
                "collision level {level}: {:?} (cumulative cpu: atoms {:.1}s, merge {:.1}s [{} pairs {:.1}s], mesh {:.1}s)",
                tl.elapsed(),
                T_ATOMS.load(AO::Relaxed) as f64 * 1e-9,
                T_MERGE.load(AO::Relaxed) as f64 * 1e-9,
                coacd::N_PAIR.load(AO::Relaxed),
                coacd::T_PAIR.load(AO::Relaxed) as f64 * 1e-9,
                T_MESH.load(AO::Relaxed) as f64 * 1e-9
            );
        }
    }
    // ---- non-overlap enforcement per level, then margin shrink
    let tn = std::time::Instant::now();
    let mut hull_recs: Vec<Vec<HullRec>> = outs
        .into_iter()
        .map(|o| o.map(|o| o.hulls.into_iter().map(|pc| HullRec { poly: pc.poly, atoms: pc.tags }).collect()).unwrap_or_default())
        .collect();
    for level in 0..nl {
        if want_level(level) && std::env::var("FRAC_DBG_NOSEP").is_err() {
            separate_level(asset, level as u8, &mut hull_recs, &atoms);
        }
    }
    if profile {
        eprintln!("collision non-overlap: {:?}", tn.elapsed());
        eprintln!(
            "  pair cpu: hull {:.1}s in {:.1}s out {:.1}s ({} evals)",
            coacd::T_HULLC.load(AO::Relaxed) as f64 * 1e-9,
            coacd::T_IN.load(AO::Relaxed) as f64 * 1e-9,
            coacd::T_OUT.load(AO::Relaxed) as f64 * 1e-9,
            coacd::N_EVAL.load(AO::Relaxed)
        );
    }
    // output
    let mut hulls = Vec::new();
    let mut ranges = vec![0..0; h.fragments.len()];
    for (fi, hp) in hull_recs.iter().enumerate() {
        let level = h.fragments[fi].level as usize;
        if !want_level(level) {
            continue;
        }
        let start = hulls.len() as u32;
        for rec in hp {
            if rec.poly.is_empty() {
                continue;
            }
            let s = rec.poly.shrunk(p.margin);
            let s = if s.is_empty() { rec.poly.clone() } else { s };
            hulls.push(to_hull(FragmentId(fi as u32), &s));
        }
        ranges[fi] = start..hulls.len() as u32;
    }
    (hulls, ranges)
}

/// A fragment hull with the atoms it was built from.
#[derive(Clone, Debug)]
struct HullRec {
    poly: ConvexPolytope,
    atoms: Vec<u32>,
}

/// Projection of an atom on a direction (for separation losses).
struct AtomProj {
    lo: f64,
    hi: f64,
    full: f64,
    frac: f64,
    id: u32,
}

fn project_atoms(atoms: &[Atom], ids: &[u32], n: DVec3) -> Vec<AtomProj> {
    ids.iter()
        .map(|&a| {
            let at = &atoms[a as usize];
            let (lo, hi) = at.pts.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), q| (lo.min(n.dot(*q)), hi.max(n.dot(*q))));
            let full = at.poly.volume();
            let frac = if full > 0.0 { at.vol / full } else { 1.0 };
            AtomProj { lo, hi, full, frac, id: a }
        })
        .collect()
}

/// Volume of the projected atoms beyond `n·x > t` (`above`) or below.
/// Atoms straddling the plane are clipped exactly, or (`approx`) estimated
/// from their projected extent with a smoothstep profile.
fn cut_loss(atoms: &[Atom], pr: &[AtomProj], n: DVec3, t: f64, above: bool, approx: bool) -> f64 {
    let mut v = 0.0;
    for p in pr {
        let removed = if (above && p.hi <= t) || (!above && p.lo >= t) {
            0.0
        } else if (above && p.lo >= t) || (!above && p.hi <= t) {
            p.full
        } else if approx {
            let u = if above { (p.hi - t) / (p.hi - p.lo) } else { (t - p.lo) / (p.hi - p.lo) };
            p.full * u * u * (3.0 - 2.0 * u)
        } else if above {
            atoms[p.id as usize].poly.clip(&HalfSpace { n: -n, d: -t }).volume()
        } else {
            atoms[p.id as usize].poly.clip(&HalfSpace { n, d: t }).volume()
        };
        v += removed * p.frac;
    }
    v
}

/// Cached data of a hull during separation.
struct HullData {
    verts: Vec<DVec3>,
    bbox: Aabb,
}

fn hull_data(p: &ConvexPolytope) -> HullData {
    let verts = p.vertices();
    let bbox = Aabb::from_points(verts.iter());
    HullData { verts, bbox }
}

/// Best separating plane (unit normal from A to B, offset) for two
/// overlapping hulls, minimizing the removed own geometry: candidate
/// normals are the bond / interface normals, the face normals of the
/// overlap region and the centroid direction; the offset is searched over
/// the overlap interval (uniform samples, then golden-section refinement).
#[allow(clippy::too_many_arguments)]
fn best_separation(atoms: &[Atom], ha: &HullRec, da: &HullData, hb: &HullRec, db: &HullData, overlap: &ConvexPolytope, extra: &[DVec3]) -> Option<(DVec3, f64, f64)> {
    let mut cands: Vec<DVec3> = extra.to_vec();
    for (hs, _) in &overlap.faces {
        cands.push(hs.n);
        cands.push(-hs.n);
    }
    let ca = ha.poly.volume_integrals().com();
    let cb = hb.poly.volume_integrals().com();
    let dc = (cb - ca).normalize_or_zero();
    if dc != DVec3::ZERO {
        cands.push(dc);
    }
    if let Ok(k) = std::env::var("FRAC_DBG_FIB") {
        let k: usize = k.parse().unwrap_or(64);
        for i in 0..k {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / k as f64;
            let r = (1.0 - z * z).sqrt();
            let a = i as f64 * 2.399963229728653;
            cands.push(DVec3::new(r * libm::cos(a), r * libm::sin(a), z));
        }
    }
    // dedupe (near-parallel)
    let mut uniq: Vec<DVec3> = Vec::new();
    for c in cands {
        let c = c.normalize_or_zero();
        if c != DVec3::ZERO && !uniq.iter().any(|u| u.dot(c) > 1.0 - 1e-9) {
            uniq.push(c);
        }
    }
    // offset search for one normal: uniform samples over the overlap
    // interval, then golden-section refinement around the best one
    let search = |n: DVec3, approx: bool| -> Option<(f64, f64)> {
        let hi_a = da.verts.iter().map(|q| n.dot(*q)).fold(f64::NEG_INFINITY, f64::max);
        let lo_b = db.verts.iter().map(|q| n.dot(*q)).fold(f64::INFINITY, f64::min);
        if !(hi_a.is_finite() && lo_b.is_finite()) {
            return None;
        }
        if hi_a <= lo_b {
            return Some((0.0, 0.5 * (hi_a + lo_b)));
        }
        // only atoms reaching into the overlap interval can be cut
        let pa: Vec<AtomProj> = project_atoms(atoms, &ha.atoms, n).into_iter().filter(|p| p.hi > lo_b).collect();
        let pb: Vec<AtomProj> = project_atoms(atoms, &hb.atoms, n).into_iter().filter(|p| p.lo < hi_a).collect();
        let loss = |t: f64| cut_loss(atoms, &pa, n, t, true, approx) + cut_loss(atoms, &pb, n, t, false, approx);
        let k = 8;
        let mut best = (f64::INFINITY, lo_b);
        for i in 0..=k {
            let t = lo_b + (hi_a - lo_b) * i as f64 / k as f64;
            let l = loss(t);
            if l < best.0 {
                best = (l, t);
            }
        }
        let step = (hi_a - lo_b) / k as f64;
        let (mut a, mut b) = ((best.1 - step).max(lo_b), (best.1 + step).min(hi_a));
        let g = 0.6180339887498949;
        for _ in 0..8 {
            let x1 = b - g * (b - a);
            let x2 = a + g * (b - a);
            let (l1, l2) = (loss(x1), loss(x2));
            if l1 < best.0 {
                best = (l1, x1);
            }
            if l2 < best.0 {
                best = (l2, x2);
            }
            if l1 <= l2 {
                b = x2;
            } else {
                a = x1;
            }
        }
        Some(best)
    };
    // screen all normals with the approximate loss, refine the best few
    // exactly
    let approx: Vec<Option<(f64, f64)>> = uniq.iter().map(|&n| search(n, true)).collect();
    let mut order: Vec<usize> = (0..uniq.len()).filter(|&i| approx[i].is_some()).collect();
    order.sort_by(|&i, &j| approx[i].unwrap().0.total_cmp(&approx[j].unwrap().0).then(i.cmp(&j)));
    order.truncate(3);
    let mut best: Option<(DVec3, f64, f64)> = None;
    for i in order {
        if let Some((l, t)) = search(uniq[i], false) {
            if best.map(|b| l < b.2).unwrap_or(true) {
                best = Some((uniq[i], t, l));
            }
        }
    }
    best
}

/// Overlap polytope of two hulls if their intersection has volume.
fn overlap_of(pa: &ConvexPolytope, da: &HullData, pb: &ConvexPolytope, db: &HullData) -> Option<ConvexPolytope> {
    if pa.is_empty() || pb.is_empty() || !da.bbox.overlaps(&db.bbox) {
        return None;
    }
    let ov = pa.clip_all(&pb.halfspaces());
    if ov.is_empty() || ov.volume() <= 0.0 { None } else { Some(ov) }
}

/// Separate overlapping hulls of neighbouring fragments at one level.
///
/// Overlapping hull pairs are detected in parallel and resolved in bond
/// order; independent pairs (touching hulls no earlier pending pair
/// touches) are resolved concurrently, which gives the same result as the
/// sequential order.
fn separate_level(asset: &Asset, level: u8, recs: &mut [Vec<HullRec>], atoms: &[Atom]) {
    let bonds: Vec<&Bond> = asset.level_bonds(level).filter(|b| matches!(b.b, FragmentOrWorld::Fragment(_))).collect();
    let fb_of = |b: &Bond| match b.b {
        FragmentOrWorld::Fragment(f) => f.idx(),
        FragmentOrWorld::World => unreachable!(),
    };
    // interface normals oriented from A to B, per bond
    let extras: Vec<Vec<DVec3>> = bonds
        .par_iter()
        .map(|b| {
            let fa_cells: BTreeSet<CellId> = asset.fragment_cells(asset.fragment(b.a)).iter().copied().collect();
            let mut extra: Vec<DVec3> = vec![b.normal];
            for &iid in &b.interfaces {
                let it = &asset.interfaces[iid.idx()];
                let s = if fa_cells.contains(&it.cells.0) { 1.0 } else { -1.0 };
                for poly in &it.polygons {
                    extra.push(poly.normal * s);
                }
                if extra.len() > 32 {
                    break;
                }
            }
            extra
        })
        .collect();
    let mut data: Vec<Vec<HullData>> = recs.iter().map(|v| v.iter().map(|r| hull_data(&r.poly)).collect()).collect();
    // detection
    let found: Vec<Vec<(usize, usize, usize)>> = bonds
        .par_iter()
        .enumerate()
        .map(|(bi, b)| {
            let (ia, ib) = (b.a.idx(), fb_of(b));
            let mut v = Vec::new();
            for x in 0..recs[ia].len() {
                for y in 0..recs[ib].len() {
                    if overlap_of(&recs[ia][x].poly, &data[ia][x], &recs[ib][y].poly, &data[ib][y]).is_some() {
                        v.push((bi, x, y));
                    }
                }
            }
            v
        })
        .collect();
    let mut pending: Vec<(usize, usize, usize)> = found.into_iter().flatten().collect();
    if std::env::var("FRAC_PROFILE").is_ok() {
        eprintln!("  level {level}: {} bonds, {} overlapping hull pairs", bonds.len(), pending.len());
    }
    // resolution in dependency-respecting parallel batches
    while !pending.is_empty() {
        let mut used: BTreeSet<(usize, usize)> = BTreeSet::new();
        let mut batch = Vec::new();
        let mut rest = Vec::new();
        for &(bi, x, y) in &pending {
            let (ia, ib) = (bonds[bi].a.idx(), fb_of(bonds[bi]));
            let free = !used.contains(&(ia, x)) && !used.contains(&(ib, y));
            used.insert((ia, x));
            used.insert((ib, y));
            if free {
                batch.push((bi, x, y));
            } else {
                rest.push((bi, x, y));
            }
        }
        let cuts: Vec<Option<(DVec3, f64)>> = batch
            .par_iter()
            .map(|&(bi, x, y)| {
                let b = bonds[bi];
                let (ia, ib) = (b.a.idx(), fb_of(b));
                let ov = overlap_of(&recs[ia][x].poly, &data[ia][x], &recs[ib][y].poly, &data[ib][y])?;
                Some(match best_separation(atoms, &recs[ia][x], &data[ia][x], &recs[ib][y], &data[ib][y], &ov, &extras[bi]) {
                    Some((n, t, _)) => (n, t),
                    None => (b.normal, b.normal.dot(ov.volume_integrals().com())),
                })
            })
            .collect();
        for (&(bi, x, y), cut) in batch.iter().zip(cuts) {
            let Some((n, t)) = cut else { continue };
            let (ia, ib) = (bonds[bi].a.idx(), fb_of(bonds[bi]));
            recs[ia][x].poly = recs[ia][x].poly.clip(&HalfSpace { n, d: t });
            recs[ib][y].poly = recs[ib][y].poly.clip(&HalfSpace { n: -n, d: -t });
            data[ia][x] = hull_data(&recs[ia][x].poly);
            data[ib][y] = hull_data(&recs[ib][y].poly);
        }
        pending = rest;
    }
    for v in recs.iter_mut() {
        v.retain(|r| !r.poly.is_empty());
    }
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
