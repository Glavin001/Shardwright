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

/// Relative volume tolerance of the convexity test (a solid is treated as
/// convex when its hull volume exceeds its volume by at most this fraction).
pub const CONVEX_TOL: f64 = 1e-6;

static T_ATOMS: AtomicU64 = AtomicU64::new(0);
static T_MERGE: AtomicU64 = AtomicU64::new(0);
static T_MESH: AtomicU64 = AtomicU64::new(0);
static T_O_SHRINK: AtomicU64 = AtomicU64::new(0);
static T_SETUP: AtomicU64 = AtomicU64::new(0);
static T_S_SAMPLES: AtomicU64 = AtomicU64::new(0);
static T_S_QUERY: AtomicU64 = AtomicU64::new(0);
static T_S_COARSEN: AtomicU64 = AtomicU64::new(0);
static T_O_LIMIT: AtomicU64 = AtomicU64::new(0);
static T_O_CONVEX: AtomicU64 = AtomicU64::new(0);
static N_O_REBUILD: AtomicU64 = AtomicU64::new(0);
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
    /// Maximum vertices per output hull (0: unlimited).
    pub max_hull_vertices: usize,
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
            carry_cap: 16,
            intrusion_k: 0.6,
            max_hull_vertices: 64,
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
    let set: BTreeSet<CellId> = cells.iter().copied().collect();
    let mut comps: BTreeSet<u32> = BTreeSet::new();
    for c in cells {
        comps.insert(asset.cells[c.idx()].component.0);
    }
    let mut out = TriMesh::default();
    for ci in comps {
        let g = &asset.components[ci as usize].geometry;
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
        out.append(&TriMesh { verts: g.verts.clone(), tris }.compact());
    }
    out
}

/// Per-cell lists of the polygons of the component geometry (so boundary
/// meshes of a few cells do not scan whole components).
struct CellIndex {
    ext: Vec<Vec<u32>>,
    patch: Vec<Vec<u32>>,
}

impl CellIndex {
    fn new(asset: &Asset) -> CellIndex {
        let n = asset.cells.len();
        let mut ext = vec![Vec::new(); n];
        let mut patch = vec![Vec::new(); n];
        for comp in &asset.components {
            let g = &comp.geometry;
            for (i, e) in g.ext_polys.iter().enumerate() {
                ext[e.cell.idx()].push(i as u32);
            }
            for (i, p) in g.patches.iter().enumerate() {
                patch[p.cells.0.idx()].push(i as u32);
                if p.cells.1 != p.cells.0 {
                    patch[p.cells.1.idx()].push(i as u32);
                }
            }
        }
        CellIndex { ext, patch }
    }

    /// Closed boundary mesh of a set of cells and the cell of each triangle
    /// (same surface as [`cells_boundary_mesh`]).
    fn boundary_mesh(&self, asset: &Asset, cells: &[CellId]) -> (TriMesh, Vec<CellId>) {
        let mut sorted: Vec<CellId> = cells.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let inside = |c: CellId| sorted.binary_search(&c).is_ok();
        let mut index: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
        let mut verts: Vec<DVec3> = Vec::new();
        let mut tris: Vec<[u32; 3]> = Vec::new();
        let mut owner: Vec<CellId> = Vec::new();
        for &c in &sorted {
            let ci = asset.cells[c.idx()].component.0;
            let g = &asset.components[ci as usize].geometry;
            let mut add = |t: [u32; 3], verts: &mut Vec<DVec3>| {
                let mut nt = [0u32; 3];
                for k in 0..3 {
                    nt[k] = *index.entry((ci, t[k])).or_insert_with(|| {
                        verts.push(g.verts[t[k] as usize]);
                        (verts.len() - 1) as u32
                    });
                }
                tris.push(nt);
                owner.push(c);
            };
            for &e in &self.ext[c.idx()] {
                for &t in &g.ext_polys[e as usize].tris {
                    add(t, &mut verts);
                }
            }
            for &pi in &self.patch[c.idx()] {
                let p = &g.patches[pi as usize];
                let (first, other) = if p.cells.0 == c { (true, p.cells.1) } else { (false, p.cells.0) };
                if inside(other) {
                    continue;
                }
                for &t in &p.tris {
                    add(if first { t } else { [t[0], t[2], t[1]] }, &mut verts);
                }
            }
        }
        (TriMesh { verts, tris }, owner)
    }

    /// Points of a cell (all boundary polygon vertices).
    fn cell_points(&self, asset: &Asset, cell: CellId) -> Vec<DVec3> {
        let g = &asset.components[asset.cells[cell.idx()].component.idx()].geometry;
        let mut ids: Vec<u32> = Vec::new();
        for &e in &self.ext[cell.idx()] {
            ids.extend(g.ext_polys[e as usize].verts.iter().copied());
        }
        for &pi in &self.patch[cell.idx()] {
            for l in &g.patches[pi as usize].loops {
                ids.extend(l.iter().copied());
            }
        }
        ids.sort_unstable();
        ids.dedup();
        ids.into_iter().map(|i| g.verts[i as usize]).collect()
    }
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
    /// Volume of `poly`.
    pvol: f64,
}

/// Atoms of a cell.
fn cell_atoms(asset: &Asset, idx: &CellIndex, cell: CellId, p: &CollisionParams) -> Vec<Atom> {
    let mesh = idx.boundary_mesh(asset, &[cell]).0;
    if mesh.tris.is_empty() {
        return Vec::new();
    }
    let pts = idx.cell_points(asset, cell);
    let Some(ch) = coacd::hull(&pts) else { return Vec::new() };
    let vol = mesh.signed_volume();
    if ch.volume - vol <= CONVEX_TOL * vol {
        return vec![Atom { cell, pvol: ch.poly.volume(), poly: ch.poly, pts: ch.pts, vol, own: 0.0 }];
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
        merge_cost: coacd::MergeCost::CollisionAware,
        hb: coacd::HbMode::Exact,
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
            Atom { cell, pvol: poly.volume(), poly, pts, vol: c.solid.volume() * h3, own: c.cost * frame.half }
        })
        .collect();
    if atoms.is_empty() {
        return vec![Atom { cell, pvol: ch.poly.volume(), poly: ch.poly, pts: ch.pts, vol, own: 0.0 }];
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

fn frag_geom(asset: &Asset, idx: &CellIndex, f: &Fragment, p: &CollisionParams) -> FragGeom {
    let cells = asset.fragment_cells(f);
    let (mesh, owner) = idx.boundary_mesh(asset, cells);
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

fn atom_piece(a: &Atom, id: u32, cell_nbrs: &[Vec<CellId>]) -> Piece {
    let ch = coacd::Ch { poly: a.poly.clone(), pts: a.pts.clone(), volume: a.pvol };
    let mut pc = Piece::new(ch, a.vol, Vec::new(), vec![id]);
    pc.own = a.own;
    pc.keys = vec![a.cell.0];
    let mut halo: Vec<u32> = cell_nbrs[a.cell.idx()].iter().map(|c| c.0).collect();
    halo.push(a.cell.0);
    halo.sort_unstable();
    halo.dedup();
    pc.halo = halo;
    pc
}

/// Per-fragment merge output: hull pieces and the pieces carried upward.
struct FragOut {
    hulls: Vec<HullRec>,
    carry: Vec<Piece>,
}

/// Compact a merge result for storage: surface samples are reassigned at
/// every level, so they are dropped.
fn frag_out(hulls: Vec<Piece>, carry: Vec<Piece>, max_v: usize) -> FragOut {
    let hulls = hulls.into_iter().map(|pc| HullRec { poly: coacd::limit_vertices(&pc.poly, max_v), atoms: pc.tags }).collect();
    let carry = carry
        .into_iter()
        .map(|mut pc| {
            pc.samples = Vec::new();
            pc
        })
        .collect();
    FragOut { hulls, carry }
}

/// Compute hulls for all fragments. Returns hulls and per-fragment ranges.
pub fn build_hulls(asset: &Asset, p: &CollisionParams) -> (Vec<Hull>, Vec<std::ops::Range<u32>>) {
    let h = &asset.hierarchy;
    let nl = h.levels as usize;
    let want_level = |l: usize| p.levels.is_empty() || p.levels.contains(&(l as u8));
    let profile = std::env::var("FRAC_PROFILE").is_ok();
    // ---- atoms (convex cells, CoACD parts of non-convex cells)
    let ta = std::time::Instant::now();
    let cidx = CellIndex::new(asset);
    let per_cell: Vec<Vec<Atom>> = (0..asset.cells.len() as u32).into_par_iter().map(|c| cell_atoms(asset, &cidx, CellId(c), p)).collect();
    let mut atoms: Vec<Atom> = Vec::new();
    let mut atoms_of_cell: Vec<std::ops::Range<u32>> = Vec::with_capacity(per_cell.len());
    for v in per_cell {
        let s = atoms.len() as u32;
        atoms.extend(v);
        atoms_of_cell.push(s..atoms.len() as u32);
    }
    tadd(&T_ATOMS, ta);
    if profile {
        let cut = atoms_of_cell.iter().filter(|r| r.len() > 1 || (r.len() == 1 && atoms[r.start as usize].own > 0.0)).count();
        eprintln!(
            "collision atoms: {} from {} cells ({} convex, {} decomposed) in {:?} (cpu: {} clips {:.1}s, hull volumes {:.1}s)",
            atoms.len(),
            asset.cells.len(),
            asset.cells.len() - cut,
            cut,
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
    // coarsest level that needs hulls (coarser ones are not computed)
    let coarsest = if p.levels.is_empty() { 0 } else { p.levels.iter().copied().min().unwrap_or(0) as usize };
    for level in (coarsest..nl).rev() {
        let tl = std::time::Instant::now();
        let r = h.level_ranges[level].clone();
        // out-term budget per merge candidate: full on levels with few
        // fragments, reduced on building-scale levels (a function of the
        // data only, so independent of the thread count)
        let max_tri_samples = (32768 / r.len().max(1)).clamp(64, 256);
        let (n_convex, n_particle, n_decomp) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
        // (pieces, ns) per decomposed fragment (FRAC_PROFILE)
        let frag_times: std::sync::Mutex<Vec<(usize, u64)>> = std::sync::Mutex::new(Vec::new());
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
                // fast path: particle candidates and convex fragments get the
                // single hull of their points (convexity test: hull volume
                // within a relative 1e-6 of the exact fragment volume)
                let single = |kind: &AtomicU64| -> Option<FragOut> {
                    let pts: Vec<DVec3> = cells.iter().flat_map(|&c| cidx.cell_points(asset, c)).collect();
                    let ch = coacd::hull(&pts)?;
                    let convex = ch.volume - f.mass.volume <= CONVEX_TOL * f.mass.volume;
                    if !(f.particle_candidate || convex) {
                        return None;
                    }
                    kind.fetch_add(1, AO::Relaxed);
                    let tags: Vec<u32> = cells.iter().flat_map(|&c| atoms_of_cell[c.idx()].clone()).collect();
                    let mut pc = Piece::new(ch, f.mass.volume, Vec::new(), tags);
                    let mut keys: Vec<u32> = cells.iter().map(|c| c.0).collect();
                    keys.sort_unstable();
                    keys.dedup();
                    let mut halo: Vec<u32> = keys.iter().flat_map(|&c| cell_nbrs[c as usize].iter().map(|n| n.0)).chain(keys.iter().copied()).collect();
                    halo.sort_unstable();
                    halo.dedup();
                    pc.keys = keys;
                    pc.halo = halo;
                    let hull = vec![pc];
                    Some(frag_out(hull.clone(), hull, p.max_hull_vertices))
                };
                let is_leaf = level == nl - 1 || f.children.is_empty();
                // (leaf fragments of convex cells are handled exactly by
                // their atoms; the test matters for unions of cells)
                if f.particle_candidate || !is_leaf || cells.len() > 1 {
                    if let Some(o) = single(if f.particle_candidate { &n_particle } else { &n_convex }) {
                        return (fi, o);
                    }
                }
                let convex_cell = is_leaf && cells.len() == 1 && {
                    let r = atoms_of_cell[cells[0].idx()].clone();
                    r.len() == 1 && atoms[r.start as usize].own == 0.0
                };
                if convex_cell { &n_convex } else { &n_decomp }.fetch_add(1, AO::Relaxed);
                let tm = std::time::Instant::now();
                let mut pieces: Vec<Piece> = if level == nl - 1 || f.children.is_empty() {
                    cells.iter().flat_map(|&c| atoms_of_cell[c.idx()].clone()).map(|a| atom_piece(&atoms[a as usize], a, &cell_nbrs)).collect()
                } else {
                    f.children.clone().flat_map(|ch| outs[ch as usize].as_ref().map(|o| o.carry.clone()).unwrap_or_default()).collect()
                };
                // a single piece (e.g. a leaf of one convex cell) is its hull
                if pieces.len() <= 1 {
                    return (fi, frag_out(pieces.clone(), pieces, p.max_hull_vertices));
                }
                let g = frag_geom(asset, &cidx, f, p);
                tadd(&T_MESH, tm);
                let tmg = std::time::Instant::now();
                assign_samples(&g, &mut pieces, &atoms);
                tadd(&T_S_SAMPLES, tmg);
                let t1 = std::time::Instant::now();
                let q = MeshQuery::new(&g.mesh);
                let foreign: Vec<coacd::Foreign> = nbr_frags
                    .get(&fi)
                    .map(|ns| {
                        ns.iter()
                            .flat_map(|&nf| asset.fragment_cells(&h.fragments[nf as usize]).iter().flat_map(|&c| atoms_of_cell[c.idx()].clone()))
                            .map(|a| {
                                let at = &atoms[a as usize];
                                let pv = at.pvol;
                                coacd::Foreign { poly: &at.poly, pts: &at.pts, bbox: Aabb::from_points(at.pts.iter()), scale: if pv > 0.0 { at.vol / pv } else { 1.0 }, pvol: pv }
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let signer = coacd::Signer::new(&g.mesh);
                tadd(&T_S_QUERY, t1);
                let ctx = MergeCtx { q: &q, signer: Some(&signer), spacing: g.spacing, rv_k: 0.3, max_tri_samples, foreign, intrusion_k: p.intrusion_k, upstream_density: None, seed: p.seed, batch: 1 };
                // Pieces keep their own costs from the finer level (atoms: from
                // the cut search): a fragment's surface is a subset of its
                // children's, so those costs bound the costs at this level.
                // adjacent: a cell of one piece is (or neighbours) a cell of
                // the other
                let adjacent = |a: &Piece, b: &Piece| -> bool { a.bbox.expanded(1e-6 * g.thresh.max(1e-12)).overlaps(&b.bbox) && coacd::sorted_intersects(&a.halo, &b.keys) };
                let budget = p.max_hulls.max(1);
                // very many pieces (large coarse fragments): cheap pre-merge
                let limit = budget.saturating_mul(16).max(64);
                let t1 = std::time::Instant::now();
                let pieces = if pieces.len() > limit { coacd::coarsen(pieces, limit, &adjacent) } else { pieces };
                tadd(&T_S_COARSEN, t1);
                let n_in = pieces.len();
                tadd(&T_SETUP, tmg);
                let res = coacd::greedy_merge(pieces, &ctx, g.thresh, budget, p.carry_cap.max(budget), &adjacent);
                tadd(&T_MERGE, tmg);
                if profile {
                    frag_times.lock().unwrap().push((n_in, tm.elapsed().as_nanos() as u64));
                }
                (fi, frag_out(res.pieces, res.carry, p.max_hull_vertices))
            })
            .collect();
        if profile {
            // decomposed-fragment CPU time by input piece count
            let ft = frag_times.lock().unwrap();
            let mut buckets: BTreeMap<usize, (usize, u64, u64)> = BTreeMap::new();
            for &(n, t) in ft.iter() {
                let b = match n {
                    0..=4 => 4,
                    5..=8 => 8,
                    9..=16 => 16,
                    17..=32 => 32,
                    33..=64 => 64,
                    65..=128 => 128,
                    _ => usize::MAX,
                };
                let e = buckets.entry(b).or_default();
                e.0 += 1;
                e.1 += t;
                e.2 = e.2.max(t);
            }
            for (b, (n, t, mx)) in buckets {
                eprintln!("    pieces <= {b}: {n} fragments, cpu {:.1}s, max {:.2}s", t as f64 * 1e-9, mx as f64 * 1e-9);
            }
            eprintln!(
                "    setup cpu so far: {:.1}s (samples {:.1}s, queries + neighbours {:.1}s, coarsening {:.1}s)",
                T_SETUP.load(AO::Relaxed) as f64 * 1e-9,
                T_S_SAMPLES.load(AO::Relaxed) as f64 * 1e-9,
                T_S_QUERY.load(AO::Relaxed) as f64 * 1e-9,
                T_S_COARSEN.load(AO::Relaxed) as f64 * 1e-9
            );
            eprintln!(
                "    pair cpu so far: hull {:.1}s in {:.1}s cover {:.1}s out {:.1}s ({} stage1, {} stage2, {} evals)",
                coacd::T_HULLC.load(AO::Relaxed) as f64 * 1e-9,
                coacd::T_IN.load(AO::Relaxed) as f64 * 1e-9,
                coacd::T_COV.load(AO::Relaxed) as f64 * 1e-9,
                coacd::T_OUT.load(AO::Relaxed) as f64 * 1e-9,
                coacd::N_PAIR.load(AO::Relaxed),
                coacd::N_STAGE2.load(AO::Relaxed),
                coacd::N_EVAL.load(AO::Relaxed)
            );
        }
        if profile {
            let (nh, nc) = results.iter().fold((0, 0), |a, (_, o)| (a.0 + o.hulls.len(), a.1 + o.carry.len()));
            let maxv = results.iter().flat_map(|(_, o)| o.carry.iter().map(|p| p.pts.len())).max().unwrap_or(0);
            eprintln!(
                "  level {level}: {} fragments ({} convex fast path, {} particles, {} decomposed) -> {nh} hulls, {nc} carried (max {maxv} hull vertices)",
                results.len(),
                n_convex.load(AO::Relaxed),
                n_particle.load(AO::Relaxed),
                n_decomp.load(AO::Relaxed)
            );
        }
        for (fi, o) in results {
            outs[fi as usize] = Some(o);
            // the children's carried pieces are no longer needed
            for ch in h.fragments[fi as usize].children.clone() {
                if let Some(c) = outs[ch as usize].as_mut() {
                    c.carry = Vec::new();
                }
            }
        }
        if profile {
            eprintln!(
                "collision level {level}: {:?} (cumulative cpu: atoms {:.1}s, merge {:.1}s [{} merged hulls], mesh {:.1}s)",
                tl.elapsed(),
                T_ATOMS.load(AO::Relaxed) as f64 * 1e-9,
                T_MERGE.load(AO::Relaxed) as f64 * 1e-9,
                coacd::N_PAIR.load(AO::Relaxed),
                T_MESH.load(AO::Relaxed) as f64 * 1e-9
            );
        }
    }
    // ---- non-overlap enforcement per level, then margin shrink
    let tn = std::time::Instant::now();
    let mut hull_recs: Vec<Vec<HullRec>> = outs
        .into_iter()
        .map(|o| o.map(|o| o.hulls).unwrap_or_default())
        .collect();
    for level in 0..nl {
        if want_level(level) {
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
    // output (per fragment in parallel, concatenated in fragment order)
    let to = std::time::Instant::now();
    let per_frag: Vec<Vec<Hull>> = hull_recs
        .par_iter()
        .enumerate()
        .map(|(fi, hp)| {
            let mut hulls = Vec::new();
            let level = h.fragments[fi].level as usize;
            if !want_level(level) {
                return hulls;
            }
            for rec in hp {
                if rec.poly.is_empty() {
                    continue;
                }
                let t1 = std::time::Instant::now();
                let s = rec.poly.shrunk(p.margin);
                let s = if s.is_empty() { rec.poly.clone() } else { s };
                tadd(&T_O_SHRINK, t1);
                // clipping and shrinking can add vertices: re-apply the cap
                // (inner approximation, keeps the separation)
                let t1 = std::time::Instant::now();
                let s = cap_shrunk(&rec.poly, &s, p.max_hull_vertices);
                let mut hull = to_hull(FragmentId(fi as u32), &s);
                tadd(&T_O_LIMIT, t1);
                // sliver faces (near-duplicate vertices from clipping) have
                // unreliable planes: rebuild from significant vertices only
                let t1 = std::time::Instant::now();
                let scale = s.scale().max(1e-300);
                for eps in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3] {
                    if coacd::stored_hull_is_convex(&hull.vertices, &hull.faces) {
                        break;
                    }
                    N_O_REBUILD.fetch_add(1, AO::Relaxed);
                    hull = to_hull(FragmentId(fi as u32), &coacd::greedy_subset(&s, p.max_hull_vertices, eps * scale));
                }
                tadd(&T_O_CONVEX, t1);
                hulls.push(hull);
            }
            hulls
        })
        .collect();
    let mut hulls = Vec::new();
    let mut ranges = vec![0..0; h.fragments.len()];
    for (fi, v) in per_frag.into_iter().enumerate() {
        let start = hulls.len() as u32;
        hulls.extend(v);
        ranges[fi] = start..hulls.len() as u32;
    }
    if profile {
        eprintln!(
            "collision output: {:?} (cpu: shrink {:.1}s, vertex cap {:.1}s, convexity {:.1}s, {} rebuilds)",
            to.elapsed(),
            T_O_SHRINK.load(AO::Relaxed) as f64 * 1e-9,
            T_O_LIMIT.load(AO::Relaxed) as f64 * 1e-9,
            T_O_CONVEX.load(AO::Relaxed) as f64 * 1e-9,
            N_O_REBUILD.load(AO::Relaxed)
        );
    }
    (hulls, ranges)
}

/// Vertex cap of a shrunk hull. Shrinking splits every vertex where more
/// than three faces meet into nearby copies; each group of copies (by the
/// nearest vertex of the unshrunk hull) is replaced by its centroid, which
/// lies inside the shrunk hull, so the result is an inner approximation
/// that keeps the margin and the separation. Falls back to
/// [`coacd::limit_vertices`] when that is still above the cap.
fn cap_shrunk(orig: &ConvexPolytope, s: &ConvexPolytope, max_v: usize) -> ConvexPolytope {
    let sv = s.vertices();
    if max_v == 0 || sv.len() <= max_v || s.is_empty() {
        return s.clone();
    }
    let ov = orig.vertices();
    let mut sum: Vec<(DVec3, usize)> = vec![(DVec3::ZERO, 0); ov.len()];
    for &v in &sv {
        let k = (0..ov.len()).min_by(|&a, &b| (ov[a] - v).length_squared().total_cmp(&(ov[b] - v).length_squared()).then(a.cmp(&b))).unwrap_or(0);
        sum[k].0 += v;
        sum[k].1 += 1;
    }
    let cents: Vec<DVec3> = sum.iter().filter(|x| x.1 > 0).map(|x| x.0 / x.1 as f64).collect();
    match coacd::hull(&cents) {
        Some(h) if h.pts.len() <= max_v => h.poly,
        Some(h) => coacd::limit_vertices(&h.poly, max_v),
        None => coacd::limit_vertices(s, max_v),
    }
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
            let full = at.pvol;
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
        } else {
            clipped_volume(&atoms[p.id as usize].poly, n, t, above)
        };
        v += removed * p.frac;
    }
    v
}

/// Volume of the part of a convex polytope with `n·x >= t` (`above`) or
/// `n·x <= t` (unit `n`): the clipped face polygons summed as cones from a
/// point on the plane, so the cap face (which contributes nothing) is
/// never built.
fn clipped_volume(poly: &ConvexPolytope, n: DVec3, t: f64, above: bool) -> f64 {
    let Some(v0) = poly.faces.first().and_then(|f| f.1.first().copied()) else { return 0.0 };
    let sgn = if above { -1.0 } else { 1.0 };
    let r = v0 - n * (n.dot(v0) - t);
    let mut vol = 0.0;
    let mut buf: Vec<DVec3> = Vec::new();
    for (_, f) in &poly.faces {
        buf.clear();
        let m = f.len();
        for i in 0..m {
            let (a, b) = (f[i], f[(i + 1) % m]);
            let (da, db) = (sgn * (n.dot(a) - t), sgn * (n.dot(b) - t));
            if da <= 0.0 {
                buf.push(a);
            }
            if (da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0) {
                buf.push(a + (b - a) * (da / (da - db)));
            }
        }
        for k in 1..buf.len().saturating_sub(1) {
            vol += (buf[0] - r).dot((buf[k] - r).cross(buf[k + 1] - r));
        }
    }
    (vol / 6.0).max(0.0)
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
    // a face plane of either hull with the other hull on its outer side
    // (touching allowed): no volume in common
    let eps = 1e-12 * da.bbox.union(&db.bbox).diagonal().max(1e-300);
    let separated = |p: &ConvexPolytope, other: &[DVec3]| p.faces.iter().any(|(h, _)| other.iter().all(|q| h.dist(*q) >= -eps));
    if separated(pb, &da.verts) || separated(pa, &db.verts) {
        return None;
    }
    let ov = coacd::clip_all_cutting(pa, &pb.halfspaces());
    if ov.is_empty() || ov.volume() <= 0.0 { None } else { Some(ov) }
}

/// Plane (normal from A to B, offset) separating two hulls with the least
/// loss of their own atoms; `None` when they do not overlap.
fn separating_plane(atoms: &[Atom], ha: &HullRec, da: &HullData, hb: &HullRec, db: &HullData, extra: &[DVec3]) -> Option<(DVec3, f64)> {
    let ov = overlap_of(&ha.poly, da, &hb.poly, db)?;
    Some(match best_separation(atoms, ha, da, hb, db, &ov, extra) {
        Some((n, t, _)) => (n, t),
        None => {
            let n = extra.first().copied().unwrap_or(DVec3::X);
            (n, n.dot(ov.volume_integrals().com()))
        }
    })
}

/// Separate overlapping hulls of neighbouring fragments at one level.
///
/// Overlapping hull pairs are detected in parallel and resolved in bond
/// order; independent pairs (touching hulls no earlier pending pair
/// touches) are resolved concurrently, which gives the same result as the
/// sequential order.
fn separate_level(asset: &Asset, level: u8, recs: &mut [Vec<HullRec>], atoms: &[Atom]) {
    let t0 = std::time::Instant::now();
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
                if extra.len() > 8 {
                    break;
                }
            }
            extra
        })
        .collect();
    let mut data: Vec<Vec<HullData>> = recs.par_iter().map(|v| v.iter().map(|r| hull_data(&r.poly)).collect()).collect();
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
    let t_detect = t0.elapsed();
    let mut rounds = 0usize;
    // resolution in dependency-respecting parallel batches
    while !pending.is_empty() {
        rounds += 1;
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
                separating_plane(atoms, &recs[ia][x], &data[ia][x], &recs[ib][y], &data[ib][y], &extras[bi])
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
    if std::env::var("FRAC_PROFILE").is_ok() {
        eprintln!("  level {level}: separation {:?} (detection {:?}, {rounds} rounds)", t0.elapsed(), t_detect);
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn atom_of(pts: &[DVec3]) -> Option<Atom> {
        let ch = coacd::hull(pts)?;
        Some(Atom { cell: CellId(0), vol: ch.volume, pvol: ch.poly.volume(), poly: ch.poly, pts: ch.pts, own: 0.0 })
    }

    /// Separate two fragment hulls built from atoms and return the clipped
    /// polytopes.
    fn separate(atoms: &[Atom], ia: Vec<u32>, ib: Vec<u32>, normal: DVec3) -> (ConvexPolytope, ConvexPolytope, f64) {
        let pa: Vec<DVec3> = ia.iter().flat_map(|&a| atoms[a as usize].pts.clone()).collect();
        let pb: Vec<DVec3> = ib.iter().flat_map(|&a| atoms[a as usize].pts.clone()).collect();
        let ha = HullRec { poly: coacd::hull(&pa).unwrap().poly, atoms: ia.clone() };
        let hb = HullRec { poly: coacd::hull(&pb).unwrap().poly, atoms: ib.clone() };
        let (da, db) = (hull_data(&ha.poly), hull_data(&hb.poly));
        match separating_plane(atoms, &ha, &da, &hb, &db, &[normal]) {
            Some((n, t)) => {
                let a = ha.poly.clip(&HalfSpace { n, d: t });
                let b = hb.poly.clip(&HalfSpace { n: -n, d: -t });
                // own atom volume kept by each hull
                let keep = |h: &ConvexPolytope, ids: &[u32]| -> f64 { ids.iter().map(|&i| atoms[i as usize].poly.intersection_volume(h)).sum() };
                let total: f64 = ia.iter().chain(&ib).map(|&i| atoms[i as usize].vol).sum();
                let kept = keep(&a, &ia) + keep(&b, &ib);
                (a, b, kept / total)
            }
            None => (ha.poly, hb.poly, 1.0),
        }
    }

    fn cube(c: DVec3, h: f64) -> Vec<DVec3> {
        (0..8).map(|i| c + DVec3::new(if i & 1 == 0 { -h } else { h }, if i & 2 == 0 { -h } else { h }, if i & 4 == 0 { -h } else { h })).collect()
    }

    #[test]
    fn clipped_volume_matches_clip() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        for _ in 0..200 {
            let c = DVec3::new(rng.gen_range(-5.0..5.0), rng.gen_range(-5.0..5.0), rng.gen_range(-5.0..5.0));
            let pts: Vec<DVec3> = (0..rng.gen_range(4..40)).map(|_| c + DVec3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0))).collect();
            let Some(ch) = coacd::hull(&pts) else { continue };
            let n = DVec3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)).normalize();
            let t = n.dot(c) + rng.gen_range(-1.0..1.0);
            let above = ch.poly.clip(&HalfSpace { n: -n, d: -t }).volume();
            let below = ch.poly.clip(&HalfSpace { n, d: t }).volume();
            let (a2, b2) = (clipped_volume(&ch.poly, n, t, true), clipped_volume(&ch.poly, n, t, false));
            assert!((above - a2).abs() <= 1e-9 * ch.volume.max(1.0), "above {above} vs {a2}");
            assert!((below - b2).abs() <= 1e-9 * ch.volume.max(1.0), "below {below} vs {b2}");
            // polygon_minus: face area outside a box equals area minus the overlap
            let bx = ConvexPolytope::from_box(c - DVec3::splat(0.3), c + DVec3::splat(0.3));
            for (h, f) in &ch.poly.faces {
                let area = |p: &[DVec3]| -> f64 { (1..p.len() - 1).map(|k| (p[k] - p[0]).cross(p[k + 1] - p[0]).dot(h.n) * 0.5).sum() };
                let outside: f64 = coacd::polygon_minus(f, &[&bx], 1e-12).iter().map(|p| area(p)).sum();
                // the part inside the box: the face clipped by the box planes
                let mut inside = f.clone();
                for (bh, _) in &bx.faces {
                    let mut out = Vec::new();
                    for i in 0..inside.len() {
                        let (a, b) = (inside[i], inside[(i + 1) % inside.len()]);
                        let (da, db) = (bh.dist(a), bh.dist(b));
                        if da <= 0.0 {
                            out.push(a);
                        }
                        if (da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0) {
                            out.push(a + (b - a) * (da / (da - db)));
                        }
                    }
                    inside = out;
                    if inside.len() < 3 {
                        inside.clear();
                        break;
                    }
                }
                let ins = if inside.len() >= 3 { area(&inside) } else { 0.0 };
                assert!((outside + ins - area(f)).abs() <= 1e-9, "face area {} = {outside} + {ins}", area(f));
            }
        }
    }

    #[test]
    fn interleaved_l_shapes_separate_with_small_loss() {
        // A: L of three unit cubes, B: the complementary L; their hulls
        // overlap in the notch region
        let a = [cube(DVec3::new(0.5, 0.5, 0.5), 0.5), cube(DVec3::new(1.5, 0.5, 0.5), 0.5), cube(DVec3::new(0.5, 1.5, 0.5), 0.5)];
        let b = [cube(DVec3::new(1.5, 1.5, 0.5), 0.5), cube(DVec3::new(2.5, 1.5, 0.5), 0.5), cube(DVec3::new(1.5, 2.5, 0.5), 0.5)];
        let atoms: Vec<Atom> = a.iter().chain(b.iter()).map(|p| atom_of(p).unwrap()).collect();
        let (pa, pb, kept) = separate(&atoms, vec![0, 1, 2], vec![3, 4, 5], DVec3::new(1.0, 1.0, 0.0).normalize());
        assert!(pa.intersection_volume(&pb) <= 1e-12);
        // the diagonal plane x + y = 2 removes 1/4 of a cube from each side
        assert!(kept > 0.9, "kept {kept}");
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        /// Random fragment pairs (atoms: random convex blobs on either side
        /// of a jagged interface, so the fragments are disjoint but their
        /// hulls overlap) never overlap after separation.
        #[test]
        fn separation_removes_overlap(seed in 0u64..1_000_000, na in 1usize..4, nb in 1usize..4, conv in proptest::bool::ANY) {
            use rand::{Rng, SeedableRng};
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
            let mut atoms = Vec::new();
            let blob = |side: f64, k: usize, rng: &mut rand_chacha::ChaCha8Rng| -> Option<Atom> {
                // a box cell in a 2-column grid; side selects the column,
                // rows are jagged by a random shift
                let y0 = k as f64 + rng.gen_range(-0.3..0.3);
                let x0 = if side < 0.0 { rng.gen_range(-1.0..-0.2) } else { rng.gen_range(0.2..1.0) };
                let w = rng.gen_range(0.4..1.2);
                let lo = DVec3::new(if side < 0.0 { x0 - w } else { x0 - 0.2 }, y0, 0.0);
                let hi = DVec3::new(if side < 0.0 { x0 + 0.2 } else { x0 + w }, y0 + 0.9, rng.gen_range(0.5..1.5));
                let mut pts = cube((lo + hi) * 0.5, 0.5);
                for p in pts.iter_mut() {
                    *p = (lo + hi) * 0.5 + (*p - (lo + hi) * 0.5) * 2.0 * (hi - lo) * 0.5;
                }
                if !conv {
                    for p in pts.iter_mut() {
                        *p += DVec3::new(rng.gen_range(-0.1..0.1), rng.gen_range(-0.1..0.1), rng.gen_range(-0.1..0.1));
                    }
                }
                atom_of(&pts)
            };
            for k in 0..na {
                if let Some(a) = blob(-1.0, k, &mut rng) { atoms.push(a); }
            }
            let n_a = atoms.len() as u32;
            for k in 0..nb {
                if let Some(a) = blob(1.0, k, &mut rng) { atoms.push(a); }
            }
            prop_assume!(n_a > 0 && (atoms.len() as u32) > n_a);
            let ia: Vec<u32> = (0..n_a).collect();
            let ib: Vec<u32> = (n_a..atoms.len() as u32).collect();
            let (pa, pb, _) = separate(&atoms, ia, ib, DVec3::X);
            let ov = pa.intersection_volume(&pb);
            prop_assert!(ov <= 1e-12, "overlap {ov}");
            // after the margin shrink the hulls are disjoint with a gap
            let (sa, sb) = (pa.shrunk(1e-3), pb.shrunk(1e-3));
            if !sa.is_empty() && !sb.is_empty() {
                prop_assert!(sa.intersection_volume(&sb) == 0.0);
            }
        }
    }
}
