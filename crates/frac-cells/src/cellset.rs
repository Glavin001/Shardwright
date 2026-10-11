//! Post-processing of clipped complexes into final cells: island splitting
//! (disconnected pieces of one convex cell become separate cells), cavity
//! attachment, mass properties, sliver merging and analysis-cluster
//! connectivity.

use crate::clip::{ClipOutput, ExtPolyOut, PatchOut};
use frac_geom::integrals::{VolumeIntegrals, sym_eigen3};
use frac_geom::{Aabb, DVec3};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
pub struct LocalCell {
    /// Complex cells merged into this cell.
    pub complex_cells: Vec<u32>,
    /// Analysis cluster (connected), assigned by [`CellSet::finalize_clusters`].
    pub cluster: u32,
    /// Physical unit (e.g. brick) for interface typing.
    pub unit: u32,
    pub vi: VolumeIntegrals,
    pub aabb: Aabb,
    pub thickness_ratio: f64,
}

#[derive(Clone, Debug, Default)]
pub struct CellSet {
    pub verts: Vec<DVec3>,
    /// Exterior polygons; `cell` is a local cell index.
    pub ext: Vec<ExtPolyOut>,
    /// Interface patches; `cells` are local cell indices (neg, pos).
    pub patches: Vec<PatchOut>,
    pub cells: Vec<LocalCell>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct CellSetParams {
    pub min_cell_volume: f64,
    pub min_thickness_ratio: f64,
    /// Smallest principal extent of the whole component (for thin parts).
    pub component_min_extent: f64,
}

fn principal_extents(vi: &VolumeIntegrals) -> DVec3 {
    if vi.volume <= 0.0 {
        return DVec3::ZERO;
    }
    let c = vi.com();
    let cov = (vi.second - frac_geom::polygon::outer(c, c) * vi.volume) / vi.volume;
    let (ev, _) = sym_eigen3(&cov);
    // extent of a uniform box with this covariance: L = sqrt(12 * var)
    DVec3::new(
        (12.0 * ev.x.max(0.0)).sqrt(),
        (12.0 * ev.y.max(0.0)).sqrt(),
        (12.0 * ev.z.max(0.0)).sqrt(),
    )
}

/// Thickness ratio: smallest principal extent over the largest one, capped
/// by the component's own thickness so that thin parts (panes, panels) are
/// not flagged as slivers.
pub fn thickness_ratio(vi: &VolumeIntegrals, comp_min_extent: f64) -> f64 {
    let e = principal_extents(vi);
    let denom = e.z.min(comp_min_extent.max(1e-300));
    if denom <= 0.0 {
        0.0
    } else {
        (e.x / denom).min(1.0)
    }
}

struct Uf(Vec<u32>);
impl Uf {
    fn new(n: usize) -> Self {
        Uf((0..n as u32).collect())
    }
    fn find(&mut self, x: u32) -> u32 {
        let mut r = x;
        while self.0[r as usize] != r {
            r = self.0[r as usize];
        }
        let mut y = x;
        while self.0[y as usize] != r {
            let n = self.0[y as usize];
            self.0[y as usize] = r;
            y = n;
        }
        r
    }
    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            let (lo, hi) = (ra.min(rb), ra.max(rb));
            self.0[hi as usize] = lo;
        }
    }
}

/// Polygon reference within a cell: exterior polygon or patch side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum PRef {
    Ext(u32),
    /// patch index, true = reversed (cell is on the positive side)
    Patch(u32, bool),
}

impl CellSet {
    /// Build cells from a clip result. `unit_of[complex_cell]` and
    /// `cluster_of[complex_cell]` label the complex cells.
    pub fn from_clip(
        out: ClipOutput,
        unit_of: &[u32],
        cluster_of: &[u32],
        params: &CellSetParams,
    ) -> CellSet {
        let ClipOutput {
            verts,
            keys: _,
            ext,
            patches,
            warnings,
        } = out;
        let mut warnings = warnings;
        // polygons per complex cell
        let mut per: BTreeMap<u32, Vec<PRef>> = BTreeMap::new();
        for (i, e) in ext.iter().enumerate() {
            per.entry(e.cell).or_default().push(PRef::Ext(i as u32));
        }
        for (i, p) in patches.iter().enumerate() {
            per.entry(p.cells[0])
                .or_default()
                .push(PRef::Patch(i as u32, false));
            per.entry(p.cells[1])
                .or_default()
                .push(PRef::Patch(i as u32, true));
        }
        let boundary_edges = |r: &PRef| -> Vec<(u32, u32)> {
            match *r {
                PRef::Ext(i) => {
                    let v = &ext[i as usize].verts;
                    (0..v.len()).map(|k| (v[k], v[(k + 1) % v.len()])).collect()
                }
                PRef::Patch(i, rev) => {
                    let mut out = Vec::new();
                    for l in &patches[i as usize].loops {
                        for k in 0..l.len() {
                            let (a, b) = (l[k], l[(k + 1) % l.len()]);
                            out.push(if rev { (b, a) } else { (a, b) });
                        }
                    }
                    out
                }
            }
        };
        // island split
        let mut cells: Vec<LocalCell> = Vec::new();
        let mut assign_ext = vec![u32::MAX; ext.len()];
        let mut assign_patch = vec![[u32::MAX; 2]; patches.len()];
        for (&cc, refs) in per.iter() {
            let n = refs.len();
            let mut uf = Uf::new(n);
            let mut owner: BTreeMap<(u32, u32), u32> = BTreeMap::new();
            for (k, r) in refs.iter().enumerate() {
                for (a, b) in boundary_edges(r) {
                    let key = (a.min(b), a.max(b));
                    if let Some(&o) = owner.get(&key) {
                        uf.union(o, k as u32);
                    } else {
                        owner.insert(key, k as u32);
                    }
                }
            }
            let mut groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
            for k in 0..n {
                groups.entry(uf.find(k as u32)).or_default().push(k);
            }
            // shells with volume
            let mut shells: Vec<(Vec<usize>, VolumeIntegrals)> = groups
                .into_values()
                .map(|g| {
                    let vi = integrate(&verts, &ext, &patches, g.iter().map(|&k| refs[k]));
                    (g, vi)
                })
                .collect();
            // attach cavities (negative shells) to the smallest containing positive shell
            let (pos, neg): (Vec<_>, Vec<_>) = shells.drain(..).partition(|s| s.1.volume >= 0.0);
            let mut pos = pos;
            for (g, vi) in neg {
                let probe = first_vertex(&ext, &patches, refs[g[0]]);
                let p = verts[probe as usize];
                let mut best: Option<usize> = None;
                for (pi, (pg, pvi)) in pos.iter().enumerate() {
                    if shell_contains(&verts, &ext, &patches, pg.iter().map(|&k| refs[k]), p)
                        && best.map(|b| pvi.volume < pos[b].1.volume).unwrap_or(true)
                    {
                        best = Some(pi);
                    }
                }
                // degenerate (≈ zero-volume) shells that are not inside any
                // positive shell join the largest shell of this cell, or
                // stand alone (and are merged as slivers later)
                let best = best.or_else(|| {
                    (0..pos.len())
                        .max_by(|&a, &b| pos[a].1.volume.partial_cmp(&pos[b].1.volume).unwrap())
                });
                match best {
                    Some(b) => {
                        pos[b].0.extend(g);
                        pos[b].1.add(&vi);
                    }
                    None => {
                        warnings.push(format!(
                            "cell {cc}: isolated degenerate shell (volume {:e})",
                            vi.volume
                        ));
                        pos.push((g, vi));
                    }
                }
            }
            for (g, vi) in pos {
                let id = cells.len() as u32;
                let mut aabb = Aabb::EMPTY;
                for &k in &g {
                    match refs[k] {
                        PRef::Ext(i) => {
                            assign_ext[i as usize] = id;
                            for &v in &ext[i as usize].verts {
                                aabb.grow(verts[v as usize]);
                            }
                        }
                        PRef::Patch(i, rev) => {
                            assign_patch[i as usize][rev as usize] = id;
                            for l in &patches[i as usize].loops {
                                for &v in l {
                                    aabb.grow(verts[v as usize]);
                                }
                            }
                        }
                    }
                }
                let tr = thickness_ratio(&vi, params.component_min_extent);
                cells.push(LocalCell {
                    complex_cells: vec![cc],
                    cluster: cluster_of.get(cc as usize).copied().unwrap_or(0),
                    unit: unit_of.get(cc as usize).copied().unwrap_or(cc),
                    vi,
                    aabb,
                    thickness_ratio: tr,
                });
            }
        }
        let ext: Vec<ExtPolyOut> = ext
            .into_iter()
            .enumerate()
            .map(|(i, mut e)| {
                e.cell = assign_ext[i];
                e
            })
            .collect();
        let patches: Vec<PatchOut> = patches
            .into_iter()
            .enumerate()
            .map(|(i, mut p)| {
                p.cells = assign_patch[i];
                p
            })
            .collect();
        let mut cs = CellSet {
            verts,
            ext,
            patches,
            cells,
            warnings,
        };
        cs.merge_slivers(params);
        cs
    }

    /// Shared patch area between pairs of cells.
    pub fn adjacency(&self) -> BTreeMap<(u32, u32), f64> {
        let mut m: BTreeMap<(u32, u32), f64> = BTreeMap::new();
        for p in &self.patches {
            let [a, b] = p.cells;
            if a == b {
                continue;
            }
            *m.entry((a.min(b), a.max(b))).or_default() += p.area.abs();
        }
        m
    }

    /// Merge cells below the volume/thickness thresholds into the neighbor
    /// with the largest shared interface. Merged cells are unions (no longer
    /// convex); internal patches are removed.
    pub fn merge_slivers(&mut self, params: &CellSetParams) {
        for _round in 0..8 {
            let adj = self.adjacency();
            let mut nbrs: BTreeMap<u32, Vec<(u32, f64)>> = BTreeMap::new();
            for (&(a, b), &w) in &adj {
                nbrs.entry(a).or_default().push((b, w));
                nbrs.entry(b).or_default().push((a, w));
            }
            let mut order: Vec<u32> = (0..self.cells.len() as u32)
                .filter(|&c| {
                    let cell = &self.cells[c as usize];
                    !cell.complex_cells.is_empty()
                        && (cell.vi.volume < params.min_cell_volume
                            || cell.thickness_ratio < params.min_thickness_ratio)
                })
                .collect();
            if order.is_empty() {
                break;
            }
            order.sort_by(|&a, &b| {
                self.cells[a as usize]
                    .vi
                    .volume
                    .partial_cmp(&self.cells[b as usize].vi.volume)
                    .unwrap()
                    .then(a.cmp(&b))
            });
            let mut target: BTreeMap<u32, u32> = BTreeMap::new();
            let mut touched: BTreeSet<u32> = BTreeSet::new();
            for c in order {
                if touched.contains(&c) {
                    continue;
                }
                let best = nbrs.get(&c).and_then(|v| {
                    v.iter()
                        .filter(|(n, _)| !touched.contains(n))
                        .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap().then(y.0.cmp(&x.0)))
                        .map(|x| x.0)
                });
                if let Some(n) = best {
                    target.insert(c, n);
                    touched.insert(c);
                    touched.insert(n);
                } else {
                    self.warnings
                        .push(format!("sliver cell {c} has no neighbor to merge into"));
                    // mark to avoid looping
                    touched.insert(c);
                }
            }
            if target.is_empty() {
                break;
            }
            self.apply_merges(&target, params);
        }
        self.compact();
    }

    /// Merge `from -> into` pairs.
    fn apply_merges(&mut self, target: &BTreeMap<u32, u32>, params: &CellSetParams) {
        for (&from, &into) in target {
            let src = std::mem::replace(&mut self.cells[from as usize].complex_cells, Vec::new());
            let vi = self.cells[from as usize].vi;
            let bb = self.cells[from as usize].aabb;
            let dst = &mut self.cells[into as usize];
            dst.complex_cells.extend(src);
            dst.complex_cells.sort_unstable();
            dst.vi.add(&vi);
            dst.aabb = dst.aabb.union(&bb);
            dst.thickness_ratio = thickness_ratio(&dst.vi, params.component_min_extent);
            self.cells[from as usize].vi = VolumeIntegrals::default();
        }
        let remap = |c: u32| -> u32 { *target.get(&c).unwrap_or(&c) };
        for e in self.ext.iter_mut() {
            e.cell = remap(e.cell);
        }
        for p in self.patches.iter_mut() {
            p.cells = [remap(p.cells[0]), remap(p.cells[1])];
        }
        self.patches.retain(|p| p.cells[0] != p.cells[1]);
    }

    /// Merge each group of cells into one cell (e.g. forbidden zones).
    pub fn merge_groups(&mut self, groups: &[Vec<u32>], params: &CellSetParams) {
        let mut target: BTreeMap<u32, u32> = BTreeMap::new();
        for g in groups {
            if g.len() < 2 {
                continue;
            }
            let into = *g.iter().min().unwrap();
            for &c in g {
                if c != into {
                    target.insert(c, into);
                }
            }
        }
        if target.is_empty() {
            return;
        }
        self.apply_merges(&target, params);
        self.compact();
    }

    /// Drop empty cells and renumber.
    fn compact(&mut self) {
        let mut map = vec![u32::MAX; self.cells.len()];
        let mut cells = Vec::new();
        for (i, c) in self.cells.iter().enumerate() {
            if !c.complex_cells.is_empty() {
                map[i] = cells.len() as u32;
                cells.push(c.clone());
            }
        }
        for e in self.ext.iter_mut() {
            e.cell = map[e.cell as usize];
        }
        for p in self.patches.iter_mut() {
            p.cells = [map[p.cells[0] as usize], map[p.cells[1] as usize]];
        }
        self.cells = cells;
    }

    /// Make analysis clusters connected: split each cluster into connected
    /// components (via patches between its cells) and renumber canonically.
    pub fn finalize_clusters(&mut self) -> u32 {
        let n = self.cells.len();
        let mut uf = Uf::new(n);
        for p in &self.patches {
            let [a, b] = p.cells;
            if self.cells[a as usize].cluster == self.cells[b as usize].cluster {
                uf.union(a, b);
            }
        }
        let mut label: BTreeMap<u32, u32> = BTreeMap::new();
        for c in 0..n {
            let r = uf.find(c as u32);
            let next = label.len() as u32;
            let l = *label.entry(r).or_insert(next);
            self.cells[c].cluster = l;
        }
        label.len() as u32
    }

    /// Reorder cells (e.g. by Morton code) with `order[new] = old`.
    pub fn reorder(&mut self, order: &[u32]) {
        let mut inv = vec![0u32; order.len()];
        for (new, &old) in order.iter().enumerate() {
            inv[old as usize] = new as u32;
        }
        self.cells = order
            .iter()
            .map(|&o| self.cells[o as usize].clone())
            .collect();
        for e in self.ext.iter_mut() {
            e.cell = inv[e.cell as usize];
        }
        for p in self.patches.iter_mut() {
            p.cells = [inv[p.cells[0] as usize], inv[p.cells[1] as usize]];
        }
    }

    /// Oriented boundary polygons of each cell (patch triangles flipped for
    /// the positive-side cell).
    pub fn cell_polygons(&self) -> Vec<Vec<Vec<u32>>> {
        let mut out: Vec<Vec<Vec<u32>>> = vec![Vec::new(); self.cells.len()];
        for e in &self.ext {
            out[e.cell as usize].push(e.verts.clone());
        }
        for p in &self.patches {
            for t in &p.tris {
                out[p.cells[0] as usize].push(t.to_vec());
                out[p.cells[1] as usize].push(vec![t[0], t[2], t[1]]);
            }
        }
        out
    }
}

fn first_vertex(ext: &[ExtPolyOut], patches: &[PatchOut], r: PRef) -> u32 {
    match r {
        PRef::Ext(i) => ext[i as usize].verts[0],
        PRef::Patch(i, _) => patches[i as usize].loops[0][0],
    }
}

fn integrate(
    verts: &[DVec3],
    ext: &[ExtPolyOut],
    patches: &[PatchOut],
    refs: impl Iterator<Item = PRef>,
) -> VolumeIntegrals {
    let refs: Vec<PRef> = refs.collect();
    let r = verts[first_vertex(ext, patches, refs[0]) as usize];
    let mut vi = VolumeIntegrals::default();
    for pr in refs {
        match pr {
            PRef::Ext(i) => {
                let pts: Vec<DVec3> = ext[i as usize]
                    .verts
                    .iter()
                    .map(|&v| verts[v as usize])
                    .collect();
                vi.add_polygon(r, &pts);
            }
            PRef::Patch(i, rev) => {
                for t in &patches[i as usize].tris {
                    let (a, b, c) = (
                        verts[t[0] as usize],
                        verts[t[1] as usize],
                        verts[t[2] as usize],
                    );
                    if rev {
                        vi.add_tet(r, a, c, b);
                    } else {
                        vi.add_tet(r, a, b, c);
                    }
                }
            }
        }
    }
    vi
}

fn shell_contains(
    verts: &[DVec3],
    ext: &[ExtPolyOut],
    patches: &[PatchOut],
    refs: impl Iterator<Item = PRef>,
    p: DVec3,
) -> bool {
    let mut w = 0.0;
    for pr in refs {
        match pr {
            PRef::Ext(i) => {
                let v = &ext[i as usize].verts;
                for k in 1..v.len() - 1 {
                    w += frac_geom::inside::tri_winding(
                        p,
                        verts[v[0] as usize],
                        verts[v[k] as usize],
                        verts[v[k + 1] as usize],
                    );
                }
            }
            PRef::Patch(i, rev) => {
                for t in &patches[i as usize].tris {
                    let (a, b, c) = (
                        verts[t[0] as usize],
                        verts[t[1] as usize],
                        verts[t[2] as usize],
                    );
                    w += if rev {
                        frac_geom::inside::tri_winding(p, a, c, b)
                    } else {
                        frac_geom::inside::tri_winding(p, a, b, c)
                    };
                }
            }
        }
    }
    w > 0.5
}
