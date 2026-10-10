//! Stage 5: the nested fracture hierarchy.
//!
//! Levels are successive refinements over the fine cells of each component:
//! L0 component → L1 structural fragments → L2 analysis cells (optionally
//! agglomerated) → L3 fine cells. Every node at level n+1 has exactly one
//! parent at level n. Fragments are ordered by (level, parent, Morton code
//! of centroid), so children of a fragment are contiguous, and cells are
//! ordered so every fragment's cells are contiguous at every level.

use frac_core::*;
use frac_geom::{Aabb, DVec3, MassProps, morton3};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

/// Per-component partition description (all indices component-local except
/// `cells`).
#[derive(Clone, Debug)]
pub struct ComponentPartition {
    pub component: ComponentId,
    pub role: ComponentRole,
    /// Global cell id range of the component.
    pub cells: std::ops::Range<u32>,
    /// Local analysis-cell index per cell.
    pub cell_analysis: Vec<u32>,
    /// L1 label per analysis cell.
    pub analysis_l1: Vec<u32>,
    /// L2 label per analysis cell (must refine L1; identity = analysis cells).
    pub analysis_l2: Vec<u32>,
}

/// Which partitions make up the levels, coarse to fine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelKind {
    Component,
    Structural,
    Analysis,
    Fine,
}

pub fn level_kinds(levels: u8) -> Vec<LevelKind> {
    match levels {
        0 | 1 => vec![LevelKind::Component],
        2 => vec![LevelKind::Component, LevelKind::Fine],
        3 => vec![LevelKind::Component, LevelKind::Structural, LevelKind::Fine],
        _ => vec![
            LevelKind::Component,
            LevelKind::Structural,
            LevelKind::Analysis,
            LevelKind::Fine,
        ],
    }
}

/// Build the hierarchy. `cells` are the asset cells (global ids).
pub fn build_hierarchy(
    cells: &[Cell],
    parts: &[ComponentPartition],
    levels: u8,
    asset_bbox: &Aabb,
    min_rigid_size: f64,
) -> Hierarchy {
    let kinds = level_kinds(levels);
    let nl = kinds.len();
    // label of each cell at each level (component-local labels first)
    // key = (component, label at this level)
    let mut cell_key: Vec<Vec<(u32, u32)>> = vec![vec![(0, 0); cells.len()]; nl];
    for part in parts {
        for c in part.cells.clone() {
            let lc = (c - part.cells.start) as usize;
            let a = part.cell_analysis[lc];
            for (li, k) in kinds.iter().enumerate() {
                let lab = match k {
                    LevelKind::Component => 0,
                    LevelKind::Structural => part.analysis_l1[a as usize],
                    LevelKind::Analysis => part.analysis_l2[a as usize],
                    LevelKind::Fine => lc as u32,
                };
                cell_key[li][c as usize] = (part.component.0, lab);
            }
        }
    }
    // group cells per node per level; compute mass + Morton
    let mut fragments: Vec<Fragment> = Vec::new();
    let mut level_ranges = Vec::new();
    let mut cell_fragment: Vec<Vec<FragmentId>> = vec![vec![FragmentId(0); cells.len()]; nl];
    let role_of: BTreeMap<u32, ComponentRole> =
        parts.iter().map(|p| (p.component.0, p.role)).collect();
    // parent ordering index of previous level fragment (for sorting)
    for li in 0..nl {
        let mut groups: BTreeMap<(u32, u32), Vec<u32>> = BTreeMap::new();
        for c in 0..cells.len() {
            groups.entry(cell_key[li][c]).or_default().push(c as u32);
        }
        let mut nodes: Vec<(u64, u64, (u32, u32), Vec<u32>, MassProps)> = groups
            .into_iter()
            .map(|(k, cs)| {
                let mp = MassProps::combine(
                    &cs.iter()
                        .map(|&c| cells[c as usize].mass)
                        .collect::<Vec<_>>(),
                );
                let parent_order = if li == 0 {
                    k.0 as u64
                } else {
                    cell_fragment[li - 1][cs[0] as usize].0 as u64
                };
                let m = morton3(mp.com, asset_bbox);
                (parent_order, m, k, cs, mp)
            })
            .collect();
        nodes.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let start = fragments.len() as u32;
        for (_, _, key, cs, mp) in nodes {
            let id = FragmentId(fragments.len() as u32);
            for &c in &cs {
                cell_fragment[li][c as usize] = id;
            }
            let parent = if li == 0 {
                None
            } else {
                Some(cell_fragment[li - 1][cs[0] as usize])
            };
            // material mix by mass
            let mut mix: BTreeMap<MaterialId, f64> = BTreeMap::new();
            for &c in &cs {
                *mix.entry(cells[c as usize].material).or_default() += cells[c as usize].mass.mass;
            }
            let total: f64 = mix.values().sum::<f64>().max(1e-300);
            let mut mixv: Vec<(MaterialId, f32)> = mix
                .into_iter()
                .map(|(m, w)| (m, (w / total) as f32))
                .collect();
            mixv.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            mixv.truncate(4);
            let mut bb = Aabb::EMPTY;
            for &c in &cs {
                bb = bb.union(&cells[c as usize].aabb);
            }
            let size = bb.extent().max_element();
            fragments.push(Fragment {
                id,
                level: li as u8,
                component: ComponentId(key.0),
                parent,
                children: 0..0,
                cells: 0..0,
                material_mix: SmallVec::from_vec(mixv),
                mass: mp,
                hulls: 0..0,
                render: RenderRefs {
                    gltf_node: -1,
                    lod_meshes: Vec::new(),
                },
                particle_candidate: size < min_rigid_size,
                role: role_of.get(&key.0).copied().unwrap_or_default(),
                interior_area: 0.0,
            });
        }
        level_ranges.push(start..fragments.len() as u32);
    }
    // children ranges (contiguous by construction of the sort)
    for li in 1..nl {
        let r = level_ranges[li].clone();
        let mut i = r.start;
        while i < r.end {
            let p = fragments[i as usize].parent.unwrap();
            let mut j = i;
            while j < r.end && fragments[j as usize].parent == Some(p) {
                j += 1;
            }
            fragments[p.idx()].children = i..j;
            i = j;
        }
    }
    // cell order: sort by fragment ids at every level (lexicographic)
    let mut order: Vec<u32> = (0..cells.len() as u32).collect();
    order.sort_by(|&a, &b| {
        for li in 0..nl {
            let o = cell_fragment[li][a as usize].cmp(&cell_fragment[li][b as usize]);
            if o != Ordering::Equal {
                return o;
            }
        }
        a.cmp(&b)
    });
    let cell_order: Vec<CellId> = order.iter().map(|&c| CellId(c)).collect();
    // ranges per fragment
    let mut first: Vec<u32> = vec![u32::MAX; fragments.len()];
    let mut last: Vec<u32> = vec![0; fragments.len()];
    for (pos, &c) in order.iter().enumerate() {
        for li in 0..nl {
            let f = cell_fragment[li][c as usize].idx();
            first[f] = first[f].min(pos as u32);
            last[f] = last[f].max(pos as u32 + 1);
        }
    }
    for (i, f) in fragments.iter_mut().enumerate() {
        f.cells = first[i]..last[i];
    }
    Hierarchy {
        levels: nl as u8,
        fragments,
        level_ranges,
        cell_order,
        cell_fragment,
    }
}

#[derive(PartialEq)]
struct Cand {
    score: f64,
    node: u32,
    region: u32,
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        self.score
            .partial_cmp(&o.score)
            .unwrap_or(Ordering::Equal)
            .then(o.node.cmp(&self.node))
            .then(o.region.cmp(&self.region))
    }
}

/// Agglomeration method B (deterministic region growing).
///
/// * `adj`: undirected edges `(a, b, shared_area, w_g)`.
/// * seeds by farthest-point sampling on centroids (start: lowest Morton);
/// * merge priority: shared area weighted by `1 / w_g` (weak interfaces
///   become fragment boundaries), minus a compactness penalty;
/// * growth stops at the target region volume; leftovers join their
///   strongest neighbor region.
/// * `constraint`: optional labels; regions never cross constraint labels.
pub fn agglomerate(
    centroids: &[DVec3],
    volumes: &[f64],
    adj: &[(u32, u32, f64, f64)],
    target: usize,
    compactness: f64,
    constraint: Option<&[u32]>,
) -> Vec<u32> {
    let n = centroids.len();
    if n == 0 {
        return Vec::new();
    }
    let target = target.clamp(1, n);
    let bb = Aabb::from_points(centroids.iter());
    let scale = bb.diagonal().max(1e-12);
    let mut nbrs: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    for &(a, b, area, w) in adj {
        if a == b {
            continue;
        }
        if let Some(c) = constraint {
            if c[a as usize] != c[b as usize] {
                continue;
            }
        }
        let s = area / w.max(1e-12);
        nbrs[a as usize].push((b, s));
        nbrs[b as usize].push((a, s));
    }
    for v in nbrs.iter_mut() {
        v.sort_by(|x, y| x.0.cmp(&y.0));
    }
    // seeds per constraint group, proportional to group volume
    let groups: BTreeMap<u32, Vec<u32>> = {
        let mut g: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for i in 0..n {
            g.entry(constraint.map(|c| c[i]).unwrap_or(0))
                .or_default()
                .push(i as u32);
        }
        g
    };
    let total_vol: f64 = volumes.iter().sum::<f64>().max(1e-300);
    let mut seeds: Vec<u32> = Vec::new();
    let mut remaining = target;
    let ng = groups.len();
    for (gi, (_, members)) in groups.iter().enumerate() {
        let gv: f64 = members.iter().map(|&i| volumes[i as usize]).sum();
        let mut k = ((gv / total_vol) * target as f64).round() as usize;
        k = k.max(1).min(members.len());
        if gi == ng - 1 {
            k = k.max(1).min(members.len()).min(remaining.max(1));
        }
        remaining = remaining.saturating_sub(k);
        // FPS
        let start = *members
            .iter()
            .min_by_key(|&&i| (morton3(centroids[i as usize], &bb), i))
            .unwrap();
        let mut chosen = vec![start];
        let mut dist: Vec<f64> = members
            .iter()
            .map(|&i| (centroids[i as usize] - centroids[start as usize]).length_squared())
            .collect();
        while chosen.len() < k {
            let (bi, _) = dist
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap().then(b.0.cmp(&a.0)))
                .unwrap();
            let s = members[bi];
            chosen.push(s);
            for (j, &m) in members.iter().enumerate() {
                dist[j] =
                    dist[j].min((centroids[m as usize] - centroids[s as usize]).length_squared());
            }
        }
        seeds.extend(chosen);
    }
    let k = seeds.len();
    let target_vol = total_vol / k as f64;
    let mut label = vec![u32::MAX; n];
    let mut rvol = vec![0.0f64; k];
    let mut rcent = vec![DVec3::ZERO; k];
    let mut heap = BinaryHeap::new();
    for (r, &s) in seeds.iter().enumerate() {
        label[s as usize] = r as u32;
        rvol[r] = volumes[s as usize];
        rcent[r] = centroids[s as usize];
    }
    let push_nbrs =
        |node: u32, r: u32, heap: &mut BinaryHeap<Cand>, label: &[u32], rcent: &[DVec3]| {
            for &(nb, s) in &nbrs[node as usize] {
                if label[nb as usize] == u32::MAX {
                    let d = (centroids[nb as usize] - rcent[r as usize]).length() / scale;
                    heap.push(Cand {
                        score: s - compactness * d * s.max(1e-300).max(1e-12),
                        node: nb,
                        region: r,
                    });
                }
            }
        };
    for (r, &s) in seeds.iter().enumerate() {
        push_nbrs(s, r as u32, &mut heap, &label, &rcent);
    }
    while let Some(c) = heap.pop() {
        if label[c.node as usize] != u32::MAX {
            continue;
        }
        let r = c.region as usize;
        if rvol[r] >= target_vol * 1.15 {
            continue;
        }
        label[c.node as usize] = c.region;
        let v = volumes[c.node as usize];
        rcent[r] =
            (rcent[r] * rvol[r] + centroids[c.node as usize] * v) / (rvol[r] + v).max(1e-300);
        rvol[r] += v;
        push_nbrs(c.node, c.region, &mut heap, &label, &rcent);
    }
    // leftovers: attach to strongest labeled neighbor, iterate
    loop {
        let mut changed = false;
        for i in 0..n {
            if label[i] != u32::MAX {
                continue;
            }
            let best = nbrs[i]
                .iter()
                .filter(|(nb, _)| label[*nb as usize] != u32::MAX)
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(b.0.cmp(&a.0)));
            if let Some(&(nb, _)) = best {
                label[i] = label[nb as usize];
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // isolated nodes become their own regions
    let mut next = k as u32;
    for l in label.iter_mut() {
        if *l == u32::MAX {
            *l = next;
            next += 1;
        }
    }
    canonical_labels(&label)
}

/// Relabel so labels appear in order of first occurrence.
pub fn canonical_labels(l: &[u32]) -> Vec<u32> {
    let mut map: BTreeMap<u32, u32> = BTreeMap::new();
    l.iter()
        .map(|&x| {
            let n = map.len() as u32;
            *map.entry(x).or_insert(n)
        })
        .collect()
}

/// Split labels into connected components over `adj` (labels canonical).
pub fn connected_labels(labels: &[u32], adj: &[(u32, u32, f64, f64)]) -> Vec<u32> {
    let n = labels.len();
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
    for &(a, b, _, _) in adj {
        if labels[a as usize] == labels[b as usize] {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra.max(rb) as usize] = ra.min(rb);
            }
        }
    }
    let roots: Vec<u32> = (0..n as u32).map(|i| find(&mut parent, i)).collect();
    canonical_labels(&roots)
}

/// Helper: set of distinct labels.
pub fn n_labels(l: &[u32]) -> usize {
    l.iter().collect::<BTreeSet<_>>().len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agglomerate_line_respects_weak_link() {
        // 10 nodes in a line, weak link between 4 and 5
        let c: Vec<DVec3> = (0..10).map(|i| DVec3::new(i as f64, 0.0, 0.0)).collect();
        let v = vec![1.0; 10];
        let adj: Vec<(u32, u32, f64, f64)> = (0..9)
            .map(|i| (i, i + 1, 1.0, if i == 4 { 100.0 } else { 1.0 }))
            .collect();
        let l = agglomerate(&c, &v, &adj, 2, 0.1, None);
        assert_eq!(n_labels(&l), 2);
        assert!(l[..5].iter().all(|&x| x == l[0]));
        assert!(l[5..].iter().all(|&x| x == l[5]));
    }

    #[test]
    fn hierarchy_nesting() {
        let mk = |x: f64| Cell {
            id: CellId(0),
            component: ComponentId(0),
            analysis_cell: 0,
            material: MaterialId(0),
            mass: MassProps {
                volume: 1.0,
                mass: 1.0,
                com: DVec3::new(x, 0.0, 0.0),
                inertia: glam::DMat3::ZERO,
            },
            aabb: Aabb {
                min: DVec3::new(x - 0.5, -0.5, -0.5),
                max: DVec3::new(x + 0.5, 0.5, 0.5),
            },
            thickness_ratio: 1.0,
        };
        let cells: Vec<Cell> = (0..8).map(|i| mk(i as f64)).collect();
        let part = ComponentPartition {
            component: ComponentId(0),
            role: ComponentRole::Generic,
            cells: 0..8,
            cell_analysis: vec![0, 0, 1, 1, 2, 2, 3, 3],
            analysis_l1: vec![0, 0, 1, 1],
            analysis_l2: vec![0, 1, 2, 3],
        };
        let bb = Aabb {
            min: DVec3::splat(-1.0),
            max: DVec3::new(8.0, 1.0, 1.0),
        };
        let h = build_hierarchy(&cells, &[part], 4, &bb, 0.0);
        assert_eq!(h.levels, 4);
        assert_eq!(h.level_ranges[1].len(), 2);
        assert_eq!(h.level_ranges[2].len(), 4);
        assert_eq!(h.level_ranges[3].len(), 8);
        for f in &h.fragments {
            if let Some(p) = f.parent {
                let pf = &h.fragments[p.idx()];
                assert!(pf.children.contains(&f.id.0));
                assert!(pf.cells.start <= f.cells.start && f.cells.end <= pf.cells.end);
            }
            let total: f64 = h.cell_order[f.cells.start as usize..f.cells.end as usize]
                .iter()
                .map(|c| cells[c.idx()].mass.mass)
                .sum();
            assert!((total - f.mass.mass).abs() < 1e-12);
        }
    }
}
