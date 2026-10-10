//! Manifold repair of cell groupings.
//!
//! A fragment is the union of the cells of one group. In box complexes
//! (masonry) two cells of a group can touch only along an edge or at a
//! vertex while the cells filling the other quadrants belong to other groups
//! (bricks of a stretcher bond meeting diagonally). The union is then not a
//! 2-manifold, so it fails the fragment-validity gate and is unusable as a
//! rigid body surface. This pass finds the non-manifold edges and vertices of
//! every group's boundary and moves single analysis cells across them (add an
//! outside cell that fills the gap, or hand an inside cell to a neighbouring
//! group). A move is accepted only when it strictly reduces the defects of the
//! two groups it touches and keeps both groups non-empty and connected.

use crate::assemble::ComponentCells;
use frac_core::*;
use std::collections::{BTreeMap, BTreeSet};

/// Outward boundary triangles of every local fine cell with the local cell
/// on the other side (`None` on the component surface).
type CellTris = Vec<Vec<([u32; 3], Option<u32>)>>;

fn cell_tris(comp: &Component, info: &ComponentCells) -> CellTris {
    let base = info.cell_range.start;
    let g = &comp.geometry;
    let mut out: CellTris = vec![Vec::new(); info.cell_range.len()];
    for e in &g.ext_polys {
        let c = (e.cell.0 - base) as usize;
        out[c].extend(e.tris.iter().map(|&t| (t, None)));
    }
    for p in &g.patches {
        let (a, b) = (p.cells.0.0 - base, p.cells.1.0 - base);
        if a == b {
            continue;
        }
        out[a as usize].extend(p.tris.iter().map(|&t| (t, Some(b))));
        out[b as usize].extend(p.tris.iter().map(|&t| ([t[0], t[2], t[1]], Some(a))));
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Defect {
    Edge(u32, u32),
    Vertex(u32),
}

struct Ctx<'a> {
    tris: &'a CellTris,
    /// Local fine cells of each analysis cell.
    fine: Vec<Vec<u32>>,
    /// Analysis cell of each local fine cell.
    analysis: &'a [u32],
    /// Analysis adjacency lists (positive shared area).
    nbr: Vec<Vec<u32>>,
}

impl Ctx<'_> {
    /// Boundary triangles of group `g`: (tri, inside analysis cell, outside analysis cell).
    fn boundary(
        &self,
        labels: &[u32],
        members: &[u32],
        g: u32,
    ) -> Vec<([u32; 3], u32, Option<u32>)> {
        let mut out = Vec::new();
        for &a in members {
            for &c in &self.fine[a as usize] {
                for &(t, o) in &self.tris[c as usize] {
                    match o {
                        None => out.push((t, a, None)),
                        Some(o) => {
                            let oa = self.analysis[o as usize];
                            if labels[oa as usize] != g {
                                out.push((t, a, Some(oa)));
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn defects(bd: &[([u32; 3], u32, Option<u32>)]) -> Vec<Defect> {
        let mut edges: BTreeMap<(u32, u32), (u32, u32)> = BTreeMap::new();
        let mut link: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();
        for (t, _, _) in bd {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                let e = edges.entry((a.min(b), a.max(b))).or_default();
                if a < b {
                    e.0 += 1;
                } else {
                    e.1 += 1;
                }
                link.entry(t[k])
                    .or_default()
                    .push((t[(k + 1) % 3], t[(k + 2) % 3]));
            }
        }
        let mut out: Vec<Defect> = edges
            .iter()
            .filter(|(_, fr)| fr.0 != 1 || fr.1 != 1)
            .map(|(&(a, b), _)| Defect::Edge(a, b))
            .collect();
        for (&v, l) in &link {
            // the link of a manifold vertex is one cycle
            let mut ids: Vec<u32> = l.iter().flat_map(|&(a, b)| [a, b]).collect();
            ids.sort_unstable();
            ids.dedup();
            let mut parent: Vec<usize> = (0..ids.len()).collect();
            fn find(p: &mut [usize], x: usize) -> usize {
                let mut r = x;
                while p[r] != r {
                    r = p[r];
                }
                p[x] = r;
                r
            }
            let idx = |x: u32| ids.binary_search(&x).unwrap();
            for &(a, b) in l {
                let (ra, rb) = (find(&mut parent, idx(a)), find(&mut parent, idx(b)));
                parent[ra.max(rb)] = ra.min(rb);
            }
            let roots = (0..ids.len())
                .filter(|&i| find(&mut parent, i) == i)
                .count();
            if roots > 1 {
                out.push(Defect::Vertex(v));
            }
        }
        out
    }

    fn connected(&self, labels: &[u32], members: &[u32], g: u32) -> bool {
        if members.is_empty() {
            return false;
        }
        let mut seen: BTreeSet<u32> = BTreeSet::new();
        let mut stack = vec![members[0]];
        seen.insert(members[0]);
        while let Some(a) = stack.pop() {
            for &b in &self.nbr[a as usize] {
                if labels[b as usize] == g && seen.insert(b) {
                    stack.push(b);
                }
            }
        }
        seen.len() == members.len()
    }
}

/// Repair `labels` (one group label per analysis cell) so that every group's
/// union is a closed 2-manifold where a single-cell move can achieve it.
/// When `within` is given, moves keep each analysis cell inside its
/// `within` group (nested levels). Returns the number of moves made and the
/// number of defects left.
pub fn repair_labels(
    comp: &Component,
    info: &ComponentCells,
    adj: &[(u32, u32, f64, f64)],
    labels: &mut [u32],
    within: Option<&[u32]>,
) -> (usize, usize) {
    let na = info.n_analysis as usize;
    if na <= 1 {
        return (0, 0);
    }
    let tris = cell_tris(comp, info);
    let mut fine = vec![Vec::new(); na];
    for (c, &a) in info.cell_analysis.iter().enumerate() {
        fine[a as usize].push(c as u32);
    }
    let mut nbr = vec![Vec::new(); na];
    for &(a, b, area, _) in adj {
        if area > 0.0 {
            nbr[a as usize].push(b);
            nbr[b as usize].push(a);
        }
    }
    for l in nbr.iter_mut() {
        l.sort_unstable();
        l.dedup();
    }
    let ctx = Ctx {
        tris: &tris,
        fine,
        analysis: &info.cell_analysis,
        nbr,
    };
    let members_of = |labels: &[u32], g: u32| -> Vec<u32> {
        (0..na as u32)
            .filter(|&a| labels[a as usize] == g)
            .collect()
    };
    let count = |labels: &[u32], g: u32| -> (usize, bool) {
        let m = members_of(labels, g);
        if m.is_empty() {
            return (usize::MAX, false);
        }
        let n = Ctx::defects(&ctx.boundary(labels, &m, g)).len();
        (n, ctx.connected(labels, &m, g))
    };
    let mut groups: Vec<u32> = labels.to_vec();
    groups.sort_unstable();
    groups.dedup();
    let mut defects: BTreeMap<u32, usize> = BTreeMap::new();
    for &g in &groups {
        let n = count(labels, g).0;
        if n > 0 {
            defects.insert(g, n);
        }
    }
    let mut moves = 0usize;
    let budget = 8 * defects.values().sum::<usize>() + 8;
    'outer: while moves < budget {
        let Some((&g, _)) = defects.iter().next() else {
            break;
        };
        let m = members_of(labels, g);
        let bd = ctx.boundary(labels, &m, g);
        let list = Ctx::defects(&bd);
        if list.is_empty() {
            defects.remove(&g);
            continue;
        }
        // candidate moves around the first defect (deterministic order)
        let mut cands: BTreeSet<(u32, u32)> = BTreeSet::new(); // (analysis cell, new label)
        for d in &list {
            for (t, a, o) in &bd {
                let touches = match *d {
                    Defect::Edge(x, y) => t.contains(&x) && t.contains(&y),
                    Defect::Vertex(v) => t.contains(&v),
                };
                if !touches {
                    continue;
                }
                if let Some(o) = *o {
                    cands.insert((o, g));
                }
                for &b in &ctx.nbr[*a as usize] {
                    let h = labels[b as usize];
                    if h != g {
                        cands.insert((*a, h));
                    }
                }
            }
            if !cands.is_empty() {
                break;
            }
        }
        let before_g = list.len();
        for (a, h) in cands {
            let old = labels[a as usize];
            if old == h {
                continue;
            }
            // nested levels: both groups must share the parent group
            if let Some(w) = within {
                let parent_h = (0..na).find(|&b| labels[b] == h).map(|b| w[b]);
                if parent_h != Some(w[a as usize]) {
                    continue;
                }
            }
            let other = if old == g { h } else { old };
            let before_o = defects.get(&other).copied().unwrap_or(0);
            labels[a as usize] = h;
            let (ng, cg) = count(labels, g);
            let (no, co) = count(labels, other);
            if cg && co && ng != usize::MAX && no != usize::MAX && ng + no < before_g + before_o {
                moves += 1;
                for (grp, n) in [(g, ng), (other, no)] {
                    if n > 0 {
                        defects.insert(grp, n);
                    } else {
                        defects.remove(&grp);
                    }
                }
                continue 'outer;
            }
            labels[a as usize] = old;
        }
        // no improving move for this group: leave it
        defects.remove(&g);
    }
    let left = groups
        .iter()
        .map(|&g| count(labels, g).0)
        .filter(|&n| n != usize::MAX)
        .sum();
    (moves, left)
}
