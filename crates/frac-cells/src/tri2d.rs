//! Triangulation of planar polygons with holes, never inserting vertices.

use spade::handles::FixedVertexHandle;
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};
use std::collections::BTreeMap;

/// Triangulate the region bounded by `loops` (outer CCW, holes CW) over 2D
/// points `pts`. Returns triangles (CCW) as indices into `pts`.
///
/// The result is always *topologically* valid: every directed loop edge
/// appears exactly once as a triangle edge and interior edges pair up, so
/// meshes assembled from it stay watertight even for degenerate
/// (zero-area) loops produced by symbolic perturbation.
pub fn triangulate(pts: &[[f64; 2]], loops: &[Vec<usize>]) -> Vec<[usize; 3]> {
    if loops.len() == 1 && loops[0].len() == 3 {
        let l = &loops[0];
        return vec![[l[0], l[1], l[2]]];
    }
    let degenerate = loops.iter().any(|l| is_degenerate(pts, l));
    if !degenerate {
        if let Some(t) = cdt(pts, loops) {
            if topologically_valid(loops, &t) {
                return t;
            }
        }
        let t = earcut(pts, loops);
        if topologically_valid(loops, &t) {
            return t;
        }
    }
    fan(&bridge(pts, loops))
}

fn perimeter(pts: &[[f64; 2]], l: &[usize]) -> f64 {
    let n = l.len();
    (0..n)
        .map(|k| {
            let a = pts[l[k]];
            let b = pts[l[(k + 1) % n]];
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        })
        .sum()
}

/// Loops whose area is negligible compared to their perimeter.
pub fn is_degenerate(pts: &[[f64; 2]], l: &[usize]) -> bool {
    let p = perimeter(pts, l);
    signed_area(pts, l).abs() <= 1e-9 * p * p
}

/// Every directed loop edge used exactly once, every other directed edge
/// matched by its reverse.
pub fn topologically_valid(loops: &[Vec<usize>], tris: &[[usize; 3]]) -> bool {
    let mut cnt: BTreeMap<(usize, usize), i32> = BTreeMap::new();
    for t in tris {
        for k in 0..3 {
            let e = (t[k], t[(k + 1) % 3]);
            if e.0 == e.1 {
                return false;
            }
            *cnt.entry(e).or_default() += 1;
        }
    }
    let mut boundary: BTreeMap<(usize, usize), i32> = BTreeMap::new();
    for l in loops {
        let n = l.len();
        for k in 0..n {
            *boundary.entry((l[k], l[(k + 1) % n])).or_default() += 1;
        }
    }
    for (&(a, b), &c) in &cnt {
        let r = cnt.get(&(b, a)).copied().unwrap_or(0);
        let bd = boundary.get(&(a, b)).copied().unwrap_or(0);
        let bdr = boundary.get(&(b, a)).copied().unwrap_or(0);
        if c - r != bd - bdr {
            return false;
        }
    }
    for (&e, &c) in &boundary {
        if cnt.get(&e).copied().unwrap_or(0) < c {
            return false;
        }
    }
    true
}

/// Merge holes into the outer loop through bridge edges (traversed twice).
fn bridge(pts: &[[f64; 2]], loops: &[Vec<usize>]) -> Vec<usize> {
    let mut outer = loops[0].clone();
    for h in &loops[1..] {
        // closest pair (outer vertex, hole vertex)
        let mut best = (f64::INFINITY, 0usize, 0usize);
        for (i, &o) in outer.iter().enumerate() {
            for (j, &x) in h.iter().enumerate() {
                let d = (pts[o][0] - pts[x][0]).powi(2) + (pts[o][1] - pts[x][1]).powi(2);
                if d < best.0 {
                    best = (d, i, j);
                }
            }
        }
        let (_, i, j) = best;
        let mut nl = Vec::with_capacity(outer.len() + h.len() + 2);
        nl.extend_from_slice(&outer[..=i]);
        nl.extend_from_slice(&h[j..]);
        nl.extend_from_slice(&h[..=j]);
        nl.extend_from_slice(&outer[i..]);
        outer = nl;
    }
    outer
}

fn fan(l: &[usize]) -> Vec<[usize; 3]> {
    (1..l.len().saturating_sub(1))
        .map(|k| [l[0], l[k], l[k + 1]])
        .filter(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2])
        .collect()
}

fn cdt(pts: &[[f64; 2]], loops: &[Vec<usize>]) -> Option<Vec<[usize; 3]>> {
    let mut tri: ConstrainedDelaunayTriangulation<Point2<f64>> =
        ConstrainedDelaunayTriangulation::new();
    let mut handle: BTreeMap<usize, FixedVertexHandle> = BTreeMap::new();
    let mut back: BTreeMap<usize, usize> = BTreeMap::new();
    for l in loops {
        for &i in l {
            if handle.contains_key(&i) {
                continue;
            }
            let h = tri.insert(Point2::new(pts[i][0], pts[i][1])).ok()?;
            if back.contains_key(&h.index()) {
                return None; // duplicate 2D position
            }
            back.insert(h.index(), i);
            handle.insert(i, h);
        }
    }
    for l in loops {
        let n = l.len();
        for k in 0..n {
            let (a, b) = (handle[&l[k]], handle[&l[(k + 1) % n]]);
            if a == b {
                continue;
            }
            if tri.exists_constraint(a, b) {
                continue;
            }
            let added = tri.try_add_constraint(a, b);
            if added.is_empty() && !tri.exists_constraint(a, b) {
                return None;
            }
        }
    }
    // seed faces left of each directed loop edge, flood fill
    let nf = tri.num_all_faces();
    let mut inside = vec![0u8; nf]; // 0 unknown, 1 inside
    let mut stack = Vec::new();
    for l in loops {
        let n = l.len();
        for k in 0..n {
            let (a, b) = (handle[&l[k]], handle[&l[(k + 1) % n]]);
            if let Some(e) = tri.get_edge_from_neighbors(a, b) {
                let f = e.face();
                if let Some(inner) = f.as_inner() {
                    let fi = inner.fix().index();
                    if inside[fi] == 0 {
                        inside[fi] = 1;
                        stack.push(inner.fix());
                    }
                }
            }
        }
    }
    while let Some(fh) = stack.pop() {
        let f = tri.face(fh);
        for e in f.adjacent_edges() {
            if e.is_constraint_edge() {
                continue;
            }
            if let Some(nb) = e.rev().face().as_inner() {
                let ni = nb.fix().index();
                if inside[ni] == 0 {
                    inside[ni] = 1;
                    stack.push(nb.fix());
                }
            }
        }
    }
    let mut out = Vec::new();
    for f in tri.inner_faces() {
        if inside[f.fix().index()] != 1 {
            continue;
        }
        let vs = f.vertices();
        let ids = [
            back[&vs[0].fix().index()],
            back[&vs[1].fix().index()],
            back[&vs[2].fix().index()],
        ];
        out.push(ids);
    }
    // sanity: area must match polygon area
    let poly_area: f64 = loops.iter().map(|l| signed_area(pts, l)).sum();
    let tri_area: f64 = out.iter().map(|t| signed_area(pts, &t[..])).sum();
    if (poly_area - tri_area).abs() > 1e-9 * poly_area.abs().max(1e-300) + 1e-300 {
        return None;
    }
    out.sort_unstable();
    Some(out)
}

pub fn signed_area(pts: &[[f64; 2]], l: &[usize]) -> f64 {
    let n = l.len();
    let mut a = 0.0;
    for k in 0..n {
        let p = pts[l[k]];
        let q = pts[l[(k + 1) % n]];
        a += p[0] * q[1] - q[0] * p[1];
    }
    0.5 * a
}

fn earcut(pts: &[[f64; 2]], loops: &[Vec<usize>]) -> Vec<[usize; 3]> {
    let mut data = Vec::new();
    let mut holes = Vec::new();
    let mut map = Vec::new();
    for (k, l) in loops.iter().enumerate() {
        if k > 0 {
            holes.push(map.len());
        }
        for &i in l {
            data.push(pts[i][0]);
            data.push(pts[i][1]);
            map.push(i);
        }
    }
    match earcutr::earcut(&data, &holes, 2) {
        Ok(idx) => {
            let mut out: Vec<[usize; 3]> = idx
                .chunks(3)
                .map(|c| {
                    let t = [map[c[0]], map[c[1]], map[c[2]]];
                    if signed_area(pts, &t) < 0.0 {
                        [t[0], t[2], t[1]]
                    } else {
                        t
                    }
                })
                .collect();
            out.sort_unstable();
            out
        }
        Err(_) => {
            // last resort: fan of the outer loop
            let l = &loops[0];
            (1..l.len() - 1).map(|k| [l[0], l[k], l[k + 1]]).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn square_with_hole() {
        let pts = vec![
            [0., 0.],
            [4., 0.],
            [4., 4.],
            [0., 4.],
            [1., 1.],
            [1., 3.],
            [3., 3.],
            [3., 1.],
        ];
        let t = triangulate(&pts, &[vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
        let a: f64 = t.iter().map(|t| signed_area(&pts, &t[..])).sum();
        assert!((a - 12.0).abs() < 1e-12);
        assert!(t.iter().all(|t| signed_area(&pts, &t[..]) > 0.0));
    }
}
