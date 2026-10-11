//! Cross-component contact interfaces (spec Stage 1, measured at the cell
//! level) and anchor interfaces to the world.

use frac_core::*;
use frac_geom::bvh::Bvh;
use frac_geom::polygon::{newell, plane_basis};
use frac_geom::{Aabb, DVec3};

/// A planar exterior polygon of a cell (convex: clipped triangle).
#[derive(Clone, Debug)]
pub struct CellFace {
    pub cell: CellId,
    pub pts: Vec<DVec3>,
    pub normal: DVec3,
    pub aabb: Aabb,
}

pub fn cell_faces(geom: &ComponentGeometry) -> Vec<CellFace> {
    geom.ext_polys
        .iter()
        .filter_map(|e| {
            let pts: Vec<DVec3> = e.verts.iter().map(|&v| geom.verts[v as usize]).collect();
            let n = newell(&pts);
            if n.length_squared() == 0.0 {
                return None;
            }
            Some(CellFace {
                cell: e.cell,
                aabb: Aabb::from_points(pts.iter()),
                normal: n.normalize(),
                pts,
            })
        })
        .collect()
}

/// Clip a convex 2D polygon by another convex 2D polygon (both CCW).
fn clip_convex(subject: &[[f64; 2]], clip: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut out = subject.to_vec();
    let n = clip.len();
    for i in 0..n {
        if out.is_empty() {
            break;
        }
        let a = clip[i];
        let b = clip[(i + 1) % n];
        let side = |p: [f64; 2]| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        let inp = out.clone();
        out.clear();
        for k in 0..inp.len() {
            let p = inp[k];
            let q = inp[(k + 1) % inp.len()];
            let (sp, sq) = (side(p), side(q));
            if sp >= 0.0 {
                out.push(p);
            }
            if (sp >= 0.0) != (sq >= 0.0) {
                let t = sp / (sp - sq);
                out.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]);
            }
        }
    }
    out
}

fn area2(p: &[[f64; 2]]) -> f64 {
    let n = p.len();
    (0..n)
        .map(|k| p[k][0] * p[(k + 1) % n][1] - p[(k + 1) % n][0] * p[k][1])
        .sum::<f64>()
        * 0.5
}

/// Contact polygons between two components: pairs of opposed, nearly
/// coplanar exterior faces within `tol`, intersected in A's plane.
/// Returns (cell_a, cell_b, polygon with normal pointing from A to B).
pub fn contact_polygons(
    a: &[CellFace],
    b: &[CellFace],
    tol: f64,
    cos: f64,
) -> Vec<(CellId, CellId, Polygon3)> {
    let boxes: Vec<Aabb> = b.iter().map(|f| f.aabb.expanded(tol)).collect();
    let bvh = Bvh::build(&boxes);
    let mut out = Vec::new();
    for fa in a {
        let cand = bvh.query_vec(&fa.aabb.expanded(tol));
        for j in cand {
            let fb = &b[j as usize];
            if fa.normal.dot(fb.normal) > -cos {
                continue;
            }
            let r = fa.pts[0];
            if fb.pts.iter().any(|p| (*p - r).dot(fa.normal).abs() > tol) {
                continue;
            }
            let (u, v) = plane_basis(fa.normal);
            let pa: Vec<[f64; 2]> = fa
                .pts
                .iter()
                .map(|p| [(*p - r).dot(u), (*p - r).dot(v)])
                .collect();
            // B's polygon reversed so it is CCW about A's normal
            let pb: Vec<[f64; 2]> = fb
                .pts
                .iter()
                .rev()
                .map(|p| [(*p - r).dot(u), (*p - r).dot(v)])
                .collect();
            if area2(&pa) <= 0.0 || area2(&pb) <= 0.0 {
                continue;
            }
            let c = clip_convex(&pa, &pb);
            if c.len() < 3 || area2(&c) <= 1e-12 * area2(&pa) {
                continue;
            }
            let pts: Vec<DVec3> = c.iter().map(|q| r + u * q[0] + v * q[1]).collect();
            out.push((
                fa.cell,
                fb.cell,
                Polygon3 {
                    loops: vec![pts],
                    normal: fa.normal,
                },
            ));
        }
    }
    out
}

/// Anchor polygons: exterior faces facing down whose vertices lie at or
/// below `height + tol`. Normal points from the cell into the world.
pub fn anchor_polygons(faces: &[CellFace], height: f64, tol: f64) -> Vec<(CellId, Polygon3)> {
    faces
        .iter()
        .filter(|f| f.normal.y < -0.7 && f.pts.iter().all(|p| p.y <= height + tol))
        .map(|f| {
            (
                f.cell,
                Polygon3 {
                    loops: vec![f.pts.clone()],
                    normal: f.normal,
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stacked_squares_contact() {
        let sq = |y: f64, flip: bool, off: f64| {
            let mut pts = vec![
                DVec3::new(off, y, 0.0),
                DVec3::new(off + 1.0, y, 0.0),
                DVec3::new(off + 1.0, y, 1.0),
                DVec3::new(off, y, 1.0),
            ];
            if flip {
                pts.reverse();
            }
            let n = newell(&pts).normalize();
            CellFace {
                cell: CellId(0),
                aabb: Aabb::from_points(pts.iter()),
                normal: n,
                pts,
            }
        };
        // A's top face (normal +y) and B's bottom face (normal -y), offset by 0.5 in x
        let a = vec![sq(1.0, true, 0.0)];
        let b = vec![sq(1.0, false, 0.5)];
        assert!(a[0].normal.y > 0.9 && b[0].normal.y < -0.9);
        let c = contact_polygons(&a, &b, 1e-3, 0.95);
        assert_eq!(c.len(), 1);
        let area = newell(&c[0].2.loops[0]).length();
        assert!((area - 0.5).abs() < 1e-12);
    }
}
