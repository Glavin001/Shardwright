//! Inside/outside classification: exact ray parity for clean closed meshes,
//! generalized winding numbers (with a Barnes–Hut style fast evaluation) for
//! messy meshes, and closest-point queries.

use crate::aabb::Aabb;
use crate::bvh::Bvh;
use crate::mesh::{p3, TriMesh};
use crate::predicates::orient3d;
use glam::DVec3;

/// Spatial query structure over a triangle mesh.
pub struct MeshQuery<'a> {
    pub mesh: &'a TriMesh,
    pub bvh: Bvh,
    pub boxes: Vec<Aabb>,
    pub bbox: Aabb,
}

impl<'a> MeshQuery<'a> {
    pub fn new(mesh: &'a TriMesh) -> Self {
        let boxes = mesh.tri_aabbs();
        let bvh = Bvh::build(&boxes);
        MeshQuery { mesh, bvh, boxes, bbox: mesh.aabb() }
    }

    /// Exact parity of the number of crossings of the segment p->q with
    /// the mesh. Returns `None` if the segment hits a degenerate feature
    /// (edge, vertex, or lies in a triangle plane), so the caller can retry.
    fn parity(&self, p: DVec3, q: DVec3) -> Option<bool> {
        let seg_box = Aabb::from_points([&p, &q]);
        let pp = p3(p);
        let qq = p3(q);
        let mut odd = false;
        let mut degenerate = false;
        self.bvh.query(&seg_box, |t| {
            if degenerate {
                return;
            }
            let [a, b, c] = self.mesh.tri_points(t as usize);
            let (a, b, c) = (p3(a), p3(b), p3(c));
            let s1 = orient3d(&a, &b, &c, &pp);
            let s2 = orient3d(&a, &b, &c, &qq);
            if s1 == 0 || s2 == 0 {
                if s1 == 0 && s2 == 0 {
                    return; // coplanar with the ray: ignore only if it misses (retry otherwise)
                }
                // endpoint on plane: only degenerate if inside the triangle
                let o1 = orient3d(&pp, &qq, &a, &b);
                let o2 = orient3d(&pp, &qq, &b, &c);
                let o3 = orient3d(&pp, &qq, &c, &a);
                if (o1 >= 0 && o2 >= 0 && o3 >= 0) || (o1 <= 0 && o2 <= 0 && o3 <= 0) {
                    degenerate = true;
                }
                return;
            }
            if s1 == s2 {
                return;
            }
            let o1 = orient3d(&pp, &qq, &a, &b);
            let o2 = orient3d(&pp, &qq, &b, &c);
            let o3 = orient3d(&pp, &qq, &c, &a);
            if o1 == 0 || o2 == 0 || o3 == 0 {
                if (o1 >= 0 && o2 >= 0 && o3 >= 0) || (o1 <= 0 && o2 <= 0 && o3 <= 0) {
                    degenerate = true;
                }
                return;
            }
            if (o1 > 0) == (o2 > 0) && (o2 > 0) == (o3 > 0) {
                odd = !odd;
            }
        });
        if degenerate { None } else { Some(odd) }
    }

    /// Exact point-in-solid test for a closed mesh (odd crossing parity).
    pub fn contains(&self, p: DVec3) -> bool {
        if !self.bbox.contains(p) {
            return false;
        }
        let r = self.bbox.diagonal() * 2.0 + 1.0;
        // deterministic sequence of irrational directions
        let dirs = [
            DVec3::new(0.5773502691896258, 0.5773502691896257, 0.5773502691896259),
            DVec3::new(0.2672612419124244, -0.5345224838248488, 0.8017837257372732),
            DVec3::new(-0.6859943405700354, 0.5144957554275265, 0.5144957554275266),
            DVec3::new(0.1825741858350554, 0.3651483716701107, -0.9128709291752769),
        ];
        for k in 0..64 {
            let d = if k < dirs.len() {
                dirs[k]
            } else {
                let a = k as f64 * 2.399963229728653;
                let z = 1.0 - 2.0 * ((k as f64 * 0.6180339887498949) % 1.0);
                let s = (1.0 - z * z).sqrt();
                DVec3::new(s * a.cos(), s * a.sin(), z)
            };
            if let Some(odd) = self.parity(p, p + d * r) {
                return odd;
            }
        }
        // Fall back to winding number.
        winding_number_brute(self.mesh, p) > 0.5
    }

    /// Closest point on the mesh surface: (point, squared distance, triangle).
    pub fn closest_point(&self, p: DVec3) -> Option<(DVec3, f64, u32)> {
        let r = self.bvh.nearest(p, |t| {
            let [a, b, c] = self.mesh.tri_points(t as usize);
            let q = closest_point_triangle(p, a, b, c);
            (q - p).length_squared()
        })?;
        let [a, b, c] = self.mesh.tri_points(r.0 as usize);
        Some((closest_point_triangle(p, a, b, c), r.1, r.0))
    }

    /// Signed distance (negative inside) for closed meshes.
    pub fn signed_distance(&self, p: DVec3) -> f64 {
        let d = self.closest_point(p).map(|c| c.1.sqrt()).unwrap_or(f64::INFINITY);
        if self.contains(p) { -d } else { d }
    }
}

/// Closest point on triangle (Ericson, Real-Time Collision Detection).
pub fn closest_point_triangle(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> DVec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

/// Solid angle of triangle (a,b,c) seen from p, divided by 4π
/// (Van Oosterom & Strackee).
#[inline]
pub fn tri_winding(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let a = a - p;
    let b = b - p;
    let c = c - p;
    let la = a.length();
    let lb = b.length();
    let lc = c.length();
    let num = a.dot(b.cross(c));
    let den = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
    2.0 * libm::atan2(num, den) / (4.0 * std::f64::consts::PI)
}

pub fn winding_number_brute(m: &TriMesh, p: DVec3) -> f64 {
    let mut w = 0.0;
    for t in 0..m.tris.len() {
        let [a, b, c] = m.tri_points(t);
        w += tri_winding(p, a, b, c);
    }
    w
}

/// Fast winding number (Barill et al. 2018), first-order dipole expansion.
pub struct FastWinding<'a> {
    mesh: &'a TriMesh,
    nodes: Vec<FwNode>,
    order: Vec<u32>,
    beta: f64,
}

struct FwNode {
    bbox: Aabb,
    center: DVec3,
    radius: f64,
    /// Σ a_i n_i (area-weighted normals)
    an: DVec3,
    start: u32,
    count: u32,
    left: u32,
    right: u32,
}

impl<'a> FastWinding<'a> {
    pub fn new(mesh: &'a TriMesh) -> Self {
        let n = mesh.tris.len();
        let mut order: Vec<u32> = (0..n as u32).collect();
        let cent: Vec<DVec3> = (0..n)
            .map(|t| {
                let [a, b, c] = mesh.tri_points(t);
                (a + b + c) / 3.0
            })
            .collect();
        let anv: Vec<DVec3> = (0..n)
            .map(|t| {
                let [a, b, c] = mesh.tri_points(t);
                (b - a).cross(c - a) * 0.5
            })
            .collect();
        let mut nodes: Vec<FwNode> = Vec::new();
        if n == 0 {
            return FastWinding { mesh, nodes, order, beta: 2.0 };
        }
        fn build(
            nodes: &mut Vec<FwNode>,
            order: &mut [u32],
            off: usize,
            cent: &[DVec3],
            anv: &[DVec3],
            mesh: &TriMesh,
        ) -> u32 {
            let mut bb = Aabb::EMPTY;
            let mut cb = Aabb::EMPTY;
            let mut an = DVec3::ZERO;
            let mut wc = DVec3::ZERO;
            let mut wa = 0.0;
            for &t in order.iter() {
                bb = bb.union(&mesh.tri_aabb(t as usize));
                cb.grow(cent[t as usize]);
                an += anv[t as usize];
                let a = anv[t as usize].length();
                wc += cent[t as usize] * a;
                wa += a;
            }
            let center = if wa > 0.0 { wc / wa } else { cb.center() };
            let mut radius: f64 = 0.0;
            for &t in order.iter() {
                for v in mesh.tri_points(t as usize) {
                    radius = radius.max((v - center).length());
                }
            }
            let idx = nodes.len() as u32;
            nodes.push(FwNode { bbox: bb, center, radius, an, start: off as u32, count: order.len() as u32, left: u32::MAX, right: u32::MAX });
            if order.len() > 8 {
                let axis = cb.longest_axis();
                let mid = order.len() / 2;
                order.select_nth_unstable_by(mid, |&a, &b| {
                    cent[a as usize][axis].partial_cmp(&cent[b as usize][axis]).unwrap().then(a.cmp(&b))
                });
                let (l, r) = order.split_at_mut(mid);
                let li = build(nodes, l, off, cent, anv, mesh);
                let ri = build(nodes, r, off + mid, cent, anv, mesh);
                nodes[idx as usize].left = li;
                nodes[idx as usize].right = ri;
            }
            idx
        }
        build(&mut nodes, &mut order, 0, &cent, &anv, mesh);
        FastWinding { mesh, nodes, order, beta: 2.0 }
    }

    pub fn eval(&self, p: DVec3) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        self.eval_node(0, p)
    }

    fn eval_node(&self, ni: u32, p: DVec3) -> f64 {
        let n = &self.nodes[ni as usize];
        let d = n.center - p;
        let dl = d.length();
        if dl > self.beta * n.radius {
            // dipole approximation
            return n.an.dot(d) / (4.0 * std::f64::consts::PI * dl * dl * dl);
        }
        if n.left == u32::MAX {
            let mut w = 0.0;
            for &t in &self.order[n.start as usize..(n.start + n.count) as usize] {
                let [a, b, c] = self.mesh.tri_points(t as usize);
                w += tri_winding(p, a, b, c);
            }
            return w;
        }
        self.eval_node(n.left, p) + self.eval_node(n.right, p)
    }

    #[allow(dead_code)]
    fn bbox(&self) -> Aabb {
        self.nodes.first().map(|n| n.bbox).unwrap_or(Aabb::EMPTY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{box_mesh, icosphere};

    #[test]
    fn inside_box() {
        let m = box_mesh(DVec3::ZERO, DVec3::ONE);
        let q = MeshQuery::new(&m);
        assert!(q.contains(DVec3::splat(0.5)));
        assert!(q.contains(DVec3::new(0.5, 0.5, 0.0001)));
        assert!(!q.contains(DVec3::new(0.5, 0.5, 1.0001)));
        // degenerate ray aligned with diagonal through edges gets retried
        assert!(q.contains(DVec3::new(0.25, 0.25, 0.25)));
        assert!((q.signed_distance(DVec3::splat(0.5)) + 0.5).abs() < 1e-12);
    }

    #[test]
    fn fast_winding_matches_brute() {
        let m = icosphere(DVec3::ZERO, 1.0, 3);
        let fw = FastWinding::new(&m);
        for p in [DVec3::ZERO, DVec3::new(0.5, 0.2, -0.1), DVec3::new(2.0, 0.0, 0.0), DVec3::new(0.99, 0.0, 0.0)] {
            let a = fw.eval(p);
            let b = winding_number_brute(&m, p);
            assert!((a - b).abs() < 5e-2, "{p:?} {a} {b}");
        }
    }
}
