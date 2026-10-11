//! Procedural closed-mesh builders for benchmarks and tests.

use frac_geom::{DVec3, TriMesh};

/// Extrude a polygon with holes (outer CCW, holes CW, in the xy plane)
/// between z0 and z1.
pub fn extrude(loops: &[Vec<[f64; 2]>], z0: f64, z1: f64) -> TriMesh {
    let mut pts2: Vec<[f64; 2]> = Vec::new();
    let mut idx_loops: Vec<Vec<usize>> = Vec::new();
    for l in loops {
        let start = pts2.len();
        pts2.extend_from_slice(l);
        idx_loops.push((start..start + l.len()).collect());
    }
    let n = pts2.len() as u32;
    let mut verts: Vec<DVec3> = pts2.iter().map(|p| DVec3::new(p[0], p[1], z0)).collect();
    verts.extend(pts2.iter().map(|p| DVec3::new(p[0], p[1], z1)));
    let caps = frac_cells::tri2d::triangulate(&pts2, &idx_loops);
    let mut tris = Vec::new();
    for t in &caps {
        tris.push([t[0] as u32, t[2] as u32, t[1] as u32]);
        tris.push([t[0] as u32 + n, t[1] as u32 + n, t[2] as u32 + n]);
    }
    for l in &idx_loops {
        for k in 0..l.len() {
            let a = l[k] as u32;
            let b = l[(k + 1) % l.len()] as u32;
            tris.push([a, b, b + n]);
            tris.push([a, b + n, a + n]);
        }
    }
    TriMesh { verts, tris }
}

/// Map an extrusion built in (u, v, w) to world axes.
pub fn reorient(m: &TriMesh, f: impl Fn(DVec3) -> DVec3, flip: bool) -> TriMesh {
    let mut out = m.transformed(f);
    if flip {
        out = out.flipped();
    }
    out
}

pub fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<[f64; 2]> {
    vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
}

pub fn rect_hole(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<[f64; 2]> {
    vec![[x0, y0], [x0, y1], [x1, y1], [x1, y0]]
}

/// Revolve a closed profile (r, y) around the Y axis. The profile must be
/// CCW in (r, y) with r > 0, except that consecutive points on the axis
/// (r = 0) are collapsed into poles.
pub fn revolve(profile: &[[f64; 2]], segs: usize) -> TriMesh {
    let np = profile.len();
    let mut verts: Vec<DVec3> = Vec::new();
    let mut id: Vec<Vec<u32>> = vec![Vec::new(); np];
    for (k, p) in profile.iter().enumerate() {
        if p[0] <= 0.0 {
            verts.push(DVec3::new(0.0, p[1], 0.0));
            id[k] = vec![(verts.len() - 1) as u32; segs];
        } else {
            for s in 0..segs {
                let a = std::f64::consts::TAU * s as f64 / segs as f64;
                verts.push(DVec3::new(p[0] * a.cos(), p[1], -p[0] * a.sin()));
                id[k].push((verts.len() - 1) as u32);
            }
        }
    }
    let mut tris = Vec::new();
    for k in 0..np {
        let k2 = (k + 1) % np;
        for s in 0..segs {
            let s2 = (s + 1) % segs;
            let (a, b, c, d) = (id[k][s], id[k2][s], id[k2][s2], id[k][s2]);
            let mut push = |t: [u32; 3]| {
                if t[0] != t[1] && t[1] != t[2] && t[0] != t[2] {
                    tris.push(t);
                }
            };
            push([a, b, c]);
            push([a, c, d]);
        }
    }
    let m = TriMesh { verts, tris };
    if m.signed_volume() < 0.0 {
        m.flipped()
    } else {
        m
    }
}

pub fn box_at(lo: DVec3, hi: DVec3) -> TriMesh {
    frac_geom::mesh::box_mesh(lo, hi)
}
