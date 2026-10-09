//! Synthetic solids and helpers for the known-answer tests.
#![allow(dead_code)]

use frac_fem::{tetrahedralize, ElasticMaterial, TetMesh};
use frac_geom::{DVec2, DVec3, TriMesh};
use frac_modes::*;

fn cross2(o: DVec2, a: DVec2, b: DVec2) -> f64 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

/// Ear-clipping triangulation of a simple CCW polygon.
pub fn triangulate(poly: &[DVec2]) -> Vec<[u32; 3]> {
    let mut idx: Vec<usize> = (0..poly.len()).collect();
    let mut tris = Vec::new();
    while idx.len() > 3 {
        let n = idx.len();
        let mut found = false;
        for i in 0..n {
            let (a, b, c) = (idx[(i + n - 1) % n], idx[i], idx[(i + 1) % n]);
            if cross2(poly[a], poly[b], poly[c]) <= 1e-14 {
                continue;
            }
            let inside = idx.iter().any(|&j| {
                j != a && j != b && j != c && {
                    let p = poly[j];
                    cross2(poly[a], poly[b], p) >= 0.0 && cross2(poly[b], poly[c], p) >= 0.0 && cross2(poly[c], poly[a], p) >= 0.0
                }
            });
            if inside {
                continue;
            }
            tris.push([a as u32, b as u32, c as u32]);
            idx.remove(i);
            found = true;
            break;
        }
        assert!(found, "ear clipping failed");
    }
    tris.push([idx[0] as u32, idx[1] as u32, idx[2] as u32]);
    tris
}

/// Extrudes a simple CCW polygon between z0 < z1 into a closed outward-oriented mesh.
pub fn extrude(poly: &[DVec2], z0: f64, z1: f64) -> TriMesh {
    let n = poly.len() as u32;
    let mut verts: Vec<DVec3> = poly.iter().map(|p| DVec3::new(p.x, p.y, z0)).collect();
    verts.extend(poly.iter().map(|p| DVec3::new(p.x, p.y, z1)));
    let mut tris = Vec::new();
    for t in triangulate(poly) {
        tris.push([t[0], t[2], t[1]]); // bottom faces down
        tris.push([t[0] + n, t[1] + n, t[2] + n]);
    }
    for i in 0..n {
        let j = (i + 1) % n;
        tris.push([i, j, j + n]);
        tris.push([i, j + n, i + n]);
    }
    TriMesh { verts, tris }
}

/// Extrudes the region between an outer and an inner CCW ring with equal vertex
/// counts (both star-shaped w.r.t. the same center, matched by angle).
pub fn extrude_ring(outer: &[DVec2], inner: &[DVec2], z0: f64, z1: f64) -> TriMesh {
    let n = outer.len() as u32;
    assert_eq!(outer.len(), inner.len());
    let mut verts = Vec::new();
    for z in [z0, z1] {
        verts.extend(outer.iter().map(|p| DVec3::new(p.x, p.y, z)));
        verts.extend(inner.iter().map(|p| DVec3::new(p.x, p.y, z)));
    }
    let (ob, ib, ot, it) = (0, n, 2 * n, 3 * n);
    let mut tris = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        // top (normal +z): outer ccw, inner
        tris.push([ot + i, ot + j, it + j]);
        tris.push([ot + i, it + j, it + i]);
        // bottom (normal -z)
        tris.push([ob + i, ib + j, ob + j]);
        tris.push([ob + i, ib + i, ib + j]);
        // outer wall (outward)
        tris.push([ob + i, ob + j, ot + j]);
        tris.push([ob + i, ot + j, ot + i]);
        // inner wall (normal towards the hole center)
        tris.push([ib + i, it + j, ib + j]);
        tris.push([ib + i, it + i, it + j]);
    }
    TriMesh { verts, tris }
}

/// Labels tets by centroid into a `dims` grid over [lo, hi]; empty cells are
/// removed and labels compacted (sorted by grid index).
pub fn grid_cells(m: &TetMesh, lo: [f64; 3], hi: [f64; 3], dims: [usize; 3]) -> (Vec<u32>, u32) {
    let raw: Vec<usize> = (0..m.tets.len())
        .map(|t| {
            let c = m.tet_centroid(t);
            let mut id = [0usize; 3];
            for k in 0..3 {
                let f = ((c[k] - lo[k]) / (hi[k] - lo[k]) * dims[k] as f64).floor();
                id[k] = (f.max(0.0) as usize).min(dims[k] - 1);
            }
            id[0] + dims[0] * (id[1] + dims[1] * id[2])
        })
        .collect();
    let mut used: Vec<usize> = raw.clone();
    used.sort_unstable();
    used.dedup();
    let labels = raw.iter().map(|r| used.binary_search(r).unwrap() as u32).collect();
    (labels, used.len() as u32)
}

/// Area-weighted centroid of each group's fault faces.
pub fn group_centroids(m: &TetMesh, cells: &[u32], groups: &[(u32, u32)]) -> Vec<[f64; 3]> {
    let mut acc = vec![[0.0; 4]; groups.len()];
    let f = m.sorted_faces();
    for w in f.windows(2) {
        if w[0].0 != w[1].0 {
            continue;
        }
        let (a, b) = (cells[w[0].1 as usize], cells[w[1].1 as usize]);
        if a == b {
            continue;
        }
        let key = (a.min(b), a.max(b));
        let g = groups.binary_search(&key).unwrap();
        let p = w[0].0.map(|v| m.verts[v as usize]);
        let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
        let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
        let c = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let ar = 0.5 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        for k in 0..3 {
            acc[g][k] += ar * (p[0][k] + p[1][k] + p[2][k]) / 3.0;
        }
        acc[g][3] += ar;
    }
    acc.iter().map(|a| [a[0] / a[3], a[1] / a[3], a[2] / a[3]]).collect()
}

pub fn steel() -> ElasticMaterial {
    ElasticMaterial::isotropic(200e9, 0.3, 7850.0)
}

pub struct Case {
    pub mesh: TetMesh,
    pub cells: Vec<u32>,
    pub n_cells: u32,
    pub mats: Vec<ElasticMaterial>,
}

pub fn case(solid: &TriMesh, h: f64, lo: [f64; 3], hi: [f64; 3], dims: [usize; 3]) -> Case {
    let mesh = tetrahedralize(solid, h, 0);
    let (cells, n_cells) = grid_cells(&mesh, lo, hi, dims);
    let mats = vec![steel(); mesh.tets.len()];
    Case { mesh, cells, n_cells, mats }
}

pub fn run(c: &Case, w: &(dyn Fn(u32, u32) -> f64 + Sync), anchors: &[u32], params: ModesParams) -> ModesOutput {
    let input = ModesInput {
        mesh: &c.mesh,
        tet_material: &c.mats,
        tet_cell: &c.cells,
        group_weight: w,
        anchored_vertices: anchors,
        params,
    };
    let out = compute_modes(&input).unwrap();
    eprintln!(
        "tets {} cells {} groups {} dofs {} solver {} iters {:?} conv {:?}",
        c.mesh.tets.len(),
        c.n_cells,
        out.groups.len(),
        out.n_dofs,
        out.solver_used,
        out.iterations,
        out.converged
    );
    eprintln!("timings {:?}", out.timings_ms);
    out
}

pub fn notched_bar() -> TriMesh {
    // 4 x 1 x 1 bar with a 0.2-wide, 0.5-deep notch from the top at x = 2
    let poly = [
        DVec2::new(0.0, 0.0),
        DVec2::new(4.0, 0.0),
        DVec2::new(4.0, 1.0),
        DVec2::new(2.1, 1.0),
        DVec2::new(2.1, 0.5),
        DVec2::new(1.9, 0.5),
        DVec2::new(1.9, 1.0),
        DVec2::new(0.0, 1.0),
    ];
    extrude(&poly, 0.0, 1.0)
}
