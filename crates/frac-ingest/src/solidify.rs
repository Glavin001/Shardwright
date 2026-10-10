//! Robust solidification of messy meshes (spec Stage 0): classify space with
//! the generalized winding number (fast Barnes–Hut evaluation) on a grid and
//! extract the 0.5 level set with marching tetrahedra over a consistent
//! Freudenthal subdivision, which yields a closed 2-manifold by construction.
//! Vertices are then projected onto the original surface where that does not
//! fold triangles, to preserve sharp features.

use frac_geom::inside::{FastWinding, MeshQuery};
use frac_geom::{DVec3, TriMesh};
use rayon::prelude::*;
use std::collections::BTreeMap;

/// The 6 tetrahedra of the Freudenthal (Kuhn) subdivision of a cube, as
/// corner indices (bit 0 = x, bit 1 = y, bit 2 = z). All share the 0-7
/// diagonal, which makes the subdivision consistent across cubes.
const KUHN: [[usize; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 3, 2, 7],
    [0, 2, 6, 7],
    [0, 6, 4, 7],
    [0, 4, 5, 7],
    [0, 5, 1, 7],
];

pub fn solidify(m: &TriMesh, resolution: u32) -> Result<TriMesh, String> {
    if m.tris.is_empty() {
        return Err("empty mesh".into());
    }
    let bb = m.aabb();
    let ext = bb.extent();
    let longest = ext.x.max(ext.y).max(ext.z).max(1e-12);
    let h = longest / resolution.max(8) as f64;
    let pad = 2.0 * h;
    let lo = bb.min - DVec3::splat(pad);
    let dims = [0, 1, 2].map(|k| (((ext[k] + 2.0 * pad) / h).ceil() as usize + 1).max(2));
    let fw = FastWinding::new(m);
    let n = dims[0] * dims[1] * dims[2];
    let idx = |i: usize, j: usize, k: usize| (k * dims[1] + j) * dims[0] + i;
    let pos = |i: usize, j: usize, k: usize| lo + DVec3::new(i as f64, j as f64, k as f64) * h;
    let vals: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|g| {
            let i = g % dims[0];
            let j = (g / dims[0]) % dims[1];
            let k = g / (dims[0] * dims[1]);
            fw.eval(pos(i, j, k)) - 0.5
        })
        .collect();
    // inside iff value >= 0 (ties count as inside: avoids iso-valued nodes)
    let inside = |g: usize| vals[g] >= 0.0;
    let mut verts: Vec<DVec3> = Vec::new();
    let mut edge_vert: BTreeMap<(usize, usize), u32> = BTreeMap::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut get_vert = |a: usize, b: usize, pa: DVec3, pb: DVec3, verts: &mut Vec<DVec3>| -> u32 {
        let key = (a.min(b), a.max(b));
        *edge_vert.entry(key).or_insert_with(|| {
            let (va, vb) = (vals[a], vals[b]);
            // keep vertices strictly inside grid edges so no triangle degenerates
            let t = (va / (va - vb)).clamp(1e-3, 1.0 - 1e-3);
            verts.push(pa + (pb - pa) * t);
            (verts.len() - 1) as u32
        })
    };
    for k in 0..dims[2] - 1 {
        for j in 0..dims[1] - 1 {
            for i in 0..dims[0] - 1 {
                let corner = |c: usize| (i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
                let cg: [usize; 8] = std::array::from_fn(|c| {
                    let (a, b, cc) = corner(c);
                    idx(a, b, cc)
                });
                let ins: [bool; 8] = std::array::from_fn(|c| inside(cg[c]));
                if ins.iter().all(|&x| x) || ins.iter().all(|&x| !x) {
                    continue;
                }
                let cp: [DVec3; 8] = std::array::from_fn(|c| {
                    let (a, b, cc) = corner(c);
                    pos(a, b, cc)
                });
                for tet in KUHN {
                    let tin: Vec<usize> = tet.iter().copied().filter(|&c| ins[c]).collect();
                    let tout: Vec<usize> = tet.iter().copied().filter(|&c| !ins[c]).collect();
                    if tin.is_empty() || tout.is_empty() {
                        continue;
                    }
                    let cin = tin.iter().fold(DVec3::ZERO, |a, &c| a + cp[c]) / tin.len() as f64;
                    let cout = tout.iter().fold(DVec3::ZERO, |a, &c| a + cp[c]) / tout.len() as f64;
                    let outward = cout - cin;
                    let emit =
                        |a: u32, b: u32, c: u32, verts: &Vec<DVec3>, tris: &mut Vec<[u32; 3]>| {
                            let n = (verts[b as usize] - verts[a as usize])
                                .cross(verts[c as usize] - verts[a as usize]);
                            if n.dot(outward) >= 0.0 {
                                tris.push([a, b, c]);
                            } else {
                                tris.push([a, c, b]);
                            }
                        };
                    if tin.len() == 1 || tout.len() == 1 {
                        let (lone, others) = if tin.len() == 1 {
                            (tin[0], &tout)
                        } else {
                            (tout[0], &tin)
                        };
                        let v: Vec<u32> = others
                            .iter()
                            .map(|&o| get_vert(cg[lone], cg[o], cp[lone], cp[o], &mut verts))
                            .collect();
                        emit(v[0], v[1], v[2], &verts, &mut tris);
                    } else {
                        // 2-2 split: quad
                        let (a, b) = (tin[0], tin[1]);
                        let (c, d) = (tout[0], tout[1]);
                        let v_ac = get_vert(cg[a], cg[c], cp[a], cp[c], &mut verts);
                        let v_ad = get_vert(cg[a], cg[d], cp[a], cp[d], &mut verts);
                        let v_bc = get_vert(cg[b], cg[c], cp[b], cp[c], &mut verts);
                        let v_bd = get_vert(cg[b], cg[d], cp[b], cp[d], &mut verts);
                        // quad cycle: ac - ad - bd - bc
                        emit(v_ac, v_ad, v_bd, &verts, &mut tris);
                        emit(v_ac, v_bd, v_bc, &verts, &mut tris);
                    }
                }
            }
        }
    }
    let mut out = TriMesh { verts, tris };
    out = out.compact();
    let topo = out.topology();
    if !topo.is_closed_manifold() {
        return Err(format!("level set not manifold: {topo:?}"));
    }
    // feature-preserving projection onto the source surface
    let projected = project_to_surface(&out, m, 0.5 * h);
    if projected.topology().is_closed_manifold()
        && projected.self_intersections(1).is_empty()
        && projected.signed_volume() > 0.0
    {
        out = projected;
    }
    if out.signed_volume() <= 0.0 {
        return Err("empty level set".into());
    }
    Ok(out)
}

/// Move vertices to the closest point of `src` when within `maxd`, rejecting
/// moves that flip adjacent triangle normals.
fn project_to_surface(m: &TriMesh, src: &TriMesh, maxd: f64) -> TriMesh {
    let q = MeshQuery::new(src);
    let targets: Vec<Option<DVec3>> = m
        .verts
        .par_iter()
        .map(|&v| {
            q.closest_point(v)
                .and_then(|(p, d2, _)| if d2.sqrt() <= maxd { Some(p) } else { None })
        })
        .collect();
    let mut out = m.clone();
    let mut vt: Vec<Vec<u32>> = vec![Vec::new(); m.verts.len()];
    for (t, tri) in m.tris.iter().enumerate() {
        for &v in tri {
            vt[v as usize].push(t as u32);
        }
    }
    for (v, tgt) in targets.iter().enumerate() {
        let Some(p) = *tgt else { continue };
        let old = out.verts[v];
        let normals_before: Vec<DVec3> = vt[v].iter().map(|&t| tri_normal(&out, t)).collect();
        out.verts[v] = p;
        let ok = vt[v].iter().zip(normals_before.iter()).all(|(&t, nb)| {
            let na = tri_normal(&out, t);
            na.length_squared() > 0.0 && na.normalize().dot(nb.normalize()) > 0.2
        });
        if !ok {
            out.verts[v] = old;
        }
    }
    out
}

fn tri_normal(m: &TriMesh, t: u32) -> DVec3 {
    let [a, b, c] = m.tri_points(t as usize);
    (b - a).cross(c - a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use frac_geom::mesh::{box_mesh, icosphere};

    #[test]
    fn solidify_open_box() {
        // remove two triangles: open box (non-watertight)
        let mut m = box_mesh(DVec3::ZERO, DVec3::ONE);
        m.tris.truncate(10);
        let s = solidify(&m, 32).unwrap();
        assert!(s.topology().is_closed_manifold());
        let v = s.signed_volume();
        assert!((v - 1.0).abs() < 0.1, "volume {v}");
    }

    #[test]
    fn solidify_overlapping_spheres() {
        let mut m = icosphere(DVec3::ZERO, 1.0, 2);
        m.append(&icosphere(DVec3::new(0.8, 0.0, 0.0), 0.7, 2));
        assert!(!m.self_intersections(1).is_empty());
        let s = solidify(&m, 40).unwrap();
        assert!(s.topology().is_closed_manifold());
        assert!(s.self_intersections(1).is_empty());
    }
}
