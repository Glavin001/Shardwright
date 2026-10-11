mod common;

use common::*;
use frac_geom::TriMesh;
use frac_io::*;
use glam::DVec3;

#[test]
fn asset_json_is_deterministic_and_exact() {
    let mut asset = sample_asset();
    asset.bonds[0].area = 0.1 + 0.2; // not representable as a short decimal
    let a = asset_to_json(&asset);
    assert_eq!(a, asset_to_json(&asset));
    let back: frac_core::Asset = serde_json::from_str(&a).unwrap();
    assert_eq!(back.bonds[0].area, 0.1 + 0.2);
    assert_eq!(asset_to_json(&back), a);
}

fn tet() -> TriMesh {
    TriMesh {
        verts: vec![
            DVec3::ZERO,
            DVec3::X,
            DVec3::Y,
            DVec3::new(0.1, 0.2, 1.0 / 3.0),
        ],
        tris: vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]],
    }
}

/// Triangles as position triples (independent of vertex numbering).
fn soup(m: &TriMesh) -> Vec<[DVec3; 3]> {
    m.tris
        .iter()
        .map(|t| t.map(|i| m.verts[i as usize]))
        .collect()
}

#[test]
fn ply_and_obj_dumps_round_trip() {
    let dir = temp_dir("dumps");
    let m = tet();
    write_ply(dir.join("t.ply"), &m).unwrap();
    write_obj(dir.join("t.obj"), &m).unwrap();
    let opts = ImportOptions::default();
    let p = load_scene(&dir.join("t.ply"), &opts).unwrap();
    assert_eq!(
        soup(&p.parts[0].mesh),
        soup(&m),
        "PLY dump is exact (doubles)"
    );
    let o = load_scene(&dir.join("t.obj"), &opts).unwrap();
    assert_eq!(
        soup(&o.parts[0].mesh),
        soup(&m),
        "OBJ dump is exact (f64 tobj)"
    );

    let polys = vec![vec![0, 1, 2], vec![0, 1, 3]];
    write_ply_polys(
        dir.join("sub/polys.ply"),
        &m.verts,
        &polys,
        Some(&[[255, 0, 0], [0, 255, 0]]),
    )
    .unwrap();
    let text = std::fs::read_to_string(dir.join("sub/polys.ply")).unwrap();
    assert!(text.contains("property uchar red"));
    assert!(text.trim_end().ends_with("3 0 1 3 0 255 0"));
    let q = load_scene(&dir.join("sub/polys.ply"), &opts).unwrap();
    assert_eq!(q.parts[0].mesh.tris.len(), 2);
    assert!(write_ply_polys(dir.join("x.ply"), &m.verts, &polys, Some(&[[1, 2, 3]])).is_err());
}
