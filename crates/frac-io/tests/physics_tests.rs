mod common;

use common::*;
use frac_io::*;

#[test]
fn physics_round_trip() {
    let asset = sample_asset();
    let bytes = write_physics(&asset);
    assert_eq!(&bytes[4..8], b"FRAC");
    let v = read_physics(&bytes).unwrap();
    assert_eq!(v.schema_version, frac_core::SCHEMA_VERSION);
    assert_eq!(v.name, "sample");
    assert_eq!(v.seed, 42);
    assert_eq!(v.variant, 1);
    assert_eq!(v.levels, 2);
    assert_eq!(v.fragment_count, 5);
    assert_eq!(v.bond_count, 6);
    assert_eq!(v.hull_count, 5);
    assert_eq!(v.loop_count, 6);
    assert_eq!(v.spawn_count, 12);
    assert_eq!(v.bond_children_count, 3);
    let lab: Vec<(u8, u32, i32)> = v.bonds.iter().map(|b| (b.level, b.a, b.b)).collect();
    assert_eq!(
        lab,
        vec![
            (0, 0, 1),
            (0, 0, -1),
            (1, 2, 3),
            (1, 2, -1),
            (1, 3, 4),
            (1, 3, -1)
        ]
    );
    assert_eq!(v.bonds[1].area, 2.0);
    let fr: Vec<(u8, f32, i32)> = v
        .fragments
        .iter()
        .map(|f| (f.level, f.mass, f.parent))
        .collect();
    assert_eq!(
        fr,
        vec![
            (0, 200.0, -1),
            (0, 100.0, -1),
            (1, 120.0, 0),
            (1, 80.0, 0),
            (1, 100.0, 1)
        ]
    );

    // Field-level checks through the generated bindings.
    let root = fb::root_as_asset(&bytes).unwrap();
    let b0 = root.bonds().unwrap().get(0);
    assert_eq!(b0.rebar_dir().unwrap().x(), 1.0, "rebar dir normalized");
    assert_eq!(b0.rebar_count(), 2);
    assert_eq!(b0.extent_half().unwrap().x(), 0.5);
    assert_eq!(b0.composition().unwrap().len(), 2);
    assert_eq!(b0.composition().unwrap().get(0).interface_material(), -1);
    assert_eq!(b0.composition().unwrap().get(1).interface_material(), 9);
    assert_eq!(b0.composition().unwrap().get(1).kind(), 1);
    assert_eq!((b0.child_begin(), b0.child_count()), (0, 1));
    assert_eq!(b0.interface_count(), 2);
    assert!(root.bonds().unwrap().get(1).anchor());
    assert_eq!(root.bonds().unwrap().get(4).parent_bond(), 0);
    assert_eq!(
        root.bond_children().unwrap().iter().collect::<Vec<_>>(),
        vec![4, 3, 5]
    );
    let f0 = root.fragments().unwrap().get(0);
    assert_eq!((f0.child_begin(), f0.child_count()), (2, 2));
    assert_eq!((f0.cell_begin(), f0.cell_count()), (0, 2));
    assert_eq!(f0.gltf_node(), 1);
    assert_eq!(f0.role(), frac_core::ComponentRole::Wall as u8);
    assert_eq!(
        f0.material_ids().unwrap().iter().collect::<Vec<_>>(),
        vec![3, 7]
    );
    let inertia: Vec<f32> = f0.inertia().unwrap().m().iter().collect();
    assert_eq!(
        inertia,
        vec![200.0, 0.0, 0.0, 0.0, 400.0, 0.0, 0.0, 0.0, 600.0]
    );
    let hf = root.hull_faces().unwrap().get(0);
    assert_eq!(hf.face_sizes().unwrap().len(), 6);
    assert_eq!(hf.indices().unwrap().len(), 24);
    assert_eq!(
        root.level_fragment_begin()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![0, 2]
    );
    assert_eq!(
        root.level_fragment_count()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![2, 3]
    );

    // JSON dump parses and carries the data.
    let j: serde_json::Value = serde_json::from_str(&physics_to_json(&bytes).unwrap()).unwrap();
    assert_eq!(j["bonds"].as_array().unwrap().len(), 6);
    assert_eq!(j["bonds"][1]["b"], -1);
    assert_eq!(j["fragments"][2]["parent"], 0);
}

#[test]
fn physics_is_deterministic() {
    let a = write_physics(&sample_asset());
    let b = write_physics(&sample_asset());
    assert_eq!(a, b);
    assert_eq!(physics_to_json(&a).unwrap(), physics_to_json(&b).unwrap());
}

#[test]
fn physics_rejects_garbage() {
    assert!(read_physics(b"not a flatbuffer").is_err());
    let mut bytes = write_physics(&sample_asset());
    bytes[4..8].copy_from_slice(b"XXXX");
    assert!(read_physics(&bytes).is_err());
    let bytes = write_physics(&sample_asset());
    assert!(read_physics(&bytes[..bytes.len() / 2]).is_err());
    // Empty asset is valid.
    let e = write_physics(&frac_core::Asset::default());
    assert_eq!(read_physics(&e).unwrap().fragment_count, 0);
}

#[test]
fn physics_flatc_schema_gate() {
    let dir = temp_dir("flatc");
    let path = dir.join("sample.fracphys");
    std::fs::write(&path, write_physics(&sample_asset())).unwrap();
    match flatc_validate(&path) {
        None => eprintln!("SKIP: flatc not available"),
        Some(r) => r.unwrap(),
    }
    let bad = dir.join("bad.fracphys");
    std::fs::write(&bad, b"FRACFRACFRAC").unwrap();
    if let Some(r) = flatc_validate(&bad) {
        assert!(r.is_err());
    }
}
