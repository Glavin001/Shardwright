mod common;
use common::*;
use frac_io::*;
use serde_json::json;

fn inputs() -> (Vec<u8>, Vec<u8>, serde_json::Value) {
    let mut physics = sample_asset();
    // The shared render fixture has three visible fragments; others are physics-only.
    for f in &mut physics.hierarchy.fragments[3..] {
        f.render.gltf_node = -1;
    }
    (
        write_glb(
            &sample_render_scene(),
            &GltfOptions {
                meshopt_compression: false,
            },
        )
        .unwrap(),
        write_physics(&physics),
        json!({"library_id":"builtin","library_version":"1","material":(0..8).map(|i| json!({"id":format!("material-{i}"),"density":1000.0})).collect::<Vec<_>>(),"interface_material":(0..10).map(|i| json!({"id":format!("joint-{i}")})).collect::<Vec<_>>()}),
    )
}
#[test]
fn standalone_roundtrip_preserves_render_and_every_physics_byte() {
    let (glb, physics, materials) = inputs();
    let out = write_fracture_glb(&glb, &physics, &materials).unwrap();
    let (extracted, library) = read_fracture_glb(&out).unwrap();
    assert_eq!(extracted, physics);
    assert_eq!(library, materials);
    let before = read_glb_summary(&glb).unwrap();
    let after = read_glb_summary(&out).unwrap();
    assert_eq!(before.mesh_triangles, after.mesh_triangles);
    assert_eq!(before.node_extras, after.node_extras);
    assert_eq!(before.node_parent, after.node_parent);
    assert!(after.extensions_used.contains(&FRACTURE_EXTENSION.into()));
    assert!(
        !after
            .extensions_required
            .contains(&FRACTURE_EXTENSION.into())
    );
    assert_eq!(out, write_fracture_glb(&glb, &physics, &materials).unwrap());
    let path = temp_dir("fracture-glb").join("sample.glb");
    std::fs::write(&path, out).unwrap();
    if let Some(result) = khronos_validate(&path) {
        assert_eq!(result.unwrap()["issues"]["numErrors"], 0);
    }
}
#[test]
fn rejects_wrong_materials_wrong_model_and_corrupt_payload() {
    let (glb, physics, mut materials) = inputs();
    materials["library_version"] = json!("other");
    assert!(write_fracture_glb(&glb, &physics, &materials).is_err());
    materials["library_version"] = json!("1");
    let mut asset = sample_asset();
    asset.hierarchy.fragments[0].render.gltf_node = 0; // root has no fragment ID
    assert!(write_fracture_glb(&glb, &write_physics(&asset), &materials).is_err());
    assert!(write_fracture_glb(&glb, b"garbage", &materials).is_err());
    assert!(read_fracture_glb(&glb).is_err());
    let out = write_fracture_glb(&glb, &physics, &materials).unwrap();
    assert!(read_fracture_glb(&out[..out.len() - 32]).is_err());
    assert!(write_fracture_glb(&out, &physics, &materials).is_err());
}

#[test]
fn preserves_self_contained_meshopt_rendering() {
    let (_, physics, materials) = inputs();
    let glb = write_glb(
        &sample_render_scene(),
        &GltfOptions {
            meshopt_compression: true,
        },
    )
    .unwrap();
    let bundled = write_fracture_glb(&glb, &physics, &materials).unwrap();
    assert_eq!(read_fracture_glb(&bundled).unwrap().0, physics);
    let decoded = decompress_meshopt_glb(&bundled).unwrap().unwrap();
    assert_eq!(
        read_glb_summary(&decoded).unwrap().mesh_triangles,
        read_glb_summary(
            &write_glb(
                &sample_render_scene(),
                &GltfOptions {
                    meshopt_compression: false
                }
            )
            .unwrap()
        )
        .unwrap()
        .mesh_triangles
    );
}
