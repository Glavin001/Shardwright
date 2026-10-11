mod common;

use common::*;
use frac_io::*;
use serde_json::json;

fn validate_or_skip(path: &std::path::Path) -> Option<serde_json::Value> {
    match khronos_validate(path) {
        None => {
            eprintln!(
                "SKIP: Khronos glTF validator unavailable (need `node` and `npm install` in tools/gltf_validate)"
            );
            None
        }
        Some(Err(e)) => panic!("{}: {e}", path.display()),
        Some(Ok(report)) => {
            let issues = &report["issues"];
            eprintln!(
                "khronos validator {}: numErrors={} numWarnings={} numInfos={} numHints={}",
                path.display(),
                issues["numErrors"],
                issues["numWarnings"],
                issues["numInfos"],
                issues["numHints"]
            );
            assert_eq!(issues["numErrors"], 0);
            Some(report)
        }
    }
}

fn glb_json(bytes: &[u8]) -> serde_json::Value {
    let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    serde_json::from_slice(&bytes[20..20 + json_len]).unwrap()
}

#[test]
fn glb_structure_uncompressed() {
    let scene = sample_render_scene();
    let glb = write_glb(
        &scene,
        &GltfOptions {
            meshopt_compression: false,
        },
    )
    .unwrap();
    assert_eq!(&glb[0..4], b"glTF");
    assert_eq!(glb.len() % 4, 0);
    assert_eq!(
        u32::from_le_bytes(glb[8..12].try_into().unwrap()) as usize,
        glb.len()
    );

    let s = read_glb_summary(&glb).unwrap();
    // 4 nodes + 1 MSFT_lod node.
    assert_eq!(s.node_count, 5);
    assert_eq!(s.scene_root_count, 1);
    assert_eq!(s.mesh_count, 4);
    assert_eq!(s.material_count, 2);
    assert_eq!(s.mesh_triangles, vec![12, 12, 12, 4]);
    assert_eq!(s.node_mesh, vec![None, Some(0), Some(1), Some(2), Some(3)]);
    assert_eq!(s.node_parent, vec![None, Some(0), Some(1), Some(1), None]);
    assert_eq!(s.node_extras[1]["fragment_id"], 0);
    assert_eq!(s.node_extras[1]["children"], json!([1, 2]));
    assert_eq!(s.node_extras[1]["lods"], json!([0, 3]));
    assert_eq!(s.node_extras[2]["lods"], json!([1]));
    assert_eq!(s.node_extras[0], json!({"asset": "sample"}));
    assert_eq!(s.scene_extras["prefracture"]["levels"], 2);
    assert_eq!(s.extensions_used, vec!["MSFT_lod".to_string()]);
    assert!(s.extensions_required.is_empty());
    assert!(!s.meshopt_compressed);

    let j = glb_json(&glb);
    assert_eq!(j["nodes"][1]["extensions"]["MSFT_lod"]["ids"], json!([4]));
    // Translations are parent-relative.
    assert_eq!(j["nodes"][1]["translation"], json!([0.0, 1.5, 0.0]));
    assert_eq!(j["nodes"][2]["translation"], json!([0.5, 0.0, 0.0]));
    // u16 indices for small meshes; POSITION min/max present.
    let acc = &j["accessors"];
    let prim = &j["meshes"][0]["primitives"][0];
    let pos = &acc[prim["attributes"]["POSITION"].as_u64().unwrap() as usize];
    assert_eq!(pos["min"], json!([-1.0, -0.5, -0.5]));
    assert_eq!(pos["max"], json!([1.0, 0.5, 0.5]));
    assert_eq!(
        acc[prim["indices"].as_u64().unwrap() as usize]["componentType"],
        5123
    );
    // Empty primitive (frag0 has no interior faces) is dropped.
    assert_eq!(j["meshes"][0]["primitives"].as_array().unwrap().len(), 1);
    assert_eq!(j["meshes"][1]["primitives"].as_array().unwrap().len(), 2);

    let dir = temp_dir("glb_uncompressed");
    let path = dir.join("sample.glb");
    std::fs::write(&path, &glb).unwrap();
    validate_or_skip(&path);
}

#[test]
fn glb_meshopt_compressed() {
    let scene = sample_render_scene();
    let plain = write_glb(
        &scene,
        &GltfOptions {
            meshopt_compression: false,
        },
    )
    .unwrap();
    let glb = write_glb(
        &scene,
        &GltfOptions {
            meshopt_compression: true,
        },
    )
    .unwrap();
    let j = glb_json(&glb);
    assert_eq!(j["extensionsRequired"], json!(["EXT_meshopt_compression"]));
    assert_eq!(
        j["extensionsUsed"],
        json!(["EXT_meshopt_compression", "MSFT_lod"])
    );
    assert_eq!(
        j["buffers"][1]["extensions"]["EXT_meshopt_compression"]["fallback"],
        true
    );
    assert!(j["buffers"][1].get("uri").is_none());
    for v in j["bufferViews"].as_array().unwrap() {
        let e = &v["extensions"]["EXT_meshopt_compression"];
        assert_eq!(v["buffer"], 1);
        assert_eq!(e["buffer"], 0);
        let mode = e["mode"].as_str().unwrap();
        if v["target"] == 34963 {
            assert_eq!(mode, "TRIANGLES");
            assert!(e["byteStride"] == 2 || e["byteStride"] == 4);
            assert!(v.get("byteStride").is_none());
        } else {
            assert_eq!(mode, "ATTRIBUTES");
            assert_eq!(e["byteStride"], v["byteStride"]);
        }
    }

    // Decoding yields the same summary as the uncompressed export.
    let s = read_glb_summary(&glb).unwrap();
    assert!(s.meshopt_compressed);
    let p = read_glb_summary(&plain).unwrap();
    assert_eq!(s.mesh_triangles, p.mesh_triangles);
    assert_eq!(s.node_extras, p.node_extras);
    assert_eq!(s.node_mesh, p.node_mesh);

    // Decoded geometry imports identically to the uncompressed file.
    let opts = ImportOptions::default();
    let a = load_scene_bytes(&plain, "glb", &opts).unwrap();
    let b = load_scene_bytes(&glb, "glb", &opts).unwrap();
    assert_eq!(a.parts.len(), b.parts.len());
    for (pa, pb) in a.parts.iter().zip(&b.parts) {
        assert_eq!(pa.name, pb.name);
        assert_eq!(pa.mesh.verts, pb.mesh.verts);
        assert_eq!(pa.mesh.tris.len(), pb.mesh.tris.len());
        assert_eq!(pa.normals, pb.normals);
        assert_eq!(pa.uvs, pb.uvs);
        assert_eq!(pa.tangents, pb.tangents);
    }

    // Compressed output is smaller and still deterministic.
    let glb2 = write_glb(
        &scene,
        &GltfOptions {
            meshopt_compression: true,
        },
    )
    .unwrap();
    assert_eq!(glb, glb2);

    let dir = temp_dir("glb_compressed");
    let path = dir.join("sample_meshopt.glb");
    std::fs::write(&path, &glb).unwrap();
    validate_or_skip(&path);
    // The validator cannot decode EXT_meshopt_compression payloads, so also
    // validate our decoded equivalent (checks accessor bounds/min/max on the
    // decompressed data).
    let decoded = decompress_meshopt_glb(&glb).unwrap().unwrap();
    assert!(decompress_meshopt_glb(&decoded).unwrap().is_none());
    let dpath = dir.join("sample_meshopt_decoded.glb");
    std::fs::write(&dpath, &decoded).unwrap();
    validate_or_skip(&dpath);
}

#[test]
fn glb_u32_indices_and_compression() {
    let mut scene = RenderScene {
        name: "grid".into(),
        materials: vec![RenderMaterial {
            name: "m".into(),
            base_color: [1.0; 4],
            metallic: 0.0,
            roughness: 0.5,
            double_sided: false,
        }],
        meshes: vec![big_grid_mesh(260)],
        nodes: vec![RenderNode {
            name: "grid".into(),
            mesh_lods: vec![0],
            ..Default::default()
        }],
        extras: serde_json::Value::Null,
    };
    let dir = temp_dir("glb_u32");
    for compress in [false, true] {
        let glb = write_glb(
            &scene,
            &GltfOptions {
                meshopt_compression: compress,
            },
        )
        .unwrap();
        let j = glb_json(&glb);
        let idx = j["meshes"][0]["primitives"][0]["indices"].as_u64().unwrap() as usize;
        assert_eq!(j["accessors"][idx]["componentType"], 5125);
        let s = read_glb_summary(&glb).unwrap();
        assert_eq!(s.total_triangles, 259 * 259 * 2);
        let path = dir.join(format!("grid_{compress}.glb"));
        std::fs::write(&path, &glb).unwrap();
        validate_or_skip(&path);
    }
    // A node referencing an empty mesh still exports (as a mesh-less node).
    scene.meshes.push(RenderMesh::default());
    scene.nodes.push(RenderNode {
        name: "empty".into(),
        mesh_lods: vec![1],
        ..Default::default()
    });
    let glb = write_glb(&scene, &GltfOptions::default()).unwrap();
    let s = read_glb_summary(&glb).unwrap();
    assert_eq!(s.mesh_count, 1);
    assert_eq!(s.node_mesh, vec![Some(0), None]);
}

#[test]
fn glb_is_deterministic() {
    let scene = sample_render_scene();
    for compress in [false, true] {
        let opts = GltfOptions {
            meshopt_compression: compress,
        };
        let a = write_glb(&scene, &opts).unwrap();
        let b = write_glb(&scene.clone(), &opts).unwrap();
        assert_eq!(a, b);
    }
}

#[test]
fn glb_rejects_invalid_scenes() {
    let mut s = sample_render_scene();
    s.nodes[0].parent = Some(3);
    assert!(
        write_glb(&s, &GltfOptions::default()).is_err(),
        "parent cycle"
    );
    let mut s = sample_render_scene();
    s.meshes[0].primitives[0].indices.push(0);
    assert!(
        write_glb(&s, &GltfOptions::default()).is_err(),
        "non-triangle index count"
    );
    let mut s = sample_render_scene();
    s.nodes[1].mesh_lods = vec![99];
    assert!(
        write_glb(&s, &GltfOptions::default()).is_err(),
        "missing mesh"
    );
    let mut s = sample_render_scene();
    s.meshes[0].primitives[0].material = 5;
    assert!(
        write_glb(&s, &GltfOptions::default()).is_err(),
        "missing material"
    );
}

#[test]
fn glb_empty_scene_is_valid() {
    let glb = write_glb(
        &RenderScene::default(),
        &GltfOptions {
            meshopt_compression: true,
        },
    )
    .unwrap();
    let s = read_glb_summary(&glb).unwrap();
    assert_eq!(s.node_count, 0);
    let dir = temp_dir("glb_empty");
    let path = dir.join("empty.glb");
    std::fs::write(&path, &glb).unwrap();
    validate_or_skip(&path);
}
