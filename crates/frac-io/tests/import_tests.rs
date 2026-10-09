mod common;

use common::*;
use frac_core::input::{InputPart, InputScene};
use frac_io::*;
use glam::DVec3;
use serde_json::json;

fn opts() -> ImportOptions {
    ImportOptions::default()
}

fn face_normal(p: &InputPart, t: usize) -> DVec3 {
    let [a, b, c] = p.mesh.tris[t].map(|i| p.mesh.verts[i as usize]);
    (b - a).cross(c - a).normalize()
}

fn canon(s: &InputScene) -> String {
    serde_json::to_string(s).unwrap()
}

const OBJ: &str = "# test
mtllib missing.mtl
o boxy
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
usemtl red
f 1/1/1 2/2/1 3/3/1 4/4/1
o tri
vn 0 0 -1
v 0 0 1
v 1 0 1
v 0 1 1
usemtl blue
f 5/1/2 6/2/2 7/3/2
usemtl red
f 7/3/2 6/2/2 5/1/2
";

#[test]
fn obj_import() {
    let dir = temp_dir("obj");
    let path = dir.join("scene.obj");
    std::fs::write(&path, OBJ).unwrap();
    let s = load_scene(&path, &opts()).unwrap();
    assert_eq!(s.parts.len(), 2);
    let a = &s.parts[0];
    assert_eq!(a.name, "boxy");
    assert_eq!(a.mesh.verts.len(), 4);
    assert_eq!(a.mesh.tris.len(), 2, "quad triangulated");
    assert_eq!(a.material_names, vec!["red"]);
    assert_eq!(a.material_slot, vec![0, 0]);
    let n = a.normals.as_ref().unwrap();
    assert_eq!(n[0][0], [0.0, 0.0, 1.0]);
    let uv = a.uvs.as_ref().unwrap();
    // v flipped to the glTF convention.
    assert_eq!(uv[0][0], [0.0, 1.0]);
    assert!(face_normal(a, 0).z > 0.99);
    let b = &s.parts[1];
    assert_eq!(b.name, "tri");
    assert_eq!(b.material_names, vec!["blue", "red"]);
    assert_eq!(b.material_slot, vec![0, 1]);
    assert_eq!(b.normals.as_ref().unwrap()[1][0], [0.0, 0.0, -1.0]);
    assert!(b.uvs.is_some());
    // Mixed faces (some without vn/vt) drop the attribute entirely.
    let mixed = OBJ.replace("f 5/1/2 6/2/2 7/3/2", "f 5 6 7");
    let mx = load_scene_bytes(mixed.as_bytes(), "obj", &opts()).unwrap();
    assert!(
        mx.parts
            .iter()
            .all(|p| p.normals.is_none() && p.uvs.is_none())
    );
    assert_eq!(b.mesh.verts.len(), 3);

    // From memory: identical result (apart from the source string).
    let mut m = load_scene_bytes(OBJ.as_bytes(), "obj", &opts()).unwrap();
    m.source = s.source.clone();
    assert_eq!(canon(&m), canon(&s));
}

#[test]
fn unit_scale_and_z_up() {
    let o = ImportOptions {
        unit_scale: 0.01,
        z_up: true,
    };
    let s = load_scene_bytes(OBJ.as_bytes(), "obj", &o).unwrap();
    let b = &s.parts[1];
    // (0,0,1) -> (0, 1, -0) * 0.01
    assert!(
        b.mesh
            .verts
            .iter()
            .any(|v| (*v - DVec3::new(0.0, 0.01, 0.0)).length() < 1e-15)
    );
    // (1,0,1) -> (1, 1, 0) * 0.01, (0,1,1) -> (0, 1, -1) * 0.01
    assert!(
        b.mesh
            .verts
            .iter()
            .any(|v| (*v - DVec3::new(0.0, 0.01, -0.01)).length() < 1e-15)
    );
    let a = &s.parts[0];
    // Normal +Z -> +Y; winding preserved (rotation).
    assert_eq!(a.normals.as_ref().unwrap()[0][0], [0.0, 1.0, 0.0]);
    assert!(face_normal(a, 0).y > 0.99);
    assert!(
        load_scene_bytes(
            OBJ.as_bytes(),
            "obj",
            &ImportOptions {
                unit_scale: 0.0,
                z_up: false
            }
        )
        .is_err()
    );
}

const PLY_ASCII: &str = "ply
format ascii 1.0
comment test
element vertex 5
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float s
property float t
property float paint
element face 2
property list uchar int vertex_indices
end_header
0 0 0 0 0 1 0 0 0.5
1 0 0 0 0 1 1 0 1
1 1 0 0 0 1 1 1 1.5
0 1 0 0 0 1 0 1 2
0 0 1 0 0 1 0 0 9
4 0 1 2 3
3 0 2 1
";

/// Write the PLY_ASCII content as binary.
fn ply_binary(big: bool) -> Vec<u8> {
    let fmt = if big {
        "binary_big_endian"
    } else {
        "binary_little_endian"
    };
    let header = PLY_ASCII.replace("format ascii 1.0", &format!("format {fmt} 1.0"));
    let header = &header[..header.find("end_header\n").unwrap() + "end_header\n".len()];
    let mut out = header.as_bytes().to_vec();
    let body = &PLY_ASCII[PLY_ASCII.find("end_header\n").unwrap() + 11..];
    let lines: Vec<&str> = body.lines().collect();
    let f = |out: &mut Vec<u8>, v: f32| {
        out.extend_from_slice(&if big {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        })
    };
    for l in &lines[..5] {
        for t in l.split_whitespace() {
            f(&mut out, t.parse().unwrap());
        }
    }
    for l in &lines[5..] {
        let toks: Vec<i32> = l.split_whitespace().map(|t| t.parse().unwrap()).collect();
        out.push(toks[0] as u8);
        for &i in &toks[1..] {
            out.extend_from_slice(&if big {
                i.to_be_bytes()
            } else {
                i.to_le_bytes()
            });
        }
    }
    out
}

#[test]
fn ply_import_ascii_and_binary() {
    let s = load_scene_bytes(PLY_ASCII.as_bytes(), "ply", &opts()).unwrap();
    assert_eq!(s.parts.len(), 1);
    let p = &s.parts[0];
    assert_eq!(p.mesh.tris.len(), 3);
    assert_eq!(p.mesh.verts.len(), 4, "unreferenced vertex dropped");
    assert_eq!(p.material_names, vec!["default"]);
    assert_eq!(p.vertex_paint.as_ref().unwrap(), &vec![0.5, 1.0, 1.5, 2.0]);
    assert_eq!(p.uvs.as_ref().unwrap()[0][2], [1.0, 0.0], "t flipped");
    assert!(face_normal(p, 0).z > 0.99);
    assert!(face_normal(p, 2).z < -0.99);
    for big in [false, true] {
        let b = load_scene_bytes(&ply_binary(big), "ply", &opts()).unwrap();
        assert_eq!(canon(&b), canon(&s), "binary big={big}");
    }
    // Via path; name from the file stem.
    let dir = temp_dir("ply");
    let path = dir.join("painted.ply");
    std::fs::write(&path, ply_binary(false)).unwrap();
    let f = load_scene(&path, &opts()).unwrap();
    assert_eq!(f.parts[0].name, "painted");
    // Truncated data errors.
    let bin = ply_binary(true);
    assert!(load_scene_bytes(&bin[..bin.len() - 3], "ply", &opts()).is_err());
}

fn cube_tris() -> Vec<[[f32; 3]; 3]> {
    let v = |i: u32| [(i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32];
    let quads = [
        [0, 2, 3, 1],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 4, 6, 2],
        [1, 3, 7, 5],
    ];
    let mut t = Vec::new();
    for q in quads {
        t.push([v(q[0]), v(q[1]), v(q[2])]);
        t.push([v(q[0]), v(q[2]), v(q[3])]);
    }
    t
}

#[test]
fn stl_import_ascii_and_binary() {
    let mut ascii = String::from("solid cube\n");
    for t in cube_tris() {
        ascii.push_str(" facet normal 0 0 0\n  outer loop\n");
        for p in t {
            ascii.push_str(&format!("   vertex {} {} {}\n", p[0], p[1], p[2]));
        }
        ascii.push_str("  endloop\n endfacet\n");
    }
    ascii.push_str("endsolid cube\n");
    let mut bin = vec![0u8; 80];
    bin[..5].copy_from_slice(b"solid"); // binary header starting with "solid"
    bin.extend_from_slice(&12u32.to_le_bytes());
    for t in cube_tris() {
        bin.extend_from_slice(&[0u8; 12]);
        for p in t {
            for c in p {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        bin.extend_from_slice(&[0u8; 2]);
    }
    let a = load_scene_bytes(ascii.as_bytes(), "stl", &opts()).unwrap();
    let b = load_scene_bytes(&bin, "stl", &opts()).unwrap();
    assert_eq!(a.parts.len(), 1);
    assert_eq!(a.parts[0].name, "cube");
    for s in [&a, &b] {
        let p = &s.parts[0];
        assert_eq!(p.mesh.verts.len(), 8, "exact duplicates merged");
        assert_eq!(p.mesh.tris.len(), 12);
        assert!((p.mesh.signed_volume() - 1.0).abs() < 1e-12);
    }
    assert_eq!(a.parts[0].mesh, b.parts[0].mesh);
    let dir = temp_dir("stl");
    std::fs::write(dir.join("c.stl"), &bin).unwrap();
    assert_eq!(
        load_scene(&dir.join("c.stl"), &opts()).unwrap().parts[0].name,
        "c"
    );
}

#[test]
fn gltf_round_trip_from_exporter() {
    let scene = sample_render_scene();
    let glb = write_glb(&scene, &GltfOptions::default()).unwrap();
    let dir = temp_dir("gltf_rt");
    let path = dir.join("sample.glb");
    std::fs::write(&path, &glb).unwrap();
    let s = load_scene(&path, &opts()).unwrap();
    // "asset" has no mesh; the MSFT_lod node is not part of the scene.
    let names: Vec<&str> = s.parts.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, vec!["frag_0", "frag_1", "frag_2"]);
    let p0 = &s.parts[0];
    assert_eq!(p0.mesh.verts.len(), 8);
    assert_eq!(p0.mesh.tris.len(), 12);
    assert_eq!(p0.material_names, vec!["concrete"]);
    assert!((p0.mesh.signed_volume() - 2.0 * 1.0 * 1.0).abs() < 1e-9);
    let c = p0.mesh.aabb().center();
    assert!(
        (c - DVec3::new(0.0, 1.5, 0.0)).length() < 1e-9,
        "world transform applied: {c:?}"
    );
    let p1 = &s.parts[1];
    assert_eq!(p1.material_names, vec!["concrete", "concrete_interior"]);
    assert_eq!(p1.material_slot.iter().filter(|&&m| m == 1).count(), 2);
    let c1 = p1.mesh.aabb().center();
    assert!((c1 - DVec3::new(0.5, 1.5, 0.0)).length() < 1e-9, "{c1:?}");
    assert!(p1.normals.is_some() && p1.uvs.is_some() && p1.tangents.is_some());
    // Extras: scene extras merged with node extras (node wins).
    assert_eq!(p1.extras["fragment_id"], 1);
    assert_eq!(p1.extras["prefracture"]["levels"], 2);
    assert_eq!(p1.extras["lods"], json!([1]));
    // Normals agree with geometric winding.
    for t in 0..p1.mesh.tris.len() {
        let n = p1.normals.as_ref().unwrap()[t][0];
        assert!(face_normal(p1, t).dot(DVec3::new(n[0] as f64, n[1] as f64, n[2] as f64)) > 0.99);
    }
    // Deterministic.
    assert_eq!(canon(&load_scene(&path, &opts()).unwrap()), canon(&s));
}

fn glb(json: serde_json::Value, bin: &[u8]) -> Vec<u8> {
    let mut j = serde_json::to_vec(&json).unwrap();
    while !j.len().is_multiple_of(4) {
        j.push(b' ');
    }
    let mut b = bin.to_vec();
    while !b.len().is_multiple_of(4) {
        b.push(0);
    }
    let total = 12 + 8 + j.len() + 8 + b.len();
    let mut out = Vec::new();
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(j.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&j);
    out.extend_from_slice(&(b.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&b);
    out
}

#[test]
fn gltf_hand_built_mirror_strip_fan_extras() {
    // Positions: 4 verts of a unit quad in z=0; normals +Z; paint.
    let pos: [[f32; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let nrm: [[f32; 3]; 4] = [[0.0, 0.0, 1.0]; 4];
    let paint: [f32; 4] = [0.1, 0.2, 0.3, 0.4];
    let strip: [u16; 4] = [0, 1, 2, 3]; // CCW strip: (0,1,2), (2,1,3)
    let fan: [u16; 4] = [0, 1, 3, 2]; // CCW fan: (0,1,3), (0,3,2)
    let mut bin = Vec::new();
    bin.extend_from_slice(bytemuck::cast_slice(&pos)); // 0..48
    bin.extend_from_slice(bytemuck::cast_slice(&nrm)); // 48..96
    bin.extend_from_slice(bytemuck::cast_slice(&paint)); // 96..112
    bin.extend_from_slice(bytemuck::cast_slice(&strip)); // 112..120
    bin.extend_from_slice(bytemuck::cast_slice(&fan)); // 120..128
    let j = json!({
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0], "extras": {"a": "scene", "s": 1}}],
        "nodes": [
            {"name": "parent", "children": [1], "translation": [10.0, 0.0, 0.0]},
            {"mesh": 0, "scale": [-1.0, 1.0, 1.0], "extras": {"a": "node", "n": 3}}
        ],
        "meshes": [{
            "extras": {"a": "mesh", "m": 2},
            "primitives": [
                {"attributes": {"POSITION": 0, "NORMAL": 1, "_PAINT": 2}, "indices": 3, "mode": 5},
                {"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 4, "mode": 6, "material": 0},
                {"attributes": {"POSITION": 0}, "mode": 0}
            ]
        }],
        "materials": [{}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]},
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 4, "type": "SCALAR"},
            {"bufferView": 3, "componentType": 5123, "count": 4, "type": "SCALAR"},
            {"bufferView": 4, "componentType": 5123, "count": 4, "type": "SCALAR"}
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 48},
            {"buffer": 0, "byteOffset": 48, "byteLength": 48},
            {"buffer": 0, "byteOffset": 96, "byteLength": 16},
            {"buffer": 0, "byteOffset": 112, "byteLength": 8},
            {"buffer": 0, "byteOffset": 120, "byteLength": 8}
        ],
        "buffers": [{"byteLength": bin.len()}]
    });
    let bytes = glb(j, &bin);
    let s = load_scene_bytes(&bytes, "glb", &opts()).unwrap();
    assert_eq!(s.parts.len(), 1);
    let p = &s.parts[0];
    assert_eq!(p.name, "node_1");
    assert_eq!(p.mesh.tris.len(), 4, "strip (2) + fan (2); points skipped");
    assert_eq!(p.material_names, vec!["default", "material_0"]);
    assert_eq!(p.material_slot, vec![0, 0, 1, 1]);
    // Mirror (det < 0): winding flipped so geometry agrees with the normals.
    for t in 0..4 {
        let n = p.normals.as_ref().unwrap()[t][0];
        assert_eq!(n, [0.0, 0.0, 1.0]);
        assert!(face_normal(p, t).z > 0.99, "triangle {t} winding");
    }
    // World transform: x -> 10 - x.
    let xs: Vec<f64> = p.mesh.verts.iter().map(|v| v.x).collect();
    assert!(xs.iter().all(|&x| x == 10.0 || x == 9.0), "{xs:?}");
    // Extras: node > mesh > scene.
    assert_eq!(p.extras, json!({"a": "node", "s": 1, "m": 2, "n": 3}));
    // Paint from the custom _PAINT attribute (first primitive defines it).
    let paint = p.vertex_paint.as_ref().unwrap();
    assert_eq!(paint.len(), 4);
    let idx0 = p
        .mesh
        .verts
        .iter()
        .position(|v| *v == DVec3::new(10.0, 0.0, 0.0))
        .unwrap();
    assert_eq!(paint[idx0], 0.1);
}

#[test]
fn unsupported_extension() {
    assert!(matches!(
        load_scene_bytes(b"", "fbx", &opts()),
        Err(IoError::UnsupportedFormat(_))
    ));
}
