//! glTF 2.0 / GLB import.
//!
//! One part per node with a mesh (default scene, else scene 0, else all root
//! nodes), visited depth-first in node order. World transforms are composed
//! in f64; negative-determinant transforms flip the triangle winding (and
//! tangent handedness). Primitives become material slots; strips and fans
//! are converted to triangle lists; point and line primitives are skipped.
//! Scalar float vertex attributes named `_PAINT` or `_DENSITY` become
//! `vertex_paint`. GLBs using `EXT_meshopt_compression` are decoded first.

use super::{Corner, PartBuilder, dir_f32};
use crate::{ImportOptions, IoError};
use frac_core::input::InputPart;
use glam::{DMat3, DMat4, DQuat, DVec3};
use gltf::mesh::Mode;
use serde_json::Value;
use std::path::Path;

fn raw_to_value(raw: &Option<Box<serde_json::value::RawValue>>) -> Value {
    raw.as_ref()
        .and_then(|r| serde_json::from_str(r.get()).ok())
        .unwrap_or(Value::Null)
}

/// Shallow-merge object extras (later sources win).
fn merge_extras(sources: &[Value]) -> Value {
    let mut out = serde_json::Map::new();
    for s in sources {
        if let Value::Object(m) = s {
            for (k, v) in m {
                out.insert(k.clone(), v.clone());
            }
        }
    }
    if out.is_empty() {
        Value::Null
    } else {
        Value::Object(out)
    }
}

fn local_matrix(node: &gltf::Node) -> DMat4 {
    match node.transform() {
        gltf::scene::Transform::Matrix { matrix } => {
            DMat4::from_cols_array_2d(&matrix.map(|c| c.map(|v| v as f64)))
        }
        gltf::scene::Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => {
            let t = DVec3::new(
                translation[0] as f64,
                translation[1] as f64,
                translation[2] as f64,
            );
            let s = DVec3::new(scale[0] as f64, scale[1] as f64, scale[2] as f64);
            let q = DQuat::from_xyzw(
                rotation[0] as f64,
                rotation[1] as f64,
                rotation[2] as f64,
                rotation[3] as f64,
            );
            let q = if q.length_squared() > 0.0 {
                q.normalize()
            } else {
                DQuat::IDENTITY
            };
            DMat4::from_scale_rotation_translation(s, q, t)
        }
    }
}

pub(super) fn load(
    bytes: &[u8],
    base: Option<&Path>,
    opts: &ImportOptions,
) -> Result<Vec<InputPart>, IoError> {
    let decompressed = if bytes.starts_with(b"glTF") {
        crate::decompress_meshopt_glb(bytes)?
    } else {
        None
    };
    let data = decompressed.as_deref().unwrap_or(bytes);
    let g = gltf::Gltf::from_slice(data).map_err(|e| IoError::Gltf(e.to_string()))?;
    let buffers = gltf::import_buffers(&g.document, base, g.blob.clone())
        .map_err(|e| IoError::Gltf(e.to_string()))?;
    let doc = &g.document;

    let scene = doc.default_scene().or_else(|| doc.scenes().next());
    let scene_extras = scene
        .as_ref()
        .map(|s| raw_to_value(s.extras()))
        .unwrap_or(Value::Null);
    let roots: Vec<gltf::Node> = match &scene {
        Some(s) => s.nodes().collect(),
        None => {
            let mut is_child = vec![false; doc.nodes().len()];
            for n in doc.nodes() {
                for c in n.children() {
                    is_child[c.index()] = true;
                }
            }
            doc.nodes().filter(|n| !is_child[n.index()]).collect()
        }
    };

    let frame = DMat4::from_mat3(opts.frame());
    let mut visited = vec![false; doc.nodes().len()];
    let mut parts = Vec::new();
    // Explicit DFS stack (node, parent world matrix), preserving node order.
    let mut stack: Vec<(gltf::Node, DMat4)> = roots
        .into_iter()
        .rev()
        .map(|n| (n, DMat4::IDENTITY))
        .collect();
    while let Some((node, parent)) = stack.pop() {
        if std::mem::replace(&mut visited[node.index()], true) {
            continue; // malformed graph (shared node / cycle)
        }
        let world = parent * local_matrix(&node);
        let children: Vec<gltf::Node> = node.children().collect();
        for c in children.into_iter().rev() {
            stack.push((c, world));
        }
        let Some(mesh) = node.mesh() else { continue };
        let name = node
            .name()
            .map(str::to_string)
            .unwrap_or_else(|| format!("node_{}", node.index()));
        let mut pb = PartBuilder::new(name);
        pb.extras = merge_extras(&[
            scene_extras.clone(),
            raw_to_value(mesh.extras()),
            raw_to_value(node.extras()),
        ]);
        add_mesh(&mut pb, &mesh, frame * world, &buffers)?;
        if pb.is_empty() {
            tracing::warn!(
                node = node.index(),
                "glTF node mesh has no triangles; skipped"
            );
            continue;
        }
        parts.push(pb.finish());
    }
    Ok(parts)
}

fn add_mesh(
    pb: &mut PartBuilder,
    mesh: &gltf::Mesh,
    m: DMat4,
    buffers: &[gltf::buffer::Data],
) -> Result<(), IoError> {
    let lin = DMat3::from_mat4(m);
    let det = lin.determinant();
    let flip = det < 0.0;
    let nmat = if det.abs() > 1e-300 {
        lin.inverse().transpose()
    } else {
        lin
    };
    let get = |b: gltf::Buffer| buffers.get(b.index()).map(|d| &d.0[..]);
    for prim in mesh.primitives() {
        let mode = prim.mode();
        if !matches!(
            mode,
            Mode::Triangles | Mode::TriangleStrip | Mode::TriangleFan
        ) {
            continue;
        }
        let reader = prim.reader(get);
        let Some(pos) = reader.read_positions() else {
            continue;
        };
        let pos: Vec<[f32; 3]> = pos.collect();
        let nrm: Option<Vec<[f32; 3]>> = reader.read_normals().map(|i| i.collect());
        let uvs: Option<Vec<[f32; 2]>> = reader.read_tex_coords(0).map(|i| i.into_f32().collect());
        let tan: Option<Vec<[f32; 4]>> = reader.read_tangents().map(|i| i.collect());
        let mut paint: Option<Vec<f32>> = None;
        for (sem, acc) in prim.attributes() {
            if let gltf::Semantic::Extras(name) = &sem {
                let n = name.to_ascii_uppercase();
                if (n == "PAINT" || n == "DENSITY")
                    && acc.dimensions() == gltf::accessor::Dimensions::Scalar
                    && acc.data_type() == gltf::accessor::DataType::F32
                {
                    paint = gltf::accessor::Iter::<f32>::new(acc, get).map(|it| it.collect());
                }
            }
        }
        let idx: Vec<u32> = match reader.read_indices() {
            Some(i) => i.into_u32().collect(),
            None => (0..pos.len() as u32).collect(),
        };
        let mut tris: Vec<[u32; 3]> = Vec::new();
        match mode {
            Mode::Triangles => tris.extend(idx.chunks_exact(3).map(|c| [c[0], c[1], c[2]])),
            Mode::TriangleStrip => {
                for i in 0..idx.len().saturating_sub(2) {
                    let t = if i % 2 == 0 {
                        [idx[i], idx[i + 1], idx[i + 2]]
                    } else {
                        [idx[i + 1], idx[i], idx[i + 2]]
                    };
                    if t[0] != t[1] && t[1] != t[2] && t[0] != t[2] {
                        tris.push(t);
                    }
                }
            }
            Mode::TriangleFan => {
                for i in 1..idx.len().saturating_sub(1) {
                    let t = [idx[i], idx[i + 1], idx[0]];
                    if t[0] != t[1] && t[1] != t[2] && t[0] != t[2] {
                        tris.push(t);
                    }
                }
            }
            _ => unreachable!(),
        }
        let mat_name = match prim.material().index() {
            Some(i) => prim
                .material()
                .name()
                .map(str::to_string)
                .unwrap_or_else(|| format!("material_{i}")),
            None => "default".to_string(),
        };
        let slot = pb.slot(&mat_name);
        let corner = |i: u32| -> Result<Corner, IoError> {
            let i = i as usize;
            let p = pos
                .get(i)
                .ok_or_else(|| IoError::Gltf(format!("index {i} out of range")))?;
            let p = m.transform_point3(DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64));
            let n = nrm
                .as_ref()
                .and_then(|v| v.get(i))
                .map(|n| dir_f32(nmat * DVec3::new(n[0] as f64, n[1] as f64, n[2] as f64)));
            let t = tan.as_ref().and_then(|v| v.get(i)).map(|t| {
                let d = dir_f32(lin * DVec3::new(t[0] as f64, t[1] as f64, t[2] as f64));
                let w = if t[3] < 0.0 { -1.0 } else { 1.0 };
                [d[0], d[1], d[2], if flip { -w } else { w }]
            });
            Ok(Corner {
                p,
                n,
                uv: uvs.as_ref().and_then(|v| v.get(i)).copied(),
                t,
                paint: paint.as_ref().and_then(|v| v.get(i)).copied(),
            })
        };
        for t in tris {
            let t = if flip { [t[0], t[2], t[1]] } else { t };
            pb.add_tri([corner(t[0])?, corner(t[1])?, corner(t[2])?], slot);
        }
    }
    Ok(())
}
