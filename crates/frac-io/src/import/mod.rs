//! Scene import (spec §5.1): glTF/GLB (primary), OBJ, PLY, STL →
//! [`InputScene`] in the asset frame (meters, right-handed, +Y up).
//!
//! Conventions:
//! * `unit_scale` multiplies all positions; `z_up` converts a Z-up source to
//!   Y-up with `(x, y, z) -> (x, z, -y)` (a proper rotation, so winding is
//!   kept). Both are applied after any glTF node transforms.
//! * Vertices are indexed per part by exact position (bit-identical after
//!   transformation, `-0.0 == 0.0`); no tolerance welding is done here.
//! * Per-corner attributes are `Some` when at least one source triangle of the
//!   part carries them; missing corners are filled (normals: geometric face
//!   normal, UVs: `[0, 0]`, tangents: `[1, 0, 0, 1]`, paint: `0`).
//! * UVs follow the glTF convention (origin top-left): OBJ/PLY texture
//!   coordinates are converted with `v' = 1 - v`.

mod gltf_in;
mod obj;
mod ply;
mod stl;

use crate::IoError;
use frac_core::input::{InputPart, InputScene};
use frac_geom::TriMesh;
use glam::{DMat3, DVec3};
use std::collections::BTreeMap;
use std::path::Path;

/// Import options.
#[derive(Clone, Debug)]
pub struct ImportOptions {
    /// Multiplier applied to positions (source units → meters).
    pub unit_scale: f64,
    /// Source is Z-up: convert with `(x, y, z) -> (x, z, -y)`.
    pub z_up: bool,
}

impl Default for ImportOptions {
    fn default() -> Self {
        ImportOptions {
            unit_scale: 1.0,
            z_up: false,
        }
    }
}

impl ImportOptions {
    /// Rotation part of the frame conversion.
    pub(crate) fn rotation(&self) -> DMat3 {
        if self.z_up {
            // Columns are the images of X, Y, Z: X->X, Y->-Z, Z->Y.
            DMat3::from_cols(DVec3::X, DVec3::NEG_Z, DVec3::Y)
        } else {
            DMat3::IDENTITY
        }
    }
    /// Full linear frame conversion (scale * rotation).
    pub(crate) fn frame(&self) -> DMat3 {
        self.rotation() * self.unit_scale
    }
}

/// Load a scene file, dispatching on its extension
/// (`gltf`, `glb`, `obj`, `ply`, `stl`; case-insensitive).
pub fn load_scene(path: &Path, opts: &ImportOptions) -> Result<InputScene, IoError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .ok_or_else(|| IoError::UnsupportedFormat(path.display().to_string()))?;
    let bytes = std::fs::read(path).map_err(|e| IoError::io(path, e))?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("mesh")
        .to_string();
    load_impl(
        &bytes,
        &ext,
        path.parent(),
        &stem,
        path.display().to_string(),
        opts,
    )
}

/// Load a scene from memory. `ext` is the format extension (`"glb"`,
/// `"gltf"`, `"obj"`, `"ply"`, `"stl"`, with or without a leading dot).
/// External resources (glTF external buffers) are not available.
pub fn load_scene_bytes(
    bytes: &[u8],
    ext: &str,
    opts: &ImportOptions,
) -> Result<InputScene, IoError> {
    let ext = ext.trim_start_matches('.').to_ascii_lowercase();
    load_impl(bytes, &ext, None, "mesh", format!("<memory>.{ext}"), opts)
}

fn load_impl(
    bytes: &[u8],
    ext: &str,
    base: Option<&Path>,
    stem: &str,
    source: String,
    opts: &ImportOptions,
) -> Result<InputScene, IoError> {
    if !(opts.unit_scale.is_finite() && opts.unit_scale > 0.0) {
        return Err(IoError::InvalidScene(format!(
            "unit_scale must be positive, got {}",
            opts.unit_scale
        )));
    }
    let parts = match ext {
        "gltf" | "glb" => gltf_in::load(bytes, base, opts)?,
        "obj" => obj::load(bytes, stem, opts)?,
        "ply" => ply::load(bytes, stem, opts)?,
        "stl" => stl::load(bytes, stem, opts)?,
        other => return Err(IoError::UnsupportedFormat(other.to_string())),
    };
    tracing::debug!(source = %source, parts = parts.len(), "imported scene");
    Ok(InputScene { source, parts })
}

/// One triangle corner (already in the asset frame).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Corner {
    pub p: DVec3,
    pub n: Option<[f32; 3]>,
    pub uv: Option<[f32; 2]>,
    pub t: Option<[f32; 4]>,
    pub paint: Option<f32>,
}

/// Accumulates an [`InputPart`], indexing vertices by exact position.
pub(crate) struct PartBuilder {
    name: String,
    verts: Vec<DVec3>,
    map: BTreeMap<[u64; 3], u32>,
    tris: Vec<[u32; 3]>,
    normals: Vec<[Option<[f32; 3]>; 3]>,
    uvs: Vec<[Option<[f32; 2]>; 3]>,
    tangents: Vec<[Option<[f32; 4]>; 3]>,
    paint: Vec<Option<f32>>,
    slots: Vec<u16>,
    material_names: Vec<String>,
    pub extras: serde_json::Value,
}

impl PartBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        PartBuilder {
            name: name.into(),
            verts: Vec::new(),
            map: BTreeMap::new(),
            tris: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            tangents: Vec::new(),
            paint: Vec::new(),
            slots: Vec::new(),
            material_names: Vec::new(),
            extras: serde_json::Value::Null,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn is_empty(&self) -> bool {
        self.tris.is_empty()
    }

    /// Material slot for a material name (deduplicated by name, in order of
    /// first use).
    pub fn slot(&mut self, name: &str) -> u16 {
        if let Some(i) = self.material_names.iter().position(|n| n == name) {
            return i as u16;
        }
        self.material_names.push(name.to_string());
        (self.material_names.len() - 1) as u16
    }

    fn vertex(&mut self, p: DVec3, paint: Option<f32>) -> u32 {
        // Canonicalize -0.0 to +0.0 so they index the same vertex.
        let p = p + DVec3::ZERO;
        let key = [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
        let next = self.verts.len() as u32;
        let idx = *self.map.entry(key).or_insert(next);
        if idx == next {
            self.verts.push(p);
            self.paint.push(paint);
        } else if self.paint[idx as usize].is_none() {
            self.paint[idx as usize] = paint;
        }
        idx
    }

    pub fn add_tri(&mut self, c: [Corner; 3], slot: u16) {
        let ids = [
            self.vertex(c[0].p, c[0].paint),
            self.vertex(c[1].p, c[1].paint),
            self.vertex(c[2].p, c[2].paint),
        ];
        self.tris.push(ids);
        self.normals.push([c[0].n, c[1].n, c[2].n]);
        self.uvs.push([c[0].uv, c[1].uv, c[2].uv]);
        self.tangents.push([c[0].t, c[1].t, c[2].t]);
        self.slots.push(slot);
    }

    pub fn finish(self) -> InputPart {
        let any_n = self.normals.iter().flatten().any(Option::is_some);
        let any_uv = self.uvs.iter().flatten().any(Option::is_some);
        let any_t = self.tangents.iter().flatten().any(Option::is_some);
        let any_paint = self.paint.iter().any(Option::is_some);
        let verts = &self.verts;
        let normals = any_n.then(|| {
            self.normals
                .iter()
                .zip(&self.tris)
                .map(|(ns, t)| {
                    let face = {
                        let [a, b, c] = t.map(|i| verts[i as usize]);
                        let n = (b - a).cross(c - a).normalize_or_zero();
                        [n.x as f32, n.y as f32, n.z as f32]
                    };
                    ns.map(|n| n.unwrap_or(face))
                })
                .collect()
        });
        let uvs = any_uv.then(|| {
            self.uvs
                .iter()
                .map(|u| u.map(|x| x.unwrap_or([0.0, 0.0])))
                .collect()
        });
        let tangents = any_t.then(|| {
            self.tangents
                .iter()
                .map(|u| u.map(|x| x.unwrap_or([1.0, 0.0, 0.0, 1.0])))
                .collect()
        });
        let vertex_paint = any_paint.then(|| self.paint.iter().map(|p| p.unwrap_or(0.0)).collect());
        let mut material_names = self.material_names;
        if material_names.is_empty() {
            material_names.push("default".to_string());
        }
        InputPart {
            name: self.name,
            mesh: TriMesh {
                verts: self.verts,
                tris: self.tris,
            },
            normals,
            uvs,
            tangents,
            material_slot: self.slots,
            material_names,
            vertex_paint,
            extras: self.extras,
        }
    }
}

/// Normalize a direction in f64 and narrow to f32.
pub(crate) fn dir_f32(v: DVec3) -> [f32; 3] {
    let n = v.normalize_or_zero();
    [n.x as f32, n.y as f32, n.z as f32]
}
