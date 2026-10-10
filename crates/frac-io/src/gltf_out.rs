//! glTF 2.0 render export (spec §6.1), GLB summary reader and an
//! `EXT_meshopt_compression` decompressor.
//!
//! Output layout (deterministic):
//! * one material per [`RenderMaterial`] (metallic-roughness PBR factors);
//! * one glTF mesh per non-empty [`RenderMesh`], one primitive per non-empty
//!   [`RenderPrimitive`]; all primitives of a mesh share the vertex
//!   attribute accessors; separate (non-interleaved) bufferViews per
//!   attribute and per index list, 4-byte aligned; indices are `u16` when
//!   the mesh has at most 65535 vertices, `u32` otherwise;
//! * glTF node `i` == [`RenderNode`] `i`; scene roots are the nodes without
//!   parent; node translation is written relative to the parent (the input
//!   is the absolute origin in the asset frame);
//! * LODs: the node's mesh is `mesh_lods[0]`; all LOD mesh indices (glTF
//!   mesh indices) are recorded in the node extras as `"lods"`; when a node
//!   has more than one LOD, `MSFT_lod` is emitted: `lods[1..]` become extra
//!   nodes appended after all regular nodes (same local transform, not in the
//!   scene) referenced from the node's `extensions.MSFT_lod.ids`;
//! * `EXT_meshopt_compression` (optional): every vertex attribute and index
//!   bufferView is encoded with meshoptimizer's codecs into buffer 0 (the GLB
//!   BIN chunk); the uncompressed bufferViews live in buffer 1, a fallback
//!   buffer without data flagged `{"EXT_meshopt_compression":{"fallback":true}}`.
//!   The extension is listed in `extensionsUsed` and `extensionsRequired`.
//!
//! Empty meshes (no vertices or no triangles) are dropped, since glTF
//! forbids empty accessors and primitive-less meshes; node references to
//! them become mesh-less nodes, so glTF mesh indices may differ from
//! `RenderScene::meshes` indices in that case (the `"lods"` extras always
//! hold glTF mesh indices).

use crate::IoError;
use crate::glb::{read_glb_container, write_glb_container};
use serde_json::{Map, Value, json};

#[derive(Clone, Debug, Default)]
pub struct RenderMaterial {
    pub name: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub double_sided: bool,
}

#[derive(Clone, Debug, Default)]
pub struct RenderPrimitive {
    pub material: u32,
    /// Triangle list, CCW front faces.
    pub indices: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct RenderMesh {
    pub name: String,
    /// Relative to the node origin.
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Option<Vec<[f32; 2]>>,
    pub tangents: Option<Vec<[f32; 4]>>,
    /// e.g. [exterior slot, interior slot]
    pub primitives: Vec<RenderPrimitive>,
}

#[derive(Clone, Debug, Default)]
pub struct RenderNode {
    pub name: String,
    /// Index into nodes; None = scene root.
    pub parent: Option<u32>,
    /// Node origin in the asset frame.
    pub translation: [f64; 3],
    /// Indices into `RenderScene::meshes`; lods[0] is the node's mesh; may be empty.
    pub mesh_lods: Vec<u32>,
    /// Written as node extras (Three.js `userData`).
    pub extras: Value,
}

#[derive(Clone, Debug, Default)]
pub struct RenderScene {
    pub name: String,
    pub materials: Vec<RenderMaterial>,
    pub meshes: Vec<RenderMesh>,
    pub nodes: Vec<RenderNode>,
    /// Written as the glTF scene's extras.
    pub extras: Value,
}

#[derive(Clone, Debug, Default)]
pub struct GltfOptions {
    pub meshopt_compression: bool,
}

const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;
const FLOAT: u32 = 5126;
const UNSIGNED_SHORT: u32 = 5123;
const UNSIGNED_INT: u32 = 5125;
const EXT_MESHOPT: &str = "EXT_meshopt_compression";
const MSFT_LOD: &str = "MSFT_lod";

/// Exact JSON number for an f32 (its f64 widening; used for min/max).
fn jexact(x: f32) -> Value {
    Value::from(x as f64)
}

/// Short JSON number for an f32 (shortest decimal that round-trips to f32).
fn jshort(x: f32) -> Value {
    let s = format!("{x}");
    Value::from(s.parse::<f64>().unwrap_or(x as f64))
}

enum ViewKind {
    Attribute { stride: usize, count: usize },
    Index { stride: usize, count: usize },
}

struct Writer {
    compress: bool,
    bin: Vec<u8>,
    fallback_len: usize,
    views: Vec<Value>,
    accessors: Vec<Value>,
    used_compression: bool,
}

fn align4(v: &mut Vec<u8>) {
    let n = v.len().next_multiple_of(4);
    v.resize(n, 0);
}

impl Writer {
    /// Add a bufferView; returns (view index, the bytes the view decodes to).
    fn add_view(&mut self, data: Vec<u8>, kind: ViewKind) -> Result<usize, IoError> {
        let (target, stride_field) = match kind {
            ViewKind::Attribute { stride, .. } => (ARRAY_BUFFER, Some(stride)),
            ViewKind::Index { .. } => (ELEMENT_ARRAY_BUFFER, None),
        };
        let mut view = Map::new();
        if self.compress {
            let (encoded, fallback, ext) = match kind {
                ViewKind::Attribute { stride, count } => {
                    let enc = encode_vertex_bytes(&data, stride, count)?;
                    (
                        enc,
                        data,
                        json!({"byteStride": stride, "count": count, "mode": "ATTRIBUTES"}),
                    )
                }
                ViewKind::Index { stride, count } => {
                    let idx: Vec<u32> = if stride == 2 {
                        data.chunks_exact(2)
                            .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
                            .collect()
                    } else {
                        data.chunks_exact(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect()
                    };
                    let vcount = idx
                        .iter()
                        .copied()
                        .max()
                        .map(|m| m as usize + 1)
                        .unwrap_or(0);
                    let enc = meshopt::encode_index_buffer(&idx, vcount)
                        .map_err(|e| IoError::Meshopt(e.to_string()))?;
                    // The index codec may rotate triangle corners (winding is
                    // preserved); make the fallback identical to the decoded data.
                    let decoded = decode_meshopt(&enc, count, stride, "TRIANGLES", None)?;
                    (
                        enc,
                        decoded,
                        json!({"byteStride": stride, "count": count, "mode": "TRIANGLES"}),
                    )
                }
            };
            align4(&mut self.bin);
            let c_off = self.bin.len();
            self.bin.extend_from_slice(&encoded);
            let fb_off = self.fallback_len.next_multiple_of(4);
            self.fallback_len = fb_off + fallback.len();
            let mut ext = ext;
            ext["buffer"] = json!(0);
            ext["byteOffset"] = json!(c_off);
            ext["byteLength"] = json!(encoded.len());
            view.insert("buffer".into(), json!(1));
            view.insert("byteOffset".into(), json!(fb_off));
            view.insert("byteLength".into(), json!(fallback.len()));
            view.insert("extensions".into(), json!({ EXT_MESHOPT: ext }));
            self.used_compression = true;
        } else {
            align4(&mut self.bin);
            view.insert("buffer".into(), json!(0));
            view.insert("byteOffset".into(), json!(self.bin.len()));
            view.insert("byteLength".into(), json!(data.len()));
            self.bin.extend_from_slice(&data);
        }
        if let Some(s) = stride_field {
            view.insert("byteStride".into(), json!(s));
        }
        view.insert("target".into(), json!(target));
        self.views.push(Value::Object(view));
        Ok(self.views.len() - 1)
    }

    fn add_accessor(&mut self, acc: Value) -> usize {
        self.accessors.push(acc);
        self.accessors.len() - 1
    }
}

fn encode_vertex_bytes(data: &[u8], stride: usize, count: usize) -> Result<Vec<u8>, IoError> {
    macro_rules! enc {
        ($n:literal) => {{
            let v: Vec<[u8; $n]> = data
                .chunks_exact($n)
                .map(|c| <[u8; $n]>::try_from(c).unwrap())
                .collect();
            debug_assert_eq!(v.len(), count);
            meshopt::encode_vertex_buffer(&v).map_err(|e| IoError::Meshopt(e.to_string()))
        }};
    }
    match stride {
        8 => enc!(8),
        12 => enc!(12),
        16 => enc!(16),
        _ => Err(IoError::Meshopt(format!(
            "unsupported vertex stride {stride}"
        ))),
    }
}

/// Decode one meshopt-compressed bufferView into `count * stride` bytes.
fn decode_meshopt(
    src: &[u8],
    count: usize,
    stride: usize,
    mode: &str,
    filter: Option<&str>,
) -> Result<Vec<u8>, IoError> {
    let mut out = vec![0u8; count * stride];
    if count == 0 {
        return Ok(out);
    }
    // Safety: `out` holds exactly `count * stride` bytes, which is what the
    // decoders write; `src` is a valid slice of the given length.
    let rc = unsafe {
        match mode {
            "ATTRIBUTES" => {
                if !stride.is_multiple_of(4) || stride > 256 {
                    return Err(IoError::Meshopt(format!(
                        "invalid ATTRIBUTES stride {stride}"
                    )));
                }
                meshopt::ffi::meshopt_decodeVertexBuffer(
                    out.as_mut_ptr().cast(),
                    count,
                    stride,
                    src.as_ptr(),
                    src.len(),
                )
            }
            "TRIANGLES" | "INDICES" => {
                if stride != 2 && stride != 4 {
                    return Err(IoError::Meshopt(format!("invalid index stride {stride}")));
                }
                if mode == "TRIANGLES" {
                    meshopt::ffi::meshopt_decodeIndexBuffer(
                        out.as_mut_ptr().cast(),
                        count,
                        stride,
                        src.as_ptr(),
                        src.len(),
                    )
                } else {
                    meshopt::ffi::meshopt_decodeIndexSequence(
                        out.as_mut_ptr().cast(),
                        count,
                        stride,
                        src.as_ptr(),
                        src.len(),
                    )
                }
            }
            m => return Err(IoError::Meshopt(format!("unknown meshopt mode '{m}'"))),
        }
    };
    if rc != 0 {
        return Err(IoError::Meshopt(format!(
            "meshopt decode failed (code {rc}, mode {mode})"
        )));
    }
    match filter.unwrap_or("NONE") {
        "NONE" => {}
        "OCTAHEDRAL" => filter_oct(&mut out, stride)?,
        "QUATERNION" => filter_quat(&mut out, stride)?,
        "EXPONENTIAL" => filter_exp(&mut out, stride)?,
        f => return Err(IoError::Meshopt(format!("unknown meshopt filter '{f}'"))),
    }
    Ok(out)
}

// Decode filters of EXT_meshopt_compression, ported from the extension
// specification's reference code (the `meshopt` crate does not build the
// C++ filter sources).

/// Octahedral filter: 4x i8 (stride 4) or 4x i16 (stride 8) per element.
pub(crate) fn filter_oct(data: &mut [u8], stride: usize) -> Result<(), IoError> {
    fn run<const N: usize>(data: &mut [u8], get: fn(&[u8]) -> f32, put: fn(&mut [u8], f32)) {
        let max = ((1i32 << (N * 8 - 1)) - 1) as f32;
        for e in data.chunks_exact_mut(4 * N) {
            let mut x = get(&e[0..N]);
            let mut y = get(&e[N..2 * N]);
            let z = get(&e[2 * N..3 * N]) - x.abs() - y.abs();
            let t = if z >= 0.0 { 0.0 } else { z };
            x += if x >= 0.0 { t } else { -t };
            y += if y >= 0.0 { t } else { -t };
            let l = (x * x + y * y + z * z).sqrt();
            let s = if l > 0.0 { max / l } else { 0.0 };
            put(&mut e[0..N], (x * s).round());
            put(&mut e[N..2 * N], (y * s).round());
            put(&mut e[2 * N..3 * N], (z * s).round());
        }
    }
    match stride {
        4 => run::<1>(data, |b| b[0] as i8 as f32, |b, v| b[0] = v as i8 as u8),
        8 => run::<2>(
            data,
            |b| i16::from_le_bytes([b[0], b[1]]) as f32,
            |b, v| b.copy_from_slice(&(v as i16).to_le_bytes()),
        ),
        _ => {
            return Err(IoError::Meshopt(format!(
                "OCTAHEDRAL filter needs stride 4 or 8, got {stride}"
            )));
        }
    }
    Ok(())
}

/// Quaternion filter: 4x i16 per element (stride 8).
pub(crate) fn filter_quat(data: &mut [u8], stride: usize) -> Result<(), IoError> {
    if stride != 8 {
        return Err(IoError::Meshopt(format!(
            "QUATERNION filter needs stride 8, got {stride}"
        )));
    }
    let scale = 1.0f32 / 2.0f32.sqrt();
    for e in data.chunks_exact_mut(8) {
        let c: [i16; 4] = std::array::from_fn(|k| i16::from_le_bytes([e[2 * k], e[2 * k + 1]]));
        let sf = (c[3] as i32) | 3;
        let ss = scale / sf as f32;
        let x = c[0] as f32 * ss;
        let y = c[1] as f32 * ss;
        let z = c[2] as f32 * ss;
        let ww = 1.0 - x * x - y * y - z * z;
        let w = if ww >= 0.0 { ww } else { 0.0 }.sqrt();
        let r = |v: f32| (v * 32767.0 + if v >= 0.0 { 0.5 } else { -0.5 }) as i32 as i16;
        let qc = (c[3] & 3) as usize;
        let mut o = [0i16; 4];
        o[(qc + 1) & 3] = r(x);
        o[(qc + 2) & 3] = r(y);
        o[(qc + 3) & 3] = r(z);
        o[qc] = r(w);
        for k in 0..4 {
            e[2 * k..2 * k + 2].copy_from_slice(&o[k].to_le_bytes());
        }
    }
    Ok(())
}

/// Exponential filter: each 32-bit value is 2^E * M (8-bit signed exponent,
/// 24-bit signed mantissa).
pub(crate) fn filter_exp(data: &mut [u8], stride: usize) -> Result<(), IoError> {
    if !stride.is_multiple_of(4) {
        return Err(IoError::Meshopt(format!(
            "EXPONENTIAL filter needs stride % 4 == 0, got {stride}"
        )));
    }
    for e in data.chunks_exact_mut(4) {
        let v = i32::from_le_bytes([e[0], e[1], e[2], e[3]]);
        let exp = v >> 24;
        let mant = (v << 8) >> 8;
        let f = (mant as f32) * 2.0f32.powi(exp);
        e.copy_from_slice(&f.to_le_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod filter_tests {
    use super::*;

    #[test]
    fn exp_filter() {
        // 3 * 2^-2 = 0.75
        let v: i32 = ((-2i32) << 24) | 3;
        let mut d = v.to_le_bytes().to_vec();
        filter_exp(&mut d, 4).unwrap();
        assert_eq!(f32::from_le_bytes([d[0], d[1], d[2], d[3]]), 0.75);
        let v: i32 = (1 << 24) | (0x00ff_ffff & -5i32);
        let mut d = v.to_le_bytes().to_vec();
        filter_exp(&mut d, 4).unwrap();
        assert_eq!(f32::from_le_bytes([d[0], d[1], d[2], d[3]]), -10.0);
    }

    #[test]
    fn oct_filter_axis() {
        // +Z: x = y = 0, z = one (127); w preserved.
        let mut d = vec![0u8, 0, 127, 9];
        filter_oct(&mut d, 4).unwrap();
        assert_eq!(d, vec![0, 0, 127, 9]);
        // +X: x = one, y = 0.
        let mut d = vec![127u8, 0, 127, 0];
        filter_oct(&mut d, 4).unwrap();
        assert_eq!(d, vec![127, 0, 0, 0]);
        // -Z folds to the corner x = y = one.
        let mut d = vec![127u8, 127, 127, 0];
        filter_oct(&mut d, 4).unwrap();
        assert_eq!(
            d.iter().map(|&b| b as i8).collect::<Vec<_>>(),
            vec![0, 0, -127, 0]
        );
    }

    #[test]
    fn quat_filter_identity() {
        // Identity quaternion: max component w (index 3) reconstructed; x=y=z=0.
        let qc = 3i16;
        let c = [0i16, 0, 0, (0x7ff0 & !3) | qc];
        let mut d: Vec<u8> = c.iter().flat_map(|v| v.to_le_bytes()).collect();
        filter_quat(&mut d, 8).unwrap();
        let o: Vec<i16> = d
            .chunks(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(o, vec![0, 0, 0, 32767]);
    }
}

fn check_scene(scene: &RenderScene) -> Result<(), IoError> {
    let n = scene.nodes.len();
    for (i, node) in scene.nodes.iter().enumerate() {
        if let Some(p) = node.parent
            && p as usize >= n
        {
            return Err(IoError::InvalidScene(format!(
                "node {i} has out-of-range parent {p}"
            )));
        }
        // Cycle check: the parent chain must terminate within n steps.
        let mut cur = node.parent;
        let mut steps = 0;
        while let Some(p) = cur {
            steps += 1;
            if steps > n {
                return Err(IoError::InvalidScene(format!(
                    "node {i} is part of a parent cycle"
                )));
            }
            cur = scene.nodes[p as usize].parent;
        }
        for &m in &node.mesh_lods {
            if m as usize >= scene.meshes.len() {
                return Err(IoError::InvalidScene(format!(
                    "node {i} references missing mesh {m}"
                )));
            }
        }
        if node.translation.iter().any(|v| !v.is_finite()) {
            return Err(IoError::InvalidScene(format!(
                "node {i} has a non-finite translation"
            )));
        }
    }
    for (mi, m) in scene.meshes.iter().enumerate() {
        let nv = m.positions.len();
        if !m.normals.is_empty() && m.normals.len() != nv {
            return Err(IoError::InvalidScene(format!(
                "mesh {mi}: {} normals for {nv} positions",
                m.normals.len()
            )));
        }
        if let Some(uv) = &m.uvs
            && uv.len() != nv
        {
            return Err(IoError::InvalidScene(format!(
                "mesh {mi}: {} uvs for {nv} positions",
                uv.len()
            )));
        }
        if let Some(t) = &m.tangents
            && t.len() != nv
        {
            return Err(IoError::InvalidScene(format!(
                "mesh {mi}: {} tangents for {nv} positions",
                t.len()
            )));
        }
        if m.positions.iter().flatten().any(|v| !v.is_finite()) {
            return Err(IoError::InvalidScene(format!(
                "mesh {mi} has non-finite positions"
            )));
        }
        for (pi, p) in m.primitives.iter().enumerate() {
            if p.indices.len() % 3 != 0 {
                return Err(IoError::InvalidScene(format!(
                    "mesh {mi} primitive {pi}: index count not a multiple of 3"
                )));
            }
            if let Some(&bad) = p.indices.iter().find(|&&i| i as usize >= nv) {
                return Err(IoError::InvalidScene(format!(
                    "mesh {mi} primitive {pi}: index {bad} >= vertex count {nv}"
                )));
            }
            if !scene.materials.is_empty() && p.material as usize >= scene.materials.len() {
                return Err(IoError::InvalidScene(format!(
                    "mesh {mi} primitive {pi}: material {} missing",
                    p.material
                )));
            }
        }
    }
    Ok(())
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = ((v[0] as f64).powi(2) + (v[1] as f64).powi(2) + (v[2] as f64).powi(2)).sqrt();
    if l > 1e-20 && l.is_finite() {
        [
            (v[0] as f64 / l) as f32,
            (v[1] as f64 / l) as f32,
            (v[2] as f64 / l) as f32,
        ]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn bytes_of<T: bytemuck::Pod>(v: &[T]) -> Vec<u8> {
    // glTF is little-endian; so is every supported target.
    bytemuck::cast_slice(v).to_vec()
}

/// Write `scene` as a binary glTF 2.0 (GLB) file.
pub fn write_glb(scene: &RenderScene, opts: &GltfOptions) -> Result<Vec<u8>, IoError> {
    check_scene(scene)?;
    write_glb_core(scene, MeshList::Borrowed(&scene.meshes), opts)
}

/// [`write_glb`] that consumes the scene and frees every mesh as soon as its
/// buffers are written, so the scene and the binary buffer are not both held
/// in full (building-scale assets). Identical output.
pub fn write_glb_owned(mut scene: RenderScene, opts: &GltfOptions) -> Result<Vec<u8>, IoError> {
    check_scene(&scene)?;
    let meshes = std::mem::take(&mut scene.meshes);
    write_glb_core(&scene, MeshList::Owned(meshes), opts)
}

enum MeshList<'a> {
    Borrowed(&'a [RenderMesh]),
    Owned(Vec<RenderMesh>),
}

impl MeshList<'_> {
    fn len(&self) -> usize {
        match self {
            MeshList::Borrowed(m) => m.len(),
            MeshList::Owned(m) => m.len(),
        }
    }
    fn get(&self, i: usize) -> &RenderMesh {
        match self {
            MeshList::Borrowed(m) => &m[i],
            MeshList::Owned(m) => &m[i],
        }
    }
    fn release(&mut self, i: usize) {
        if let MeshList::Owned(m) = self {
            m[i] = RenderMesh::default();
            // the binary buffer is one large (mmap'd) allocation that cannot
            // reuse the freed mesh memory; hand it back to the OS regularly
            if i % 4096 == 4095 {
                release_free_memory();
            }
        }
    }
}

/// Return freed heap pages to the OS (glibc `malloc_trim`; no-op
/// elsewhere). Keeps the resident set of building-scale exports near the
/// live data instead of the sum of every stage's peak.
pub fn release_free_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
}

fn write_glb_core(scene: &RenderScene, mut mesh_list: MeshList, opts: &GltfOptions) -> Result<Vec<u8>, IoError> {
    // reserve the (uncompressed) binary size up front: doubling growth of a
    // multi-GB buffer would nearly double the peak memory
    let estimate: usize = (0..mesh_list.len())
        .map(|i| mesh_list.get(i))
        .map(|m| {
            let nv = m.positions.len();
            nv * (24 + m.uvs.as_ref().map_or(0, |_| 8) + m.tangents.as_ref().map_or(0, |_| 16)) + m.primitives.iter().map(|p| 4 * p.indices.len() + 4).sum::<usize>() + 64
        })
        .sum();
    let mut w = Writer {
        compress: opts.meshopt_compression,
        bin: Vec::with_capacity(if opts.meshopt_compression { estimate / 2 } else { estimate }),
        fallback_len: 0,
        views: Vec::new(),
        accessors: Vec::new(),
        used_compression: false,
    };

    // Materials.
    let clamp01 = |x: f32| {
        if x.is_finite() {
            x.clamp(0.0, 1.0)
        } else {
            1.0
        }
    };
    let materials: Vec<Value> = scene
        .materials
        .iter()
        .map(|m| {
            let mut o = Map::new();
            if !m.name.is_empty() {
                o.insert("name".into(), json!(m.name));
            }
            o.insert(
                "pbrMetallicRoughness".into(),
                json!({
                    "baseColorFactor": m.base_color.iter().map(|&c| jshort(clamp01(c))).collect::<Vec<_>>(),
                    "metallicFactor": jshort(clamp01(m.metallic)),
                    "roughnessFactor": jshort(clamp01(m.roughness)),
                }),
            );
            if m.double_sided {
                o.insert("doubleSided".into(), json!(true));
            }
            Value::Object(o)
        })
        .collect();

    // Meshes.
    let mut mesh_map: Vec<Option<usize>> = Vec::with_capacity(mesh_list.len());
    let mut meshes: Vec<Value> = Vec::new();
    for mi in 0..mesh_list.len() {
        let m = mesh_list.get(mi);
        let nv = m.positions.len();
        let prims: Vec<&RenderPrimitive> = m
            .primitives
            .iter()
            .filter(|p| !p.indices.is_empty())
            .collect();
        if nv == 0 || prims.is_empty() {
            mesh_map.push(None);
            continue;
        }
        let mut attrs = Map::new();
        // POSITION with exact min/max.
        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        for p in &m.positions {
            for k in 0..3 {
                mn[k] = mn[k].min(p[k]);
                mx[k] = mx[k].max(p[k]);
            }
        }
        // Canonicalize -0.0 so min/max match the data bit-for-bit semantics.
        let positions: Vec<[f32; 3]> = m
            .positions
            .iter()
            .map(|p| [p[0] + 0.0, p[1] + 0.0, p[2] + 0.0])
            .collect();
        let (mn, mx) = (mn.map(|v| v + 0.0), mx.map(|v| v + 0.0));
        let v = w.add_view(
            bytes_of(&positions),
            ViewKind::Attribute {
                stride: 12,
                count: nv,
            },
        )?;
        let a = w.add_accessor(json!({
            "bufferView": v, "componentType": FLOAT, "count": nv, "type": "VEC3",
            "min": mn.iter().map(|&x| jexact(x)).collect::<Vec<_>>(),
            "max": mx.iter().map(|&x| jexact(x)).collect::<Vec<_>>(),
        }));
        attrs.insert("POSITION".into(), json!(a));
        let has_normals = !m.normals.is_empty();
        if has_normals {
            let normals: Vec<[f32; 3]> = m.normals.iter().map(|&n| normalize3(n)).collect();
            let v = w.add_view(
                bytes_of(&normals),
                ViewKind::Attribute {
                    stride: 12,
                    count: nv,
                },
            )?;
            let a = w.add_accessor(
                json!({"bufferView": v, "componentType": FLOAT, "count": nv, "type": "VEC3"}),
            );
            attrs.insert("NORMAL".into(), json!(a));
        }
        if let Some(uvs) = &m.uvs {
            let uvs: Vec<[f32; 2]> = uvs
                .iter()
                .map(|t| t.map(|x| if x.is_finite() { x } else { 0.0 }))
                .collect();
            let v = w.add_view(
                bytes_of(&uvs),
                ViewKind::Attribute {
                    stride: 8,
                    count: nv,
                },
            )?;
            let a = w.add_accessor(
                json!({"bufferView": v, "componentType": FLOAT, "count": nv, "type": "VEC2"}),
            );
            attrs.insert("TEXCOORD_0".into(), json!(a));
        }
        if let (Some(tangents), true) = (&m.tangents, has_normals) {
            let tangents: Vec<[f32; 4]> = tangents
                .iter()
                .map(|t| {
                    let d = normalize3([t[0], t[1], t[2]]);
                    [d[0], d[1], d[2], if t[3] < 0.0 { -1.0 } else { 1.0 }]
                })
                .collect();
            let v = w.add_view(
                bytes_of(&tangents),
                ViewKind::Attribute {
                    stride: 16,
                    count: nv,
                },
            )?;
            let a = w.add_accessor(
                json!({"bufferView": v, "componentType": FLOAT, "count": nv, "type": "VEC4"}),
            );
            attrs.insert("TANGENT".into(), json!(a));
        }
        let use_u16 = nv <= 65535;
        let mut prim_json = Vec::new();
        for p in prims {
            let count = p.indices.len();
            let (data, stride, ct) = if use_u16 {
                let v: Vec<u16> = p.indices.iter().map(|&i| i as u16).collect();
                (bytes_of(&v), 2, UNSIGNED_SHORT)
            } else {
                (bytes_of(&p.indices), 4, UNSIGNED_INT)
            };
            let v = w.add_view(data, ViewKind::Index { stride, count })?;
            let a = w.add_accessor(
                json!({"bufferView": v, "componentType": ct, "count": count, "type": "SCALAR"}),
            );
            let mut pj = Map::new();
            pj.insert("attributes".into(), Value::Object(attrs.clone()));
            pj.insert("indices".into(), json!(a));
            if !scene.materials.is_empty() {
                pj.insert("material".into(), json!(p.material));
            }
            pj.insert("mode".into(), json!(4));
            prim_json.push(Value::Object(pj));
        }
        let mut mj = Map::new();
        if !m.name.is_empty() {
            mj.insert("name".into(), json!(m.name));
        }
        mj.insert("primitives".into(), Value::Array(prim_json));
        mesh_map.push(Some(meshes.len()));
        meshes.push(Value::Object(mj));
        mesh_list.release(mi);
    }

    // Nodes.
    let n = scene.nodes.len();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut roots = Vec::new();
    for (i, node) in scene.nodes.iter().enumerate() {
        match node.parent {
            Some(p) => children[p as usize].push(i),
            None => roots.push(i),
        }
    }
    let local_t = |i: usize| -> [f64; 3] {
        let t = scene.nodes[i].translation;
        match scene.nodes[i].parent {
            Some(p) => {
                let pt = scene.nodes[p as usize].translation;
                [t[0] - pt[0], t[1] - pt[1], t[2] - pt[2]]
            }
            None => t,
        }
    };
    let mut nodes: Vec<Value> = Vec::with_capacity(n);
    let mut lod_nodes: Vec<Value> = Vec::new();
    let mut used_lod = false;
    for (i, node) in scene.nodes.iter().enumerate() {
        let mut o = Map::new();
        let name = if node.name.is_empty() {
            format!("node_{i}")
        } else {
            node.name.clone()
        };
        o.insert("name".into(), json!(name));
        if !children[i].is_empty() {
            o.insert("children".into(), json!(children[i]));
        }
        let t = local_t(i);
        let translation = if t.iter().any(|&v| v != 0.0) {
            Some(Value::Array(t.iter().map(|&v| jshort(v as f32)).collect()))
        } else {
            None
        };
        if let Some(t) = &translation {
            o.insert("translation".into(), t.clone());
        }
        let lods: Vec<usize> = node
            .mesh_lods
            .iter()
            .filter_map(|&m| mesh_map[m as usize])
            .collect();
        if let Some(&m0) = lods.first() {
            o.insert("mesh".into(), json!(m0));
        }
        if lods.len() > 1 {
            let mut ids = Vec::new();
            for (k, &m) in lods.iter().enumerate().skip(1) {
                ids.push(n + lod_nodes.len());
                let mut lo = Map::new();
                lo.insert("name".into(), json!(format!("{name}_lod{k}")));
                if let Some(t) = &translation {
                    lo.insert("translation".into(), t.clone());
                }
                lo.insert("mesh".into(), json!(m));
                lod_nodes.push(Value::Object(lo));
            }
            o.insert("extensions".into(), json!({ MSFT_LOD: { "ids": ids } }));
            used_lod = true;
        }
        let extras = match &node.extras {
            Value::Null if lods.is_empty() => Value::Null,
            Value::Null => json!({ "lods": lods }),
            Value::Object(map) => {
                let mut map = map.clone();
                if !lods.is_empty() {
                    map.insert("lods".into(), json!(lods));
                }
                Value::Object(map)
            }
            other => other.clone(),
        };
        if !extras.is_null() {
            o.insert("extras".into(), extras);
        }
        nodes.push(Value::Object(o));
    }
    nodes.extend(lod_nodes);

    // Root.
    let mut root = Map::new();
    root.insert(
        "asset".into(),
        json!({"version": "2.0", "generator": format!("Shardwright frac-io {}", env!("CARGO_PKG_VERSION"))}),
    );
    let mut ext_used = Vec::new();
    if w.used_compression {
        ext_used.push(EXT_MESHOPT);
        root.insert("extensionsRequired".into(), json!([EXT_MESHOPT]));
    }
    if used_lod {
        ext_used.push(MSFT_LOD);
    }
    if !ext_used.is_empty() {
        ext_used.sort_unstable();
        root.insert("extensionsUsed".into(), json!(ext_used));
    }
    let mut sc = Map::new();
    if !scene.name.is_empty() {
        sc.insert("name".into(), json!(scene.name));
    }
    sc.insert("nodes".into(), json!(roots));
    if !scene.extras.is_null() {
        sc.insert("extras".into(), scene.extras.clone());
    }
    // glTF forbids empty `scenes[].nodes`; a scene without nodes has no
    // glTF scene at all (its extras then go to `asset.extras`).
    if !roots.is_empty() {
        root.insert("scene".into(), json!(0));
        root.insert("scenes".into(), json!([Value::Object(sc)]));
    } else if !scene.extras.is_null() {
        root["asset"]["extras"] = scene.extras.clone();
    }
    if !nodes.is_empty() {
        root.insert("nodes".into(), Value::Array(nodes));
    }
    if !materials.is_empty() {
        root.insert("materials".into(), Value::Array(materials));
    }
    if !meshes.is_empty() {
        root.insert("meshes".into(), Value::Array(meshes));
    }
    if !w.accessors.is_empty() {
        root.insert(
            "accessors".into(),
            Value::Array(std::mem::take(&mut w.accessors)),
        );
    }
    if !w.views.is_empty() {
        root.insert(
            "bufferViews".into(),
            Value::Array(std::mem::take(&mut w.views)),
        );
    }
    let has_bin = !w.bin.is_empty();
    let mut buffers = Vec::new();
    if has_bin {
        buffers.push(json!({"byteLength": w.bin.len()}));
    }
    if w.used_compression {
        buffers.push(json!({"byteLength": w.fallback_len, "extensions": { EXT_MESHOPT: {"fallback": true} }}));
    }
    if !buffers.is_empty() {
        root.insert("buffers".into(), Value::Array(buffers));
    }
    let json_bytes = serde_json::to_vec(&Value::Object(root))?;
    Ok(write_glb_container(
        &json_bytes,
        has_bin.then_some(&w.bin[..]),
    ))
}

/// Rewrite a GLB that uses `EXT_meshopt_compression` into an equivalent
/// uncompressed GLB (all data in the BIN chunk). Returns `Ok(None)` when the
/// file does not use the extension. Only GLB-embedded data is supported.
pub fn decompress_meshopt_glb(bytes: &[u8]) -> Result<Option<Vec<u8>>, IoError> {
    let (mut root, bin) = read_glb_container(bytes)?;
    let uses = root
        .get("extensionsUsed")
        .and_then(|v| v.as_array())
        .is_some_and(|a| a.iter().any(|e| e.as_str() == Some(EXT_MESHOPT)));
    if !uses {
        return Ok(None);
    }
    let buffers = root
        .get("buffers")
        .and_then(|b| b.as_array())
        .cloned()
        .unwrap_or_default();
    // Data per buffer: Some(bytes) for the GLB BIN buffer, None for fallback.
    let mut data: Vec<Option<&[u8]>> = Vec::new();
    for (i, b) in buffers.iter().enumerate() {
        let fallback = b
            .pointer("/extensions/EXT_meshopt_compression/fallback")
            .and_then(|v| v.as_bool())
            == Some(true);
        if fallback {
            data.push(None);
        } else if b.get("uri").is_none() && i == 0 {
            data.push(Some(
                bin.as_deref()
                    .ok_or_else(|| IoError::Gltf("GLB has no BIN chunk".into()))?,
            ));
        } else {
            return Err(IoError::Gltf(format!(
                "buffer {i}: external/data-URI buffers are not supported with meshopt"
            )));
        }
    }
    let get_slice = |buf: usize, off: usize, len: usize| -> Result<&[u8], IoError> {
        data.get(buf)
            .copied()
            .flatten()
            .and_then(|d| d.get(off..off + len))
            .ok_or_else(|| {
                IoError::Gltf(format!("bufferView range {off}+{len} outside buffer {buf}"))
            })
    };
    let uint = |v: &Value, k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
    let mut new_bin = Vec::new();
    if let Some(views) = root.get_mut("bufferViews").and_then(|v| v.as_array_mut()) {
        for (vi, view) in views.iter_mut().enumerate() {
            let ext = view.pointer("/extensions/EXT_meshopt_compression").cloned();
            let bytes_out = match ext {
                Some(e) => {
                    let buf = uint(&e, "buffer")
                        .ok_or_else(|| IoError::Gltf(format!("view {vi}: missing buffer")))?;
                    let off = uint(&e, "byteOffset").unwrap_or(0);
                    let len = uint(&e, "byteLength")
                        .ok_or_else(|| IoError::Gltf(format!("view {vi}: missing byteLength")))?;
                    let stride = uint(&e, "byteStride")
                        .ok_or_else(|| IoError::Gltf(format!("view {vi}: missing byteStride")))?;
                    let count = uint(&e, "count")
                        .ok_or_else(|| IoError::Gltf(format!("view {vi}: missing count")))?;
                    let mode = e
                        .get("mode")
                        .and_then(|m| m.as_str())
                        .unwrap_or("ATTRIBUTES");
                    let filter = e.get("filter").and_then(|m| m.as_str());
                    let mut out =
                        decode_meshopt(get_slice(buf, off, len)?, count, stride, mode, filter)?;
                    let want = uint(view, "byteLength").unwrap_or(out.len());
                    out.resize(want, 0);
                    out
                }
                None => {
                    let buf = uint(view, "buffer").unwrap_or(0);
                    let off = uint(view, "byteOffset").unwrap_or(0);
                    let len = uint(view, "byteLength").unwrap_or(0);
                    get_slice(buf, off, len)?.to_vec()
                }
            };
            align4(&mut new_bin);
            let o = view
                .as_object_mut()
                .ok_or_else(|| IoError::Gltf("bufferView is not an object".into()))?;
            o.insert("buffer".into(), json!(0));
            o.insert("byteOffset".into(), json!(new_bin.len()));
            o.insert("byteLength".into(), json!(bytes_out.len()));
            if let Some(exts) = o.get_mut("extensions").and_then(|e| e.as_object_mut()) {
                exts.remove(EXT_MESHOPT);
                if exts.is_empty() {
                    o.remove("extensions");
                }
            }
            new_bin.extend_from_slice(&bytes_out);
        }
    }
    let o = root
        .as_object_mut()
        .ok_or_else(|| IoError::Gltf("glTF root is not an object".into()))?;
    for key in ["extensionsUsed", "extensionsRequired"] {
        if let Some(a) = o.get_mut(key).and_then(|v| v.as_array_mut()) {
            a.retain(|e| e.as_str() != Some(EXT_MESHOPT));
            if a.is_empty() {
                o.remove(key);
            }
        }
    }
    if new_bin.is_empty() {
        o.remove("buffers");
    } else {
        o.insert("buffers".into(), json!([{"byteLength": new_bin.len()}]));
    }
    let json_bytes = serde_json::to_vec(&root)?;
    Ok(Some(write_glb_container(
        &json_bytes,
        (!new_bin.is_empty()).then_some(&new_bin[..]),
    )))
}

/// Summary of a GLB file (for tests and `inspect`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GlbSummary {
    /// All nodes, including LOD nodes that are not part of the scene.
    pub node_count: usize,
    pub scene_root_count: usize,
    pub mesh_count: usize,
    pub material_count: usize,
    /// Triangles per glTF mesh (sum over its triangle primitives).
    pub mesh_triangles: Vec<usize>,
    pub total_triangles: usize,
    pub node_names: Vec<String>,
    pub node_mesh: Vec<Option<usize>>,
    pub node_parent: Vec<Option<usize>>,
    /// Node extras (`Value::Null` when absent).
    pub node_extras: Vec<Value>,
    pub scene_extras: Value,
    pub extensions_used: Vec<String>,
    pub extensions_required: Vec<String>,
    /// True if the file used `EXT_meshopt_compression` (decoded for the summary).
    pub meshopt_compressed: bool,
}

fn raw_to_value(raw: &Option<Box<serde_json::value::RawValue>>) -> Value {
    raw.as_ref()
        .and_then(|r| serde_json::from_str(r.get()).ok())
        .unwrap_or(Value::Null)
}

/// Read a GLB and summarize it using the `gltf` crate (compressed files are
/// decoded with [`decompress_meshopt_glb`] first). Also verifies that all
/// index data can be read.
pub fn read_glb_summary(bytes: &[u8]) -> Result<GlbSummary, IoError> {
    let decompressed = decompress_meshopt_glb(bytes)?;
    let compressed = decompressed.is_some();
    let data = decompressed.as_deref().unwrap_or(bytes);
    let g = gltf::Gltf::from_slice(data).map_err(|e| IoError::Gltf(e.to_string()))?;
    let buffers = gltf::import_buffers(&g.document, None, g.blob.clone())
        .map_err(|e| IoError::Gltf(e.to_string()))?;
    let doc = &g.document;
    let mut s = GlbSummary {
        node_count: doc.nodes().len(),
        mesh_count: doc.meshes().len(),
        material_count: doc.materials().len(),
        meshopt_compressed: compressed,
        extensions_used: doc.extensions_used().map(|e| e.to_string()).collect(),
        extensions_required: doc.extensions_required().map(|e| e.to_string()).collect(),
        ..Default::default()
    };
    if compressed {
        s.extensions_used.push(EXT_MESHOPT.to_string());
        s.extensions_used.sort();
        s.extensions_required.push(EXT_MESHOPT.to_string());
        s.extensions_required.sort();
    }
    for mesh in doc.meshes() {
        let mut tris = 0usize;
        for prim in mesh.primitives() {
            let reader = prim.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
            let nidx = match reader.read_indices() {
                Some(it) => it.into_u32().count(),
                None => prim
                    .get(&gltf::Semantic::Positions)
                    .map(|a| a.count())
                    .unwrap_or(0),
            };
            tris += match prim.mode() {
                gltf::mesh::Mode::Triangles => nidx / 3,
                gltf::mesh::Mode::TriangleStrip | gltf::mesh::Mode::TriangleFan => {
                    nidx.saturating_sub(2)
                }
                _ => 0,
            };
        }
        s.mesh_triangles.push(tris);
    }
    s.total_triangles = s.mesh_triangles.iter().sum();
    s.node_parent = vec![None; s.node_count];
    for node in doc.nodes() {
        s.node_names.push(node.name().unwrap_or("").to_string());
        s.node_mesh.push(node.mesh().map(|m| m.index()));
        s.node_extras.push(raw_to_value(node.extras()));
        for c in node.children() {
            s.node_parent[c.index()] = Some(node.index());
        }
    }
    if let Some(scene) = doc.default_scene().or_else(|| doc.scenes().next()) {
        s.scene_root_count = scene.nodes().len();
        s.scene_extras = raw_to_value(scene.extras());
    }
    Ok(s)
}
