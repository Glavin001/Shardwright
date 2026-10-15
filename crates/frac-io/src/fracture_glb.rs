//! Self-contained glTF delivery of a baked fracture asset. No fracture cooking.
use crate::{
    IoError, fb,
    glb::{read_glb_container, write_glb_container_owned},
};
use serde_json::{Value, json};

/// Project/vendor extension, not a ratified Khronos extension.
pub const FRACTURE_EXTENSION: &str = "SHARDWRIGHT_fracture";

/// Embed a verified FRAC payload and the resolved material library in a GLB.
/// Existing rendering resources are retained. External resources are rejected.
/// The extension is optional for rendering, required by a destruction consumer.
pub fn write_fracture_glb(
    glb: &[u8],
    physics: &[u8],
    materials: &Value,
) -> Result<Vec<u8>, IoError> {
    if physics.len() < 8 {
        return Err(IoError::Physics("truncated FRAC header".into()));
    }
    crate::read_physics(physics)?;
    let asset = fb::root_as_asset(physics).map_err(|e| IoError::Physics(e.to_string()))?;
    if asset.schema_version() != frac_core::SCHEMA_VERSION {
        return Err(IoError::Physics("unsupported FRAC schema version".into()));
    }
    for (field, expected) in [
        ("library_id", asset.material_library_id().unwrap_or("")),
        (
            "library_version",
            asset.material_library_version().unwrap_or(""),
        ),
    ] {
        if materials[field].as_str() != Some(expected) {
            return Err(IoError::Physics(format!(
                "material {field} does not match baked physics"
            )));
        }
    }
    let material_count = materials["material"].as_array().map_or(0, Vec::len);
    let interface_count = materials["interface_material"]
        .as_array()
        .map_or(0, Vec::len);
    let (mut doc, bin) = read_glb_container(glb)?;
    if u32::from_le_bytes(glb[8..12].try_into().unwrap()) as usize != glb.len() {
        return Err(IoError::Gltf("GLB length mismatch".into()));
    }
    if doc
        .get("extensions")
        .and_then(|e| e.get(FRACTURE_EXTENSION))
        .is_some()
    {
        return Err(IoError::Gltf("GLB already contains fracture data".into()));
    }
    let buffers = doc["buffers"]
        .as_array()
        .ok_or_else(|| IoError::Gltf("missing buffers".into()))?;
    // Meshopt may declare URI-less virtual fallback buffers; all encoded
    // bytes still live in embedded buffer 0. No external files are needed.
    if buffers.is_empty()
        || buffers[0].get("uri").is_some()
        || buffers.iter().skip(1).any(|b| {
            b.get("uri").is_some() || b["extensions"]["EXT_meshopt_compression"]["fallback"] != true
        })
        || doc["images"]
            .as_array()
            .is_some_and(|images| images.iter().any(|v| v.get("uri").is_some()))
    {
        return Err(IoError::Gltf(
            "standalone export requires all buffers and images embedded".into(),
        ));
    }
    let nodes = doc["nodes"]
        .as_array()
        .ok_or_else(|| IoError::Gltf("missing nodes".into()))?;
    if let Some(fs) = asset.fragments() {
        for f in fs {
            let node = f.gltf_node();
            if node >= 0 {
                let n = nodes.get(node as usize).ok_or_else(|| {
                    IoError::Physics(format!("fragment {} has invalid glTF node {node}", f.id()))
                })?;
                if n["extras"]["fragment_id"].as_u64() != Some(f.id() as u64) {
                    return Err(IoError::Physics(format!(
                        "fragment {} does not match this render GLB",
                        f.id()
                    )));
                }
            }
            if f.material_ids()
                .is_some_and(|ids| ids.iter().any(|id| id as usize >= material_count))
            {
                return Err(IoError::Physics(
                    "fragment material index out of range".into(),
                ));
            }
        }
    }
    if let Some(bs) = asset.bonds() {
        for b in bs {
            if b.composition().is_some_and(|cs| {
                cs.iter().any(|c| {
                    c.interface_material() >= 0
                        && ((c.interface_material() as usize) < material_count
                            || c.interface_material() as usize >= material_count + interface_count)
                })
            }) {
                return Err(IoError::Physics(
                    "bond interface material index out of range".into(),
                ));
            }
        }
    }
    let mut bin = bin.ok_or_else(|| IoError::Gltf("missing BIN chunk".into()))?;
    let old_len = buffers[0]["byteLength"]
        .as_u64()
        .ok_or_else(|| IoError::Gltf("invalid buffer length".into()))? as usize;
    if old_len > bin.len() {
        return Err(IoError::Gltf("truncated binary buffer".into()));
    }
    bin.truncate(old_len);
    // FlatBuffers contains u64 fields. Eight-byte alignment also satisfies glTF.
    bin.resize(bin.len().next_multiple_of(8), 0);
    let offset = bin.len();
    let final_len = offset
        .checked_add(physics.len())
        .ok_or_else(|| IoError::Gltf("asset too large".into()))?;
    if final_len > u32::MAX as usize - 1_048_576 {
        return Err(IoError::Gltf("asset exceeds GLB 32-bit size limit".into()));
    }
    bin.extend_from_slice(physics);
    let views = doc["bufferViews"]
        .as_array_mut()
        .ok_or_else(|| IoError::Gltf("missing bufferViews".into()))?;
    let view = views.len();
    views.push(json!({"buffer":0,"byteOffset":offset,"byteLength":physics.len(),"name":"Shardwright baked physics"}));
    doc["buffers"][0]["byteLength"] = json!(bin.len());
    if doc.get("extensions").is_none() {
        doc["extensions"] = json!({});
    }
    doc["extensions"][FRACTURE_EXTENSION] = json!({
        "version":1,"encoding":"flatbuffers","bufferView":view,
        "physicsSchemaVersion":asset.schema_version(),
        "coordinates":{"units":"meters","up":"Y","handedness":"right"},
        "materialLibrary":materials,
    });
    if doc.get("extensionsUsed").is_none() {
        doc["extensionsUsed"] = json!([]);
    }
    let used = doc["extensionsUsed"]
        .as_array_mut()
        .ok_or_else(|| IoError::Gltf("invalid extensionsUsed".into()))?;
    used.push(json!(FRACTURE_EXTENSION));
    let json = serde_json::to_vec(&doc)?;
    if 28u64 + json.len().next_multiple_of(4) as u64 + bin.len().next_multiple_of(4) as u64
        > u32::MAX as u64
    {
        return Err(IoError::Gltf("asset exceeds GLB 32-bit size limit".into()));
    }
    Ok(write_glb_container_owned(&json, Some(bin)))
}

/// Extract and validate the embedded physics bytes (useful to existing readers).
pub fn read_fracture_glb(glb: &[u8]) -> Result<(Vec<u8>, Value), IoError> {
    let (doc, bin) = read_glb_container(glb)?;
    if u32::from_le_bytes(glb[8..12].try_into().unwrap()) as usize != glb.len() {
        return Err(IoError::Gltf("GLB length mismatch".into()));
    }
    let ext = &doc["extensions"][FRACTURE_EXTENSION];
    if ext["version"] != 1
        || ext["encoding"] != "flatbuffers"
        || ext["physicsSchemaVersion"] != frac_core::SCHEMA_VERSION
    {
        return Err(IoError::Physics(
            "missing or unsupported SHARDWRIGHT_fracture extension".into(),
        ));
    }
    let index = ext["bufferView"]
        .as_u64()
        .ok_or_else(|| IoError::Physics("missing physics bufferView".into()))?
        as usize;
    let view = doc["bufferViews"]
        .get(index)
        .ok_or_else(|| IoError::Physics("invalid physics bufferView".into()))?;
    if view["buffer"] != 0 {
        return Err(IoError::Physics(
            "physics must be embedded in buffer 0".into(),
        ));
    }
    let start = view["byteOffset"].as_u64().unwrap_or(0) as usize;
    let size = view["byteLength"]
        .as_u64()
        .ok_or_else(|| IoError::Physics("missing physics byteLength".into()))?
        as usize;
    let end = start
        .checked_add(size)
        .ok_or_else(|| IoError::Physics("physics range overflow".into()))?;
    let bin = bin.ok_or_else(|| IoError::Gltf("missing BIN chunk".into()))?;
    let bytes = bin
        .get(start..end)
        .ok_or_else(|| IoError::Physics("physics range outside BIN".into()))?;
    if bytes.len() < 8 {
        return Err(IoError::Physics("truncated FRAC header".into()));
    }
    crate::read_physics(bytes)?;
    let asset = fb::root_as_asset(bytes).map_err(|e| IoError::Physics(e.to_string()))?;
    if asset.schema_version() != frac_core::SCHEMA_VERSION
        || ext["materialLibrary"]["library_id"].as_str() != asset.material_library_id()
        || ext["materialLibrary"]["library_version"].as_str() != asset.material_library_version()
    {
        return Err(IoError::Physics(
            "embedded schema or material library mismatch".into(),
        ));
    }
    Ok((bytes.to_vec(), ext["materialLibrary"].clone()))
}
