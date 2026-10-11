//! Minimal GLB container read/write (glTF 2.0 binary).

use crate::IoError;

const MAGIC: u32 = 0x4654_6C67; // "glTF"
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;

/// Assemble a GLB from JSON text and an optional BIN chunk payload. The JSON
/// chunk is padded with spaces, the BIN chunk with zeros (both to 4 bytes).
pub(crate) fn write_glb_container(json: &[u8], bin: Option<&[u8]>) -> Vec<u8> {
    let json_len = json.len().next_multiple_of(4);
    let bin_len = bin.map(|b| b.len().next_multiple_of(4));
    let total = 12 + 8 + json_len + bin_len.map(|l| 8 + l).unwrap_or(0);
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json_len as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_JSON.to_le_bytes());
    out.extend_from_slice(json);
    out.resize(20 + json_len, b' ');
    if let (Some(b), Some(l)) = (bin, bin_len) {
        out.extend_from_slice(&(l as u32).to_le_bytes());
        out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
        let start = out.len();
        out.extend_from_slice(b);
        out.resize(start + l, 0);
    }
    debug_assert_eq!(out.len(), total);
    out
}

/// [`write_glb_container`] that takes the BIN chunk by value and builds the
/// file in place (header and JSON are inserted in front of it), avoiding a
/// fresh multi-GB allocation and copy for building-scale scenes.
pub(crate) fn write_glb_container_owned(json: &[u8], bin: Option<Vec<u8>>) -> Vec<u8> {
    let Some(mut b) = bin else {
        return write_glb_container(json, None);
    };
    let json_len = json.len().next_multiple_of(4);
    let bin_len = b.len().next_multiple_of(4);
    let total = 12 + 8 + json_len + 8 + bin_len;
    b.resize(bin_len, 0);
    let mut head = Vec::with_capacity(12 + 8 + json_len + 8);
    head.extend_from_slice(&MAGIC.to_le_bytes());
    head.extend_from_slice(&2u32.to_le_bytes());
    head.extend_from_slice(&(total as u32).to_le_bytes());
    head.extend_from_slice(&(json_len as u32).to_le_bytes());
    head.extend_from_slice(&CHUNK_JSON.to_le_bytes());
    head.extend_from_slice(json);
    head.resize(20 + json_len, b' ');
    head.extend_from_slice(&(bin_len as u32).to_le_bytes());
    head.extend_from_slice(&CHUNK_BIN.to_le_bytes());
    b.splice(0..0, head);
    debug_assert_eq!(b.len(), total);
    b
}

/// Split a GLB into (JSON value, BIN chunk).
pub(crate) fn read_glb_container(
    bytes: &[u8],
) -> Result<(serde_json::Value, Option<Vec<u8>>), IoError> {
    let rd = |o: usize| -> Result<u32, IoError> {
        bytes
            .get(o..o + 4)
            .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
            .ok_or_else(|| IoError::Gltf("truncated GLB".into()))
    };
    if rd(0)? != MAGIC {
        return Err(IoError::Gltf("not a GLB file (bad magic)".into()));
    }
    if rd(4)? != 2 {
        return Err(IoError::Gltf(format!("unsupported GLB version {}", rd(4)?)));
    }
    let total = (rd(8)? as usize).min(bytes.len());
    let mut off = 12;
    let mut json = None;
    let mut bin = None;
    while off + 8 <= total {
        let len = rd(off)? as usize;
        let ty = rd(off + 4)?;
        let data = bytes
            .get(off + 8..off + 8 + len)
            .ok_or_else(|| IoError::Gltf("truncated GLB chunk".into()))?;
        match ty {
            CHUNK_JSON if json.is_none() => json = Some(serde_json::from_slice(data)?),
            CHUNK_BIN if bin.is_none() => bin = Some(data.to_vec()),
            _ => {}
        }
        off += 8 + len.next_multiple_of(4);
    }
    let json = json.ok_or_else(|| IoError::Gltf("GLB has no JSON chunk".into()))?;
    Ok((json, bin))
}
