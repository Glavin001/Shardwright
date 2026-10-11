//! Minimal STL reader (ASCII and binary). Exact-identical positions are
//! merged. Facet normals are ignored (often unreliable); ingest recomputes
//! normals. ASCII files with several `solid` blocks yield one part each.

use super::{Corner, PartBuilder};
use crate::{ImportOptions, IoError};
use frac_core::input::InputPart;
use glam::DVec3;

pub(super) fn load(
    bytes: &[u8],
    stem: &str,
    opts: &ImportOptions,
) -> Result<Vec<InputPart>, IoError> {
    let frame = opts.frame();
    // Binary if the size matches the triangle count exactly (binary headers
    // may also start with "solid").
    let is_binary = bytes.len() >= 84 && {
        let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        n.checked_mul(50).and_then(|s| s.checked_add(84)) == Some(bytes.len())
    };
    if is_binary || !bytes.trim_ascii_start().starts_with(b"solid") {
        return load_binary(bytes, stem, frame);
    }
    load_ascii(bytes, stem, frame)
}

fn load_binary(bytes: &[u8], stem: &str, frame: glam::DMat3) -> Result<Vec<InputPart>, IoError> {
    if bytes.len() < 84 {
        return Err(IoError::Stl("file too short for binary STL".into()));
    }
    let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    if bytes.len() < 84 + 50 * n {
        return Err(IoError::Stl(format!(
            "binary STL truncated: {n} triangles need {} bytes, have {}",
            84 + 50 * n,
            bytes.len()
        )));
    }
    let f =
        |o: usize| f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as f64;
    let mut pb = PartBuilder::new(stem);
    let slot = pb.slot("default");
    for t in 0..n {
        let base = 84 + 50 * t + 12; // skip the facet normal
        let c = |k: usize| {
            let o = base + 12 * k;
            Corner {
                p: frame * DVec3::new(f(o), f(o + 4), f(o + 8)),
                ..Default::default()
            }
        };
        pb.add_tri([c(0), c(1), c(2)], slot);
    }
    if pb.is_empty() {
        return Err(IoError::Stl("no triangles".into()));
    }
    Ok(vec![pb.finish()])
}

fn load_ascii(bytes: &[u8], stem: &str, frame: glam::DMat3) -> Result<Vec<InputPart>, IoError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| IoError::Stl("ASCII STL is not valid UTF-8".into()))?;
    let mut parts = Vec::new();
    let mut cur: Option<PartBuilder> = None;
    let mut verts: Vec<DVec3> = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let mut toks = line.split_ascii_whitespace();
        match toks.next() {
            Some("solid") => {
                if let Some(pb) = cur.take()
                    && !pb.is_empty()
                {
                    parts.push(pb.finish());
                }
                let name = line.trim_start()["solid".len()..].trim();
                let name = if name.is_empty() {
                    format!("{stem}_{}", parts.len())
                } else {
                    name.to_string()
                };
                cur = Some(PartBuilder::new(name));
            }
            Some("vertex") => {
                let mut v = [0.0f64; 3];
                for x in &mut v {
                    let t = toks
                        .next()
                        .ok_or_else(|| IoError::Stl(format!("line {}: short vertex", ln + 1)))?;
                    *x = t
                        .parse()
                        .map_err(|_| IoError::Stl(format!("line {}: bad number '{t}'", ln + 1)))?;
                }
                verts.push(DVec3::from_array(v));
            }
            Some("endloop") => {
                let pb = cur.get_or_insert_with(|| PartBuilder::new(stem));
                let slot = pb.slot("default");
                // Fan-triangulate (loops are triangles in practice).
                for k in 1..verts.len().saturating_sub(1) {
                    let c = |p: DVec3| Corner {
                        p: frame * p,
                        ..Default::default()
                    };
                    pb.add_tri([c(verts[0]), c(verts[k]), c(verts[k + 1])], slot);
                }
                verts.clear();
            }
            Some("endsolid") => {
                if let Some(pb) = cur.take()
                    && !pb.is_empty()
                {
                    parts.push(pb.finish());
                }
            }
            _ => {}
        }
    }
    if let Some(pb) = cur.take()
        && !pb.is_empty()
    {
        parts.push(pb.finish());
    }
    if parts.is_empty() {
        return Err(IoError::Stl("no triangles".into()));
    }
    // If there is a single solid with the default name, use the file stem.
    if parts.len() == 1 && parts[0].name == format!("{stem}_0") {
        parts[0].name = stem.to_string();
    }
    Ok(parts)
}
