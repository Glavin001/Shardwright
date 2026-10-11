//! Wavefront OBJ import (via `tobj`, triangulated, f64 positions).
//!
//! One part per object/group name (`o`/`g`; models of the same name are
//! merged, in order of first appearance). Material slots are named after the
//! `usemtl` names; `.mtl` files are not needed (only names are used), so the
//! material library is synthesized from the `usemtl` statements.
//! Normals (texture coordinates) are imported only when every face corner in
//! the file specifies them; mixed files drop that attribute (tobj would
//! otherwise fabricate values for the missing corners).

use super::{Corner, PartBuilder, dir_f32};
use crate::{ImportOptions, IoError};
use frac_core::input::InputPart;
use glam::DVec3;

const SYNTH_MTL: &str = "__frac_io_synthesized__.mtl";

pub(super) fn load(
    bytes: &[u8],
    stem: &str,
    opts: &ImportOptions,
) -> Result<Vec<InputPart>, IoError> {
    let text = String::from_utf8_lossy(bytes);
    // Collect usemtl names (first appearance order) for the synthetic library.
    let mut names: Vec<String> = Vec::new();
    for line in text.lines() {
        let l = line.trim_start();
        if let Some(rest) = l.strip_prefix("usemtl")
            && rest.starts_with(char::is_whitespace)
        {
            let n = rest.trim().to_string();
            if !n.is_empty() && !names.contains(&n) {
                names.push(n);
            }
        }
    }
    // tobj fabricates normals/texcoords for face corners that lack them when
    // other faces have them; if the file mixes faces with and without `vn`
    // (or `vt`), that attribute is dropped for the whole file instead.
    let (mut n_with, mut n_without, mut t_with, mut t_without) = (0usize, 0usize, 0usize, 0usize);
    for line in text.lines() {
        let mut toks = line.split_ascii_whitespace();
        if toks.next() != Some("f") {
            continue;
        }
        for tok in toks {
            let mut it = tok.split('/');
            let _v = it.next();
            let vt = it.next().unwrap_or("");
            let vn = it.next().unwrap_or("");
            if vt.is_empty() {
                t_without += 1
            } else {
                t_with += 1
            }
            if vn.is_empty() {
                n_without += 1
            } else {
                n_with += 1
            }
        }
    }
    let use_normals = n_with > 0 && n_without == 0;
    let use_uvs = t_with > 0 && t_without == 0;
    let synth: String = names.iter().map(|n| format!("newmtl {n}\n")).collect();
    let src = format!("mtllib {SYNTH_MTL}\n{text}");
    let load_opts = tobj::LoadOptions {
        triangulate: true,
        single_index: false,
        ..Default::default()
    };
    let (models, materials) = tobj::load_obj_buf(&mut src.as_bytes(), &load_opts, |p| {
        if p.to_str() == Some(SYNTH_MTL) {
            tobj::load_mtl_buf(&mut synth.as_bytes())
        } else {
            tobj::load_mtl_buf(&mut "".as_bytes())
        }
    })
    .map_err(|e| IoError::Obj(e.to_string()))?;
    let materials = materials.map_err(|e| IoError::Obj(e.to_string()))?;

    let frame = opts.frame();
    let rot = opts.rotation();
    let mut parts: Vec<PartBuilder> = Vec::new();
    for model in &models {
        let name = if model.name.is_empty() || model.name == "unnamed_object" {
            stem.to_string()
        } else {
            model.name.clone()
        };
        let pi = match parts.iter().position(|p| p.name() == name) {
            Some(i) => i,
            None => {
                parts.push(PartBuilder::new(name));
                parts.len() - 1
            }
        };
        let pb = &mut parts[pi];
        let m = &model.mesh;
        let mat_name = match m.material_id {
            Some(id) => materials
                .get(id)
                .map(|mm| mm.name.clone())
                .unwrap_or_else(|| format!("material_{id}")),
            None => "default".to_string(),
        };
        let slot = pb.slot(&mat_name);
        let has_n = use_normals && !m.normal_indices.is_empty() && !m.normals.is_empty();
        let has_t = use_uvs && !m.texcoord_indices.is_empty() && !m.texcoords.is_empty();
        let corner = |k: usize| -> Result<Corner, IoError> {
            let vi = m.indices[k] as usize;
            let p = m
                .positions
                .get(3 * vi..3 * vi + 3)
                .ok_or_else(|| IoError::Obj(format!("vertex {vi} out of range")))?;
            let n = if has_n {
                let ni = m.normal_indices[k] as usize;
                m.normals
                    .get(3 * ni..3 * ni + 3)
                    .map(|n| dir_f32(rot * DVec3::new(n[0], n[1], n[2])))
            } else {
                None
            };
            let uv = if has_t {
                let ti = m.texcoord_indices[k] as usize;
                m.texcoords
                    .get(2 * ti..2 * ti + 2)
                    .map(|t| [t[0] as f32, (1.0 - t[1]) as f32])
            } else {
                None
            };
            Ok(Corner {
                p: frame * DVec3::new(p[0], p[1], p[2]),
                n,
                uv,
                t: None,
                paint: None,
            })
        };
        for k in (0..m.indices.len() / 3).map(|t| 3 * t) {
            pb.add_tri([corner(k)?, corner(k + 1)?, corner(k + 2)?], slot);
        }
    }
    Ok(parts
        .into_iter()
        .filter(|p| !p.is_empty())
        .map(PartBuilder::finish)
        .collect())
}
