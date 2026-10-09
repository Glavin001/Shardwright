//! Minimal PLY reader: ASCII and binary (little/big endian).
//!
//! Vertex properties: `x y z` (required), optional `nx ny nz`, texture
//! coordinates `s t` / `u v` / `texture_u texture_v` (converted to the glTF
//! convention `v' = 1 - v`), and a scalar `paint` or `density` property that
//! becomes `vertex_paint`. Faces: list property `vertex_indices` (or
//! `vertex_index`); polygons are fan-triangulated. A face `material_index`
//! (or `material`) scalar property, when present, selects the material slot
//! `material_<k>`. Other elements and properties are skipped.

use super::{Corner, PartBuilder, dir_f32};
use crate::{ImportOptions, IoError};
use frac_core::input::InputPart;
use glam::DVec3;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Ty {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Ty {
    fn parse(s: &str) -> Result<Ty, IoError> {
        Ok(match s {
            "char" | "int8" => Ty::I8,
            "uchar" | "uint8" => Ty::U8,
            "short" | "int16" => Ty::I16,
            "ushort" | "uint16" => Ty::U16,
            "int" | "int32" => Ty::I32,
            "uint" | "uint32" => Ty::U32,
            "float" | "float32" => Ty::F32,
            "double" | "float64" => Ty::F64,
            _ => return Err(IoError::Ply(format!("unknown property type '{s}'"))),
        })
    }
    fn size(self) -> usize {
        match self {
            Ty::I8 | Ty::U8 => 1,
            Ty::I16 | Ty::U16 => 2,
            Ty::I32 | Ty::U32 | Ty::F32 => 4,
            Ty::F64 => 8,
        }
    }
}

#[derive(Debug)]
enum Prop {
    Scalar { name: String, ty: Ty },
    List { name: String, count: Ty, item: Ty },
}

impl Prop {
    fn name(&self) -> &str {
        match self {
            Prop::Scalar { name, .. } | Prop::List { name, .. } => name,
        }
    }
}

#[derive(Debug)]
struct Element {
    name: String,
    count: usize,
    props: Vec<Prop>,
}

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Ascii,
    Le,
    Be,
}

enum Body<'a> {
    Ascii(std::str::SplitAsciiWhitespace<'a>),
    Bin {
        data: &'a [u8],
        pos: usize,
        big: bool,
    },
}

impl Body<'_> {
    fn read(&mut self, ty: Ty) -> Result<f64, IoError> {
        match self {
            Body::Ascii(it) => {
                let tok = it
                    .next()
                    .ok_or_else(|| IoError::Ply("unexpected end of ASCII data".into()))?;
                tok.parse::<f64>()
                    .map_err(|_| IoError::Ply(format!("invalid number '{tok}'")))
            }
            Body::Bin { data, pos, big } => {
                let n = ty.size();
                let s = data
                    .get(*pos..*pos + n)
                    .ok_or_else(|| IoError::Ply("unexpected end of binary data".into()))?;
                *pos += n;
                let mut b = [0u8; 8];
                b[..n].copy_from_slice(s);
                if *big {
                    b[..n].reverse();
                }
                Ok(match ty {
                    Ty::I8 => b[0] as i8 as f64,
                    Ty::U8 => b[0] as f64,
                    Ty::I16 => i16::from_le_bytes([b[0], b[1]]) as f64,
                    Ty::U16 => u16::from_le_bytes([b[0], b[1]]) as f64,
                    Ty::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
                    Ty::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
                    Ty::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
                    Ty::F64 => f64::from_le_bytes(b),
                })
            }
        }
    }
}

fn parse_header(bytes: &[u8]) -> Result<(Format, Vec<Element>, usize), IoError> {
    if !bytes.starts_with(b"ply") {
        return Err(IoError::Ply("missing 'ply' magic".into()));
    }
    let mut pos = 0;
    let mut format = None;
    let mut elements: Vec<Element> = Vec::new();
    loop {
        let end = bytes[pos..]
            .iter()
            .position(|&c| c == b'\n')
            .map(|e| pos + e)
            .ok_or_else(|| IoError::Ply("unterminated header".into()))?;
        let line = std::str::from_utf8(&bytes[pos..end])
            .map_err(|_| IoError::Ply("non-UTF-8 header".into()))?;
        pos = end + 1;
        let toks: Vec<&str> = line.split_ascii_whitespace().collect();
        match toks.first().copied() {
            Some("ply") | Some("comment") | Some("obj_info") | None => {}
            Some("format") => {
                format = Some(match toks.get(1).copied() {
                    Some("ascii") => Format::Ascii,
                    Some("binary_little_endian") => Format::Le,
                    Some("binary_big_endian") => Format::Be,
                    f => return Err(IoError::Ply(format!("unknown format {f:?}"))),
                })
            }
            Some("element") => {
                let (Some(name), Some(count)) = (toks.get(1), toks.get(2)) else {
                    return Err(IoError::Ply(format!("bad element line '{line}'")));
                };
                let count = count
                    .parse()
                    .map_err(|_| IoError::Ply(format!("bad element count '{count}'")))?;
                elements.push(Element {
                    name: name.to_string(),
                    count,
                    props: Vec::new(),
                });
            }
            Some("property") => {
                let el = elements
                    .last_mut()
                    .ok_or_else(|| IoError::Ply("property before element".into()))?;
                let prop = if toks.get(1) == Some(&"list") {
                    if toks.len() < 5 {
                        return Err(IoError::Ply(format!("bad list property '{line}'")));
                    }
                    Prop::List {
                        name: toks[4].to_string(),
                        count: Ty::parse(toks[2])?,
                        item: Ty::parse(toks[3])?,
                    }
                } else {
                    if toks.len() < 3 {
                        return Err(IoError::Ply(format!("bad property '{line}'")));
                    }
                    Prop::Scalar {
                        name: toks[2].to_string(),
                        ty: Ty::parse(toks[1])?,
                    }
                };
                el.props.push(prop);
            }
            Some("end_header") => break,
            Some(other) => return Err(IoError::Ply(format!("unknown header keyword '{other}'"))),
        }
    }
    Ok((
        format.ok_or_else(|| IoError::Ply("missing format line".into()))?,
        elements,
        pos,
    ))
}

pub(super) fn load(
    bytes: &[u8],
    stem: &str,
    opts: &ImportOptions,
) -> Result<Vec<InputPart>, IoError> {
    let (format, elements, body_start) = parse_header(bytes)?;
    let body = &bytes[body_start..];
    let mut rd = match format {
        Format::Ascii => Body::Ascii(
            std::str::from_utf8(body)
                .map_err(|_| IoError::Ply("non-UTF-8 ASCII body".into()))?
                .split_ascii_whitespace(),
        ),
        Format::Le | Format::Be => Body::Bin {
            data: body,
            pos: 0,
            big: format == Format::Be,
        },
    };

    let mut pos: Vec<DVec3> = Vec::new();
    let mut nrm: Vec<[f64; 3]> = Vec::new();
    let mut uv: Vec<[f64; 2]> = Vec::new();
    let mut paint: Vec<f32> = Vec::new();
    let (mut has_n, mut has_uv, mut has_paint) = (false, false, false);
    let mut faces: Vec<(Vec<u32>, u16)> = Vec::new();

    for el in &elements {
        let find = |names: &[&str]| {
            el.props
                .iter()
                .position(|p| matches!(p, Prop::Scalar { .. }) && names.contains(&p.name()))
        };
        let is_vertex = el.name == "vertex";
        let is_face = el.name == "face";
        let (ix, iy, iz) = (find(&["x"]), find(&["y"]), find(&["z"]));
        let (inx, iny, inz) = (find(&["nx"]), find(&["ny"]), find(&["nz"]));
        let iu = find(&["s", "u", "texture_u", "texture_s"]);
        let iv = find(&["t", "v", "texture_v", "texture_t"]);
        let ip = find(&["paint", "density"]);
        let imat = find(&["material_index", "material"]);
        let ilist = el.props.iter().position(|p| {
            matches!(p, Prop::List { .. })
                && (p.name() == "vertex_indices" || p.name() == "vertex_index")
        });
        if is_vertex {
            if ix.is_none() || iy.is_none() || iz.is_none() {
                return Err(IoError::Ply("vertex element lacks x/y/z".into()));
            }
            has_n = inx.is_some() && iny.is_some() && inz.is_some();
            has_uv = iu.is_some() && iv.is_some();
            has_paint = ip.is_some();
        }
        if is_face && ilist.is_none() {
            return Err(IoError::Ply("face element lacks vertex_indices".into()));
        }
        let mut vals = vec![0.0f64; el.props.len()];
        for _ in 0..el.count {
            let mut list: Vec<u32> = Vec::new();
            for (k, p) in el.props.iter().enumerate() {
                match p {
                    Prop::Scalar { ty, .. } => vals[k] = rd.read(*ty)?,
                    Prop::List { count, item, .. } => {
                        let n = rd.read(*count)?;
                        if !(n >= 0.0 && n.fract() == 0.0) {
                            return Err(IoError::Ply(format!("invalid list length {n}")));
                        }
                        let keep = is_face && Some(k) == ilist;
                        for _ in 0..n as usize {
                            let v = rd.read(*item)?;
                            if keep {
                                if !(v >= 0.0 && v.fract() == 0.0 && v <= u32::MAX as f64) {
                                    return Err(IoError::Ply(format!("invalid vertex index {v}")));
                                }
                                list.push(v as u32);
                            }
                        }
                    }
                }
            }
            if is_vertex {
                let g = |i: Option<usize>| i.map(|i| vals[i]).unwrap_or(0.0);
                pos.push(DVec3::new(g(ix), g(iy), g(iz)));
                if has_n {
                    nrm.push([g(inx), g(iny), g(inz)]);
                }
                if has_uv {
                    uv.push([g(iu), g(iv)]);
                }
                if has_paint {
                    paint.push(g(ip) as f32);
                }
            } else if is_face {
                let slot = imat
                    .map(|i| vals[i].clamp(0.0, u16::MAX as f64) as u16)
                    .unwrap_or(0);
                faces.push((list, slot));
            }
        }
    }

    let frame = opts.frame();
    let rot = opts.rotation();
    let mut pb = PartBuilder::new(stem);
    let has_mat = faces.iter().any(|f| f.1 != 0);
    // Slot names in increasing material index order (deterministic).
    let mut mat_ids: Vec<u16> = faces.iter().map(|f| f.1).collect();
    mat_ids.sort_unstable();
    mat_ids.dedup();
    let mut slot_of = std::collections::BTreeMap::new();
    for &m in &mat_ids {
        let name = if has_mat {
            format!("material_{m}")
        } else {
            "default".to_string()
        };
        slot_of.insert(m, pb.slot(&name));
    }
    let corner = |i: u32| -> Result<Corner, IoError> {
        let i = i as usize;
        let p = *pos.get(i).ok_or_else(|| {
            IoError::Ply(format!(
                "face index {i} out of range ({} vertices)",
                pos.len()
            ))
        })?;
        Ok(Corner {
            p: frame * p,
            n: has_n.then(|| dir_f32(rot * DVec3::from_array(nrm[i]))),
            uv: has_uv.then(|| [uv[i][0] as f32, (1.0 - uv[i][1]) as f32]),
            t: None,
            paint: has_paint.then(|| paint[i]),
        })
    };
    for (f, m) in &faces {
        let slot = slot_of[m];
        for k in 1..f.len().saturating_sub(1) {
            pb.add_tri([corner(f[0])?, corner(f[k])?, corner(f[k + 1])?], slot);
        }
    }
    if pb.is_empty() {
        return Err(IoError::Ply("no faces".into()));
    }
    Ok(vec![pb.finish()])
}
