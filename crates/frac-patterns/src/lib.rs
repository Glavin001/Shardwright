//! Stage 11: runtime pattern library. For each material and failure mode a
//! canonical pattern of convex cells is generated in a unit frame, clipped
//! by the exact kernel (so interfaces are exact), and exported with internal
//! bond data so runtime stamping produces pre-bonded pieces.

#[allow(clippy::all, unused_imports, dead_code, non_snake_case, unsafe_op_in_unsafe_fn, mismatched_lifetime_syntaxes)]
#[path = "generated/fracpat_generated.rs"]
mod fracpat_generated;
pub use fracpat_generated::fracpat as fb;

use frac_cells::cellset::{CellSet, CellSetParams};
use frac_cells::clip::Clipper;
use frac_cells::complex::Complex;
use frac_core::determinism::{rng_for, RngStage};
use frac_geom::hull::ConvexPolytope;
use frac_geom::polygon::{area_integrals, plane_basis, AreaIntegrals};
use frac_geom::{DVec3, TriMesh};
use frac_material::MaterialLibrary;
use rand::Rng;
use std::collections::BTreeMap;

pub const MODES: [&str; 6] = ["punch_cone", "bending", "shear", "crater", "crush", "glass_radial"];

#[derive(Clone, Debug)]
pub struct PatternCell {
    pub hull: ConvexPolytope,
    pub volume: f64,
    pub com: DVec3,
    pub inertia: frac_geom::DMat3,
}

#[derive(Clone, Debug)]
pub struct PatternBond {
    pub a: u32,
    pub b: u32,
    pub area: f64,
    pub centroid: DVec3,
    pub normal: DVec3,
    pub frame_u: DVec3,
    pub frame_v: DVec3,
    pub i_uu: f64,
    pub i_vv: f64,
    pub i_uv: f64,
    pub j: f64,
}

#[derive(Clone, Debug)]
pub struct Pattern {
    pub material: String,
    pub mode: String,
    pub thickness: f64,
    pub cells: Vec<PatternCell>,
    pub bonds: Vec<PatternBond>,
    /// Total cell volume / frame volume - 1 (validation).
    pub volume_error: f64,
}

fn sample(rng: &mut rand_chacha::ChaCha8Rng, n: usize, lo: DVec3, hi: DVec3, density: impl Fn(DVec3) -> f64) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    let mut tries = 0;
    while out.len() < n && tries < n * 2000 {
        tries += 1;
        let p = DVec3::new(rng.gen_range(lo.x..hi.x), rng.gen_range(lo.y..hi.y), rng.gen_range(lo.z..hi.z));
        if rng.gen_range(0.0..1.0) < density(p).clamp(0.0, 1.0) {
            out.push(p.to_array());
        }
    }
    out
}

/// Build one pattern.
pub fn build_pattern(material: &str, mode: &str, glass: bool, seed: u64) -> Result<Pattern, String> {
    let mut rng = rng_for(seed, frac_core::stable_hash(&[material.len() as u64, mode.len() as u64]) as u32, RngStage::Patterns, 0, MODES.iter().position(|m| *m == mode).unwrap_or(0) as u32);
    let thickness = if glass || mode == "glass_radial" { 0.02 } else { 0.0 };
    let (lo, hi) = if thickness > 0.0 { (DVec3::new(-1.0, -1.0, -thickness), DVec3::new(1.0, 1.0, 0.0)) } else { (DVec3::new(-1.0, -1.0, -1.0), DVec3::new(1.0, 1.0, 0.0)) };
    let mid = -0.5 * thickness;
    let seeds: Vec<[f64; 3]> = match mode {
        "punch_cone" => sample(&mut rng, 140, lo, hi, |p| {
            let r = (p.x * p.x + p.y * p.y).sqrt();
            let cone = 0.25 + (-p.z) * 1.0; // 45° cone radius at depth
            libm::exp(-((r - cone).abs() / 0.12)) + 0.08
        }),
        "bending" => sample(&mut rng, 120, lo, hi, |p| libm::exp(-p.x.abs() / 0.15) + 0.05),
        "shear" => sample(&mut rng, 120, lo, hi, |p| libm::exp(-((p.x + p.z + 0.5) / std::f64::consts::SQRT_2).abs() / 0.15) + 0.05),
        "crater" => sample(&mut rng, 160, lo, hi, |p| 0.05 / (p.length() + 0.05).powi(2)),
        "crush" => sample(&mut rng, 150, lo, hi, |_| 1.0),
        "glass_radial" | _ => {
            let mut v = vec![[0.0, 0.0, mid]];
            let rings = 7;
            for k in 1..=rings {
                let r = 1.45 * (k as f64 / rings as f64).powf(1.4);
                let m = 10 + 2 * k;
                let ph = rng.gen_range(0.0..std::f64::consts::TAU);
                for j in 0..m {
                    let a = ph + std::f64::consts::TAU * (j as f64 + rng.gen_range(-0.2..0.2)) / m as f64;
                    let rr = r * rng.gen_range(0.94..1.06);
                    let p = [rr * libm::cos(a), rr * libm::sin(a), mid];
                    if p[0].abs() < 1.0 && p[1].abs() < 1.0 {
                        v.push(p);
                    }
                }
            }
            v
        }
    };
    let solid = frac_geom::mesh::box_mesh(lo, hi);
    let pad = DVec3::splat(0.05);
    let cx = Complex::voronoi(&seeds, (lo - pad).to_array(), (hi + pad).to_array())?;
    let out = Clipper::new(&solid, &cx).run()?;
    let params = CellSetParams { min_cell_volume: 1e-6 * (hi - lo).element_product(), min_thickness_ratio: 0.02, component_min_extent: (hi - lo).min_element() };
    let unit: Vec<u32> = (0..cx.cells.len() as u32).collect();
    let cs = CellSet::from_clip(out, &unit, &unit, &params);
    Ok(assemble(material, mode, thickness, &cs, (hi - lo).element_product()))
}

fn assemble(material: &str, mode: &str, thickness: f64, cs: &CellSet, frame_volume: f64) -> Pattern {
    let mut cells = Vec::new();
    let polys = cs.cell_polygons();
    for (ci, c) in cs.cells.iter().enumerate() {
        let pts: Vec<DVec3> = polys[ci].iter().flatten().map(|&v| cs.verts[v as usize]).collect();
        let hull = ConvexPolytope::from_points(&pts).unwrap_or_default();
        let mp = c.vi.mass_props(1.0);
        cells.push(PatternCell { hull, volume: mp.volume, com: mp.com, inertia: mp.inertia });
    }
    let mut acc: BTreeMap<(u32, u32), AreaIntegrals> = BTreeMap::new();
    for p in &cs.patches {
        let (a, b) = (p.cells[0], p.cells[1]);
        let loops: Vec<Vec<DVec3>> = p.loops.iter().map(|l| l.iter().map(|&v| cs.verts[v as usize]).collect()).collect();
        let mut ai = area_integrals(&loops, p.normal);
        let key = if a < b { (a, b) } else {
            ai.area_vec = -ai.area_vec;
            (b, a)
        };
        acc.entry(key).or_default().add(&ai);
    }
    let bonds = acc
        .into_iter()
        .map(|((a, b), ai)| {
            let n = ai.area_vec.normalize_or_zero();
            let c2 = ai.central_second();
            let (u0, v0) = plane_basis(n);
            let m = [[u0.dot(c2 * u0), u0.dot(c2 * v0)], [v0.dot(c2 * u0), v0.dot(c2 * v0)]];
            let th = 0.5 * libm::atan2(2.0 * m[0][1], m[0][0] - m[1][1]);
            let u = u0 * libm::cos(th) + v0 * libm::sin(th);
            let v = n.cross(u);
            let (iuu, ivv) = (v.dot(c2 * v), u.dot(c2 * u));
            PatternBond { a, b, area: ai.area, centroid: ai.centroid(), normal: n, frame_u: u, frame_v: v, i_uu: iuu, i_vv: ivv, i_uv: u.dot(c2 * v), j: iuu + ivv }
        })
        .collect();
    let total: f64 = cells.iter().map(|c| c.volume).sum();
    Pattern { material: material.into(), mode: mode.into(), thickness, cells, bonds, volume_error: (total / frame_volume - 1.0).abs() }
}

/// Build every (material, mode) pattern and serialize the library.
pub fn build_library(lib: &MaterialLibrary, seed: u64) -> Result<Vec<u8>, String> {
    let mut patterns = Vec::new();
    for m in &lib.materials {
        let glass = m.recipe.as_deref().map(|r| r.starts_with("glass")).unwrap_or(false);
        let modes: Vec<&str> = if glass { vec!["glass_radial"] } else { MODES.iter().copied().filter(|m| *m != "glass_radial").collect() };
        for mode in modes {
            patterns.push(build_pattern(&m.id, mode, glass, seed)?);
        }
    }
    Ok(serialize(&patterns, lib, seed))
}

fn v3(p: DVec3) -> fb::Vec3 {
    fb::Vec3::new(p.x as f32, p.y as f32, p.z as f32)
}

pub fn serialize(patterns: &[Pattern], lib: &MaterialLibrary, seed: u64) -> Vec<u8> {
    let mut b = flatbuffers::FlatBufferBuilder::new();
    let mut pats = Vec::new();
    for p in patterns {
        let mut cells = Vec::new();
        for c in &p.cells {
            let verts = c.hull.vertices();
            let index_of = |q: &DVec3| verts.iter().position(|v| v == q).unwrap() as u32;
            let mut fi = Vec::new();
            let mut fs = Vec::new();
            for (_, f) in &c.hull.faces {
                fs.push(f.len() as u16);
                fi.extend(f.iter().map(index_of));
            }
            let vv: Vec<fb::Vec3> = verts.iter().map(|v| v3(*v)).collect();
            let vertices = b.create_vector(&vv);
            let face_indices = b.create_vector(&fi);
            let face_sizes = b.create_vector(&fs);
            let it: Vec<f32> = (0..9).map(|k| c.inertia.col(k % 3)[k / 3] as f32).collect();
            let inertia = b.create_vector(&it);
            let com = v3(c.com);
            cells.push(fb::Cell::create(&mut b, &fb::CellArgs { vertices: Some(vertices), face_indices: Some(face_indices), face_sizes: Some(face_sizes), volume: c.volume as f32, com: Some(&com), inertia: Some(inertia) }));
        }
        let cells = b.create_vector(&cells);
        let mut bonds = Vec::new();
        for bd in &p.bonds {
            let (c, n, u, v) = (v3(bd.centroid), v3(bd.normal), v3(bd.frame_u), v3(bd.frame_v));
            bonds.push(fb::Bond::create(&mut b, &fb::BondArgs {
                a: bd.a,
                b: bd.b,
                area: bd.area as f32,
                centroid: Some(&c),
                normal: Some(&n),
                frame_u: Some(&u),
                frame_v: Some(&v),
                i_uu: bd.i_uu as f32,
                i_vv: bd.i_vv as f32,
                i_uv: bd.i_uv as f32,
                j: bd.j as f32,
                kind: 0,
                interface_material: -1,
            }));
        }
        let bonds = b.create_vector(&bonds);
        let material = b.create_string(&p.material);
        let mode = b.create_string(&p.mode);
        pats.push(fb::Pattern::create(&mut b, &fb::PatternArgs { material: Some(material), mode: Some(mode), thickness: p.thickness as f32, cells: Some(cells), bonds: Some(bonds) }));
    }
    let pats = b.create_vector(&pats);
    let tv = b.create_string(frac_core::TOOL_VERSION);
    let lid = b.create_string(&lib.library_id);
    let lv = b.create_string(&lib.library_version);
    let root = fb::PatternLibrary::create(&mut b, &fb::PatternLibraryArgs { schema_version: 1, tool_version: Some(tv), material_library_id: Some(lid), material_library_version: Some(lv), seed, patterns: Some(pats) });
    fb::finish_pattern_library_buffer(&mut b, root);
    b.finished_data().to_vec()
}

/// Read back a pattern library (verifies the buffer).
pub fn read_library(bytes: &[u8]) -> Result<Vec<(String, String, usize, usize)>, String> {
    let lib = fb::root_as_pattern_library(bytes).map_err(|e| e.to_string())?;
    Ok(lib
        .patterns()
        .map(|ps| ps.iter().map(|p| (p.material().unwrap_or("").to_string(), p.mode().unwrap_or("").to_string(), p.cells().map(|c| c.len()).unwrap_or(0), p.bonds().map(|b| b.len()).unwrap_or(0))).collect())
        .unwrap_or_default())
}

/// Unused helper kept for API symmetry.
pub fn pattern_mesh(p: &Pattern) -> TriMesh {
    let mut m = TriMesh::default();
    for c in &p.cells {
        m.append(&c.hull.to_mesh());
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn patterns_validate() {
        for mode in MODES {
            let p = build_pattern("concrete_c30", mode, false, 7).unwrap();
            assert!(p.volume_error < 1e-9, "{mode}: volume error {}", p.volume_error);
            assert!(p.cells.len() > 20, "{mode}: {} cells", p.cells.len());
            // convex cells: hull volume equals cell volume
            for c in &p.cells {
                assert!((c.hull.volume() / c.volume - 1.0).abs() < 1e-6, "{mode}: non-convex cell");
            }
            // each bond's cells are distinct; area positive
            assert!(p.bonds.iter().all(|b| b.a != b.b && b.area > 0.0));
        }
        let lib = MaterialLibrary::builtin();
        let bytes = build_library(&lib, 1).unwrap();
        let back = read_library(&bytes).unwrap();
        assert!(back.len() >= lib.materials.len());
        assert_eq!(bytes, build_library(&lib, 1).unwrap());
    }
}
