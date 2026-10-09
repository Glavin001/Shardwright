//! Stage 0 (ingest & repair) and Stage 1 (components & contacts).

pub mod solidify;

use frac_core::input::{AuthoringMeta, InputPart, InputScene, PartMeta};
use frac_core::{ComponentRole, MaterialId, SurfaceAttributes};
use frac_geom::inside::MeshQuery;
use frac_geom::{Aabb, DVec3, TriMesh};
use frac_material::MaterialLibrary;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct IngestSettings {
    pub weld_tolerance: f64,
    pub solidify_resolution: u32,
    pub contact_tolerance: f64,
}

#[derive(Clone, Debug)]
pub struct IngestedPart {
    pub name: String,
    pub meta: PartMeta,
    pub material: MaterialId,
    pub role: ComponentRole,
    pub solid: TriMesh,
    /// Attributes per solid triangle corner (when the solid is the input).
    pub surface: SurfaceAttributes,
    pub reconstructed: bool,
    /// Original render surface and attributes for reconstructed parts.
    pub render_surface: Option<(TriMesh, SurfaceAttributes)>,
    pub volume: f64,
    pub aabb: Aabb,
    pub failed: Option<String>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct IngestOut {
    pub parts: Vec<IngestedPart>,
    /// Pairs of parts whose solids come within the contact tolerance.
    pub contacts: Vec<(usize, usize)>,
    pub warnings: Vec<String>,
}

/// Resolve a part's library material.
pub fn resolve_material(part: &InputPart, meta: &PartMeta, lib: &MaterialLibrary) -> Result<MaterialId, String> {
    if let Some(m) = &meta.material {
        return lib.material_id(m).ok_or_else(|| format!("part '{}': unknown material '{m}'", part.name));
    }
    // dominant source slot through the mapping table, or a slot named like a library material
    let mut counts = vec![0usize; part.material_names.len().max(1)];
    for &s in &part.material_slot {
        if (s as usize) < counts.len() {
            counts[s as usize] += 1;
        }
    }
    let mut order: Vec<usize> = (0..counts.len()).collect();
    order.sort_by(|&a, &b| counts[b].cmp(&counts[a]).then(a.cmp(&b)));
    for i in order {
        if let Some(name) = part.material_names.get(i) {
            if let Some(mapped) = meta.material_map.get(name) {
                if let Some(id) = lib.material_id(mapped) {
                    return Ok(id);
                }
            }
            if let Some(id) = lib.material_id(name) {
                return Ok(id);
            }
        }
    }
    Err(format!("part '{}': cannot resolve a library material (set `material` in metadata)", part.name))
}

/// Stage 0: weld, validate, and solidify every part.
pub fn ingest(scene: &InputScene, meta: &AuthoringMeta, lib: &MaterialLibrary, s: &IngestSettings) -> IngestOut {
    let parts: Vec<IngestedPart> = scene
        .parts
        .par_iter()
        .map(|p| {
            let pm = meta.for_part(&p.name, &p.extras);
            let role = pm.component_role.as_deref().and_then(ComponentRole::from_name).unwrap_or_default();
            let material = match resolve_material(p, &pm, lib) {
                Ok(m) => m,
                Err(e) => {
                    return IngestedPart {
                        name: p.name.clone(),
                        meta: pm,
                        material: MaterialId(0),
                        role,
                        solid: TriMesh::default(),
                        surface: SurfaceAttributes::default(),
                        reconstructed: false,
                        render_surface: None,
                        volume: 0.0,
                        aabb: Aabb::EMPTY,
                        failed: Some(e),
                        warnings: Vec::new(),
                    };
                }
            };
            repair_part(p, pm, material, role, s)
        })
        .collect();
    // contacts between solids (AABB prefilter + closest distance)
    let mut contacts = Vec::new();
    for i in 0..parts.len() {
        for j in i + 1..parts.len() {
            if parts[i].failed.is_some() || parts[j].failed.is_some() {
                continue;
            }
            if !parts[i].aabb.expanded(s.contact_tolerance).overlaps(&parts[j].aabb) {
                continue;
            }
            if solids_within(&parts[i].solid, &parts[j].solid, s.contact_tolerance) {
                contacts.push((i, j));
            }
        }
    }
    let mut warnings = Vec::new();
    for p in &parts {
        if let Some(f) = &p.failed {
            warnings.push(format!("part '{}' passed through unfractured: {f}", p.name));
        }
        for w in &p.warnings {
            warnings.push(format!("part '{}': {w}", p.name));
        }
    }
    IngestOut { parts, contacts, warnings }
}

fn corner_attrs(p: &InputPart, keep: &[usize]) -> SurfaceAttributes {
    let normals = match &p.normals {
        Some(n) => keep.iter().map(|&t| n[t]).collect(),
        None => Vec::new(),
    };
    SurfaceAttributes {
        normals,
        uvs: p.uvs.as_ref().map(|u| keep.iter().map(|&t| u[t]).collect()),
        tangents: p.tangents.as_ref().map(|u| keep.iter().map(|&t| u[t]).collect()),
        material_slot: Some(keep.iter().map(|&t| p.material_slot.get(t).copied().unwrap_or(0)).collect()),
    }
}

/// Weld + validate; reconstruct with the generalized winding number when the
/// input is not a clean closed manifold.
pub fn repair_part(p: &InputPart, meta: PartMeta, material: MaterialId, role: ComponentRole, s: &IngestSettings) -> IngestedPart {
    let mut warnings = Vec::new();
    let diag = p.mesh.aabb().diagonal().max(1e-12);
    // weld positions, keep per-triangle mapping to input triangles
    let welded = weld_keep_map(&p.mesh, s.weld_tolerance * diag);
    let (mesh, keep) = welded;
    let topo = mesh.topology();
    let mut clean = topo.is_closed_manifold();
    if clean {
        let si = mesh.self_intersections(1);
        if !si.is_empty() {
            clean = false;
            warnings.push("self-intersecting input".into());
        }
    } else {
        warnings.push(format!("non-manifold or open input: {topo:?}"));
    }
    let base = IngestedPart {
        name: p.name.clone(),
        meta,
        material,
        role,
        solid: TriMesh::default(),
        surface: SurfaceAttributes::default(),
        reconstructed: false,
        render_surface: None,
        volume: 0.0,
        aabb: Aabb::EMPTY,
        failed: None,
        warnings: Vec::new(),
    };
    if clean {
        let mut solid = mesh;
        let mut surface = corner_attrs(p, &keep);
        let mut vol = solid.signed_volume();
        if vol < 0.0 {
            solid = solid.flipped();
            vol = -vol;
            // flip corner order to match
            flip_corners(&mut surface);
            warnings.push("inverted orientation fixed".into());
        }
        if vol <= 0.0 {
            return IngestedPart { failed: Some("zero volume".into()), warnings, ..base };
        }
        let aabb = solid.aabb();
        return IngestedPart { solid, surface, volume: vol, aabb, warnings, ..base };
    }
    match solidify::solidify(&p.mesh, s.solidify_resolution) {
        Ok(solid) => {
            let vol = solid.signed_volume();
            let aabb = solid.aabb();
            let all: Vec<usize> = (0..p.mesh.tris.len()).collect();
            let rs = (p.mesh.clone(), corner_attrs(p, &all));
            warnings.push(format!("reconstructed solid ({} tris) from non-watertight input", solid.tris.len()));
            IngestedPart { solid, reconstructed: true, render_surface: Some(rs), volume: vol, aabb, warnings, ..base }
        }
        Err(e) => IngestedPart { failed: Some(format!("solidification failed: {e}")), warnings, ..base },
    }
}

fn flip_corners(s: &mut SurfaceAttributes) {
    for n in s.normals.iter_mut() {
        n.swap(1, 2);
    }
    if let Some(u) = s.uvs.as_mut() {
        for c in u.iter_mut() {
            c.swap(1, 2);
        }
    }
    if let Some(u) = s.tangents.as_mut() {
        for c in u.iter_mut() {
            c.swap(1, 2);
        }
    }
}

/// Weld vertices within `eps`, dropping degenerate triangles; returns the
/// mesh and, for each output triangle, its input triangle index.
pub fn weld_keep_map(m: &TriMesh, eps: f64) -> (TriMesh, Vec<usize>) {
    // reuse TriMesh::weld but track triangles by recomputing the remap
    let w = if eps > 0.0 { weld_remap(m, eps) } else { weld_remap(m, 0.0) };
    let (verts, remap) = w;
    let mut tris = Vec::new();
    let mut keep = Vec::new();
    for (i, t) in m.tris.iter().enumerate() {
        let nt = [remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]];
        if nt[0] == nt[1] || nt[1] == nt[2] || nt[0] == nt[2] {
            continue;
        }
        tris.push(nt);
        keep.push(i);
    }
    let mut out = TriMesh { verts, tris };
    // drop exactly degenerate (collinear) triangles
    let mut k2 = Vec::new();
    let mut t2 = Vec::new();
    for (j, t) in out.tris.iter().enumerate() {
        let deg = {
            let mm = TriMesh { verts: vec![out.verts[t[0] as usize], out.verts[t[1] as usize], out.verts[t[2] as usize]], tris: vec![[0, 1, 2]] };
            mm.is_degenerate(0)
        };
        if !deg {
            t2.push(*t);
            k2.push(keep[j]);
        }
    }
    out.tris = t2;
    (out.compact(), k2)
}

fn weld_remap(m: &TriMesh, eps: f64) -> (Vec<DVec3>, Vec<u32>) {
    use std::collections::BTreeMap;
    let mut verts: Vec<DVec3> = Vec::new();
    let mut remap = vec![0u32; m.verts.len()];
    if eps <= 0.0 {
        let mut map: BTreeMap<[u64; 3], u32> = BTreeMap::new();
        for (i, v) in m.verts.iter().enumerate() {
            let k = [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()];
            remap[i] = *map.entry(k).or_insert_with(|| {
                verts.push(*v);
                (verts.len() - 1) as u32
            });
        }
        return (verts, remap);
    }
    let key = |v: DVec3| [(v.x / eps).floor() as i64, (v.y / eps).floor() as i64, (v.z / eps).floor() as i64];
    let mut grid: BTreeMap<[i64; 3], Vec<u32>> = BTreeMap::new();
    for (i, &v) in m.verts.iter().enumerate() {
        let k = key(v);
        let mut found = None;
        'o: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(c) = grid.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) {
                        for &id in c {
                            if (verts[id as usize] - v).length() <= eps {
                                found = Some(id);
                                break 'o;
                            }
                        }
                    }
                }
            }
        }
        remap[i] = match found {
            Some(id) => id,
            None => {
                verts.push(v);
                let id = (verts.len() - 1) as u32;
                grid.entry(k).or_default().push(id);
                id
            }
        };
    }
    (verts, remap)
}

/// Do two closed solids come within `tol` of each other?
pub fn solids_within(a: &TriMesh, b: &TriMesh, tol: f64) -> bool {
    let qb = MeshQuery::new(b);
    // vertex-to-surface distance both ways (adequate for faceted contacts),
    // plus containment (interpenetration)
    let close = |m: &TriMesh, q: &MeshQuery| {
        m.verts.iter().any(|&v| q.closest_point(v).map(|c| c.1 <= tol * tol).unwrap_or(false) || q.contains(v))
    };
    if close(a, &qb) {
        return true;
    }
    let qa = MeshQuery::new(a);
    close(b, &qa)
}
