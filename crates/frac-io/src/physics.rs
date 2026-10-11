//! FlatBuffers physics payload (`.fracphys`, spec §6.2 / §12), schema
//! `schemas/frac.fbs` (namespace `frac`, file identifier `FRAC`).
//!
//! Fragments and bonds are written in the order stored in the asset (the
//! data model keeps fragments sorted by (level, id) with `fragments[i].id ==
//! i`, and bonds sorted by (level, a, b)); all ranges index into the stored
//! order. Positions are in the asset frame, narrowed to f32.

use crate::IoError;
use crate::fb;
use flatbuffers::{FlatBufferBuilder, WIPOffset};
use frac_core::{Asset, FragmentOrWorld};
use glam::{DMat3, DVec3};
use serde_json::{Value, json};

fn v3(v: DVec3) -> fb::Vec3 {
    fb::Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}

/// Row-major m[3*row + col].
fn m3(m: &DMat3) -> fb::Mat3 {
    let mut a = [0f32; 9];
    for r in 0..3 {
        for c in 0..3 {
            a[3 * r + c] = m.col(c)[r] as f32;
        }
    }
    fb::Mat3::new(&a)
}

fn sat_u16(v: usize, what: &str) -> u16 {
    u16::try_from(v).unwrap_or_else(|_| {
        tracing::warn!("{what} = {v} exceeds u16; saturated");
        u16::MAX
    })
}

fn range_len(r: &std::ops::Range<u32>) -> u32 {
    r.end.saturating_sub(r.start)
}

/// Serialize the physics payload. Deterministic: the same asset always
/// yields the same bytes.
pub fn write_physics(asset: &Asset) -> Vec<u8> {
    let mut b = FlatBufferBuilder::with_capacity(1024);
    let h = &asset.hierarchy;

    // Fragments.
    let mut frags: Vec<WIPOffset<fb::Fragment>> = Vec::with_capacity(h.fragments.len());
    for f in &h.fragments {
        let ids: Vec<u16> = f.material_mix.iter().map(|(m, _)| m.0).collect();
        let fracs: Vec<f32> = f.material_mix.iter().map(|&(_, w)| w).collect();
        let material_ids = b.create_vector(&ids);
        let material_fracs = b.create_vector(&fracs);
        let com = v3(f.mass.com);
        let inertia = m3(&f.mass.inertia);
        frags.push(fb::Fragment::create(
            &mut b,
            &fb::FragmentArgs {
                id: f.id.0,
                level: f.level,
                parent: f.parent.map(|p| p.0 as i32).unwrap_or(-1),
                child_begin: f.children.start,
                child_count: range_len(&f.children),
                mass: f.mass.mass as f32,
                com: Some(&com),
                inertia: Some(&inertia),
                material_ids: Some(material_ids),
                material_fracs: Some(material_fracs),
                hull_begin: f.hulls.start,
                hull_count: sat_u16(range_len(&f.hulls) as usize, "hull_count"),
                particle_candidate: f.particle_candidate,
                role: f.role as u8,
                gltf_node: f.render.gltf_node,
                volume: f.mass.volume as f32,
                interior_area: f.interior_area as f32,
                component: f.component.0,
                cell_begin: f.cells.start,
                cell_count: range_len(&f.cells),
            },
        ));
    }
    let fragments = b.create_vector(&frags);

    // Bonds.
    let mut bonds: Vec<WIPOffset<fb::Bond>> = Vec::with_capacity(asset.bonds.len());
    for bd in &asset.bonds {
        let comps: Vec<WIPOffset<fb::BondComposition>> = bd
            .composition
            .iter()
            .map(|c| {
                fb::BondComposition::create(
                    &mut b,
                    &fb::BondCompositionArgs {
                        kind: c.kind as u8,
                        interface_material: c.interface_material.map(|m| m.0 as i32).unwrap_or(-1),
                        fraction: c.fraction,
                    },
                )
            })
            .collect();
        let composition = b.create_vector(&comps);
        let centroid = v3(bd.centroid);
        let normal = v3(bd.normal);
        let frame_u = v3(bd.frame_u);
        let frame_v = v3(bd.frame_v);
        let extent_center = v3(bd.extent.center);
        let extent_half = fb::Vec2::new(bd.extent.half[0] as f32, bd.extent.half[1] as f32);
        let rebar_dir = v3(bd.reinforcement.dir.normalize_or_zero());
        bonds.push(fb::Bond::create(
            &mut b,
            &fb::BondArgs {
                id: bd.id.0,
                level: bd.level,
                a: bd.a.0,
                b: match bd.b {
                    FragmentOrWorld::Fragment(f) => f.0 as i32,
                    FragmentOrWorld::World => -1,
                },
                area: bd.area as f32,
                centroid: Some(&centroid),
                normal: Some(&normal),
                planarity: bd.planarity,
                frame_u: Some(&frame_u),
                frame_v: Some(&frame_v),
                i_uu: bd.i_uu as f32,
                i_vv: bd.i_vv as f32,
                i_uv: bd.i_uv as f32,
                j: bd.j as f32,
                extent_center: Some(&extent_center),
                extent_half: Some(&extent_half),
                dist_a: bd.dist_a as f32,
                dist_b: bd.dist_b as f32,
                composition: Some(composition),
                rebar_count: sat_u16(bd.reinforcement.count as usize, "rebar_count"),
                rebar_area: bd.reinforcement.steel_area as f32,
                rebar_dir: Some(&rebar_dir),
                strength_scale: bd.strength_scale,
                parent_bond: bd.parent_bond.map(|p| p.0 as i32).unwrap_or(-1),
                child_begin: bd.child_bonds.start,
                child_count: range_len(&bd.child_bonds),
                loop_begin: bd.boundary_loops.start,
                loop_count: sat_u16(range_len(&bd.boundary_loops) as usize, "loop_count"),
                spawn_begin: bd.spawn.start,
                spawn_count: range_len(&bd.spawn),
                anchor: bd.anchor,
                interface_count: bd.interfaces.len() as u32,
            },
        ));
    }
    let bonds = b.create_vector(&bonds);

    // Hulls and their faces (parallel vectors).
    let mut hulls = Vec::with_capacity(asset.hulls.len());
    let mut hull_faces = Vec::with_capacity(asset.hulls.len());
    for hull in &asset.hulls {
        let verts: Vec<fb::Vec3> = hull.vertices.iter().map(|&v| v3(v)).collect();
        let vertices = b.create_vector(&verts);
        hulls.push(fb::Hull::create(
            &mut b,
            &fb::HullArgs {
                vertices: Some(vertices),
                fragment: hull.fragment.0,
            },
        ));
        let mut indices: Vec<u32> = Vec::new();
        let mut sizes: Vec<u8> = Vec::new();
        for face in &hull.faces {
            if face.len() <= u8::MAX as usize {
                indices.extend_from_slice(face);
                sizes.push(face.len() as u8);
            } else {
                // Faces larger than 255 vertices are fan-triangulated.
                for k in 1..face.len() - 1 {
                    indices.extend_from_slice(&[face[0], face[k], face[k + 1]]);
                    sizes.push(3);
                }
            }
        }
        let indices = b.create_vector(&indices);
        let face_sizes = b.create_vector(&sizes);
        hull_faces.push(fb::HullFaces::create(
            &mut b,
            &fb::HullFacesArgs {
                indices: Some(indices),
                face_sizes: Some(face_sizes),
            },
        ));
    }
    let hulls = b.create_vector(&hulls);
    let hull_faces = b.create_vector(&hull_faces);

    let loops: Vec<WIPOffset<fb::Loop>> = asset
        .loops
        .iter()
        .map(|l| {
            let pts: Vec<fb::Vec3> = l.iter().map(|&p| v3(p)).collect();
            let points = b.create_vector(&pts);
            fb::Loop::create(
                &mut b,
                &fb::LoopArgs {
                    points: Some(points),
                },
            )
        })
        .collect();
    let loops = b.create_vector(&loops);
    let spawn: Vec<fb::SpawnPoint> = asset
        .spawn
        .iter()
        .map(|s| fb::SpawnPoint::new(&v3(s.p), &v3(s.n)))
        .collect();
    let spawn = b.create_vector(&spawn);
    let lb: Vec<u32> = h.level_ranges.iter().map(|r| r.start).collect();
    let lc: Vec<u32> = h.level_ranges.iter().map(range_len).collect();
    let level_fragment_begin = b.create_vector(&lb);
    let level_fragment_count = b.create_vector(&lc);
    let bc: Vec<u32> = asset.bond_children.iter().map(|c| c.0).collect();
    let bond_children = b.create_vector(&bc);

    let m = &asset.meta;
    let tool_version = if m.tool_version.is_empty() {
        frac_core::TOOL_VERSION
    } else {
        &m.tool_version
    };
    let tool_version = b.create_string(tool_version);
    let settings_hash = b.create_string(&m.settings_hash);
    let material_library_id = b.create_string(&m.material_library_id);
    let material_library_version = b.create_string(&m.material_library_version);
    let name = b.create_string(&m.name);
    let root = fb::Asset::create(
        &mut b,
        &fb::AssetArgs {
            schema_version: frac_core::SCHEMA_VERSION,
            tool_version: Some(tool_version),
            settings_hash: Some(settings_hash),
            seed: m.seed,
            material_library_id: Some(material_library_id),
            material_library_version: Some(material_library_version),
            fragments: Some(fragments),
            bonds: Some(bonds),
            hulls: Some(hulls),
            hull_faces: Some(hull_faces),
            loops: Some(loops),
            levels: h.levels,
            spawn: Some(spawn),
            name: Some(name),
            variant: m.variant,
            level_fragment_begin: Some(level_fragment_begin),
            level_fragment_count: Some(level_fragment_count),
            bond_children: Some(bond_children),
        },
    );
    fb::finish_asset_buffer(&mut b, root);
    b.finished_data().to_vec()
}

/// Per-bond summary.
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicsBond {
    pub level: u8,
    pub a: u32,
    /// -1 = world anchor.
    pub b: i32,
    pub area: f32,
}

/// Per-fragment summary.
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicsFragment {
    pub id: u32,
    pub level: u8,
    pub parent: i32,
    pub mass: f32,
}

/// Small owned summary of a verified physics payload.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PhysicsView {
    pub schema_version: u16,
    pub tool_version: String,
    pub settings_hash: String,
    pub name: String,
    pub seed: u64,
    pub variant: u32,
    pub levels: u8,
    pub fragment_count: usize,
    pub bond_count: usize,
    pub hull_count: usize,
    pub loop_count: usize,
    pub spawn_count: usize,
    pub bond_children_count: usize,
    pub bonds: Vec<PhysicsBond>,
    pub fragments: Vec<PhysicsFragment>,
}

fn verified(bytes: &[u8]) -> Result<fb::Asset<'_>, IoError> {
    if !fb::asset_buffer_has_identifier(bytes) {
        return Err(IoError::Physics("missing 'FRAC' file identifier".into()));
    }
    let opts = flatbuffers::VerifierOptions {
        max_depth: 64,
        max_tables: usize::MAX / 2,
        max_apparent_size: 1 << 31,
        ..Default::default()
    };
    fb::root_as_asset_with_opts(&opts, bytes)
        .map_err(|e| IoError::Physics(format!("verification failed: {e}")))
}

/// Verify a `.fracphys` buffer and return an owned summary.
pub fn read_physics(bytes: &[u8]) -> Result<PhysicsView, IoError> {
    let a = verified(bytes)?;
    let mut v = PhysicsView {
        schema_version: a.schema_version(),
        tool_version: a.tool_version().unwrap_or("").to_string(),
        settings_hash: a.settings_hash().unwrap_or("").to_string(),
        name: a.name().unwrap_or("").to_string(),
        seed: a.seed(),
        variant: a.variant(),
        levels: a.levels(),
        hull_count: a.hulls().map(|h| h.len()).unwrap_or(0),
        loop_count: a.loops().map(|h| h.len()).unwrap_or(0),
        spawn_count: a.spawn().map(|h| h.len()).unwrap_or(0),
        bond_children_count: a.bond_children().map(|h| h.len()).unwrap_or(0),
        ..Default::default()
    };
    if let Some(fs) = a.fragments() {
        v.fragments = fs
            .iter()
            .map(|f| PhysicsFragment {
                id: f.id(),
                level: f.level(),
                parent: f.parent(),
                mass: f.mass(),
            })
            .collect();
    }
    if let Some(bs) = a.bonds() {
        v.bonds = bs
            .iter()
            .map(|b| PhysicsBond {
                level: b.level(),
                a: b.a(),
                b: b.b(),
                area: b.area(),
            })
            .collect();
    }
    v.fragment_count = v.fragments.len();
    v.bond_count = v.bonds.len();
    Ok(v)
}

fn jv3(v: Option<&fb::Vec3>) -> Value {
    match v {
        Some(v) => json!([v.x(), v.y(), v.z()]),
        None => Value::Null,
    }
}

/// Verify a `.fracphys` buffer and dump its full contents as pretty JSON
/// (for `prefracture inspect`).
pub fn physics_to_json(bytes: &[u8]) -> Result<String, IoError> {
    let a = verified(bytes)?;
    let vec_u32 = |v: Option<flatbuffers::Vector<'_, u32>>| -> Value {
        json!(v.map(|v| v.iter().collect::<Vec<_>>()).unwrap_or_default())
    };
    let fragments: Vec<Value> = a
        .fragments()
        .map(|fs| {
            fs.iter()
                .map(|f| {
                    json!({
                        "id": f.id(), "level": f.level(), "parent": f.parent(),
                        "child_begin": f.child_begin(), "child_count": f.child_count(),
                        "mass": f.mass(), "com": jv3(f.com()),
                        "inertia": f.inertia().map(|m| m.m().iter().collect::<Vec<f32>>()),
                        "material_ids": f.material_ids().map(|v| v.iter().collect::<Vec<_>>()),
                        "material_fracs": f.material_fracs().map(|v| v.iter().collect::<Vec<_>>()),
                        "hull_begin": f.hull_begin(), "hull_count": f.hull_count(),
                        "particle_candidate": f.particle_candidate(), "role": f.role(),
                        "gltf_node": f.gltf_node(), "volume": f.volume(), "interior_area": f.interior_area(),
                        "component": f.component(), "cell_begin": f.cell_begin(), "cell_count": f.cell_count(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let bonds: Vec<Value> = a
        .bonds()
        .map(|bs| {
            bs.iter()
                .map(|b| {
                    let comp: Vec<Value> = b
                        .composition()
                        .map(|cs| {
                            cs.iter()
                                .map(|c| json!({"kind": c.kind(), "interface_material": c.interface_material(), "fraction": c.fraction()}))
                                .collect()
                        })
                        .unwrap_or_default();
                    json!({
                        "id": b.id(), "level": b.level(), "a": b.a(), "b": b.b(), "area": b.area(),
                        "centroid": jv3(b.centroid()), "normal": jv3(b.normal()), "planarity": b.planarity(),
                        "frame_u": jv3(b.frame_u()), "frame_v": jv3(b.frame_v()),
                        "i_uu": b.i_uu(), "i_vv": b.i_vv(), "i_uv": b.i_uv(), "j": b.j(),
                        "extent_center": jv3(b.extent_center()),
                        "extent_half": b.extent_half().map(|h| json!([h.x(), h.y()])),
                        "dist_a": b.dist_a(), "dist_b": b.dist_b(), "composition": comp,
                        "rebar_count": b.rebar_count(), "rebar_area": b.rebar_area(), "rebar_dir": jv3(b.rebar_dir()),
                        "strength_scale": b.strength_scale(), "parent_bond": b.parent_bond(),
                        "child_begin": b.child_begin(), "child_count": b.child_count(),
                        "loop_begin": b.loop_begin(), "loop_count": b.loop_count(),
                        "spawn_begin": b.spawn_begin(), "spawn_count": b.spawn_count(),
                        "anchor": b.anchor(), "interface_count": b.interface_count(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let hulls: Vec<Value> = a
        .hulls()
        .map(|hs| {
            hs.iter()
                .map(|h| {
                    let vs: Vec<Value> = h
                        .vertices()
                        .map(|v| v.iter().map(|p| jv3(Some(p))).collect())
                        .unwrap_or_default();
                    json!({"fragment": h.fragment(), "vertices": vs})
                })
                .collect()
        })
        .unwrap_or_default();
    let hull_faces: Vec<Value> = a
        .hull_faces()
        .map(|hs| {
            hs.iter()
                .map(|h| {
                    json!({
                        "indices": vec_u32(h.indices()),
                        "face_sizes": h.face_sizes().map(|v| v.iter().collect::<Vec<_>>()).unwrap_or_default(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let loops: Vec<Value> = a
        .loops()
        .map(|ls| {
            ls.iter()
                .map(|l| {
                    Value::Array(
                        l.points()
                            .map(|v| v.iter().map(|p| jv3(Some(p))).collect())
                            .unwrap_or_default(),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    let spawn: Vec<Value> = a
        .spawn()
        .map(|ss| {
            ss.iter()
                .map(|s| json!({"p": jv3(Some(s.p())), "n": jv3(Some(s.n()))}))
                .collect()
        })
        .unwrap_or_default();
    let root = json!({
        "schema_version": a.schema_version(),
        "tool_version": a.tool_version(),
        "settings_hash": a.settings_hash(),
        "seed": a.seed(),
        "material_library_id": a.material_library_id(),
        "material_library_version": a.material_library_version(),
        "name": a.name(),
        "variant": a.variant(),
        "levels": a.levels(),
        "level_fragment_begin": vec_u32(a.level_fragment_begin()),
        "level_fragment_count": vec_u32(a.level_fragment_count()),
        "fragments": fragments,
        "bonds": bonds,
        "bond_children": vec_u32(a.bond_children()),
        "hulls": hulls,
        "hull_faces": hull_faces,
        "loops": loops,
        "spawn": spawn,
    });
    Ok(serde_json::to_string_pretty(&root)?)
}
