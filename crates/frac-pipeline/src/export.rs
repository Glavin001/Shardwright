//! Conversion of render output to the glTF scene description.

use frac_core::*;
use frac_io::{RenderMaterial, RenderMesh, RenderNode, RenderPrimitive, RenderScene};
use frac_material::MaterialLibrary;
use frac_render::RenderOut;
use serde_json::json;
use std::collections::BTreeMap;

pub fn render_scene(asset: &mut Asset, render: &RenderOut, lib: &MaterialLibrary) -> RenderScene {
    let mut scene = RenderScene { name: asset.meta.name.clone(), ..Default::default() };
    // two render materials per library material: exterior, interior
    let mut mat_slot: BTreeMap<MaterialId, (u32, u32)> = BTreeMap::new();
    for comp in &asset.components {
        mat_slot.entry(comp.material).or_insert_with(|| {
            let m = lib.material(comp.material);
            let ext = scene.materials.len() as u32;
            scene.materials.push(RenderMaterial {
                name: format!("{}_exterior", m.id),
                base_color: m.exterior_color.unwrap_or([0.7, 0.7, 0.7, 1.0]),
                metallic: if m.id.starts_with("steel") { 0.9 } else { 0.0 },
                roughness: 0.8,
                double_sided: false,
            });
            scene.materials.push(RenderMaterial {
                name: m.interior_render_material.clone().unwrap_or_else(|| format!("{}_interior", m.id)),
                base_color: m.interior_color.unwrap_or([0.6, 0.6, 0.6, 1.0]),
                metallic: 0.0,
                roughness: 0.95,
                double_sided: false,
            });
            (ext, ext + 1)
        });
    }
    let rebar_mat = scene.materials.len() as u32;
    scene.materials.push(RenderMaterial { name: "rebar".into(), base_color: [0.35, 0.25, 0.2, 1.0], metallic: 0.8, roughness: 0.6, double_sided: false });
    let h = &asset.hierarchy;
    let nf = h.fragments.len();
    for fi in 0..nf {
        let f = &h.fragments[fi];
        let (ext_m, int_m) = mat_slot[&asset.components[f.component.idx()].material];
        let origin = f.mass.com;
        let mut lods = Vec::new();
        for (li, fm) in render.fragments[fi].iter().enumerate() {
            let mut prims = Vec::new();
            if !fm.ext_indices.is_empty() {
                prims.push(RenderPrimitive { material: ext_m, indices: fm.ext_indices.clone() });
            }
            if !fm.int_indices.is_empty() {
                prims.push(RenderPrimitive { material: int_m, indices: fm.int_indices.clone() });
            }
            if prims.is_empty() {
                continue;
            }
            let mi = scene.meshes.len() as u32;
            scene.meshes.push(RenderMesh {
                name: format!("frag_{}_lod{}", f.id.0, li),
                positions: fm.positions.iter().map(|p| (*p - origin).as_vec3().to_array()).collect(),
                normals: fm.normals.iter().map(|n| n.normalize_or_zero().to_array()).collect(),
                uvs: Some(fm.uvs.clone()),
                tangents: None,
                primitives: prims,
            });
            lods.push(mi);
        }
        let parent_com = f.parent.map(|p| h.fragments[p.idx()].mass.com).unwrap_or(glam::DVec3::ZERO);
        let children: Vec<u32> = f.children.clone().collect();
        scene.nodes.push(RenderNode {
            name: format!("frag_{}_L{}", f.id.0, f.level),
            parent: f.parent.map(|p| p.0),
            translation: (origin - parent_com).to_array(),
            mesh_lods: lods.clone(),
            extras: json!({
                "fragment_id": f.id.0,
                "level": f.level,
                "parent": f.parent.map(|p| p.0 as i64).unwrap_or(-1),
                "children": children,
                "component": f.component.0,
                "mass": f.mass.mass,
                "particle_candidate": f.particle_candidate,
            }),
        });
    }
    for fi in 0..nf {
        let lods: Vec<u32> = scene.nodes[fi].mesh_lods.clone();
        let f = &mut asset.hierarchy.fragments[fi];
        f.render = RenderRefs { gltf_node: fi as i32, lod_meshes: lods };
    }
    // rebar stubs under the finest fragment of the interface's first cell
    let leaf = asset.hierarchy.levels.saturating_sub(1) as usize;
    for st in &render.stubs {
        let it = &asset.interfaces[st.interface.idx()];
        let f = asset.hierarchy.cell_fragment[leaf][it.cells.0.idx()];
        let origin = asset.hierarchy.fragments[f.idx()].mass.com;
        let mi = scene.meshes.len() as u32;
        scene.meshes.push(RenderMesh {
            name: format!("rebar_{}", st.interface.0),
            positions: st.positions.iter().map(|p| (*p - origin).as_vec3().to_array()).collect(),
            normals: st.normals.iter().map(|n| n.to_array()).collect(),
            uvs: None,
            tangents: None,
            primitives: vec![RenderPrimitive { material: rebar_mat, indices: st.indices.clone() }],
        });
        scene.nodes.push(RenderNode {
            name: format!("rebar_{}", st.interface.0),
            parent: Some(f.0),
            translation: [0.0; 3],
            mesh_lods: vec![mi],
            extras: json!({"rebar_stub": true, "interface": st.interface.0, "cells": [it.cells.0 .0, match it.cells.1 { CellOrWorld::Cell(c) => c.0 as i64, CellOrWorld::World => -1 }]}),
        });
    }
    scene.extras = json!({
        "generator": format!("prefracture {}", TOOL_VERSION),
        "levels": asset.hierarchy.levels,
        "seed": asset.meta.seed,
        "variant": asset.meta.variant,
        "settings_hash": asset.meta.settings_hash,
    });
    scene
}
