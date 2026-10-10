//! Procedural benchmark suite (spec §13.8). Each asset is written as a GLB
//! plus a `.meta.json` authoring sidecar.

use crate::meshgen::*;
use frac_core::input::{AuthoringMeta, BoxVolume, ConnectionSpec, PartMeta, RebarSpec};
use frac_geom::{DVec3, TriMesh};
use frac_io::{GltfOptions, RenderMaterial, RenderMesh, RenderNode, RenderPrimitive, RenderScene};
use serde_json::json;
use std::path::Path;

struct Part {
    name: String,
    mesh: TriMesh,
    material: String,
    meta: PartMeta,
}

fn part(name: &str, mesh: TriMesh, material: &str) -> Part {
    Part { name: name.into(), mesh, material: material.into(), meta: PartMeta { material: Some(material.into()), ..Default::default() } }
}

fn write_asset(dir: &Path, name: &str, parts: Vec<Part>, connections: Vec<ConnectionSpec>, ground: Option<f64>) -> Result<(), String> {
    let mut scene = RenderScene { name: name.into(), ..Default::default() };
    let mut mats: Vec<String> = Vec::new();
    let mut meta = AuthoringMeta { connections, ground_height: ground, ..Default::default() };
    for p in &parts {
        let mi = match mats.iter().position(|m| *m == p.material) {
            Some(i) => i,
            None => {
                mats.push(p.material.clone());
                scene.materials.push(RenderMaterial { name: p.material.clone(), base_color: [0.7, 0.7, 0.7, 1.0], metallic: 0.0, roughness: 0.8, double_sided: false });
                mats.len() - 1
            }
        };
        // flat-shaded triangle soup
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        for t in 0..p.mesh.tris.len() {
            let [a, b, c] = p.mesh.tri_points(t);
            let n = (b - a).cross(c - a).normalize_or_zero();
            for v in [a, b, c] {
                indices.push(positions.len() as u32);
                positions.push(v.as_vec3().to_array());
                normals.push(n.as_vec3().to_array());
            }
        }
        let uvs = positions.iter().map(|p: &[f32; 3]| [p[0] + p[2], p[1]]).collect();
        scene.meshes.push(RenderMesh { name: p.name.clone(), positions, normals, uvs: Some(uvs), tangents: None, primitives: vec![RenderPrimitive { material: mi as u32, indices }] });
        scene.nodes.push(RenderNode { name: p.name.clone(), parent: None, translation: [0.0; 3], mesh_lods: vec![(scene.meshes.len() - 1) as u32], extras: json!({}) });
        meta.parts.insert(p.name.clone(), p.meta.clone());
    }
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let glb = frac_io::write_glb(&scene, &GltfOptions { meshopt_compression: false }).map_err(|e| e.to_string())?;
    std::fs::write(dir.join(format!("{name}.glb")), glb).map_err(|e| e.to_string())?;
    std::fs::write(dir.join(format!("{name}.glb.meta.json")), serde_json::to_string_pretty(&meta).unwrap()).map_err(|e| e.to_string())?;
    println!("wrote {name} ({} parts)", parts.len());
    Ok(())
}

/// Wall in the XY plane (length along X, height along Y, thickness along Z)
/// with rectangular openings.
fn wall(x0: f64, x1: f64, y0: f64, y1: f64, z0: f64, z1: f64, holes: &[[f64; 4]]) -> TriMesh {
    let mut loops = vec![rect(x0, y0, x1, y1)];
    for h in holes {
        loops.push(rect_hole(h[0], h[1], h[2], h[3]));
    }
    extrude(&loops, z0, z1)
}

/// Same wall but along Z (length along Z, thickness along X).
fn wall_z(z0: f64, z1: f64, y0: f64, y1: f64, x0: f64, x1: f64, holes: &[[f64; 4]]) -> TriMesh {
    let w = wall(z0, z1, y0, y1, x0, x1, holes);
    // (u, v, w) -> (x = w, y = v, z = u): a reflection, so flip winding
    reorient(&w, |p| DVec3::new(p.z, p.y, p.x), true)
}

pub fn generate(dir: &Path) -> Result<(), String> {
    // 1. ceramic bowl (thin curved shell)
    {
        let r = 0.12;
        let h = 0.07;
        let t = 0.004;
        let n = 24;
        // CCW profile in (r, y): bottom pole, outer wall up to the rim,
        // across the rim, inner wall down to the inner pole
        let mut prof: Vec<[f64; 2]> = vec![[0.0, 0.0]];
        for k in 1..=n {
            let a = std::f64::consts::FRAC_PI_2 * k as f64 / n as f64;
            prof.push([r * a.sin(), h * (1.0 - a.cos())]);
        }
        for k in (1..=n).rev() {
            let a = std::f64::consts::FRAC_PI_2 * k as f64 / n as f64;
            prof.push([(r - t) * a.sin(), t + (h - t) * (1.0 - a.cos())]);
        }
        prof.push([0.0, t]);
        let m = revolve(&prof, 48);
        write_asset(dir, "ceramic_bowl", vec![part("bowl", m, "ceramic")], vec![], None)?;
    }
    // 2. glass panes
    {
        let mut p = part("pane", box_at(DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 1.2, 0.006)), "glass_annealed");
        p.meta.anchor_below = Some(0.0);
        p.meta.impact_center = Some([0.45, 0.65, 0.003]);
        write_asset(dir, "glass_annealed_pane", vec![p], vec![], None)?;
        let mut p = part("pane", box_at(DVec3::new(0.0, 0.0, 0.0), DVec3::new(0.8, 1.0, 0.008)), "glass_tempered");
        p.meta.anchor_below = Some(0.0);
        write_asset(dir, "glass_tempered_pane", vec![p], vec![], None)?;
    }
    // 3. brick wall 4 x 3 m with a window
    {
        let m = wall(0.0, 4.0, 0.0, 3.0, 0.0, 0.215, &[[1.4, 1.0, 2.6, 2.2]]);
        let mut p = part("wall", m, "brick_clay");
        p.meta.anchor_below = Some(0.0);
        p.meta.masonry_layout = Some(json!({"brick": [0.215, 0.065, 0.1025], "bond": "stretcher", "mortar": 0.01, "half_split_fraction": 0.2, "chip_fraction": 0.05}));
        write_asset(dir, "brick_wall_window", vec![p], vec![], Some(0.0))?;
    }
    // 4. reinforced concrete column
    {
        let m = box_at(DVec3::new(-0.2, 0.0, -0.2), DVec3::new(0.2, 3.0, 0.2));
        let mut p = part("column", m, "concrete_c30");
        p.meta.anchor_below = Some(0.0);
        for (x, z) in [(-0.15, -0.15), (0.15, -0.15), (0.15, 0.15), (-0.15, 0.15)] {
            p.meta.rebar.push(RebarSpec { points: vec![[x, -0.1, z], [x, 3.1, z]], diameter: 0.02 });
        }
        write_asset(dir, "rc_column", vec![p], vec![], Some(0.0))?;
    }
    // 5. timber beam
    {
        let mut p = part("beam", box_at(DVec3::new(0.0, 0.0, 0.0), DVec3::new(4.0, 0.3, 0.15)), "pine");
        p.meta.grain = Some([1.0, 0.0, 0.0]);
        write_asset(dir, "timber_beam", vec![p], vec![], None)?;
    }
    // 6. RC slab on 4 columns
    {
        let mut parts = Vec::new();
        for (i, (x, z)) in [(0.0, 0.0), (3.7, 0.0), (3.7, 3.7), (0.0, 3.7)].iter().enumerate() {
            let mut c = part(&format!("column_{i}"), box_at(DVec3::new(*x, 0.0, *z), DVec3::new(x + 0.3, 3.0, z + 0.3)), "concrete_c30");
            c.meta.component_role = Some("column".into());
            c.meta.anchor_below = Some(0.0);
            parts.push(c);
        }
        let mut s = part("slab", box_at(DVec3::new(0.0, 3.0, 0.0), DVec3::new(4.0, 3.2, 4.0)), "concrete_c30");
        s.meta.component_role = Some("slab".into());
        parts.push(s);
        write_asset(dir, "rc_slab_on_columns", parts, vec![], Some(0.0))?;
    }
    // 7. two-storey building (~50 components)
    {
        let mut parts = Vec::new();
        let span = 4.0;
        let (h, slab_t, col) = (3.0, 0.2, 0.3);
        for floor in 0..2 {
            let y0 = floor as f64 * (h + slab_t);
            for i in 0..3 {
                for j in 0..3 {
                    let (x, z) = (i as f64 * span, j as f64 * span);
                    let mut c = part(&format!("f{floor}_col_{i}{j}"), box_at(DVec3::new(x, y0, z), DVec3::new(x + col, y0 + h, z + col)), "concrete_c30");
                    c.meta.component_role = Some("column".into());
                    parts.push(c);
                }
            }
            // beams between columns (under the slab), x and z directions
            for i in 0..3 {
                for j in 0..2 {
                    let (x, z) = (i as f64 * span, j as f64 * span);
                    let mut b = part(&format!("f{floor}_beamz_{i}{j}"), box_at(DVec3::new(x, y0 + h - 0.4, z + col), DVec3::new(x + col, y0 + h, z + span)), "concrete_c30");
                    b.meta.component_role = Some("beam".into());
                    parts.push(b);
                    let mut b = part(&format!("f{floor}_beamx_{j}{i}"), box_at(DVec3::new(z + col, y0 + h - 0.4, x), DVec3::new(z + span, y0 + h, x + col)), "concrete_c30");
                    b.meta.component_role = Some("beam".into());
                    parts.push(b);
                }
            }
            let mut s = part(&format!("f{floor}_slab"), box_at(DVec3::new(0.0, y0 + h, 0.0), DVec3::new(2.0 * span + col, y0 + h + slab_t, 2.0 * span + col)), "concrete_c30");
            s.meta.component_role = Some("slab".into());
            parts.push(s);
            // infill walls on two facades (between columns, below beams)
            for j in 0..2 {
                let z = j as f64 * span;
                let mut w = part(&format!("f{floor}_wall_w{j}"), wall_z(z + col, z + span, y0, y0 + h - 0.4, 0.05, 0.25, &[[z + 1.5, y0 + 1.0, z + 2.8, y0 + 2.2]]), "brick_clay");
                w.meta.component_role = Some("wall".into());
                w.meta.masonry_layout = Some(json!({"brick": [0.215, 0.065, 0.1025], "bond": "stretcher", "mortar": 0.01, "half_split_fraction": 0.0, "chip_fraction": 0.0}));
                parts.push(w);
            }
            if floor == 0 {
                for p in parts.iter_mut().filter(|p| p.name.starts_with("f0_col") || p.name.starts_with("f0_wall")) {
                    p.meta.anchor_below = Some(0.0);
                }
            }
        }
        write_asset(dir, "two_storey_building", parts, vec![], Some(0.0))?;
    }
    // 8. messy scanned mesh: holes, duplicated faces, overlapping blobs, noise
    {
        let mut m = frac_geom::mesh::icosphere(DVec3::new(0.0, 0.3, 0.0), 0.3, 3);
        m.append(&frac_geom::mesh::icosphere(DVec3::new(0.2, 0.45, 0.1), 0.18, 3));
        for (i, v) in m.verts.iter_mut().enumerate() {
            let h = frac_core::stable_hash(&[i as u64, 99]);
            *v += (*v - DVec3::new(0.0, 0.3, 0.0)).normalize_or_zero() * (frac_core::unit_f64(h) - 0.5) * 0.01;
        }
        let keep: Vec<[u32; 3]> = m.tris.iter().enumerate().filter(|(i, _)| i % 97 != 0).map(|(_, t)| *t).collect();
        let dup: Vec<[u32; 3]> = keep.iter().step_by(53).copied().collect();
        m.tris = keep;
        m.tris.extend(dup);
        write_asset(dir, "messy_scan", vec![part("scan", m, "sandstone")], vec![], None)?;
    }
    // held-out set (never tuned on)
    let held = dir.join("held_out");
    {
        // arch
        let mut outline = Vec::new();
        let (ro, ri) = (1.2, 0.9);
        for k in 0..=24 {
            let a = std::f64::consts::PI * k as f64 / 24.0;
            outline.push([ro * a.cos(), 1.0 + ro * a.sin()]);
        }
        for k in (0..=24).rev() {
            let a = std::f64::consts::PI * k as f64 / 24.0;
            outline.push([ri * a.cos(), 1.0 + ri * a.sin()]);
        }
        let mut o2 = vec![[-1.2, 0.0], [-0.9, 0.0]];
        o2.extend(outline.iter().rev().skip(0).copied().filter(|p| p[1] > 1.0 - 1e-12));
        let arch_loop: Vec<[f64; 2]> = {
            let mut l = vec![[0.9, 0.0], [1.2, 0.0]];
            for k in 0..=24 {
                let a = std::f64::consts::PI * k as f64 / 24.0;
                l.push([ro * a.cos(), 1.0 + ro * a.sin()]);
            }
            l.push([-1.2, 0.0]);
            l.push([-0.9, 0.0]);
            for k in (0..=24).rev() {
                let a = std::f64::consts::PI * k as f64 / 24.0;
                l.push([ri * a.cos(), 1.0 + ri * a.sin()]);
            }
            l
        };
        let _ = o2;
        let mut a = part("arch", extrude(&[arch_loop], 0.0, 0.4), "sandstone");
        a.meta.anchor_below = Some(0.0);
        write_asset(&held, "stone_arch", vec![a], vec![], Some(0.0))?;
        // hollow pipe
        let pipe = revolve(&[[0.15, 0.0], [0.2, 0.0], [0.2, 2.0], [0.15, 2.0]], 32);
        write_asset(&held, "concrete_pipe", vec![part("pipe", pipe, "concrete_c30")], vec![], None)?;
        // drywall panel with door
        let mut d = part("drywall", wall(0.0, 2.4, 0.0, 2.5, 0.0, 0.0125, &[[0.8, 0.3, 1.7, 2.1]]), "drywall");
        d.meta.anchor_below = Some(0.0);
        write_asset(&held, "drywall_door", vec![d], vec![ConnectionSpec { a: "drywall".into(), b: "drywall".into(), kind: "adhesive".into(), interface_material: None }], Some(0.0))?;
    }
    let _ = BoxVolume { min: [0.0; 3], max: [0.0; 3] };
    Ok(())
}
