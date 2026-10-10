//! Structural soundness gates on tiny cuboid assemblies: a sound frame
//! passes; a floating beam fails `structural_support`; a slab cantilevered
//! off a single bearing pier fails `self_weight` (overturning of a
//! compression-only joint).
use frac_core::input::{AuthoringMeta, InputPart, InputScene, PartMeta};
use frac_core::Settings;
use frac_geom::mesh::box_mesh;
use frac_geom::DVec3;
use frac_material::MaterialLibrary;
use frac_pipeline::{run, InputSpec};
use frac_validate::GateStatus;

struct P {
    name: &'static str,
    lo: [f64; 3],
    hi: [f64; 3],
    material: &'static str,
    role: &'static str,
    anchored: bool,
}

fn gates(parts: &[P]) -> std::collections::BTreeMap<String, (GateStatus, String)> {
    let mut meta = AuthoringMeta { ground_height: Some(0.0), ..Default::default() };
    let mut scene = Vec::new();
    for p in parts {
        let mesh = box_mesh(DVec3::from_array(p.lo), DVec3::from_array(p.hi));
        let n = mesh.tris.len();
        scene.push(InputPart { name: p.name.into(), mesh, material_slot: vec![0; n], material_names: vec![p.material.into()], extras: serde_json::Value::Null, ..Default::default() });
        meta.parts.insert(
            p.name.into(),
            PartMeta { material: Some(p.material.into()), component_role: Some(p.role.into()), anchor_below: p.anchored.then_some(0.0), ..Default::default() },
        );
    }
    let input = InputSpec { name: "structure".into(), scene: InputScene { source: "test".into(), parts: scene }, meta, variant: 0 };
    let mut s = Settings::default();
    s.cells.target_analysis_cells_per_m3 = 10.0;
    s.cells.fine_per_analysis = 2;
    s.modes.enabled = false;
    s.render.lods = 1;
    s.render.noise = false;
    s.render.chipping = false;
    let out = run(&input, &s, &MaterialLibrary::builtin()).expect("pipeline");
    out.report.scorecard.gates.iter().map(|g| (g.name.clone(), (g.status.clone(), g.detail.clone()))).collect()
}

#[test]
fn sound_frame_passes() {
    let g = gates(&[
        P { name: "col_a", lo: [0.0, 0.0, 0.0], hi: [0.3, 2.8, 0.3], material: "concrete_c30", role: "column", anchored: true },
        P { name: "col_b", lo: [3.7, 0.0, 0.0], hi: [4.0, 2.8, 0.3], material: "concrete_c30", role: "column", anchored: true },
        P { name: "slab", lo: [0.0, 2.8, 0.0], hi: [4.0, 3.0, 0.3], material: "concrete_c30", role: "slab", anchored: false },
    ]);
    assert_eq!(g["structural_support"].0, GateStatus::Pass, "{}", g["structural_support"].1);
    assert_eq!(g["self_weight"].0, GateStatus::Pass, "{}", g["self_weight"].1);
}

#[test]
fn floating_beam_fails_support() {
    let g = gates(&[
        P { name: "col", lo: [0.0, 0.0, 0.0], hi: [0.3, 2.8, 0.3], material: "concrete_c30", role: "column", anchored: true },
        P { name: "beam", lo: [1.0, 2.0, 0.0], hi: [3.0, 2.4, 0.3], material: "concrete_c30", role: "beam", anchored: false },
    ]);
    assert_eq!(g["structural_support"].0, GateStatus::Fail, "{}", g["structural_support"].1);
    assert!(g["structural_support"].1.contains("beam"));
}

#[test]
fn cantilever_on_bearing_pier_overturns() {
    // brick pier (bearing joint to the slab above, compression only) with a
    // heavy concrete slab extending 3 m to one side
    let g = gates(&[
        P { name: "pier", lo: [0.0, 0.0, 0.0], hi: [0.3, 1.0, 0.3], material: "brick_clay", role: "column", anchored: true },
        P { name: "slab", lo: [0.0, 1.0, 0.0], hi: [3.3, 1.2, 0.3], material: "concrete_c30", role: "slab", anchored: false },
    ]);
    assert_eq!(g["structural_support"].0, GateStatus::Pass, "{}", g["structural_support"].1);
    assert_eq!(g["self_weight"].0, GateStatus::Fail, "{}", g["self_weight"].1);
}
