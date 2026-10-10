//! Reference bond-network consistency (spec §13.3): under uniform stress the
//! Tensorial network must reproduce `σ·n` on every bond, interior and next
//! to free surfaces, on a real baked Voronoi complex.
use frac_core::input::{AuthoringMeta, InputPart, InputScene, PartMeta};
use frac_core::{Fragment, Settings};
use frac_geom::mesh::box_mesh;
use frac_geom::DVec3;
use frac_material::MaterialLibrary;
use frac_pipeline::{run, InputSpec};
use frac_validate::network::{ReferenceSolver, StiffnessModel};
use glam::DMat3;

fn quantile(mut v: Vec<f64>, q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q) as usize]
}

#[test]
fn tensorial_network_passes_patch_tests() {
    let mesh = box_mesh(DVec3::new(-0.2, 0.0, -0.2), DVec3::new(0.2, 1.2, 0.2));
    let n = mesh.tris.len();
    let part = InputPart { name: "block".into(), mesh, material_slot: vec![0; n], material_names: vec!["concrete_c30".into()], extras: serde_json::Value::Null, ..Default::default() };
    let mut meta = AuthoringMeta::default();
    meta.parts.insert("block".into(), PartMeta { material: Some("concrete_c30".into()), ..Default::default() });
    let input = InputSpec { name: "block".into(), scene: InputScene { source: "test".into(), parts: vec![part] }, meta, variant: 0 };
    let mut settings = Settings::default();
    settings.cells.target_analysis_cells_per_m3 = 200.0;
    settings.cells.fine_per_analysis = 8;
    settings.modes.enabled = false;
    let lib = MaterialLibrary::builtin();
    let asset = run(&input, &settings, &lib).expect("pipeline").asset;
    let level = asset.hierarchy.levels - 1;
    let solver = ReferenceSolver { lib: &lib, model: StiffnessModel::Tensorial };

    // interior bonds, boundary fragments driven: axial and pure shear
    let axial = DMat3::from_diagonal(DVec3::new(0.0, 1e-4, 0.0));
    let shear = DMat3::from_cols(DVec3::new(0.0, 1e-4, 0.0), DVec3::new(1e-4, 0.0, 0.0), DVec3::ZERO);
    for (name, eps) in [("axial", axial), ("shear", shear)] {
        let e = solver.patch_test(&asset, level, eps);
        assert!(e.len() > 20, "too few interior bonds: {}", e.len());
        let (p50, p95) = (quantile(e.clone(), 0.5), quantile(e, 0.95));
        eprintln!("{name}: p50 {p50:.4} p95 {p95:.4}");
        assert!(p50 < 0.01 && p95 < 0.03, "{name} patch test: p50 {p50} p95 {p95}");
    }

    // uniaxial stress with a traction-free lateral surface: only the end
    // fragments are driven
    let nu = lib.elastic(asset.components[0].material).poisson;
    let uni = DMat3::from_diagonal(DVec3::new(-nu, 1.0, -nu) * 1e-4);
    let ends = |f: &Fragment| f.mass.com.y < 0.1 || f.mass.com.y > 1.1;
    let e: Vec<f64> = solver.patch_test_with(&asset, level, uni, Some(&ends)).into_iter().map(|x| x.1).collect();
    let (p50, p95) = (quantile(e.clone(), 0.5), quantile(e, 0.95));
    eprintln!("uniaxial, free lateral surface: p50 {p50:.4} p95 {p95:.4}");
    assert!(p50 < 0.01 && p95 < 0.04, "uniaxial patch test: p50 {p50} p95 {p95}");
}
