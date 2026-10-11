use frac_core::Settings;
use frac_core::input::{AuthoringMeta, InputPart, InputScene, PartMeta};
use frac_geom::DVec3;
use frac_geom::mesh::box_mesh;
use frac_material::MaterialLibrary;
use frac_pipeline::{InputSpec, run};

fn part(name: &str, mesh: frac_geom::TriMesh, material: &str) -> (InputPart, PartMeta) {
    let n = mesh.tris.len();
    (
        InputPart {
            name: name.into(),
            mesh,
            material_slot: vec![0; n],
            material_names: vec![material.into()],
            extras: serde_json::Value::Null,
            ..Default::default()
        },
        PartMeta {
            material: Some(material.into()),
            ..Default::default()
        },
    )
}

#[test]
fn column_end_to_end() {
    let (p, mut pm) = part(
        "column",
        box_mesh(DVec3::new(-0.2, 0.0, -0.2), DVec3::new(0.2, 3.0, 0.2)),
        "concrete_c30",
    );
    pm.anchor = true;
    let mut meta = AuthoringMeta::default();
    meta.parts.insert("column".into(), pm);
    let input = InputSpec {
        name: "column".into(),
        scene: InputScene {
            source: "test".into(),
            parts: vec![p],
        },
        meta,
        variant: 0,
    };
    let mut settings = Settings::default();
    settings.cells.target_analysis_cells_per_m3 = 60.0;
    settings.cells.fine_per_analysis = 6;
    settings.modes.target_level1_fragments = 4;
    settings.modes.max_tets = 3000;
    settings.modes.k = 4;
    let lib = MaterialLibrary::builtin();
    let t = std::time::Instant::now();
    let out = run(&input, &settings, &lib).expect("pipeline");
    eprintln!("elapsed {:?}", t.elapsed());
    eprintln!("{}", out.report.to_markdown());
    assert!(
        out.report.scorecard.all_pass(),
        "gate failures: {:?}",
        out.report.scorecard.failures()
    );
    assert!(!out.gltf.is_empty() && !out.physics.is_empty());
}
