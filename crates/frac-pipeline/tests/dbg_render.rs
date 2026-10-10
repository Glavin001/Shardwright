use frac_core::input::{AuthoringMeta, InputPart, InputScene, PartMeta};
use frac_core::Settings;
use frac_geom::mesh::box_mesh;
use frac_geom::DVec3;
use frac_material::MaterialLibrary;
use frac_pipeline::{run_with, InputSpec};

#[test]
#[ignore]
fn dbg_render_si() {
    let mesh = box_mesh(DVec3::new(-0.2, 0.0, -0.2), DVec3::new(0.2, 3.0, 0.2));
    let n = mesh.tris.len();
    let p = InputPart { name: "column".into(), mesh, material_slot: vec![0; n], material_names: vec!["concrete_c30".into()], extras: serde_json::Value::Null, ..Default::default() };
    let mut meta = AuthoringMeta::default();
    meta.parts.insert("column".into(), PartMeta { material: Some("concrete_c30".into()), anchor: true, ..Default::default() });
    let input = InputSpec { name: "column".into(), scene: InputScene { source: "t".into(), parts: vec![p] }, meta, variant: 0 };
    let mut s = Settings::default();
    s.cells.target_analysis_cells_per_m3 = 60.0;
    s.cells.fine_per_analysis = 6;
    s.modes.enabled = false;
    if std::env::var("NO_NOISE").is_ok() { s.render.noise = false; }
    if std::env::var("NO_CHIP").is_ok() { s.render.chipping = false; }
    let out = run_with(&input, &s, &MaterialLibrary::builtin(), true).unwrap();
    for (fi, lods) in out.render.as_ref().unwrap().fragments.iter().enumerate() {
        let m = &lods[0];
        let tm = m.as_trimesh();
        let si = tm.self_intersections(4);
        let si: Vec<_> = si.into_iter().filter(|&(a, b)| !tm.is_degenerate(a as usize) && !tm.is_degenerate(b as usize)).collect();
        if si.is_empty() { continue; }
        let (a, b) = si[0];
        let nt_ext = m.ext_indices.len() / 3;
        eprintln!("frag {fi} level {} pair {a} {b} ext_tris {nt_ext} total {}", out.asset.hierarchy.fragments[fi].level, tm.tris.len());
        for t in [a, b] {
            let p = tm.tri_points(t as usize);
            let nrm = (p[1]-p[0]).cross(p[2]-p[0]);
            eprintln!("  tri {t} {:?} n {:?} area {:e}", p, nrm.normalize(), nrm.length()*0.5);
        }
    }
}
