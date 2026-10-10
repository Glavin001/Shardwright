//! Render-validity diagnostics: bakes an asset and reports, for every
//! fragment whose LOD0 render mesh self-intersects, the offending triangles
//! (exterior/interior, owning cell, displacement from the clean geometry).
//! Usage: cargo run --release -p frac-pipeline --example render_diag -- <asset.glb> <bake.toml>
use frac_core::input::AuthoringMeta;
use frac_core::Settings;
use frac_material::MaterialLibrary;
use frac_pipeline::{run, InputSpec};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = std::path::PathBuf::from(&args[1]);
    let settings = Settings::from_toml(&std::fs::read_to_string(&args[2]).unwrap()).unwrap();
    let meta_path = format!("{}.meta.json", args[1]);
    let meta = if std::path::Path::new(&meta_path).exists() { AuthoringMeta::from_json(&std::fs::read_to_string(&meta_path).unwrap()).unwrap() } else { AuthoringMeta::default() };
    let opts = frac_io::ImportOptions { unit_scale: settings.ingest.unit_scale, z_up: settings.ingest.z_up };
    let scene = frac_io::load_scene(&input, &opts).unwrap();
    let name = input.file_stem().and_then(|s| s.to_str()).unwrap_or("asset").to_string();
    let spec = InputSpec { name, scene, meta, variant: 0 };
    let out = run(&spec, &settings, &MaterialLibrary::builtin()).unwrap();
    for (fi, lods) in out.render.fragments.iter().enumerate() {
        let m = &lods[0];
        let tm = m.as_trimesh();
        let si: Vec<_> = tm.self_intersections(8).into_iter().filter(|&(a, b)| !tm.is_degenerate(a as usize) && !tm.is_degenerate(b as usize)).collect();
        if si.is_empty() {
            continue;
        }
        let f = &out.asset.hierarchy.fragments[fi];
        let next = m.ext_indices.len() / 3;
        eprintln!("fragment {fi} level {} cells {:?} ext_tris {next} total {} pairs {:?}", f.level, f.cells, tm.tris.len(), si);
        for &(a, b) in si.iter().take(2) {
            for t in [a, b] {
                let p = tm.tri_points(t as usize);
                let kind = if (t as usize) < next { "ext" } else { "int" };
                eprintln!("   tri {t} {kind} {:?}", p);
            }
        }
    }
}
