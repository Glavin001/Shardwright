//! Minimal library use: bake a mesh, write the payloads, walk the result.
//! Usage: cargo run --release -p frac-pipeline --example bake -- \
//!     benchmarks/assets/rc_column.glb benchmarks/configs/bake.toml out/
use frac_core::input::AuthoringMeta;
use frac_core::{FragmentOrWorld, Settings};
use frac_material::MaterialLibrary;
use frac_pipeline::{InputSpec, run};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (input, config, out) = (Path::new(&args[1]), &args[2], Path::new(&args[3]));

    let settings = Settings::from_toml(&std::fs::read_to_string(config)?)?;
    let lib = MaterialLibrary::builtin();
    // authoring metadata (materials, anchors, rebar): `<input>.meta.json`
    let meta_path = format!("{}.meta.json", input.display());
    let meta = match std::fs::read_to_string(&meta_path) {
        Ok(text) => AuthoringMeta::from_json(&text)?,
        Err(_) => AuthoringMeta::default(),
    };
    let opts = frac_io::ImportOptions {
        unit_scale: settings.ingest.unit_scale,
        z_up: settings.ingest.z_up,
    };
    let spec = InputSpec {
        name: input.file_stem().unwrap().to_string_lossy().into(),
        scene: frac_io::load_scene(input, &opts)?,
        meta,
        variant: 0,
    };

    let res = run(&spec, &settings, &lib)?;
    std::fs::create_dir_all(out)?;
    std::fs::write(out.join(format!("{}.glb", spec.name)), &res.gltf)?;
    std::fs::write(out.join(format!("{}.fracphys", spec.name)), &res.physics)?;

    let a = &res.asset;
    for level in 0..a.hierarchy.levels {
        let frags = a.hierarchy.fragments.iter().filter(|f| f.level == level);
        let bonds: Vec<_> = a.bonds.iter().filter(|b| b.level == level).collect();
        let anchors = bonds
            .iter()
            .filter(|b| matches!(b.b, FragmentOrWorld::World))
            .count();
        println!(
            "L{level}: {} fragments, {} bonds ({anchors} to the ground)",
            frags.count(),
            bonds.len()
        );
    }
    let failed: Vec<_> = res
        .report
        .scorecard
        .gates
        .iter()
        .filter(|g| g.status == frac_validate::GateStatus::Fail)
        .map(|g| g.name.as_str())
        .collect();
    println!("hard gates failed: {failed:?}");
    Ok(())
}
