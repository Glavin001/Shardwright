//! Times render (stage 9), the physics payload and validation on a baked
//! asset, without re-running cells, bonds and collision.
//! Usage: cargo run --release -p frac-pipeline --example render_bench -- <X.asset.json> <bake.toml> [--no-validate]
use frac_core::{Asset, Settings};
use frac_material::MaterialLibrary;
use std::time::Instant;

/// Process CPU time (user + system, seconds) from /proc (Linux).
fn cpu_s() -> f64 {
    let st = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let f: Vec<&str> = st.rsplit(')').next().unwrap_or("").split_whitespace().collect();
    let tick = |i: usize| f.get(i).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
    (tick(11) + tick(12)) / 100.0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t = Instant::now();
    let mut asset: Asset = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let settings = Settings::from_toml(&std::fs::read_to_string(&args[2]).unwrap()).unwrap();
    let lib = MaterialLibrary::builtin();
    eprintln!("load: {:.1} s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let c = cpu_s();
    let render = frac_pipeline::render_asset(&asset, &settings, &lib);
    eprintln!("render cpu: {:.1} s", cpu_s() - c);
    let tris: usize = render.fragments.iter().flat_map(|l| l.iter()).map(|m| m.triangle_count()).sum();
    eprintln!("render: {:.1} s ({} triangles over all LODs)", t.elapsed().as_secs_f64(), tris);
    for l in 0..asset.hierarchy.levels {
        let fr = asset.level_fragments(l);
        let per_lod: Vec<usize> = (0..settings.render.lods.max(1) as usize).map(|k| fr.iter().map(|f| render.fragments[f.id.idx()].get(k).map(|m| m.triangle_count()).unwrap_or(0)).sum()).collect();
        let (ext, int): (usize, usize) = fr.iter().map(|f| render.fragments[f.id.idx()].first().map(|m| (m.ext_indices.len() / 3, m.int_indices.len() / 3)).unwrap_or((0, 0))).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        eprintln!("  level {l}: {} fragments, triangles per LOD {per_lod:?} (LOD0 exterior {ext}, interior {int})", fr.len());
    }
    if args.iter().any(|a| a == "--no-validate") {
        return;
    }
    let t = Instant::now();
    let c = cpu_s();
    frac_pipeline::export::render_refs(&mut asset, &render);
    let physics = frac_io::write_physics(&asset);
    let sc = frac_validate::validate(&asset, &render, &lib, &settings.validation, &physics);
    eprintln!("validate: {:.1} s (cpu {:.1} s)", t.elapsed().as_secs_f64(), cpu_s() - c);
    for g in &sc.gates {
        eprintln!("  {} {:?} {}", g.name, g.status, g.detail);
    }
}
