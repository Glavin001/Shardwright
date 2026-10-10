//! `prefracture` — offline pre-fracture baking tool.

mod bench;
mod buildings;
mod meshgen;
mod preview;

use clap::{Parser, Subcommand};
use frac_core::input::AuthoringMeta;
use frac_core::Settings;
use frac_material::MaterialLibrary;
use frac_pipeline::{run, InputSpec};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "prefracture", version, about = "Pre-fracture any mesh into render + physics assets with exact bonds")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Bake an input mesh/assembly into glTF + physics payloads.
    Bake {
        #[arg(long)]
        input: PathBuf,
        /// Material library TOML (default: built-in library).
        #[arg(long)]
        materials: Option<PathBuf>,
        /// Bake settings TOML.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Authoring metadata sidecar (JSON or TOML). Default: <input>.meta.json if present.
        #[arg(long)]
        meta: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        /// Do not fail the bake on hard-gate failures.
        #[arg(long)]
        allow_gate_failures: bool,
        /// Bake twice and require byte-identical outputs.
        #[arg(long)]
        check_determinism: bool,
        /// Write intermediate geometry: cells, hulls, solids (comma separated).
        #[arg(long)]
        debug_dump: Option<String>,
        /// Override the seed.
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Re-validate a bake output and export bond-network results for the oracle harness.
    Validate {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        materials: Option<PathBuf>,
        #[arg(long)]
        oracle_cache: Option<PathBuf>,
        #[arg(long)]
        report: Option<PathBuf>,
        /// Network stiffness model: tensorial (default), calibrated or spec.
        #[arg(long, default_value = "tensorial")]
        stiffness_model: String,
        /// Freeze the computed FEM oracle into benchmarks/golden/<asset>.
        #[arg(long)]
        write_golden: bool,
        /// Ignore benchmarks/golden and always run the FEM oracle.
        #[arg(long)]
        no_golden: bool,
    },
    /// Inspect a physics payload.
    Inspect {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        level: Option<u8>,
        #[arg(long)]
        bond: Option<u32>,
        #[arg(long)]
        fragment: Option<u32>,
    },
    /// Byte and metric diff between two output directories.
    Diff {
        #[arg(long)]
        a: PathBuf,
        #[arg(long)]
        b: PathBuf,
    },
    /// Build the runtime pattern library (.fracpat).
    Patterns {
        #[arg(long)]
        materials: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 1337)]
        seed: u64,
    },
    /// Recompute the collision hulls of a baked asset (`.asset.json`) and
    /// write the asset with the new hulls (for decomposition experiments and
    /// the CoACD differential harness).
    Hulls {
        #[arg(long)]
        asset: PathBuf,
        /// Bake settings TOML (collision section used).
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        /// Concavity threshold in CoACD's normalized units (overrides `concavity`).
        #[arg(long)]
        coacd_threshold: Option<f64>,
        /// Hull budget per fragment (0 = unlimited).
        #[arg(long)]
        max_hulls: Option<usize>,
        /// Levels to compute (comma separated; default: settings).
        #[arg(long)]
        levels: Option<String>,
        /// Also report the validator's hull metrics per level (overshoot,
        /// count, fit on <= 100 sampled fragments) as JSON on stdout.
        #[arg(long)]
        metrics: bool,
    },
    /// Stand-alone convex decomposition of a closed mesh with the CoACD port
    /// (upstream default parameters unless overridden). Input/output JSON:
    /// `{"vertices": [[x,y,z],...], "faces": [[i,j,k],...]}` ->
    /// `{"hulls": [[[x,y,z],...],...], "concavity": [...], "seconds": t}`.
    Decompose {
        #[arg(long)]
        mesh: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 0.05)]
        threshold: f64,
        /// Hull limit (0 = none, merge only below the threshold).
        #[arg(long, default_value_t = 0)]
        max_convex_hull: usize,
        #[arg(long, default_value_t = 150)]
        mcts_iterations: u32,
        #[arg(long, default_value_t = 3)]
        mcts_depth: u32,
        #[arg(long, default_value_t = 20)]
        mcts_nodes: u32,
        #[arg(long, default_value_t = 2000)]
        resolution: u32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Merge cost: "upstream" (CoACD 1.0.x hull-vs-hull) or "collision"
        /// (collision-aware, against the input surface).
        #[arg(long, default_value = "upstream")]
        merge_cost: String,
        /// Skip the merge step (cut parts only).
        #[arg(long)]
        no_merge: bool,
        /// Hb estimator of the cut loop: "upstream" (10-nearest-sample
        /// triangles, as CoACD 1.0.x) or "exact".
        #[arg(long, default_value = "upstream")]
        hb: String,
    },
    /// Generate the procedural benchmark suite (spec §13.8).
    /// Render a PNG preview of a baked asset (one colour per fragment).
    Preview {
        /// Baked asset JSON (`<name>.asset.json`).
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Hierarchy level (default: finest); `--all-levels` renders every level side by side.
        #[arg(long)]
        level: Option<u8>,
        #[arg(long)]
        all_levels: bool,
        /// Push fragments apart from the centre by this fraction of their offset (e.g. 0.15).
        #[arg(long, default_value_t = 0.0)]
        explode: f64,
        #[arg(long, default_value_t = 1200)]
        width: u32,
        #[arg(long, default_value_t = 900)]
        height: u32,
        #[arg(long, default_value_t = 35.0)]
        azimuth: f64,
        #[arg(long, default_value_t = 25.0)]
        elevation: f64,
        /// Draw bonds (contact surfaces) instead of fragments: `kind` or `strength`.
        #[arg(long)]
        bonds: Option<String>,
        /// Cutaway: hide geometry whose centroid z is above this value.
        #[arg(long)]
        clip_z: Option<f64>,
    },
    GenBench {
        #[arg(long, default_value = "benchmarks/assets")]
        out: PathBuf,
        /// Only (re)generate these assets (e.g. building_v0); default: all.
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
    },
}

fn load_lib(p: &Option<PathBuf>) -> Result<MaterialLibrary, String> {
    match p {
        Some(p) => MaterialLibrary::from_toml(&std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?).map_err(|e| e.to_string()),
        None => Ok(MaterialLibrary::builtin()),
    }
}

fn load_meta(input: &Path, meta: &Option<PathBuf>) -> Result<AuthoringMeta, String> {
    let path = meta.clone().or_else(|| {
        let mut s = input.as_os_str().to_owned();
        s.push(".meta.json");
        let p = PathBuf::from(s);
        if p.exists() { Some(p) } else { None }
    });
    match path {
        None => Ok(AuthoringMeta::default()),
        Some(p) => {
            let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            if p.extension().and_then(|e| e.to_str()) == Some("toml") {
                AuthoringMeta::from_toml(&text).map_err(|e| e.to_string())
            } else {
                AuthoringMeta::from_json(&text).map_err(|e| e.to_string())
            }
        }
    }
}

fn hash(b: &[u8]) -> String {
    blake3::hash(b).to_hex().to_string()
}

#[allow(clippy::too_many_arguments)]
fn bake(
    input: &Path,
    materials: &Option<PathBuf>,
    config: &Option<PathBuf>,
    meta: &Option<PathBuf>,
    out: &Path,
    allow: bool,
    check_det: bool,
    dump: &Option<String>,
    seed: Option<u64>,
) -> Result<bool, String> {
    let lib = load_lib(materials)?;
    let mut settings = match config {
        Some(c) => Settings::from_toml(&std::fs::read_to_string(c).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?,
        None => Settings::default(),
    };
    if let Some(s) = seed {
        settings.seed = s;
    }
    let meta = load_meta(input, meta)?;
    let opts = frac_io::ImportOptions { unit_scale: settings.ingest.unit_scale, z_up: settings.ingest.z_up };
    let scene = frac_io::load_scene(input, &opts).map_err(|e| e.to_string())?;
    let name = input.file_stem().and_then(|s| s.to_str()).unwrap_or("asset").to_string();
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let mut all_ok = true;
    for variant in 0..settings.variants.max(1) {
        let vname = if settings.variants > 1 { format!("{name}_v{variant}") } else { name.clone() };
        let spec = InputSpec { name: vname.clone(), scene: scene.clone(), meta: meta.clone(), variant };
        let mut res = run(&spec, &settings, &lib).map_err(|e| e.to_string())?;
        if check_det {
            let again = run(&spec, &settings, &lib).map_err(|e| e.to_string())?;
            let same = again.gltf == res.gltf && again.physics == res.physics;
            if let Some(g) = res.report.scorecard.gates.iter_mut().find(|g| g.name == "determinism") {
                g.status = if same { frac_validate::GateStatus::Pass } else { frac_validate::GateStatus::Fail };
                g.detail = format!("two in-process runs: glb {} / {}, fracphys {} / {}", &hash(&res.gltf)[..12], &hash(&again.gltf)[..12], &hash(&res.physics)[..12], &hash(&again.physics)[..12]);
                g.value = if same { 0.0 } else { 1.0 };
            }
        }
        let glb = out.join(format!("{vname}.glb"));
        let phys = out.join(format!("{vname}.fracphys"));
        std::fs::write(&glb, &res.gltf).map_err(|e| e.to_string())?;
        std::fs::write(&phys, &res.physics).map_err(|e| e.to_string())?;
        // external schema validators (when available)
        let mut schema_notes = Vec::new();
        if let Some(r) = frac_io::khronos_validate(&glb) {
            match r {
                Ok(rep) => schema_notes.push(format!("Khronos glTF validator: {} errors, {} warnings", rep["issues"]["numErrors"], rep["issues"]["numWarnings"])),
                Err(e) => {
                    schema_notes.push(format!("Khronos glTF validator FAILED: {e}"));
                    if let Some(g) = res.report.scorecard.gates.iter_mut().find(|g| g.name == "schema") {
                        g.status = frac_validate::GateStatus::Fail;
                    }
                }
            }
        }
        if let Some(r) = frac_io::flatc_validate(&phys) {
            match r {
                Ok(()) => schema_notes.push("flatc schema decode OK".into()),
                Err(e) => {
                    schema_notes.push(format!("flatc decode FAILED: {e}"));
                    if let Some(g) = res.report.scorecard.gates.iter_mut().find(|g| g.name == "schema") {
                        g.status = frac_validate::GateStatus::Fail;
                    }
                }
            }
        }
        if let Some(g) = res.report.scorecard.gates.iter_mut().find(|g| g.name == "schema") {
            if !schema_notes.is_empty() {
                g.detail = format!("{}; {}", g.detail, schema_notes.join("; "));
            }
        }
        frac_io::write_asset_json(&res.asset, &out.join(format!("{vname}.asset.json"))).map_err(|e| e.to_string())?;
        std::fs::write(out.join(format!("{vname}.report.json")), res.report.to_json()).map_err(|e| e.to_string())?;
        std::fs::write(out.join(format!("{vname}.report.md")), res.report.to_markdown()).map_err(|e| e.to_string())?;
        if let Some(d) = dump {
            debug_dump(&res.asset, out, &vname, d)?;
        }
        let ok = res.report.scorecard.all_pass();
        println!(
            "{vname}: {} cells, {} fragments, {} bonds, {} hulls in {:.1} s — gates {}",
            res.asset.cells.len(),
            res.asset.hierarchy.fragments.len(),
            res.asset.bonds.len(),
            res.asset.hulls.len(),
            res.report.total_ms / 1e3,
            if ok { "PASS".to_string() } else { format!("FAIL ({})", res.report.scorecard.failures().iter().map(|g| g.name.clone()).collect::<Vec<_>>().join(", ")) }
        );
        all_ok &= ok;
    }
    Ok(all_ok || allow)
}

#[allow(clippy::too_many_arguments)]
fn decompose_cmd(mesh: &Path, out: &Path, threshold: f64, max_ch: usize, iters: u32, depth: u32, nodes: u32, resolution: u32, seed: u64, merge_cost: &str, merge: bool, hb: &str) -> Result<bool, String> {
    use frac_pipeline::frac_collision::coacd::{HbMode, MergeCost};
    let hb = match hb {
        "upstream" => HbMode::Upstream,
        "exact" => HbMode::Exact,
        o => return Err(format!("unknown hb mode {o}")),
    };
    let merge_cost = match merge_cost {
        "upstream" => MergeCost::Upstream,
        "collision" => MergeCost::CollisionAware,
        o => return Err(format!("unknown merge cost {o}")),
    };
    let text = std::fs::read_to_string(mesh).map_err(|e| format!("{}: {e}", mesh.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let verts: Vec<glam::DVec3> = v["vertices"]
        .as_array()
        .ok_or("missing vertices")?
        .iter()
        .map(|p| glam::DVec3::new(p[0].as_f64().unwrap_or(0.0), p[1].as_f64().unwrap_or(0.0), p[2].as_f64().unwrap_or(0.0)))
        .collect();
    let tris: Vec<[u32; 3]> = v["faces"]
        .as_array()
        .ok_or("missing faces")?
        .iter()
        .map(|t| [t[0].as_u64().unwrap_or(0) as u32, t[1].as_u64().unwrap_or(0) as u32, t[2].as_u64().unwrap_or(0) as u32])
        .collect();
    let m = frac_geom::TriMesh { verts, tris };
    let p = frac_pipeline::frac_collision::coacd::CoacdParams {
        threshold,
        mcts_iterations: iters,
        mcts_depth: depth,
        mcts_nodes: nodes,
        resolution,
        seed,
        max_convex_hull: if max_ch == 0 { None } else { Some(max_ch) },
        merge_cost,
        merge,
        hb,
        ..Default::default()
    };
    let t = std::time::Instant::now();
    let res = frac_pipeline::frac_collision::coacd::decompose_detailed(&m, &p);
    let secs = t.elapsed().as_secs_f64();
    let hulls: Vec<Vec<[f64; 3]>> = res.iter().map(|(h, _)| h.vertices().iter().map(|q| [q.x, q.y, q.z]).collect()).collect();
    let conc: Vec<f64> = res.iter().map(|x| x.1).collect();
    // the same hulls under the 64-vertex physics cap
    let capped: Vec<Vec<[f64; 3]>> = res
        .iter()
        .map(|(h, _)| frac_pipeline::frac_collision::coacd::limit_vertices(h, 64).vertices().iter().map(|q| [q.x, q.y, q.z]).collect())
        .collect();
    let doc = serde_json::json!({ "hulls": hulls, "hulls_capped64": capped, "concavity": conc, "seconds": secs });
    std::fs::write(out, doc.to_string()).map_err(|e| e.to_string())?;
    println!("{}: {} hulls in {secs:.2} s", out.display(), res.len());
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn hulls_cmd(asset: &Path, config: &Option<PathBuf>, out: &Path, coacd_threshold: Option<f64>, max_hulls: Option<usize>, levels: &Option<String>, metrics: bool) -> Result<bool, String> {
    let settings = match config {
        Some(c) => Settings::from_toml(&std::fs::read_to_string(c).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?,
        None => Settings::default(),
    };
    let text = std::fs::read_to_string(asset).map_err(|e| format!("{}: {e}", asset.display()))?;
    let mut a: frac_core::Asset = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", asset.display()))?;
    let mut cp = frac_pipeline::collision_params(&settings);
    cp.coacd_threshold = coacd_threshold;
    if let Some(m) = max_hulls {
        cp.max_hulls = if m == 0 { usize::MAX } else { m };
    }
    if let Some(l) = levels {
        cp.levels = l.split(',').filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse::<u8>().map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
    }
    let t = std::time::Instant::now();
    // `FRAC_HULLS_KEEP`: evaluate the asset's own hulls (no recompute)
    if std::env::var_os("FRAC_HULLS_KEEP").is_none() {
        let (hulls, ranges) = frac_pipeline::build_hulls(&a, &cp);
        a.hulls = hulls;
        for (i, r) in ranges.into_iter().enumerate() {
            a.hierarchy.fragments[i].hulls = r;
        }
    }
    let secs = t.elapsed().as_secs_f64();
    if let Some(l) = std::fs::read_to_string("/proc/self/status").ok().and_then(|s| s.lines().find(|l| l.starts_with("VmHWM")).map(str::to_string)) {
        eprintln!("collision: {secs:.1} s, peak {}", l.split_whitespace().skip(1).collect::<Vec<_>>().join(" "));
    }
    // content hash of the hulls (determinism checks)
    let mut hasher = blake3::Hasher::new();
    for f in &a.hierarchy.fragments {
        hasher.update(&f.hulls.start.to_le_bytes());
        hasher.update(&f.hulls.end.to_le_bytes());
    }
    for h in &a.hulls {
        for v in &h.vertices {
            for c in [v.x, v.y, v.z] {
                hasher.update(&c.to_bits().to_le_bytes());
            }
        }
        for f in &h.faces {
            for &i in f {
                hasher.update(&i.to_le_bytes());
            }
            hasher.update(&u32::MAX.to_le_bytes());
        }
    }
    eprintln!("hulls hash: {}", hasher.finalize().to_hex());
    // non-overlap check (as the hard gate)
    let polys: Vec<_> = a.hulls.iter().map(frac_pipeline::hull_polytope).collect();
    let mut worst: f64 = 0.0;
    let mut pairs = 0usize;
    for b in &a.bonds {
        let frac_core::FragmentOrWorld::Fragment(fb) = b.b else { continue };
        for x in a.hierarchy.fragments[b.a.idx()].hulls.clone() {
            for y in a.hierarchy.fragments[fb.idx()].hulls.clone() {
                pairs += 1;
                worst = worst.max(polys[x as usize].intersection_volume(&polys[y as usize]));
            }
        }
    }
    // `--out -`: no output asset
    if out != Path::new("-") {
        frac_io::write_asset_json(&a, out).map_err(|e| e.to_string())?;
    }
    if metrics {
        println!("{}", serde_json::to_string(&hull_metrics(&a, &polys)).map_err(|e| e.to_string())?);
    }
    // collision-shape limits (as the collision_shapes gate)
    let nonconvex = a.hulls.iter().filter(|h| !frac_pipeline::frac_collision::coacd::stored_hull_is_convex(&h.vertices, &h.faces)).count();
    let maxv = a.hulls.iter().map(|h| h.vertices.len()).max().unwrap_or(0);
    println!("collision shapes: {nonconvex} non-convex, max {maxv} vertices");
    println!("{}: {} hulls in {secs:.2} s; max neighbour hull overlap {worst:.3e} m3 over {pairs} pairs ({})", out.display(), a.hulls.len(), if worst <= 1e-9 { "PASS" } else { "FAIL" });
    Ok(worst <= 1e-9)
}

/// The validator's hull metrics per level: overshoot (Σ hull volume / V -
/// 1) and hull count over all fragments with hulls, and the symmetric fit
/// deviation / diameter on <= 100 evenly strided fragments (same sampling as
/// `frac-validate`).
fn hull_metrics(a: &frac_core::Asset, polys: &[frac_geom::hull::ConvexPolytope]) -> serde_json::Value {
    use frac_geom::inside::MeshQuery;
    use rayon::prelude::*;
    let q = |mut v: Vec<f64>| -> serde_json::Value {
        v.sort_by(|a, b| a.total_cmp(b));
        let at = |p: f64| v.get(((v.len() as f64 - 1.0) * p).round().max(0.0) as usize).copied().unwrap_or(0.0);
        serde_json::json!({"n": v.len(), "p50": at(0.5), "p95": at(0.95), "max": at(1.0), "mean": if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }})
    };
    let pdist = |x: frac_geom::DVec3, p: &frac_geom::hull::ConvexPolytope| -> f64 {
        let tol = 1e-12 * p.scale().max(1e-300);
        if p.faces.iter().all(|(h, _)| h.dist(x) <= tol) {
            return 0.0;
        }
        let mut best = f64::INFINITY;
        for (h, poly) in &p.faces {
            let dn = h.dist(x);
            if dn <= 0.0 || poly.len() < 3 {
                continue;
            }
            let qp = x - h.n * dn;
            if (0..poly.len()).all(|i| (poly[(i + 1) % poly.len()] - poly[i]).cross(qp - poly[i]).dot(h.n) >= -tol) {
                best = best.min(dn);
                continue;
            }
            for i in 0..poly.len() {
                let (s, e) = (poly[i], poly[(i + 1) % poly.len()]);
                let se = e - s;
                let t = ((x - s).dot(se) / se.length_squared().max(1e-300)).clamp(0.0, 1.0);
                best = best.min((s + se * t - x).length());
            }
        }
        if best.is_finite() { best } else { 0.0 }
    };
    let h = &a.hierarchy;
    let mut levels = Vec::new();
    for l in 0..h.levels {
        let frs = a.level_fragments(l);
        let with: Vec<&frac_core::Fragment> = frs.iter().filter(|f| !f.hulls.is_empty()).collect();
        let over: Vec<f64> = with.iter().map(|f| f.hulls.clone().map(|i| polys[i as usize].volume()).sum::<f64>() / f.mass.volume.max(1e-300) - 1.0).collect();
        let count: Vec<f64> = with.iter().map(|f| f.hulls.len() as f64).collect();
        let fit: Vec<(u32, f64)> = frs
            .par_iter()
            .step_by(if std::env::var_os("FRAC_FIT_ALL").is_some() { 1 } else { (frs.len() / 100).max(1) })
            .filter(|f| !f.hulls.is_empty())
            .map(|f| {
                let m = frac_pipeline::frac_collision::cells_boundary_mesh(a, a.fragment_cells(f));
                let mq = MeshQuery::new(&m);
                let diam = m.aabb().diagonal().max(1e-300);
                let hp = &polys[f.hulls.start as usize..f.hulls.end as usize];
                let mut worst: f64 = 0.0;
                for hh in &a.hulls[f.hulls.start as usize..f.hulls.end as usize] {
                    for v in &hh.vertices {
                        if !mq.contains(*v) {
                            if let Some((_, d2, _)) = mq.closest_point(*v) {
                                worst = worst.max(d2.sqrt());
                            }
                        }
                    }
                }
                let cents = m.tris.iter().map(|t| (m.verts[t[0] as usize] + m.verts[t[1] as usize] + m.verts[t[2] as usize]) / 3.0);
                for x in m.verts.iter().copied().chain(cents) {
                    let d = hp.iter().map(|p| pdist(x, p)).fold(f64::INFINITY, f64::min);
                    if d.is_finite() {
                        worst = worst.max(d);
                    }
                }
                (f.id.0, worst / diam)
            })
            .collect();
        // worst fragments (for diagnosis)
        let mut top = fit.clone();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top: Vec<(u32, f64)> = top.into_iter().take(5).collect();
        let fit: Vec<f64> = fit.into_iter().map(|x| x.1).collect();
        levels.push(serde_json::json!({"level": l, "fragments": frs.len(), "hulls": count.iter().sum::<f64>(), "hull_overshoot": q(over), "hull_count": q(count), "hull_fit": q(fit), "worst_fit": top}));
    }
    serde_json::json!({ "levels": levels })
}

fn debug_dump(asset: &frac_core::Asset, out: &Path, name: &str, what: &str) -> Result<(), String> {
    let dir = out.join(format!("{name}_debug"));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    for w in what.split(',') {
        match w.trim() {
            "cells" => {
                for c in &asset.components {
                    let g = &c.geometry;
                    let mut polys: Vec<Vec<u32>> = Vec::new();
                    let mut colors = Vec::new();
                    let col = |id: u32| {
                        let h = frac_core::stable_hash(&[id as u64]);
                        [(h & 255) as u8, ((h >> 8) & 255) as u8, ((h >> 16) & 255) as u8]
                    };
                    for e in &g.ext_polys {
                        polys.push(e.verts.clone());
                        colors.push(col(e.cell.0));
                    }
                    for p in &g.patches {
                        for t in &p.tris {
                            polys.push(t.to_vec());
                            colors.push(col(p.cells.0 .0));
                        }
                    }
                    frac_io::write_ply_polys(dir.join(format!("cells_{}.ply", c.name)), &g.verts, &polys, Some(&colors)).map_err(|e| e.to_string())?;
                }
            }
            "solids" => {
                for c in &asset.components {
                    frac_io::write_ply(dir.join(format!("solid_{}.ply", c.name)), &c.solid).map_err(|e| e.to_string())?;
                }
            }
            "hulls" => {
                let mut m = frac_geom::TriMesh::default();
                for h in &asset.hulls {
                    let mut hm = frac_geom::TriMesh { verts: h.vertices.clone(), tris: Vec::new() };
                    for f in &h.faces {
                        for k in 1..f.len() - 1 {
                            hm.tris.push([f[0], f[k], f[k + 1]]);
                        }
                    }
                    m.append(&hm);
                }
                frac_io::write_ply(dir.join("hulls.ply"), &m).map_err(|e| e.to_string())?;
            }
            other => return Err(format!("unknown debug dump stage '{other}' (cells, solids, hulls)")),
        }
    }
    Ok(())
}

fn validate_cmd(input: &Path, materials: &Option<PathBuf>, cache: &Option<PathBuf>, report: &Option<PathBuf>, model: &str, write_golden: bool, no_golden: bool) -> Result<bool, String> {
    let model = match model {
        "spec" => frac_validate::network::StiffnessModel::Spec,
        "calibrated" => frac_validate::network::StiffnessModel::Calibrated,
        _ => frac_validate::network::StiffnessModel::Tensorial,
    };
    let lib = load_lib(materials)?;
    let files: Vec<PathBuf> = if input.is_dir() {
        let mut v: Vec<PathBuf> = std::fs::read_dir(input).map_err(|e| e.to_string())?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.to_string_lossy().ends_with(".asset.json")).collect();
        v.sort();
        v
    } else {
        vec![input.to_path_buf()]
    };
    let mut md = String::from("# Validation\n\n");
    for f in files {
        let text = std::fs::read_to_string(&f).map_err(|e| e.to_string())?;
        let asset: frac_core::Asset = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", f.display()))?;
        let net = frac_pipeline::oracle::network_export_with(&asset, &lib, model);
        let base = f.to_string_lossy().trim_end_matches(".asset.json").to_string();
        let net_path = PathBuf::from(format!("{base}.network.json"));
        std::fs::write(&net_path, serde_json::to_string_pretty(&net).unwrap()).map_err(|e| e.to_string())?;
        md += &format!("## {}\n\nBond-network results written to `{}`.\n\n", asset.meta.name, net_path.display());
        // the oracle runs when a cache is given, or when a golden oracle set
        // exists for this asset (then only numpy/scipy are needed)
        let golden = frac_pipeline::oracle::golden_dir(&asset.meta.name).is_some_and(|d| d.join("manifest.json").exists());
        let tmp = std::env::temp_dir().join("prefracture-oracle");
        let c = cache.as_deref().or(if golden && !no_golden { Some(tmp.as_path()) } else { None });
        if let Some(c) = c {
            let mut extra = Vec::new();
            if write_golden {
                extra.push("--write-golden");
            }
            if no_golden {
                extra.push("--no-golden");
            }
            md += &frac_pipeline::oracle::run_harness_with(&f, &net_path, c, &extra);
        }
    }
    if let Some(r) = report {
        std::fs::write(r, &md).map_err(|e| e.to_string())?;
    } else {
        println!("{md}");
    }
    Ok(true)
}

fn inspect(input: &Path, level: Option<u8>, bond: Option<u32>, frag: Option<u32>) -> Result<bool, String> {
    let bytes = std::fs::read(input).map_err(|e| e.to_string())?;
    let json = frac_io::physics_to_json(&bytes).map_err(|e| e.to_string())?;
    let v: serde_json::Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
    if let Some(b) = bond {
        println!("{}", serde_json::to_string_pretty(&v["bonds"][b as usize]).unwrap());
    } else if let Some(f) = frag {
        println!("{}", serde_json::to_string_pretty(&v["fragments"][f as usize]).unwrap());
    } else if let Some(l) = level {
        let fr: Vec<&serde_json::Value> = v["fragments"].as_array().map(|a| a.iter().filter(|x| x["level"].as_u64() == Some(l as u64)).collect()).unwrap_or_default();
        let bo: Vec<&serde_json::Value> = v["bonds"].as_array().map(|a| a.iter().filter(|x| x["level"].as_u64() == Some(l as u64)).collect()).unwrap_or_default();
        println!("level {l}: {} fragments, {} bonds", fr.len(), bo.len());
        let total_area: f64 = bo.iter().filter_map(|b| b["area"].as_f64()).sum();
        let total_mass: f64 = fr.iter().filter_map(|f| f["mass"].as_f64()).sum();
        println!("  total mass {total_mass:.4} kg, total bond area {total_area:.6} m²");
    } else {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": v["schema_version"], "tool_version": v["tool_version"], "levels": v["levels"],
            "fragments": v["fragments"].as_array().map(|a| a.len()), "bonds": v["bonds"].as_array().map(|a| a.len()),
            "hulls": v["hulls"].as_array().map(|a| a.len())
        })).unwrap());
    }
    Ok(true)
}

fn diff(a: &Path, b: &Path) -> Result<bool, String> {
    let list = |d: &Path| -> Result<Vec<String>, String> {
        let mut v: Vec<String> = std::fs::read_dir(d).map_err(|e| e.to_string())?.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().to_string()).filter(|n| n.ends_with(".glb") || n.ends_with(".fracphys") || n.ends_with(".report.json")).collect();
        v.sort();
        Ok(v)
    };
    let fa = list(a)?;
    let mut identical = true;
    for name in &fa {
        let pa = std::fs::read(a.join(name)).map_err(|e| e.to_string())?;
        let pb = std::fs::read(b.join(name)).unwrap_or_default();
        let same = pa == pb;
        if name.ends_with(".report.json") {
            let ra: serde_json::Value = serde_json::from_slice(&pa).unwrap_or_default();
            let rb: serde_json::Value = serde_json::from_slice(&pb).unwrap_or_default();
            println!("{name}:");
            for (k, va) in ra["levels"].as_array().into_iter().flatten().enumerate() {
                let vb = &rb["levels"][k];
                println!("  L{k}: fragments {} -> {}, bonds {} -> {}, hulls {} -> {}", va["fragments"], vb["fragments"], va["bonds"], vb["bonds"], va["hulls"], vb["hulls"]);
            }
            println!("  total ms {} -> {}", ra["total_ms"], rb["total_ms"]);
        } else {
            println!("{name}: {}", if same { "identical".into() } else { format!("DIFFERENT ({} vs {})", &hash(&pa)[..12], &hash(&pb)[..12]) });
            identical &= same;
        }
    }
    Ok(identical)
}

fn main() -> ExitCode {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).with_writer(std::io::stderr).init();
    let cli = Cli::parse();
    let res = match &cli.cmd {
        Cmd::Bake { input, materials, config, meta, out, allow_gate_failures, check_determinism, debug_dump, seed } => {
            bake(input, materials, config, meta, out, *allow_gate_failures, *check_determinism, debug_dump, *seed)
        }
        Cmd::Validate { input, materials, oracle_cache, report, stiffness_model, write_golden, no_golden } => validate_cmd(input, materials, oracle_cache, report, stiffness_model, *write_golden, *no_golden),
        Cmd::Inspect { input, level, bond, fragment } => inspect(input, *level, *bond, *fragment),
        Cmd::Diff { a, b } => diff(a, b),
        Cmd::Patterns { materials, out, seed } => load_lib(materials).and_then(|lib| {
            let bytes = frac_patterns::build_library(&lib, *seed).map_err(|e| e.to_string())?;
            std::fs::write(out, &bytes).map_err(|e| e.to_string())?;
            println!("wrote {} ({} bytes)", out.display(), bytes.len());
            Ok(true)
        }),
        Cmd::Preview { input, out, level, all_levels, explode, width, height, azimuth, elevation, bonds, clip_z } => (|| -> Result<bool, String> {
            let text = std::fs::read_to_string(input).map_err(|e| format!("{}: {e}", input.display()))?;
            let asset: frac_core::Asset = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", input.display()))?;
            let bonds = match bonds.as_deref() {
                None => None,
                Some("kind") => Some(preview::BondColour::Kind),
                Some("strength") => Some(preview::BondColour::Strength),
                Some(x) => return Err(format!("--bonds {x}: expected kind or strength")),
            };
            let o = preview::PreviewOptions { bonds, clip_z: *clip_z, level: *level, all_levels: *all_levels, width: *width, height: *height, explode: *explode, azimuth_deg: *azimuth, elevation_deg: *elevation };
            println!("{}", preview::preview(&asset, &MaterialLibrary::builtin(), out, &o)?);
            Ok(true)
        })(),
        Cmd::Hulls { asset, config, out, coacd_threshold, max_hulls, levels, metrics } => hulls_cmd(asset, config, out, *coacd_threshold, *max_hulls, levels, *metrics),
        Cmd::Decompose { mesh, out, threshold, max_convex_hull, mcts_iterations, mcts_depth, mcts_nodes, resolution, seed, merge_cost, no_merge, hb } => {
            decompose_cmd(mesh, out, *threshold, *max_convex_hull, *mcts_iterations, *mcts_depth, *mcts_nodes, *resolution, *seed, merge_cost, !*no_merge, hb)
        }
        Cmd::GenBench { out, only } => bench::generate(out, only).map(|_| true),
    };
    match res {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(2),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
