//! Stage orchestration (spec §7): ingest → components → cell complex →
//! (analysis mesh → fracture modes) → hierarchy → interfaces & bonds → mass
//! → collision → render → validation → export.
//!
//! Every stage is a pure function of its typed inputs; per-component work
//! runs in parallel with results collected in index order, so outputs are
//! deterministic.

pub mod assemble;
pub mod export;
pub mod level1;
pub mod manifold;
pub mod oracle;
pub mod report;

use assemble::{add_anchors, add_contacts, assemble, asset_bbox, BuiltComponent};
use frac_cells::cellset::{CellSet, CellSetParams};
use frac_cells::recipes::{build_cells, CellParams, MasonryLayout, Recipe};
use frac_core::input::{AuthoringMeta, InputScene};
use frac_core::*;
use frac_geom::DVec3;
use frac_material::MaterialLibrary;
use rayon::prelude::*;
use report::{Report, StageTiming};
use std::time::Instant;

/// Pipeline input.
#[derive(Clone, Debug, Default)]
pub struct InputSpec {
    pub name: String,
    pub scene: InputScene,
    pub meta: AuthoringMeta,
    pub variant: u32,
}

pub struct PipelineOutput {
    pub asset: Asset,
    pub gltf: Vec<u8>,
    pub physics: Vec<u8>,
    pub report: Report,
    /// Render meshes, kept only by [`run_with`] with `keep_render`.
    pub render: Option<frac_render::RenderOut>,
}

fn recipe_for(lib: &MaterialLibrary, m: MaterialId, meta: &frac_core::input::PartMeta) -> Recipe {
    let mat = lib.material(m);
    let name = meta.recipe.clone().or_else(|| mat.recipe.clone()).unwrap_or_else(|| "clustered_voronoi".into());
    let mut r = Recipe::from_name(&name).unwrap_or(Recipe::ClusteredVoronoi);
    match &mut r {
        Recipe::Masonry(layout) => {
            if let Some(v) = meta.masonry_layout.clone().or_else(|| mat.masonry_layout.clone()) {
                if let Ok(l) = serde_json::from_value::<MasonryLayout>(v) {
                    *layout = l;
                }
            }
        }
        Recipe::Wood { stretch, fine_stretch } => {
            if let Some(gs) = mat.grain_stretch {
                *stretch = gs;
                *fine_stretch = gs * 1.5;
            }
        }
        Recipe::GlassAnnealed { impact, .. } => {
            if let Some(c) = meta.impact_center {
                *impact = Some(c);
            }
        }
        _ => {}
    }
    r
}

/// Run the full pipeline for one variant.
pub fn run(input: &InputSpec, settings: &Settings, lib: &MaterialLibrary) -> Result<PipelineOutput, FracError> {
    run_with(input, settings, lib, false)
}

/// [`run`], optionally keeping the f64 render meshes in the output (for
/// diagnostics; they are dropped by default to bound peak memory).
pub fn run_with(input: &InputSpec, settings: &Settings, lib: &MaterialLibrary, keep_render: bool) -> Result<PipelineOutput, FracError> {
    let mut timings: Vec<StageTiming> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let t0 = Instant::now();
    let log = std::env::var_os("FRAC_LOG").is_some();
    let tick = |name: &str, t: &mut Instant, timings: &mut Vec<StageTiming>| {
        timings.push(StageTiming { stage: name.into(), ms: t.elapsed().as_secs_f64() * 1e3 });
        if log {
            eprintln!("[{}] {name}: {:.1} s (total {:.1} s), rss {:.0} MB", input.name, t.elapsed().as_secs_f64(), t0.elapsed().as_secs_f64(), rss_mb());
        }
        *t = Instant::now();
    };
    let mut t = Instant::now();
    // ---- Stage 0/1: ingest
    let ing_settings = frac_ingest::IngestSettings {
        weld_tolerance: settings.ingest.weld_tolerance,
        solidify_resolution: settings.ingest.solidify_resolution,
        contact_tolerance: settings.ingest.contact_tolerance,
    };
    let ing = frac_ingest::ingest(&input.scene, &input.meta, lib, &ing_settings);
    warnings.extend(ing.warnings.iter().cloned());
    tick("ingest", &mut t, &mut timings);
    // keep only parts that ingested; map indices
    let ok_parts: Vec<usize> = (0..ing.parts.len()).filter(|&i| ing.parts[i].failed.is_none()).collect();
    if ok_parts.is_empty() {
        return Err(FracError::new(Stage::Ingest, input.name.clone(), "no valid parts"));
    }
    let part_index: std::collections::BTreeMap<usize, usize> = ok_parts.iter().enumerate().map(|(k, &i)| (i, k)).collect();
    // ---- Stage 2: cells per component (parallel)
    let built: Vec<(BuiltComponent, Vec<String>)> = ok_parts
        .par_iter()
        .enumerate()
        .map(|(ci, &pi)| {
            let p = &ing.parts[pi];
            let mat = lib.material(p.material);
            let mut w = Vec::new();
            let recipe = recipe_for(lib, p.material, &p.meta);
            let density = p.meta.analysis_cells_per_m3.or(mat.analysis_cells_per_m3).unwrap_or(settings.cells.target_analysis_cells_per_m3);
            let na = ((p.volume * density).round() as usize).clamp(settings.cells.min_analysis_cells as usize, settings.cells.max_analysis_cells as usize);
            let fpa = p.meta.fine_per_analysis.or(mat.fine_per_analysis).unwrap_or(settings.cells.fine_per_analysis) as usize;
            let grain = p.meta.grain.map(DVec3::from_array).or_else(|| {
                if mat.anisotropy.is_some() || matches!(recipe, Recipe::Wood { .. }) {
                    let (_, axes, _) = frac_cells::recipes::principal_axes(&p.solid);
                    Some(axes.col(2))
                } else {
                    None
                }
            });
            let boxes = p.meta.density_boxes.clone();
            let spacing = move |x: DVec3| -> f64 {
                let mut f = 1.0f64;
                for (b, factor) in &boxes {
                    if b.contains(x) {
                        f = f.max(*factor);
                    }
                }
                f.powf(-1.0 / 3.0)
            };
            let has_density = !p.meta.density_boxes.is_empty();
            let params = CellParams {
                recipe: recipe.clone(),
                analysis_target: na,
                fine_per_analysis: fpa,
                max_fine: settings.cells.max_fine_cells as usize,
                asset_seed: settings.seed,
                component: ci as u32,
                variant: input.variant,
                grain,
                min_cell_volume: settings.cells.min_cell_volume,
                min_thickness_ratio: settings.cells.min_thickness_ratio,
                spacing: if has_density { Some(&spacing) } else { None },
            };
            let mut unfractured = false;
            let mut res = build_cells(&p.solid, &params);
            if let Err(e) = &res {
                w.push(format!("component '{}': cell construction failed ({e}); exported unfractured", p.name));
                unfractured = true;
                let single = CellParams { recipe: Recipe::Steel, ..params };
                res = build_cells(&p.solid, &single);
            }
            let mut cells: CellSet = match res {
                Ok(b) => b.cells,
                Err(e) => {
                    w.push(format!("component '{}': single-cell fallback failed: {e}", p.name));
                    CellSet::default()
                }
            };
            // forbidden zones: merge the cells inside each zone
            if !p.meta.forbidden_zones.is_empty() {
                let mut groups: Vec<Vec<u32>> = vec![Vec::new(); p.meta.forbidden_zones.len()];
                for (k, c) in cells.cells.iter().enumerate() {
                    let com = c.vi.com();
                    if let Some(z) = p.meta.forbidden_zones.iter().position(|z| z.contains(com)) {
                        groups[z].push(k as u32);
                    }
                }
                let (ev, _, _) = frac_cells::recipes::principal_axes(&p.solid);
                let csp = CellSetParams {
                    min_cell_volume: settings.cells.min_cell_volume,
                    min_thickness_ratio: settings.cells.min_thickness_ratio,
                    component_min_extent: (12.0 * ev.x.max(0.0)).sqrt(),
                };
                cells.merge_groups(&groups, &csp);
                cells.finalize_clusters();
            }
            let joint = mat.joint_interface_material.as_ref().map(|j| (InterfaceKind::MortarJoint, lib.interface_material_id(j)));
            let (surface, render_surface) = match &p.render_surface {
                Some((m, a)) => (a.clone(), Some(m.clone())),
                None => (p.surface.clone(), None),
            };
            (
                BuiltComponent {
                    name: p.name.clone(),
                    meta: p.meta.clone(),
                    role: p.role,
                    material: p.material,
                    solid: p.solid.clone(),
                    surface,
                    reconstructed: p.reconstructed,
                    render_surface,
                    cells,
                    grain,
                    unfractured,
                    joint,
                    wood: matches!(recipe, Recipe::Wood { .. }),
                },
                w,
            )
        })
        .collect();
    let mut builts = Vec::new();
    for (b, w) in built {
        warnings.extend(w);
        builts.push(b);
    }
    let metas: Vec<frac_core::input::PartMeta> = builts.iter().map(|b| b.meta.clone()).collect();
    tick("cells", &mut t, &mut timings);
    // ---- assemble asset
    let mut asset = Asset::default();
    asset.meta = AssetMeta {
        name: input.name.clone(),
        seed: settings.seed,
        variant: input.variant,
        tool_version: TOOL_VERSION.into(),
        settings_hash: settings.hash(),
        material_library_id: lib.library_id.clone(),
        material_library_version: lib.library_version.clone(),
    };
    let infos = assemble(&mut asset, builts, lib);
    let contacts: Vec<(usize, usize)> = ing.contacts.iter().filter_map(|&(a, b)| Some((*part_index.get(&a)?, *part_index.get(&b)?))).collect();
    warnings.extend(add_contacts(&mut asset, &contacts, &input.meta.connections, lib, settings.ingest.contact_tolerance, settings.ingest.contact_cos));
    let ground = settings.ground_height.or(input.meta.ground_height);
    add_anchors(&mut asset, &metas, ground, settings.ingest.contact_tolerance);
    tick("interfaces", &mut t, &mut timings);
    // ---- Stage 3/4/5: L1 (modes or agglomeration), L2, hierarchy
    let levels = frac_hierarchy::level_kinds(settings.levels);
    let has_structural = levels.contains(&frac_hierarchy::LevelKind::Structural);
    let l1: Vec<(level1::Level1Result, Vec<(u32, u32, f64, f64)>)> = (0..asset.components.len())
        .into_par_iter()
        .map(|ci| {
            let comp = &asset.components[ci];
            let info = &infos[ci];
            let adj = level1::analysis_adjacency(&asset, comp, info, lib, settings.modes.material_aware, &metas[ci].forbidden_zones);
            let anchor_h = asset.interfaces.iter().filter(|it| it.cells.1 == CellOrWorld::World && comp.cells.contains(&it.cells.0 .0)).flat_map(|it| it.polygons.iter().flat_map(|p| p.loops[0].iter().map(|v| v.y))).fold(None, |a: Option<f64>, y| Some(a.map_or(y, |a| a.max(y))));
            let r = if has_structural {
                let cfg = level1::ModesConfig { settings: &settings.modes, seed: settings.seed, component: ci as u32, anchor_height: anchor_h };
                let mut r = level1::level1(&asset, comp, info, &adj, lib, &metas[ci], &cfg, settings.modes.target_level1_fragments as usize, settings.hierarchy.compactness);
                let (moves, left) = manifold::repair_labels(comp, info, &adj, &mut r.labels, None);
                if left > 0 {
                    r.warnings.push(format!("component '{}': level 1 manifold repair moved {moves} analysis cells, {left} defects left", comp.name));
                }
                r
            } else {
                level1::Level1Result { labels: vec![0; info.n_analysis as usize], method: "none".into(), jumps: Vec::new(), warnings: Vec::new() }
            };
            (r, adj)
        })
        .collect();
    let mut parts = Vec::new();
    for (ci, (r, adj)) in l1.iter().enumerate() {
        warnings.extend(r.warnings.iter().cloned());
        asset.diagnostics.level1_method.push(r.method.clone());
        for &(a, b, j) in &r.jumps {
            asset.diagnostics.mode_jumps.push((ci as u32, a, b, j));
        }
        let comp = &asset.components[ci];
        let na = infos[ci].n_analysis as usize;
        let l2 = if settings.hierarchy.level2_target > 0 && (settings.hierarchy.level2_target as usize) < na {
            let ac = &asset.analysis_cells[comp.analysis_cells.start as usize..comp.analysis_cells.end as usize];
            let c: Vec<DVec3> = ac.iter().map(|a| a.mass.com).collect();
            let v: Vec<f64> = ac.iter().map(|a| a.mass.volume).collect();
            let l = frac_hierarchy::agglomerate(&c, &v, adj, settings.hierarchy.level2_target as usize, settings.hierarchy.compactness, Some(&r.labels));
            let mut l = frac_hierarchy::connected_labels(&l, adj);
            let (moves, left) = manifold::repair_labels(comp, &infos[ci], adj, &mut l, Some(&r.labels));
            if left > 0 {
                warnings.push(format!("component '{}': level 2 manifold repair moved {moves} analysis cells, {left} defects left", comp.name));
            }
            l
        } else {
            (0..na as u32).collect()
        };
        parts.push(frac_hierarchy::ComponentPartition {
            component: comp.id,
            role: comp.role,
            cells: infos[ci].cell_range.clone(),
            cell_analysis: infos[ci].cell_analysis.clone(),
            analysis_l1: r.labels.clone(),
            analysis_l2: l2,
        });
    }
    let bbox = asset_bbox(&asset);
    asset.hierarchy = frac_hierarchy::build_hierarchy(&asset.cells, &parts, settings.levels, &bbox, settings.collision.min_rigid_size);
    tick("hierarchy", &mut t, &mut timings);
    // ---- Stage 6/10: bonds and spawn data
    let weibull = |im: Option<MaterialId>, m: MaterialId| -> f64 { lib.weibull(im.unwrap_or(m)) };
    let bp = frac_bonds::BondParams {
        seed: settings.seed ^ (input.variant as u64).wrapping_mul(0x9e37),
        weibull: &weibull,
        loop_simplify: settings.bonds.loop_simplify,
        spawn_density: settings.bonds.spawn_density,
        max_spawn_per_bond: settings.bonds.max_spawn_per_bond,
    };
    let bo = frac_bonds::build_bonds(&asset, &bp);
    asset.bonds = bo.bonds;
    asset.bond_children = bo.bond_children;
    asset.loops = bo.loops;
    asset.spawn = bo.spawn;
    for (i, a) in bo.interior_area.into_iter().enumerate() {
        asset.hierarchy.fragments[i].interior_area = a;
    }
    tick("bonds", &mut t, &mut timings);
    // diagnostics: `FRAC_STOP_BEFORE_COLLISION=<path>` writes the asset
    // (cells, hierarchy, bonds; no hulls) and stops, for collision
    // experiments with `prefracture hulls`
    if let Some(p) = std::env::var_os("FRAC_STOP_BEFORE_COLLISION") {
        frac_io::write_asset_json(&asset, std::path::Path::new(&p)).map_err(|e| FracError::new(Stage::Export, input.name.clone(), e.to_string()))?;
        return Err(FracError::new(Stage::Export, input.name.clone(), "stopped before collision (FRAC_STOP_BEFORE_COLLISION)"));
    }
    // ---- Stage 8: collision
    let cp = collision_params(settings);
    let (hulls, ranges) = frac_collision::build_hulls(&asset, &cp);
    asset.hulls = hulls;
    for (i, r) in ranges.into_iter().enumerate() {
        asset.hierarchy.fragments[i].hulls = r;
    }
    tick("collision", &mut t, &mut timings);
    // debugging aid: the asset before render, for replaying stages 9-12
    // (examples/render_bench.rs) without re-running cells and collision
    if let Some(p) = std::env::var_os("FRAC_DUMP_ASSET") {
        if let Err(e) = frac_io::write_asset_json(&asset, std::path::Path::new(&p)) {
            warnings.push(format!("FRAC_DUMP_ASSET: {e}"));
        }
    }
    // ---- Stage 9: render
    let render = render_asset(&asset, settings, lib);
    tick("render", &mut t, &mut timings);
    // ---- export payloads
    // Peak memory matters on building-scale assets: the f64 render meshes
    // are converted into the f32 glTF scene (and freed) only after
    // validation, and the glTF buffer is written after that.
    export::render_refs(&mut asset, &render);
    let physics = frac_io::write_physics(&asset);
    tick("export", &mut t, &mut timings);
    // ---- Stage 12: validation
    let scorecard = frac_validate::validate(&asset, &render, lib, &settings.validation, &physics);
    tick("validate", &mut t, &mut timings);
    let mut report = Report::new(&asset, &render, timings, warnings, scorecard);
    let tg = Instant::now();
    let mut render = render;
    let scene = export::render_scene(&asset, &mut render, lib, !keep_render);
    let render = if keep_render { Some(render) } else { None };
    frac_io::release_free_memory();
    if log {
        eprintln!("[{}] glb scene: {:.1} s", input.name, tg.elapsed().as_secs_f64());
    }
    let gltf = frac_io::write_glb_owned(scene, &frac_io::GltfOptions { meshopt_compression: settings.render.meshopt_compression })
        .map_err(|e| FracError::new(Stage::Export, "glb", e.to_string()))?;
    if let Some(e) = report.timings.iter_mut().find(|x| x.stage == "export") {
        e.ms += tg.elapsed().as_secs_f64() * 1e3;
    }
    if log {
        eprintln!("[{}] glb: {:.1} s, rss {:.0} MB", input.name, tg.elapsed().as_secs_f64(), rss_mb());
    }
    report.total_ms = t0.elapsed().as_secs_f64() * 1e3;
    report.payload_bytes = (gltf.len(), physics.len());
    Ok(PipelineOutput { asset, gltf, physics, report, render })
}

/// Render meshes of a baked asset with the bake settings (stage 9).
pub fn render_asset(asset: &Asset, settings: &Settings, lib: &MaterialLibrary) -> frac_render::RenderOut {
    let noise_for = |c: ComponentId| -> Option<frac_render::NoiseSpec> {
        let comp = &asset.components[c.idx()];
        let mat = lib.material(comp.material);
        let np = lib.noise(mat.noise_profile.as_deref()?)?;
        Some(frac_render::NoiseSpec {
            amplitude: np.amplitude,
            hurst: np.hurst_exponent,
            min_wl: np.min_wavelength,
            max_wl: np.max_wavelength,
            grain: match (np.grain_stretch, comp.grain) {
                (Some(s), Some(g)) => Some((g, s)),
                _ => None,
            },
        })
    };
    let chip_for = |c: ComponentId| -> f64 { lib.material(asset.components[c.idx()].material).chipping_ratio.unwrap_or(0.0) };
    let uv_for = |c: ComponentId| -> f64 { lib.material(asset.components[c.idx()].material).interior_uv_scale.unwrap_or(1.0) };
    let rp = frac_render::RenderParams { settings: &settings.render, seed: settings.seed, noise_for: &noise_for, chipping_for: &chip_for, uv_scale_for: &uv_for };
    frac_render::build_render(asset, &rp)
}

/// Resident set size of this process in MB (Linux; 0 elsewhere), for the
/// `FRAC_LOG` stage log.
fn rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|x| x.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()))
        .map(|pages| pages * 4096.0 / 1048576.0)
        .unwrap_or(0.0)
}

pub use frac_collision;
pub use frac_collision::{build_hulls, hull_polytope};

/// Collision-stage parameters from the bake settings.
pub fn collision_params(settings: &Settings) -> frac_collision::CollisionParams {
    frac_collision::CollisionParams {
        concavity: settings.collision.concavity,
        max_hulls: settings.collision.max_hulls_per_fragment as usize,
        margin: settings.collision.margin,
        min_rigid_size: settings.collision.min_rigid_size,
        levels: settings.collision.levels.clone(),
        search: frac_collision::SearchEffort {
            mcts_iterations: settings.collision.mcts_iterations,
            mcts_depth: settings.collision.mcts_depth,
            mcts_nodes: settings.collision.mcts_nodes,
            resolution: settings.collision.resolution,
        },
        max_hull_vertices: settings.collision.max_hull_vertices as usize,
        seed: settings.seed,
        ..Default::default()
    }
}
