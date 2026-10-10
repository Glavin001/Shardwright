//! Level-1 structural fragments: material-aware fracture modes (Stage 3/4)
//! with agglomeration (method B) as the fallback.

use crate::assemble::ComponentCells;
use frac_core::input::PartMeta;
use frac_core::*;
use frac_geom::bvh::Bvh;
use frac_geom::inside::MeshQuery;
use frac_geom::{Aabb, DVec3, TriMesh};
use frac_material::MaterialLibrary;
use std::collections::BTreeMap;

/// Analysis-cell adjacency of one component: (a, b, shared area, w_g).
pub fn analysis_adjacency(
    asset: &Asset,
    comp: &Component,
    info: &ComponentCells,
    lib: &MaterialLibrary,
    material_aware: bool,
    forbidden: &[frac_core::input::BoxVolume],
) -> Vec<(u32, u32, f64, f64)> {
    let base = info.cell_range.start;
    let mut acc: BTreeMap<(u32, u32), (f64, f64, bool)> = BTreeMap::new(); // area, Σ area*w, forbidden
    let gref = lib.reference_fracture_energy.max(1e-12);
    for it in &asset.interfaces {
        let CellOrWorld::Cell(cb) = it.cells.1 else { continue };
        let ca = it.cells.0;
        if !comp.cells.contains(&ca.0) || !comp.cells.contains(&cb.0) {
            continue;
        }
        let a = info.cell_analysis[(ca.0 - base) as usize];
        let b = info.cell_analysis[(cb.0 - base) as usize];
        if a == b {
            continue;
        }
        let n = it.polygons.first().map(|p| p.normal.to_array());
        let gf = match it.interface_material {
            Some(m) => lib.interface_fracture_energy(m),
            None => lib.fracture_energy(comp.material, n, comp.grain.map(|g| g.to_array())),
        };
        let w = if material_aware { (gf / gref).max(1e-12).sqrt() } else { 1.0 };
        let mut c = DVec3::ZERO;
        let mut ar = 0.0;
        for p in &it.polygons {
            let ai = frac_bonds::polygon_integrals(p);
            c += ai.first;
            ar += ai.area;
        }
        let centroid = if ar > 0.0 { c / ar } else { DVec3::ZERO };
        let forb = forbidden.iter().any(|z| z.contains(centroid));
        let e = acc.entry((a.min(b), a.max(b))).or_insert((0.0, 0.0, false));
        e.0 += it.area;
        e.1 += it.area * w;
        e.2 |= forb;
    }
    acc.into_iter()
        .map(|((a, b), (area, aw, forb))| (a, b, area, if forb { f64::INFINITY } else if area > 0.0 { aw / area } else { 1.0 }))
        .collect()
}

/// Locate points in the cells of a component (exact parity per candidate
/// cell, nearest boundary as fallback). Returns local cell indices.
pub struct CellLocator {
    meshes: Vec<TriMesh>,
    boxes: Vec<Aabb>,
    bvh: Bvh,
}

impl CellLocator {
    pub fn new(asset: &Asset, comp: &Component) -> Self {
        let g = &comp.geometry;
        let n = (comp.cells.end - comp.cells.start) as usize;
        let mut tris: Vec<Vec<[u32; 3]>> = vec![Vec::new(); n];
        for e in &g.ext_polys {
            let c = (e.cell.0 - comp.cells.start) as usize;
            tris[c].extend(e.tris.iter().copied());
        }
        for p in &g.patches {
            let a = (p.cells.0 .0 - comp.cells.start) as usize;
            let b = (p.cells.1 .0 - comp.cells.start) as usize;
            for t in &p.tris {
                tris[a].push(*t);
                tris[b].push([t[0], t[2], t[1]]);
            }
        }
        let meshes: Vec<TriMesh> = tris.into_iter().map(|t| TriMesh { verts: g.verts.clone(), tris: t }.compact()).collect();
        let boxes: Vec<Aabb> = meshes.iter().map(|m| m.aabb()).collect();
        let bvh = Bvh::build(&boxes);
        let _ = asset;
        CellLocator { meshes, boxes, bvh }
    }
    pub fn locate(&self, p: DVec3) -> u32 {
        let q = Aabb { min: p, max: p };
        let cand = self.bvh.query_vec(&q);
        for &c in &cand {
            if MeshQuery::new(&self.meshes[c as usize]).contains(p) {
                return c;
            }
        }
        // nearest cell boundary
        let mut best = (f64::INFINITY, 0u32);
        let bb_d: Vec<(f64, u32)> = {
            let mut v: Vec<(f64, u32)> = self.boxes.iter().enumerate().map(|(i, b)| (b.dist2(p), i as u32)).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v
        };
        for (d, c) in bb_d.into_iter().take(16) {
            if d > best.0 {
                break;
            }
            if let Some((_, d2, _)) = MeshQuery::new(&self.meshes[c as usize]).closest_point(p) {
                if d2 < best.0 {
                    best = (d2, c);
                }
            }
        }
        best.1
    }
}

pub struct Level1Result {
    pub labels: Vec<u32>,
    pub method: String,
    pub jumps: Vec<(u32, u32, f64)>,
    pub warnings: Vec<String>,
}

pub struct ModesConfig<'a> {
    pub settings: &'a settings::ModeSettings,
    pub seed: u64,
    pub component: u32,
    pub anchor_height: Option<f64>,
}

/// Elastic material for a component (transversely isotropic wood).
pub fn elastic_material(lib: &MaterialLibrary, m: MaterialId, grain: Option<DVec3>) -> frac_fem::ElasticMaterial {
    let mat = lib.material(m);
    match (&mat.anisotropy, grain) {
        (Some(a), Some(g)) => frac_fem::ElasticMaterial {
            youngs: a.e_long,
            poisson: mat.poisson,
            density: mat.density,
            transverse: Some(frac_fem::TransverseIsotropic {
                axis: g.normalize().to_array(),
                e_long: a.e_long,
                e_trans: a.e_trans,
                g_long: a.g,
                nu_trans: a.nu_trans,
                nu_long: a.nu_long,
            }),
        },
        _ => {
            let e = lib.elastic(m);
            frac_fem::ElasticMaterial { youngs: e.youngs, poisson: e.poisson, density: e.density, transverse: None }
        }
    }
}

/// Compute Level-1 labels over the analysis cells of one component.
#[allow(clippy::too_many_arguments)]
pub fn level1(
    asset: &Asset,
    comp: &Component,
    info: &ComponentCells,
    adj: &[(u32, u32, f64, f64)],
    lib: &MaterialLibrary,
    meta: &PartMeta,
    cfg: &ModesConfig,
    target: usize,
    compactness: f64,
) -> Level1Result {
    let na = info.n_analysis as usize;
    let mut warnings = Vec::new();
    if na <= 1 {
        return Level1Result { labels: vec![0; na], method: "single".into(), jumps: Vec::new(), warnings };
    }
    let target = target.clamp(1, na);
    let centroids: Vec<DVec3> = asset.analysis_cells[comp.analysis_cells.start as usize..comp.analysis_cells.end as usize].iter().map(|a| a.mass.com).collect();
    let volumes: Vec<f64> = asset.analysis_cells[comp.analysis_cells.start as usize..comp.analysis_cells.end as usize].iter().map(|a| a.mass.volume).collect();
    if cfg.settings.enabled && !comp.unfractured {
        match run_modes(asset, comp, info, adj, lib, meta, cfg, target) {
            Ok(r) => return r,
            Err(e) => warnings.push(format!("component '{}': fracture modes failed ({e}); using agglomeration", comp.name)),
        }
    }
    let labels = frac_hierarchy::agglomerate(&centroids, &volumes, adj, target, compactness, None);
    let labels = frac_hierarchy::connected_labels(&labels, adj);
    Level1Result { labels, method: "agglomeration".into(), jumps: Vec::new(), warnings }
}

/// `modes.discretization` setting → frac-modes discretization (`None`: the
/// linear-elastic P1 model chosen by problem size).
pub fn mode_discretization(name: &str) -> Result<Option<frac_modes::Discretization>, String> {
    match name {
        "translational" => Ok(Some(frac_modes::Discretization::CellPolynomial(0))),
        "p1" => Ok(None),
        "full" => Ok(Some(frac_modes::Discretization::Full)),
        "cell-p1" => Ok(Some(frac_modes::Discretization::CellPolynomial(1))),
        other => Err(format!("unknown modes.discretization '{other}' (translational, p1, full, cell-p1)")),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_modes(
    asset: &Asset,
    comp: &Component,
    info: &ComponentCells,
    adj: &[(u32, u32, f64, f64)],
    lib: &MaterialLibrary,
    _meta: &PartMeta,
    cfg: &ModesConfig,
    target: usize,
) -> Result<Level1Result, String> {
    let s = cfg.settings;
    let t_start = std::time::Instant::now();
    // analysis resolution: tet edge <= ratio * median analysis cell diameter
    let mut diam: Vec<f64> = asset.analysis_cells[comp.analysis_cells.start as usize..comp.analysis_cells.end as usize]
        .iter()
        .map(|a| a.mass.volume.max(0.0).cbrt())
        .collect();
    diam.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = diam[diam.len() / 2].max(1e-6);
    let edge = s.tet_edge_ratio * med;
    let mesh = match &s.ftetwild {
        Some(bin) => frac_fem::tetrahedralize_external(bin, &comp.solid, edge)?,
        None => frac_fem::tetrahedralize(&comp.solid, edge, s.max_tets),
    };
    if mesh.tets.is_empty() {
        return Err("empty analysis mesh".into());
    }
    let loc = CellLocator::new(asset, comp);
    let tet_cell: Vec<u32> = mesh
        .tets
        .iter()
        .map(|t| {
            let c = t.iter().fold(DVec3::ZERO, |a, &v| a + DVec3::from_array(mesh.verts[v as usize])) / 4.0;
            let lc = loc.locate(c);
            info.cell_analysis[lc as usize]
        })
        .collect();
    let mat = elastic_material(lib, comp.material, comp.grain);
    let tet_material = vec![mat; mesh.tets.len()];
    let wmap: BTreeMap<(u32, u32), f64> = adj.iter().map(|&(a, b, _, w)| ((a, b), w)).collect();
    let weight = move |a: u32, b: u32| -> f64 { *wmap.get(&(a.min(b), a.max(b))).unwrap_or(&1.0) };
    let anchored: Vec<u32> = match cfg.anchor_height {
        Some(h) => (0..mesh.verts.len() as u32).filter(|&v| mesh.verts[v as usize][1] <= h + 0.5 * edge).collect(),
        None => Vec::new(),
    };
    let solver = match s.solver.as_str() {
        "clarabel" => frac_modes::Solver::Clarabel,
        "admm" => frac_modes::Solver::Admm,
        _ => frac_modes::Solver::Auto,
    };
    let input = frac_modes::ModesInput {
        mesh: &mesh,
        tet_material: &tet_material,
        tet_cell: &tet_cell,
        group_weight: &weight,
        anchored_vertices: &anchored,
        params: frac_modes::ModesParams {
            k: s.k,
            omega: s.omega,
            eps: s.iccm_tolerance,
            max_iters: s.max_iccm_iters,
            solver,
            seed: stable_hash(&[cfg.seed, cfg.component as u64, 4]),
            large_dofs: s.large_problem_dofs,
            eps_large: s.large_iccm_tolerance,
            discretization: mode_discretization(&s.discretization)?,
            area_weighted: s.area_weighting,
            multi_start: s.multi_start,
        },
    };
    let volumes: Vec<f64> = asset.analysis_cells[comp.analysis_cells.start as usize..comp.analysis_cells.end as usize].iter().map(|a| a.mass.volume).collect();
    // fragments below a quarter of the mean target size are merged
    let min_volume = 0.25 * volumes.iter().sum::<f64>() / target.max(1) as f64;
    if let Some(dir) = std::env::var_os("FRAC_MODES_DUMP") {
        // diagnostics: exact problem dump for offline reruns (frac-modes `modes_bench` example)
        let dump = frac_modes::dump::ModesDump {
            mesh: mesh.clone(),
            tet_material: tet_material.clone(),
            tet_cell: tet_cell.clone(),
            adjacency: adj.to_vec(),
            anchored_vertices: anchored.clone(),
            params: input.params,
            n_analysis: info.n_analysis,
            cell_volume: volumes.clone(),
            target: target as u32,
            min_volume,
        };
        let path = std::path::Path::new(&dir).join(format!("component_{}.modes.txt", cfg.component));
        if let Err(e) = std::fs::write(&path, dump.to_text()) {
            eprintln!("FRAC_MODES_DUMP: cannot write {}: {e}", path.display());
        }
    }
    let t_modes = std::time::Instant::now();
    let out = frac_modes::compute_modes(&input)?;
    if std::env::var_os("FRAC_LOG").is_some() {
        let stages: Vec<String> = out.timings_ms.iter().map(|(k, v)| format!("{k} {:.2}", v / 1e3)).collect();
        eprintln!(
            "    [modes] component {} '{}': {} analysis cells, {} tets, {} unknowns, {}: setup {:.2} s, modes {:.2} s ({})",
            cfg.component,
            comp.name,
            info.n_analysis,
            mesh.tets.len(),
            out.n_dofs,
            out.solver_used,
            (t_modes - t_start).as_secs_f64(),
            t_modes.elapsed().as_secs_f64(),
            stages.join(", ")
        );
    }
    // size-balanced segmentation over the exact adjacency. Translational
    // modes give the jump of every adjacent pair from the per-cell
    // displacements; for P1, groups missing from the tet staircase get zero
    // jump and are never cut first
    let adj_pairs: Vec<(u32, u32)> = adj.iter().map(|&(a, b, _, _)| (a, b)).collect();
    let (seg_groups, max_jump) = match out.pair_max_jump(&adj_pairs) {
        Some(mj) => (adj_pairs, mj),
        None => (out.groups.clone(), out.max_jump()),
    };
    let (l1, groups, mj) = frac_modes::segment_from_jumps(info.n_analysis, &seg_groups, &max_jump, adj, &volumes, target as u32, min_volume);
    let mut warnings = Vec::new();
    if !l1.hit_target {
        warnings.push(format!("component '{}': modes segmentation reached {} fragments (target {target})", comp.name, l1.n_fragments));
    }
    let not_conv = out.converged.iter().filter(|&&c| !c).count();
    if not_conv > 0 {
        warnings.push(format!("component '{}': {not_conv} ICCM modes did not converge", comp.name));
    }
    let jumps = groups.iter().zip(mj.iter()).map(|(&(a, b), &j)| (a, b, j)).collect();
    // ensure labels are connected over the exact adjacency
    let labels = frac_hierarchy::connected_labels(&l1.labels, adj);
    if let Some(dir) = std::env::var_os("FRAC_MODES_DUMP") {
        let text: String = labels.iter().map(|l| format!("{l}\n")).collect();
        let path = std::path::Path::new(&dir).join(format!("component_{}.labels.txt", cfg.component));
        if let Err(e) = std::fs::write(&path, text) {
            eprintln!("FRAC_MODES_DUMP: cannot write {}: {e}", path.display());
        }
    }
    Ok(Level1Result { labels, method: format!("modes({})", out.solver_used), jumps, warnings })
}
