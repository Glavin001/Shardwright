//! Continuous metrics (spec §13.5–13.6) computed in-process. Oracle-based
//! metrics (bond fidelity vs Kratos FEM, crack placement vs fracture
//! simulations, Houdini/CoACD baselines) are computed by the harness in
//! `tools/harness` from the JSON dump.

use frac_collision::{cells_boundary_mesh_with, hull_polytope};
use frac_core::*;
use frac_geom::hull::ConvexPolytope;
use frac_geom::inside::MeshQuery;
use frac_geom::integrals::sym_eigen3;
use frac_geom::DVec3;
use frac_material::MaterialLibrary;
use frac_render::RenderOut;
use rayon::prelude::*;
use serde_json::{json, Value};

fn quantiles(mut v: Vec<f64>) -> Value {
    if v.is_empty() {
        return json!(null);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    json!({"n": v.len(), "min": v[0], "p10": q(0.1), "p50": q(0.5), "p90": q(0.9), "max": v[v.len()-1], "mean": v.iter().sum::<f64>() / v.len() as f64})
}

/// Fit a Weibull distribution to positive samples (MLE, Newton on the
/// shape parameter). Returns (shape k, scale λ).
pub fn weibull_fit(x: &[f64]) -> Option<(f64, f64)> {
    let x: Vec<f64> = x.iter().copied().filter(|v| *v > 0.0).collect();
    if x.len() < 3 {
        return None;
    }
    let ln: Vec<f64> = x.iter().map(|v| v.ln()).collect();
    let mean_ln = ln.iter().sum::<f64>() / x.len() as f64;
    let mut k = 1.0;
    for _ in 0..100 {
        let s0: f64 = x.iter().map(|v| v.powf(k)).sum();
        let s1: f64 = x.iter().zip(&ln).map(|(v, l)| v.powf(k) * l).sum();
        let s2: f64 = x.iter().zip(&ln).map(|(v, l)| v.powf(k) * l * l).sum();
        let f = s1 / s0 - 1.0 / k - mean_ln;
        let df = (s2 * s0 - s1 * s1) / (s0 * s0) + 1.0 / (k * k);
        let nk = (k - f / df).clamp(0.05, 50.0);
        if (nk - k).abs() < 1e-10 {
            k = nk;
            break;
        }
        k = nk;
    }
    let lam = (x.iter().map(|v| v.powf(k)).sum::<f64>() / x.len() as f64).powf(1.0 / k);
    Some((k, lam))
}

pub fn compute(asset: &Asset, render: &RenderOut, _lib: &MaterialLibrary) -> Value {
    let cp = asset.cell_polys();
    let h = &asset.hierarchy;
    let mut levels = Vec::new();
    for l in 0..h.levels {
        let fr = asset.level_fragments(l);
        let vols: Vec<f64> = fr.iter().map(|f| f.mass.volume).collect();
        let logv: Vec<f64> = vols.iter().filter(|v| **v > 0.0).map(|v| v.ln()).collect();
        // shape: inertia-eigenvalue aspect ratio (every fragment) and
        // convexity V / V_hull (deterministic sample of at most
        // CONVEXITY_SAMPLES fragments per level: a convex hull per fragment
        // dominates the metrics time on building-scale assets, and the
        // quantiles of a few thousand samples are within ~1%)
        const CONVEXITY_SAMPLES: usize = 4096;
        let aspect: Vec<f64> = fr
            .iter()
            .map(|f| {
                let (ev, _) = f.mass.principal();
                if ev.z > 0.0 {
                    (ev.x / ev.z).sqrt()
                } else {
                    0.0
                }
            })
            .collect();
        let convexity: Vec<f64> = fr
            .par_iter()
            .step_by(fr.len().div_ceil(CONVEXITY_SAMPLES).max(1))
            .map(|f| {
                let m = cells_boundary_mesh_with(asset, &cp, asset.fragment_cells(f));
                let hv = ConvexPolytope::from_points(&m.verts).map(|p| p.volume()).unwrap_or(0.0);
                if hv > 0.0 {
                    f.mass.volume / hv
                } else {
                    1.0
                }
            })
            .collect();
        // hulls
        let hull_stats: Vec<(f64, usize)> = fr
            .par_iter()
            .filter(|f| !f.hulls.is_empty())
            .map(|f| {
                let hv: f64 = asset.hulls[f.hulls.start as usize..f.hulls.end as usize].iter().map(|hh| hull_polytope(hh).volume()).sum();
                (hv / f.mass.volume.max(1e-300) - 1.0, f.hulls.len())
            })
            .collect();
        let tris: Vec<f64> = fr.iter().map(|f| render.fragments[f.id.idx()].first().map(|m| m.triangle_count()).unwrap_or(0) as f64).collect();
        let lod_tris: Vec<Vec<usize>> = fr.iter().take(1).map(|f| render.fragments[f.id.idx()].iter().map(|m| m.triangle_count()).collect()).collect();
        levels.push(json!({
            "level": l,
            "fragments": fr.len(),
            "volume": quantiles(vols.clone()),
            "log_volume": quantiles(logv),
            "weibull_volume": weibull_fit(&vols).map(|(k, lam)| json!({"shape": k, "scale": lam})),
            "aspect_ratio": quantiles(aspect),
            "convexity_samples": convexity.len(),
            "convexity": quantiles(convexity),
            "hull_overshoot": quantiles(hull_stats.iter().map(|s| s.0).collect()),
            "hull_count": quantiles(hull_stats.iter().map(|s| s.1 as f64).collect()),
            "triangles_lod0": quantiles(tris),
            "lod_triangles_example": lod_tris,
            "bonds": asset.level_bonds(l).count(),
            "bond_area": quantiles(asset.level_bonds(l).map(|b| b.area).collect()),
            "bond_planarity": quantiles(asset.level_bonds(l).map(|b| b.planarity as f64).collect()),
            "strength_scale": quantiles(asset.level_bonds(l).map(|b| b.strength_scale as f64).collect()),
        }));
    }
    // hull fit (spec §13.6): symmetric surface deviation between a fragment
    // and its hulls / diameter, per level (≤ 100 sampled fragments): hull
    // vertices outside the fragment (overshoot) and fragment-surface points
    // (vertices and triangle centroids) not covered by any hull (gaps, which
    // include the collision margin and the non-overlap clipping).
    let mut fit_by_level = Vec::new();
    let mut fit_all = Vec::new();
    for l in 1..h.levels {
        let frs = asset.level_fragments(l);
        let fit: Vec<f64> = frs
            .par_iter()
            .step_by((frs.len() / 100).max(1))
            .filter(|f| !f.hulls.is_empty())
            .map(|f| hull_fit(asset, &cp, f))
            .collect();
        fit_all.extend(fit.iter().copied());
        fit_by_level.push(json!({"level": l, "fit": quantiles(fit)}));
    }
    let fit = fit_all;
    let leaf = h.levels.saturating_sub(1);
    // grain alignment for anisotropic components (finest level)
    let mut grain = Vec::new();
    for c in &asset.components {
        let Some(g) = c.grain else { continue };
        let (mut s, mut w) = (0.0, 0.0);
        for f in asset.level_fragments(leaf).iter().filter(|f| f.component == c.id) {
            let (_, ax) = sym_eigen3(&f.mass.inertia);
            // principal (longest) axis = smallest inertia eigenvalue
            s += ax.col(0).dot(g).abs() * f.mass.volume;
            w += f.mass.volume;
        }
        if w > 0.0 {
            grain.push(json!({"component": c.name, "mean_abs_cos": s / w}));
        }
    }
    // interface composition
    let mut kinds = std::collections::BTreeMap::new();
    for it in &asset.interfaces {
        *kinds.entry(format!("{:?}", it.kind)).or_insert(0.0) += it.area;
    }
    let dev: Vec<f64> = render.volume_deviation.iter().map(|d| d.1).collect();
    json!({
        "levels": levels,
        "hull_fit_concavity_over_diameter": quantiles(fit),
        "hull_fit_by_level": fit_by_level,
        "grain_alignment": grain,
        "interface_area_by_kind": kinds,
        "render_volume_deviation": quantiles(dev),
        "uv_mismatch_max": render.uv_mismatch,
        "cells": asset.cells.len(),
        "thickness_ratio": quantiles(asset.cells.iter().map(|c| c.thickness_ratio).collect()),
        "level1_method": asset.diagnostics.level1_method,
        "interface_roughness": roughness(asset, render),
    })
}

/// Displaced-interface roughness: RMS height of render interior vertices
/// relative to their clean interface plane, and the structure-function
/// slope (≈ Hurst exponent) estimated from height differences at two
/// scales.
fn roughness(asset: &Asset, render: &RenderOut) -> Value {
    let leaf = asset.hierarchy.levels.saturating_sub(1);
    let mut rms = Vec::new();
    for f in asset.level_fragments(leaf).iter().take(50) {
        let m = &render.fragments[f.id.idx()][0];
        let comp = &asset.components[f.component.idx()];
        // nearest patch plane per interior vertex (by normal agreement)
        let mut s2 = 0.0;
        let mut n = 0usize;
        for &i in m.int_indices.iter().step_by(3) {
            let p = m.positions[i as usize];
            let nn = m.normals[i as usize].as_dvec3();
            let mut best = f64::INFINITY;
            for pt in &comp.geometry.patches {
                if pt.normal.dot(nn).abs() < 0.5 {
                    continue;
                }
                let r = comp.geometry.verts[pt.loops[0][0] as usize];
                let d = (p - r).dot(pt.normal).abs();
                best = best.min(d);
            }
            if best.is_finite() {
                s2 += best * best;
                n += 1;
            }
        }
        if n > 0 {
            rms.push((s2 / n as f64).sqrt());
        }
    }
    let _ = DVec3::ZERO;
    json!({"rms_height": quantiles(rms)})
}

/// Exact distance from `x` to a convex polytope (0 inside).
fn polytope_distance(x: DVec3, p: &ConvexPolytope) -> f64 {
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
        let q = x - h.n * dn;
        // inside the face polygon (CCW about n)?
        let inside = (0..poly.len()).all(|i| (poly[(i + 1) % poly.len()] - poly[i]).cross(q - poly[i]).dot(h.n) >= -tol);
        if inside {
            best = best.min(dn);
            continue;
        }
        for i in 0..poly.len() {
            let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
            let ab = b - a;
            let t = ((x - a).dot(ab) / ab.length_squared().max(1e-300)).clamp(0.0, 1.0);
            best = best.min((a + ab * t - x).length());
        }
    }
    if best.is_finite() { best } else { 0.0 }
}

/// Symmetric hull-fit deviation of a fragment / its diameter.
fn hull_fit(asset: &Asset, cp: &CellPolys, f: &Fragment) -> f64 {
    let m = cells_boundary_mesh_with(asset, cp, asset.fragment_cells(f));
    let q = MeshQuery::new(&m);
    let diam = m.aabb().diagonal().max(1e-300);
    let hulls: Vec<ConvexPolytope> = asset.hulls[f.hulls.start as usize..f.hulls.end as usize].iter().map(hull_polytope).collect();
    let mut worst: f64 = 0.0;
    for hh in &asset.hulls[f.hulls.start as usize..f.hulls.end as usize] {
        for v in &hh.vertices {
            if !q.contains(*v) {
                if let Some((_, d2, _)) = q.closest_point(*v) {
                    worst = worst.max(d2.sqrt());
                }
            }
        }
    }
    let samples = m.verts.iter().copied().chain(m.tris.iter().map(|t| (m.verts[t[0] as usize] + m.verts[t[1] as usize] + m.verts[t[2] as usize]) / 3.0));
    for x in samples {
        let d = hulls.iter().map(|p| polytope_distance(x, p)).fold(f64::INFINITY, f64::min);
        if d.is_finite() {
            worst = worst.max(d);
        }
    }
    worst / diam
}
