//! Bridge to the oracle harness (spec §13.3): run the reference bond-network
//! solver on standard load cases at every level and export results with a
//! geometric description of supports and loads, so the FEM oracle (Kratos)
//! can apply identical boundary conditions to the unfractured solid.

use frac_bonds::polygon_integrals;
use frac_core::*;
use frac_geom::polygon::plane_basis;
use frac_geom::{Aabb, DVec3};
use frac_material::MaterialLibrary;
use frac_validate::network::{BondNetworkSolver, LoadCase, ReferenceSolver};
use serde_json::{json, Value};
use smallvec::SmallVec;

/// Principal axis (longest bbox extent) and its range.
fn axis_of(asset: &Asset) -> (usize, f64, f64, Aabb) {
    let bb = asset.components.iter().fold(Aabb::EMPTY, |a, c| a.union(&c.aabb));
    let ax = bb.longest_axis();
    (ax, bb.min[ax], bb.max[ax], bb)
}

/// Exterior polygons of fragment cells lying on the end plane `x[ax] = v`.
fn end_face_polys(asset: &Asset, level: u8, ax: usize, v: f64, tol: f64) -> Vec<(FragmentId, Polygon3)> {
    let h = &asset.hierarchy;
    let mut out = Vec::new();
    for c in &asset.components {
        let g = &c.geometry;
        for e in &g.ext_polys {
            let pts: Vec<DVec3> = e.verts.iter().map(|&i| g.verts[i as usize]).collect();
            if pts.iter().all(|p| (p[ax] - v).abs() <= tol) {
                let n = frac_geom::polygon::newell(&pts).normalize_or_zero();
                let f = h.cell_fragment[level as usize][e.cell.idx()];
                out.push((f, Polygon3 { loops: vec![pts], normal: n }));
            }
        }
    }
    out
}

/// Synthetic world bonds clamping the fragments that touch an end face.
fn support_bonds(asset: &Asset, level: u8, polys: &[(FragmentId, Polygon3)]) -> Vec<Bond> {
    let mut by_frag: std::collections::BTreeMap<FragmentId, Vec<Polygon3>> = std::collections::BTreeMap::new();
    for (f, p) in polys {
        by_frag.entry(*f).or_default().push(p.clone());
    }
    let mut out = Vec::new();
    for (f, ps) in by_frag {
        let mut ai = frac_geom::polygon::AreaIntegrals::default();
        for p in &ps {
            ai.add(&polygon_integrals(p));
        }
        let n = ai.area_vec.normalize_or_zero();
        let c = ai.centroid();
        let c2 = ai.central_second();
        let (u, v) = plane_basis(n);
        let fr = asset.fragment(f);
        out.push(Bond {
            id: BondId(u32::MAX),
            level,
            a: f,
            b: FragmentOrWorld::World,
            area: ai.area,
            centroid: c,
            normal: n,
            planarity: 1.0,
            frame_u: u,
            frame_v: v,
            i_uu: v.dot(c2 * v),
            i_vv: u.dot(c2 * u),
            i_uv: u.dot(c2 * v),
            j: v.dot(c2 * v) + u.dot(c2 * u),
            extent: Obb2::default(),
            dist_a: (fr.mass.com - c).dot(n).abs(),
            dist_b: 0.0,
            composition: SmallVec::new(),
            reinforcement: RebarCrossing::default(),
            strength_scale: 1.0,
            parent_bond: None,
            child_bonds: 0..0,
            boundary_loops: 0..0,
            spawn: 0..0,
            anchor: true,
            interfaces: Vec::new(),
        });
    }
    out
}

/// Export network results for all levels >= 1.
pub fn network_export(asset: &Asset, lib: &MaterialLibrary) -> Value {
    let (ax, lo, hi, bb) = axis_of(asset);
    let len = hi - lo;
    let tol = 1e-6 * bb.diagonal();
    let mut axis = DVec3::ZERO;
    axis[ax] = 1.0;
    // transverse directions
    let t1 = if ax == 1 { DVec3::X } else { DVec3::Y };
    let t2 = axis.cross(t1);
    let total_mass: f64 = asset.level_fragments(0).iter().map(|f| f.mass.mass).sum();
    let force = (total_mass * 9.81).max(1.0) * 10.0;
    let cases = json!([
        {"name": "axial", "kind": "force", "direction": axis.to_array(), "magnitude": force},
        {"name": "bending", "kind": "force", "direction": t1.to_array(), "magnitude": force * 0.1},
        {"name": "shear", "kind": "force", "direction": t2.to_array(), "magnitude": force * 0.1},
        {"name": "torsion", "kind": "torque", "direction": axis.to_array(), "magnitude": force * 0.05 * len.max(1e-3)},
    ]);
    let solver = ReferenceSolver { lib };
    let mut levels = Vec::new();
    for level in 1..asset.hierarchy.levels {
        let fixed_polys = end_face_polys(asset, level, ax, lo, tol);
        let load_polys = end_face_polys(asset, level, ax, hi, tol);
        let mut a2 = asset.clone();
        a2.bonds.retain(|b| b.level == level);
        let supports = support_bonds(asset, level, &fixed_polys);
        for (k, mut s) in supports.into_iter().enumerate() {
            s.id = BondId((a2.bonds.len() + k) as u32);
            a2.bonds.push(s);
        }
        // reindex ids so solver lookups by id work
        for (i, b) in a2.bonds.iter_mut().enumerate() {
            b.id = BondId(i as u32);
        }
        // load distribution: by end-face area per fragment
        let mut area: std::collections::BTreeMap<FragmentId, (f64, DVec3)> = std::collections::BTreeMap::new();
        for (f, p) in &load_polys {
            let ai = polygon_integrals(p);
            let e = area.entry(*f).or_insert((0.0, DVec3::ZERO));
            e.0 += ai.area;
            e.1 += ai.first;
        }
        let total_area: f64 = area.values().map(|v| v.0).sum::<f64>().max(1e-300);
        let face_c = area.values().fold(DVec3::ZERO, |s, v| s + v.1) / total_area;
        let mut results = Vec::new();
        for case in cases.as_array().unwrap() {
            let dir = DVec3::from_array(serde_json::from_value(case["direction"].clone()).unwrap());
            let mag = case["magnitude"].as_f64().unwrap();
            let forces: Vec<(FragmentId, DVec3)> = if case["kind"] == "torque" {
                // tangential forces ∝ r giving the total torque
                let r2: f64 = area.iter().map(|(_, v)| {
                    let c = v.1 / v.0.max(1e-300) - face_c;
                    (c - dir * c.dot(dir)).length_squared() * v.0
                }).sum::<f64>().max(1e-300);
                area.iter().map(|(f, v)| {
                    let c = v.1 / v.0.max(1e-300) - face_c;
                    let r = c - dir * c.dot(dir);
                    (*f, dir.cross(r) * (mag * v.0 / r2))
                }).collect()
            } else {
                area.iter().map(|(f, v)| (*f, dir * (mag * v.0 / total_area))).collect()
            };
            let lc = LoadCase { name: case["name"].as_str().unwrap().into(), gravity: DVec3::ZERO, forces: forces.clone(), fixed: Vec::new() };
            let r = solver.static_solve(&a2, level, &lc);
            let r0 = a2.hierarchy.level_ranges[level as usize].start;
            // response at the loaded end: area-weighted displacement (rotation for torque)
            let mut resp = 0.0;
            for (f, v) in &area {
                let (t, rot) = r.displacements[(f.0 - r0) as usize];
                let c = v.1 / v.0.max(1e-300);
                let fr = a2.fragment(*f);
                let disp = t + rot.cross(c - fr.mass.com);
                resp += if case["kind"] == "torque" { rot.dot(dir) } else { disp.dot(dir) } * v.0 / total_area;
            }
            let tractions: Vec<Value> = r
                .bond_forces
                .iter()
                .filter(|bf| (bf.bond.0 as usize) < a2.bonds.len() && !a2.bonds[bf.bond.idx()].interfaces.is_empty())
                .map(|bf| {
                    let b = &a2.bonds[bf.bond.idx()];
                    json!({"interfaces": b.interfaces, "a": b.a.0, "b": b.b.as_i64(), "area": b.area, "normal": b.normal.to_array(), "centroid": b.centroid.to_array(), "traction": bf.traction, "shear": (bf.shear_force / b.area.max(1e-300)).to_array()})
                })
                .collect();
            results.push(json!({"case": case["name"], "response": resp, "stiffness": if resp != 0.0 { mag / resp } else { 0.0 }, "bonds": tractions}));
        }
        // self weight with the same clamp
        let lc = LoadCase { name: "self_weight".into(), gravity: DVec3::new(0.0, -9.81, 0.0), forces: Vec::new(), fixed: Vec::new() };
        let r = solver.static_solve(&a2, level, &lc);
        let sw: Vec<Value> = r
            .bond_forces
            .iter()
            .filter(|bf| !a2.bonds[bf.bond.idx()].interfaces.is_empty())
            .map(|bf| {
                let b = &a2.bonds[bf.bond.idx()];
                json!({"interfaces": b.interfaces, "area": b.area, "normal": b.normal.to_array(), "traction": bf.traction})
            })
            .collect();
        results.push(json!({"case": "self_weight", "bonds": sw}));
        let freqs = solver.modal(&a2, level, 10);
        levels.push(json!({"level": level, "fragments": a2.level_fragments(level).len(), "results": results, "modal_hz": freqs, "supports": fixed_polys.len()}));
    }
    let materials: Vec<Value> = asset
        .components
        .iter()
        .map(|c| {
            let e = lib.elastic(c.material);
            json!({"component": c.name, "material": lib.material(c.material).id, "E": e.youngs, "nu": e.poisson, "G": e.shear, "rho": e.density})
        })
        .collect();
    json!({
        "asset": asset.meta.name,
        "materials": materials,
        "axis": ax,
        "fixed_plane": lo,
        "loaded_plane": hi,
        "length": len,
        "load_cases": cases,
        "levels": levels,
    })
}

/// Run the Python oracle harness (Kratos FEM) if available.
pub fn run_harness(asset_json: &std::path::Path, network_json: &std::path::Path, cache: &std::path::Path) -> String {
    let py = std::env::var("PREFRACTURE_PYTHON").unwrap_or_else(|_| {
        if std::path::Path::new("/opt/fracenv/bin/python").exists() { "/opt/fracenv/bin/python".into() } else { "python3".into() }
    });
    let script = std::env::var("PREFRACTURE_HARNESS").unwrap_or_else(|_| "tools/harness/bond_fidelity.py".into());
    let out = std::process::Command::new(&py)
        .arg(&script)
        .arg("--asset")
        .arg(asset_json)
        .arg("--network")
        .arg(network_json)
        .arg("--cache")
        .arg(cache)
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
        Ok(o) => format!("harness failed:\n```\n{}\n```\n", String::from_utf8_lossy(&o.stderr)),
        Err(e) => format!("harness not runnable ({py} {script}): {e}\n"),
    }
}
