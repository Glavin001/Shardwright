//! Structural soundness of multi-component assets (buildings):
//!
//! * `structural_support`: every structural component (column, beam, slab,
//!   connection, generic) has a load path to the ground through structural
//!   components only; every non-structural component (wall, cosmetic, prop,
//!   glazing) is carried by something that reaches the ground.
//! * `self_weight`: the component-level (L0) bond network under gravity,
//!   with every joint checked against its capacity: peak normal stress from
//!   the axial force and the bending moments, tension against the joint's
//!   tensile strength (interface material, else the weaker side), compression
//!   against the weaker side's compressive strength, shear against cohesion
//!   plus friction (μ = 0.6). Compression-only bearing joints are checked for
//!   uplift and overturning (resultant inside the contact: e ≤ h/2, i.e.
//!   σ_bending ≤ 3|σ_axial| for a rectangular section).
//!
//! Both are not applicable to assets without ground anchors.

use crate::network::{BondNetworkSolver, LoadCase, ReferenceSolver, StiffnessModel};
use frac_core::*;
use frac_material::MaterialLibrary;
use glam::DVec3;

const FRICTION: f64 = 0.6;

pub fn is_structural(r: ComponentRole) -> bool {
    matches!(r, ComponentRole::Column | ComponentRole::Beam | ComponentRole::Slab | ComponentRole::Connection | ComponentRole::Generic)
}

pub struct SupportReport {
    pub applicable: bool,
    pub unsupported_structural: Vec<String>,
    pub unsupported_other: Vec<String>,
}

/// Load-path connectivity on the L0 (component) bond graph.
pub fn support(asset: &Asset) -> SupportReport {
    let n = asset.level_fragments(0).len();
    let r0 = asset.hierarchy.level_ranges[0].start;
    let comp_of = |f: FragmentId| asset.fragment(f).component.idx();
    let mut adj = vec![Vec::new(); n];
    let mut grounded = vec![false; n];
    for b in asset.level_bonds(0) {
        let a = (b.a.0 - r0) as usize;
        match b.b {
            FragmentOrWorld::World => grounded[a] = true,
            FragmentOrWorld::Fragment(f) => {
                let c = (f.0 - r0) as usize;
                adj[a].push(c);
                adj[c].push(a);
            }
        }
    }
    if !grounded.iter().any(|&g| g) {
        return SupportReport { applicable: false, unsupported_structural: Vec::new(), unsupported_other: Vec::new() };
    }
    let role = |i: usize| asset.components[comp_of(FragmentId(r0 + i as u32))].role;
    let reach = |only_structural: bool| -> Vec<bool> {
        let ok = |i: usize| !only_structural || is_structural(role(i));
        let mut seen = vec![false; n];
        let mut stack: Vec<usize> = (0..n).filter(|&i| grounded[i] && ok(i)).collect();
        for &i in &stack {
            seen[i] = true;
        }
        while let Some(i) = stack.pop() {
            for &j in &adj[i] {
                if !seen[j] && ok(j) {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        seen
    };
    let (rs, ra) = (reach(true), reach(false));
    let name = |i: usize| asset.components[comp_of(FragmentId(r0 + i as u32))].name.clone();
    SupportReport {
        applicable: true,
        unsupported_structural: (0..n).filter(|&i| is_structural(role(i)) && !rs[i]).map(name).collect(),
        unsupported_other: (0..n).filter(|&i| !is_structural(role(i)) && !ra[i]).map(name).collect(),
    }
}

pub struct JointCheck {
    pub a: String,
    pub b: String,
    pub kind: InterfaceKind,
    pub utilization: f64,
    pub mode: &'static str,
}

/// Self-weight joint utilization at L0 (empty when not applicable).
pub fn self_weight(asset: &Asset, lib: &MaterialLibrary) -> Vec<JointCheck> {
    if asset.hierarchy.levels == 0 || !asset.level_bonds(0).any(|b| matches!(b.b, FragmentOrWorld::World)) {
        return Vec::new();
    }
    let solver = ReferenceSolver { lib, model: StiffnessModel::Spec };
    let lc = LoadCase { name: "self_weight".into(), gravity: DVec3::new(0.0, -9.81, 0.0), forces: Vec::new(), force_points: Vec::new(), moments: Vec::new(), fixed: Vec::new() };
    let res = solver.static_solve(asset, 0, &lc);
    let mat_of = |f: FragmentId| asset.fragment(f).material_mix.first().map(|x| x.0).unwrap_or(MaterialId(0));
    let strengths = |m: MaterialId| {
        let mm = lib.material(m);
        let ft = mm.tensile_strength.unwrap_or(1e6);
        (ft, mm.compressive_strength.unwrap_or(10.0 * ft))
    };
    let mut out = Vec::new();
    for bf in &res.bond_forces {
        let b = &asset.bonds[bf.bond.idx()];
        let FragmentOrWorld::Fragment(fb) = b.b else { continue };
        let (fta, fca) = strengths(mat_of(b.a));
        let (ftb, fcb) = strengths(mat_of(fb));
        let comp = b.composition.first();
        let kind = comp.map(|c| c.kind).unwrap_or(InterfaceKind::ComponentConnection);
        let ft = comp
            .and_then(|c| c.interface_material)
            .and_then(|m| lib.interface_material(m))
            .and_then(|im| im.tensile_strength)
            .unwrap_or(fta.min(ftb));
        let fc = fca.min(fcb);
        let area = b.area.max(1e-300);
        let t = bf.traction_vec;
        let sn = t.dot(b.normal); // > 0 tension
        let tau = (t - b.normal * sn).length();
        let c = b.extent.half[0].max(b.extent.half[1]);
        let sb = bf.moment.dot(b.frame_u).abs() * c / b.i_uu.max(1e-300) + bf.moment.dot(b.frame_v).abs() * c / b.i_vv.max(1e-300);
        let _ = area;
        let (ut, mode_t) = if ft > 0.0 {
            ((sn + sb).max(0.0) / ft, "tension")
        } else if sn > 1e-9 * fc {
            (f64::INFINITY, "uplift")
        } else {
            (sb / (3.0 * sn.abs()).max(1e-300), "overturning")
        };
        let uc = (-sn + sb).max(0.0) / fc;
        let us = tau / (ft + FRICTION * (-sn).max(0.0)).max(1e-300);
        let (u, mode) = [(ut, mode_t), (uc, "compression"), (us, "shear")].into_iter().fold((0.0, "none"), |acc, x| if x.0 > acc.0 { x } else { acc });
        let name = |f: FragmentId| asset.components[asset.fragment(f).component.idx()].name.clone();
        out.push(JointCheck { a: name(b.a), b: name(fb), kind, utilization: if tau == 0.0 && sn == 0.0 && sb == 0.0 { 0.0 } else { u }, mode });
    }
    out
}
