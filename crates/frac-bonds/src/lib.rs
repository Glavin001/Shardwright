//! Stage 6 (interfaces & bonds) and Stage 10 (crack spawn data).
//!
//! All bond quantities are measured on the clean interface polygons, which
//! are exact unions of fine-cell interfaces; hence child bond areas sum to
//! parent bond areas up to floating-point summation order.

pub mod contacts;

use frac_core::*;
use frac_geom::integrals::sym_eigen3;
use frac_geom::polygon::{area_integrals, newell, outer, plane_basis, simplify_loop, AreaIntegrals};
use frac_geom::DVec3;
use glam::DMat3;
use smallvec::SmallVec;
use std::collections::BTreeMap;

/// Build a clean polygon from a patch (optionally flipped).
pub fn patch_polygon(verts: &[DVec3], p: &Patch, flip: bool) -> Polygon3 {
    let loops: Vec<Vec<DVec3>> = p
        .loops
        .iter()
        .map(|l| {
            let mut v: Vec<DVec3> = l.iter().map(|&i| verts[i as usize]).collect();
            if flip {
                v.reverse();
            }
            v
        })
        .collect();
    Polygon3 { loops, normal: if flip { -p.normal } else { p.normal } }
}

pub fn polygon_integrals(p: &Polygon3) -> AreaIntegrals {
    area_integrals(&p.loops, p.normal)
}

/// Intersect rebar polylines with an interface's polygons. Returns the
/// crossing summary (count, steel area, area-weighted direction oriented
/// along the interface normal) and the crossing points with diameters.
pub fn rebar_crossings(polys: &[Polygon3], rebar: &[frac_core::input::RebarSpec]) -> (RebarCrossing, Vec<(DVec3, f64)>) {
    let mut rc = RebarCrossing::default();
    let mut pts = Vec::new();
    for bar in rebar {
        let area = std::f64::consts::PI * bar.diameter * bar.diameter * 0.25;
        for w in bar.points.windows(2) {
            let a = DVec3::from_array(w[0]);
            let b = DVec3::from_array(w[1]);
            for poly in polys {
                let n = poly.normal;
                let r = poly.loops[0][0];
                let da = (a - r).dot(n);
                let db = (b - r).dot(n);
                if (da > 0.0) == (db > 0.0) || da == db {
                    continue;
                }
                let t = da / (da - db);
                let x = a + (b - a) * t;
                if point_in_polygon3(poly, x) {
                    let mut d = (b - a).normalize_or_zero();
                    if d.dot(n) < 0.0 {
                        d = -d;
                    }
                    rc.count += 1;
                    rc.steel_area += area;
                    rc.dir += d * area;
                    pts.push((x, bar.diameter));
                }
            }
        }
    }
    (rc, pts)
}

/// Point-in-polygon (with holes) for a point on the polygon plane.
pub fn point_in_polygon3(p: &Polygon3, x: DVec3) -> bool {
    let (u, v) = plane_basis(p.normal);
    let q = [x.dot(u), x.dot(v)];
    let mut inside = false;
    for l in &p.loops {
        let n = l.len();
        for k in 0..n {
            let a = [l[k].dot(u), l[k].dot(v)];
            let b = [l[(k + 1) % n].dot(u), l[(k + 1) % n].dot(v)];
            if (a[1] > q[1]) != (b[1] > q[1]) {
                let xi = a[0] + (q[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
                if q[0] < xi {
                    inside = !inside;
                }
            }
        }
    }
    inside
}

/// Inputs to bond construction.
pub struct BondParams<'a> {
    pub seed: u64,
    pub weibull: &'a dyn Fn(Option<MaterialId>, MaterialId) -> f64,
    pub loop_simplify: f64,
    pub spawn_density: f64,
    pub max_spawn_per_bond: u32,
}

pub struct BondOutput {
    pub bonds: Vec<Bond>,
    pub bond_children: Vec<BondId>,
    pub loops: Vec<Vec<DVec3>>,
    pub spawn: Vec<SpawnPoint>,
    /// Interior area per fragment.
    pub interior_area: Vec<f64>,
}

#[derive(Clone)]
struct Acc {
    ai: AreaIntegrals,
    /// Σ A_i n_i using true polygon normals oriented A->B
    an: DVec3,
    comp: BTreeMap<(InterfaceKind, Option<MaterialId>), f64>,
    rebar: RebarCrossing,
    interfaces: Vec<InterfaceId>,
    polys: Vec<Polygon3>,
}

/// Build bonds at every level from the asset's interfaces and hierarchy.
pub fn build_bonds(asset: &Asset, p: &BondParams) -> BondOutput {
    let h = &asset.hierarchy;
    let nl = h.levels as usize;
    // precompute oriented polygons/integrals per interface
    let iface_ai: Vec<AreaIntegrals> = asset
        .interfaces
        .iter()
        .map(|it| {
            let mut a = AreaIntegrals::default();
            for poly in &it.polygons {
                a.add(&polygon_integrals(poly));
            }
            a
        })
        .collect();
    let mut bonds: Vec<Bond> = Vec::new();
    let mut loops: Vec<Vec<DVec3>> = Vec::new();
    let mut spawn: Vec<SpawnPoint> = Vec::new();
    let mut interior_area = vec![0.0f64; h.fragments.len()];
    let mut per_level_index: Vec<BTreeMap<(u32, i64), u32>> = vec![BTreeMap::new(); nl];
    for level in 0..nl {
        let mut accs: BTreeMap<(u32, i64), Acc> = BTreeMap::new();
        for (ii, it) in asset.interfaces.iter().enumerate() {
            let fa = h.cell_fragment[level][it.cells.0.idx()];
            let (fb, world) = match it.cells.1 {
                CellOrWorld::Cell(c) => (Some(h.cell_fragment[level][c.idx()]), false),
                CellOrWorld::World => (None, true),
            };
            if !world && fb == Some(fa) {
                continue;
            }
            if !it.patches.is_empty() {
                interior_area[fa.idx()] += it.area;
                if let Some(b) = fb {
                    interior_area[b.idx()] += it.area;
                }
            }
            // orientation: A = smaller fragment id (world always B)
            let (a, b, flip) = match fb {
                Some(b) if b < fa => (b, b_or(fa), true),
                Some(b) => (fa, b_or(b), false),
                None => (fa, -1i64, false),
            };
            let acc = accs.entry((a.0, b)).or_insert_with(|| Acc {
                ai: AreaIntegrals::default(),
                an: DVec3::ZERO,
                comp: BTreeMap::new(),
                rebar: RebarCrossing::default(),
                interfaces: Vec::new(),
                polys: Vec::new(),
            });
            let sgn = if flip { -1.0 } else { 1.0 };
            let mut ai = iface_ai[ii];
            ai.area_vec *= sgn;
            acc.ai.add(&ai);
            acc.an += ai.area_vec;
            *acc.comp.entry((it.kind, it.interface_material)).or_default() += it.area;
            let mut rb = it.rebar;
            rb.dir *= sgn;
            acc.rebar.add(&rb);
            acc.interfaces.push(it.id);
            for poly in &it.polygons {
                let mut q = poly.clone();
                if flip {
                    q.normal = -q.normal;
                    for l in q.loops.iter_mut() {
                        l.reverse();
                    }
                }
                acc.polys.push(q);
            }
        }
        for ((a, b), acc) in accs {
            let id = BondId(bonds.len() as u32);
            per_level_index[level].insert((a, b), id.0);
            let fa = &h.fragments[a as usize];
            let area = acc.ai.area;
            let centroid = acc.ai.centroid();
            let anl = acc.an.length();
            let normal = if anl > 0.0 { acc.an / anl } else { DVec3::Y };
            let planarity = if area > 0.0 { (anl / area).min(1.0) } else { 1.0 };
            // in-plane second moment tensor about centroid, projected
            let c2 = acc.ai.central_second();
            let (u0, v0) = plane_basis(normal);
            let m2 = [
                [u0.dot(c2 * u0), u0.dot(c2 * v0)],
                [v0.dot(c2 * u0), v0.dot(c2 * v0)],
            ];
            // principal in-plane directions
            let theta = 0.5 * libm::atan2(2.0 * m2[0][1], m2[0][0] - m2[1][1]);
            let (ct, st) = (libm::cos(theta), libm::sin(theta));
            let mut fu = u0 * ct + v0 * st;
            // canonical sign for determinism
            let am = fu.abs();
            let lead = if am.x >= am.y && am.x >= am.z { fu.x } else if am.y >= am.z { fu.y } else { fu.z };
            if lead < 0.0 {
                fu = -fu;
            }
            let fv = normal.cross(fu);
            // I_uu = ∫ v² dA (about the u axis), I_vv = ∫ u² dA
            let i_uu = fv.dot(c2 * fv);
            let i_vv = fu.dot(c2 * fu);
            let i_uv = fu.dot(c2 * fv);
            let j = i_uu + i_vv;
            // extent: project polygon vertices into the frame
            let (mut umin, mut umax, mut vmin, mut vmax) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
            for poly in &acc.polys {
                for l in &poly.loops {
                    for &x in l {
                        let d = x - centroid;
                        let (uu, vv) = (d.dot(fu), d.dot(fv));
                        umin = umin.min(uu);
                        umax = umax.max(uu);
                        vmin = vmin.min(vv);
                        vmax = vmax.max(vv);
                    }
                }
            }
            let extent = if umin.is_finite() {
                Obb2 {
                    center: centroid + fu * (0.5 * (umin + umax)) + fv * (0.5 * (vmin + vmax)),
                    axis_u: fu,
                    axis_v: fv,
                    half: [0.5 * (umax - umin), 0.5 * (vmax - vmin)],
                }
            } else {
                Obb2::default()
            };
            let dist_a = (fa.mass.com - centroid).dot(normal).abs();
            let dist_b = if b >= 0 { (h.fragments[b as usize].mass.com - centroid).dot(normal).abs() } else { 0.0 };
            // composition: dominant first, max 4
            let mut comp: Vec<((InterfaceKind, Option<MaterialId>), f64)> = acc.comp.into_iter().collect();
            comp.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap().then(x.0.cmp(&y.0)));
            let comp_total: f64 = comp.iter().map(|c| c.1).sum::<f64>().max(1e-300);
            let composition: SmallVec<[BondComposition; 4]> = comp
                .iter()
                .take(4)
                .map(|((k, m), w)| BondComposition { kind: *k, interface_material: *m, fraction: (w / comp_total) as f32 })
                .collect();
            // Weibull strength scale (mean-normalized)
            let dom_mat = composition.first().and_then(|c| c.interface_material);
            let base_mat = fa.material_mix.first().map(|m| m.0).unwrap_or(MaterialId(0));
            let m = (p.weibull)(dom_mat, base_mat).max(0.5);
            let u = unit_f64(stable_hash(&[p.seed, level as u64, a as u64, b as u64, 0x5eed])).clamp(1e-12, 1.0 - 1e-12);
            let raw = libm::pow(-libm::log(u), 1.0 / m);
            let strength_scale = (raw / libm::tgamma(1.0 + 1.0 / m)) as f32;
            // boundary loops
            let bl = boundary_loops(&acc.polys);
            let loop_begin = loops.len() as u32;
            let tol = p.loop_simplify * area.sqrt();
            for l in bl {
                let s = simplify_loop(&l, tol);
                if s.len() >= 2 {
                    loops.push(s);
                }
            }
            let loop_end = loops.len() as u32;
            // spawn samples
            let sp_begin = spawn.len() as u32;
            let n_sp = ((area * p.spawn_density).ceil() as u32).clamp(1, p.max_spawn_per_bond.max(1));
            sample_polys(&acc.polys, n_sp, stable_hash(&[p.seed, level as u64, a as u64, b as u64, 0x5a]), &mut spawn);
            let sp_end = spawn.len() as u32;
            bonds.push(Bond {
                id,
                level: level as u8,
                a: FragmentId(a),
                b: if b >= 0 { FragmentOrWorld::Fragment(FragmentId(b as u32)) } else { FragmentOrWorld::World },
                area,
                centroid,
                normal,
                planarity: planarity as f32,
                frame_u: fu,
                frame_v: fv,
                i_uu,
                i_vv,
                i_uv,
                j,
                extent,
                dist_a,
                dist_b,
                composition,
                reinforcement: acc.rebar,
                strength_scale,
                parent_bond: None,
                child_bonds: 0..0,
                boundary_loops: loop_begin..loop_end,
                spawn: sp_begin..sp_end,
                anchor: b < 0,
                interfaces: acc.interfaces,
            });
        }
    }
    // parent/child links
    let mut children: Vec<Vec<BondId>> = vec![Vec::new(); bonds.len()];
    for i in 0..bonds.len() {
        let level = bonds[i].level as usize;
        if level == 0 {
            continue;
        }
        let pa = h.fragments[bonds[i].a.idx()].parent.unwrap();
        let pb: i64 = match bonds[i].b {
            FragmentOrWorld::Fragment(f) => h.fragments[f.idx()].parent.unwrap().0 as i64,
            FragmentOrWorld::World => -1,
        };
        if pb >= 0 && pb as u32 == pa.0 {
            continue; // internal to the parent fragment
        }
        let key = if pb >= 0 && (pb as u32) < pa.0 { (pb as u32, pa.0 as i64) } else { (pa.0, pb) };
        if let Some(&pid) = per_level_index[level - 1].get(&key) {
            bonds[i].parent_bond = Some(BondId(pid));
            children[pid as usize].push(BondId(i as u32));
        }
    }
    let mut bond_children = Vec::new();
    for (i, ch) in children.into_iter().enumerate() {
        let s = bond_children.len() as u32;
        bond_children.extend(ch);
        bonds[i].child_bonds = s..bond_children.len() as u32;
    }
    BondOutput { bonds, bond_children, loops, spawn, interior_area }
}

#[inline]
fn b_or(f: FragmentId) -> i64 {
    f.0 as i64
}

/// Boundary of a union of polygons: edges used once (keyed by exact
/// coordinates), chained into closed loops.
pub fn boundary_loops(polys: &[Polygon3]) -> Vec<Vec<DVec3>> {
    type K = [u64; 3];
    let key = |p: DVec3| -> K { [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()] };
    let mut pos: BTreeMap<K, DVec3> = BTreeMap::new();
    let mut cnt: BTreeMap<(K, K), i32> = BTreeMap::new();
    for poly in polys {
        for l in &poly.loops {
            let n = l.len();
            for k in 0..n {
                let (a, b) = (key(l[k]), key(l[(k + 1) % n]));
                if a == b {
                    continue;
                }
                pos.insert(a, l[k]);
                pos.insert(b, l[(k + 1) % n]);
                *cnt.entry((a, b)).or_default() += 1;
            }
        }
    }
    let mut next: BTreeMap<K, Vec<K>> = BTreeMap::new();
    for (&(a, b), &c) in &cnt {
        let r = cnt.get(&(b, a)).copied().unwrap_or(0);
        for _ in 0..(c - r).max(0) {
            next.entry(a).or_default().push(b);
        }
    }
    let mut out = Vec::new();
    while let Some((&start, _)) = next.iter().find(|(_, v)| !v.is_empty()) {
        let mut lp = vec![pos[&start]];
        let mut cur = start;
        let mut guard = 0;
        loop {
            let v = next.get_mut(&cur).unwrap();
            let nx = v.remove(0);
            if nx == start {
                break;
            }
            lp.push(pos[&nx]);
            cur = nx;
            guard += 1;
            if guard > 1_000_000 || next.get(&cur).map(|v| v.is_empty()).unwrap_or(true) {
                break;
            }
        }
        out.push(lp);
    }
    out
}

/// Deterministic area-weighted samples on polygons (fan triangulation).
fn sample_polys(polys: &[Polygon3], n: u32, seed: u64, out: &mut Vec<SpawnPoint>) {
    let mut tris: Vec<(DVec3, DVec3, DVec3, DVec3, f64)> = Vec::new();
    for p in polys {
        let r = p.loops[0][0];
        for l in &p.loops {
            for k in 0..l.len() {
                let a = l[k];
                let b = l[(k + 1) % l.len()];
                let ar = 0.5 * (a - r).cross(b - r).dot(p.normal);
                if ar.abs() > 0.0 {
                    tris.push((r, a, b, p.normal, ar));
                }
            }
        }
    }
    let total: f64 = tris.iter().map(|t| t.4.max(0.0)).sum();
    if total <= 0.0 {
        return;
    }
    let mut taken = 0u32;
    let mut k = 0u64;
    while taken < n && k < (n as u64) * 64 {
        let u = unit_f64(stable_hash(&[seed, k, 1])) * total;
        let r1 = unit_f64(stable_hash(&[seed, k, 2]));
        let r2 = unit_f64(stable_hash(&[seed, k, 3]));
        k += 1;
        let mut acc = 0.0;
        for t in &tris {
            if t.4 <= 0.0 {
                continue;
            }
            acc += t.4;
            if u <= acc {
                let s = r1.sqrt();
                let x = t.0 * (1.0 - s) + t.1 * (s * (1.0 - r2)) + t.2 * (s * r2);
                // reject samples falling in holes (negative fan triangles cover them)
                if polys.iter().any(|p| p.normal.dot(t.3) > 0.999 && point_in_polygon3(p, x)) {
                    out.push(SpawnPoint { p: x, n: t.3 });
                    taken += 1;
                }
                break;
            }
        }
    }
}

/// Newell-normal helper for loops (re-exported for tooling).
pub fn loop_normal(l: &[DVec3]) -> DVec3 {
    newell(l).normalize_or_zero()
}

/// Second-moment helper (outer product), re-exported.
pub fn outer_product(a: DVec3, b: DVec3) -> DMat3 {
    outer(a, b)
}

/// Eigen helper re-export for validation.
pub fn eigen_sym(m: &DMat3) -> (DVec3, DMat3) {
    sym_eigen3(m)
}
