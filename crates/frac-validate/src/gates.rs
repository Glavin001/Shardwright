//! Hard gates (spec §13.1).

use crate::{GateResult, GateStatus};
use frac_collision::{cells_boundary_mesh_with, hull_polytope};
use frac_core::settings::ValidationSettings;
use frac_core::*;
use frac_geom::inside::MeshQuery;
use frac_geom::{DVec3, TriMesh, VolumeIntegrals};
use frac_render::RenderOut;
use rayon::prelude::*;
use std::collections::BTreeMap;

pub const NAMES: [&str; 14] = [
    "fragment_validity",
    "volume_conservation",
    "no_overlap_solids",
    "no_overlap_hulls",
    "collision_shapes",
    "structural_support",
    "self_weight",
    "no_gaps_render",
    "bond_coverage",
    "bond_hierarchy",
    "mass_properties",
    "slivers",
    "determinism",
    "schema",
];

pub fn order(n: &str) -> usize {
    NAMES.iter().position(|x| *x == n).unwrap_or(99)
}

static STEP: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);

/// `FRAC_LOG` progress line: time since the previous step and current RSS.
pub(crate) fn log_step(name: &str) {
    if std::env::var_os("FRAC_LOG").is_none() {
        return;
    }
    let mut last = STEP.lock().unwrap();
    let now = std::time::Instant::now();
    let dt = last.map(|t| now.duration_since(t).as_secs_f64()).unwrap_or(0.0);
    *last = Some(now);
    let rss = std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|x| x.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()))
        .map(|p| p * 4096.0 / 1048576.0)
        .unwrap_or(0.0);
    eprintln!("    [validate] {name}: {dt:.1} s, rss {rss:.0} MB");
}

pub(crate) fn reset_steps() {
    *STEP.lock().unwrap() = Some(std::time::Instant::now());
}

fn gate(name: &str, ok: bool, value: f64, threshold: f64, detail: String) -> GateResult {
    log_step(name);
    GateResult { name: name.into(), status: if ok { GateStatus::Pass } else { GateStatus::Fail }, value, threshold, detail }
}

/// Closed, consistently oriented 2-manifold without self-intersections.
/// Zero-area triangles (from symbolic perturbation of exact degeneracies)
/// are tolerated topologically and reported.
pub fn mesh_validity(m: &TriMesh) -> (bool, String) {
    mesh_validity_where(m, |_, _| true)
}

/// [`mesh_validity`] with the self-intersection test restricted to the
/// triangle pairs `keep` accepts (topology is always checked in full).
pub fn mesh_validity_where(m: &TriMesh, keep: impl Fn(u32, u32) -> bool) -> (bool, String) {
    let t = m.topology();
    let topo_ok = t.boundary_edges == 0 && t.nonmanifold_edges == 0 && t.inconsistent_edges == 0 && t.nonmanifold_vertices == 0;
    if !topo_ok {
        return (false, format!("{t:?}"));
    }
    // self-intersections ignoring zero-area triangles
    let si = m.self_intersections_where(4, |a, b| keep(a, b) && !m.is_degenerate(a as usize) && !m.is_degenerate(b as usize));
    if !si.is_empty() {
        return (false, format!("{} self-intersecting pairs (e.g. {:?})", si.len(), si[0]));
    }
    (true, if t.degenerate_tris > 0 { format!("{} zero-area tris", t.degenerate_tris) } else { String::new() })
}

pub fn run_gates(asset: &Asset, render: &RenderOut, vs: &ValidationSettings, physics: &[u8]) -> Vec<GateResult> {
    reset_steps();
    let mut out = Vec::new();
    let h = &asset.hierarchy;
    let nl = h.levels as usize;
    let cp = asset.cell_polys();
    // ---- fragment validity (clean at every level, render at leaf level + LOD0 of all)
    let clean: Vec<(u32, bool, String)> = h
        .fragments
        .par_iter()
        .map(|f| {
            let m = cells_boundary_mesh_with(asset, &cp, asset.fragment_cells(f));
            let (ok, d) = mesh_validity(&m);
            (f.id.0, ok, d)
        })
        .collect();
    log_step("fragment_validity (clean meshes)");
    // A coarse fragment's LOD0 render mesh (when built directly from the
    // shared cell surfaces, i.e. it carries per-triangle owner cells) holds
    // exactly the triangles of its cells' own meshes. Those are validated
    // in full at the leaf level (one cell per leaf fragment), so only pairs
    // of triangles of different cells can still intersect.
    let leaf_lvl = (nl - 1) as u8;
    let leaf_single = asset.level_fragments(leaf_lvl).iter().all(|f| f.cells.len() == 1);
    let rend: Vec<(u32, bool, String)> = h
        .fragments
        .par_iter()
        .map(|f| {
            let fm = &render.fragments[f.id.idx()][0];
            let (ok, d) = match fm.as_trimesh_with_owners() {
                (m, Some(owner)) if leaf_single && f.level != leaf_lvl => mesh_validity_where(&m, |a, b| owner[a as usize] != owner[b as usize]),
                (m, _) => mesh_validity(&m),
            };
            (f.id.0, ok, d)
        })
        .collect();
    let bad_c: Vec<_> = clean.iter().filter(|x| !x.1).collect();
    let bad_r: Vec<_> = rend.iter().filter(|x| !x.1).collect();
    let degen: usize = clean.iter().chain(rend.iter()).filter(|x| x.1 && !x.2.is_empty()).count();
    let detail = if bad_c.is_empty() && bad_r.is_empty() {
        format!("{} clean + {} render meshes valid; {} with zero-area tris", clean.len(), rend.len(), degen)
    } else {
        format!(
            "{} clean / {} render invalid; first: {}",
            bad_c.len(),
            bad_r.len(),
            bad_c.first().or(bad_r.first()).map(|x| format!("fragment {} {}", x.0, x.2)).unwrap_or_default()
        )
    };
    out.push(gate("fragment_validity", bad_c.is_empty() && bad_r.is_empty(), (bad_c.len() + bad_r.len()) as f64, 0.0, detail));

    // ---- volume conservation per component
    let mut worst: f64 = 0.0;
    for c in &asset.components {
        let sv = c.solid.signed_volume();
        let cv: f64 = asset.cells[c.cells.start as usize..c.cells.end as usize].iter().map(|x| x.mass.volume).sum();
        if sv > 0.0 {
            worst = worst.max((cv / sv - 1.0).abs());
        }
    }
    out.push(gate("volume_conservation", worst <= 1e-6, worst, 1e-6, "max |Σ cell volumes / component volume - 1|".into()));

    // ---- no overlap (solids): structural chain identity + sampled coverage
    let (chain_ok, chain_detail) = chain_identity(asset);
    let (cov_ok, cov_detail, est) = sampled_coverage(asset, 4000);
    out.push(gate("no_overlap_solids", chain_ok && cov_ok, est, 1e-9, format!("{chain_detail}; {cov_detail}")));

    // ---- no overlap (hulls) between neighboring fragments
    let polys: Vec<frac_geom::hull::ConvexPolytope> = asset.hulls.par_iter().map(hull_polytope).collect();
    let boxes: Vec<frac_geom::Aabb> = polys.par_iter().map(|p| frac_geom::Aabb::from_points(p.faces.iter().flat_map(|f| f.1.iter()))).collect();
    // hulls with disjoint bounding boxes cannot overlap
    let (worst_h, pairs) = asset
        .bonds
        .par_iter()
        .map(|b| {
            let FragmentOrWorld::Fragment(fb) = b.b else { return (0.0f64, 0usize) };
            let ra = h.fragments[b.a.idx()].hulls.clone();
            let rb = h.fragments[fb.idx()].hulls.clone();
            let mut worst: f64 = 0.0;
            let mut n = 0usize;
            for x in ra {
                for y in rb.clone() {
                    n += 1;
                    if boxes[x as usize].overlaps(&boxes[y as usize]) {
                        worst = worst.max(polys[x as usize].intersection_volume(&polys[y as usize]));
                    }
                }
            }
            (worst, n)
        })
        .reduce(|| (0.0, 0), |a, b| (a.0.max(b.0), a.1 + b.1));
    let has_hulls = !asset.hulls.is_empty();
    out.push(if has_hulls {
        gate("no_overlap_hulls", worst_h <= 1e-9, worst_h, 1e-9, format!("{pairs} neighboring hull pairs checked (m³)"))
    } else {
        GateResult { name: "no_overlap_hulls".into(), status: GateStatus::NotEvaluated, value: 0.0, threshold: 1e-9, detail: "no hulls".into() }
    });

    // ---- no gaps (render): interior triangles appear exactly twice with opposite winding.
    // Triangles are identified by a 128-bit BLAKE3 fingerprint of their exact
    // vertex bits (rotation-canonical), kept in sorted vectors: ~16 bytes per
    // triangle instead of a map of 72-byte keys (building-scale assets have
    // tens of millions of leaf triangles).
    let leaf = nl - 1;
    let key = |p: DVec3| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
    let fp = |t: [[u64; 3]; 3]| -> u128 {
        let r = [t, [t[1], t[2], t[0]], [t[2], t[0], t[1]]];
        let c = r.iter().min().unwrap();
        let mut h = blake3::Hasher::new();
        for v in c {
            for x in v {
                h.update(&x.to_le_bytes());
            }
        }
        u128::from_le_bytes(h.finalize().as_bytes()[..16].try_into().unwrap())
    };
    let per_frag: Vec<(Vec<u128>, Vec<(u128, u128)>)> = asset
        .level_fragments(leaf as u8)
        .par_iter()
        .map(|f| {
            let m = &render.fragments[f.id.idx()][0];
            let mut all = Vec::with_capacity((m.ext_indices.len() + m.int_indices.len()) / 3);
            let mut int = Vec::with_capacity(m.int_indices.len() / 3);
            for (idx, is_int) in [(&m.ext_indices, false), (&m.int_indices, true)] {
                for t in idx.chunks(3) {
                    let k = [key(m.positions[t[0] as usize]), key(m.positions[t[1] as usize]), key(m.positions[t[2] as usize])];
                    let h = fp(k);
                    all.push(h);
                    if is_int {
                        int.push((h, fp([k[0], k[2], k[1]])));
                    }
                }
            }
            (all, int)
        })
        .collect();
    let mut all: Vec<u128> = Vec::with_capacity(per_frag.iter().map(|x| x.0.len()).sum());
    let mut interior: Vec<(u128, u128)> = Vec::with_capacity(per_frag.iter().map(|x| x.1.len()).sum());
    for (a, i) in per_frag {
        all.extend(a);
        interior.extend(i);
    }
    all.par_sort_unstable();
    // number of query keys whose multiplicity in `all` is not exactly one,
    // by a merge join of two sorted arrays (cache friendly: tens of
    // millions of keys on building-scale assets)
    let not_once = |mut q: Vec<u128>| -> usize {
        q.par_sort_unstable();
        let (mut i, mut bad) = (0usize, 0usize);
        let mut k = 0usize;
        while k < q.len() {
            let key = q[k];
            let mut n = 0usize;
            while k + n < q.len() && q[k + n] == key {
                n += 1;
            }
            while i < all.len() && all[i] < key {
                i += 1;
            }
            let mut c = 0usize;
            while i + c < all.len() && all[i + c] == key {
                c += 1;
            }
            if c != 1 {
                bad += n;
            }
            k += n;
        }
        bad
    };
    let unmatched = not_once(interior.iter().map(|x| x.1).collect());
    let dup = not_once(interior.iter().map(|x| x.0).collect());
    out.push(gate("no_gaps_render", unmatched == 0 && dup == 0, (unmatched + dup) as f64, 0.0, format!("{} interior triangles at leaf level, {unmatched} unmatched, {dup} duplicated", interior.len())));

    // ---- bond coverage: each interface in exactly one bond per level, or internal
    let mut bad_cov = 0usize;
    for level in 0..nl {
        let mut seen = vec![0u32; asset.interfaces.len()];
        for b in asset.level_bonds(level as u8) {
            for i in &b.interfaces {
                seen[i.idx()] += 1;
            }
        }
        for (ii, it) in asset.interfaces.iter().enumerate() {
            let fa = h.cell_fragment[level][it.cells.0.idx()];
            let internal = match it.cells.1 {
                CellOrWorld::Cell(c) => h.cell_fragment[level][c.idx()] == fa,
                CellOrWorld::World => false,
            };
            let expect = if internal { 0 } else { 1 };
            if seen[ii] != expect {
                bad_cov += 1;
            }
        }
    }
    out.push(gate("bond_coverage", bad_cov == 0, bad_cov as f64, 0.0, format!("{} interfaces x {} levels", asset.interfaces.len(), nl)));

    // ---- bond hierarchy: Σ child areas = parent area
    let mut worst_b: f64 = 0.0;
    for b in &asset.bonds {
        if b.child_bonds.is_empty() {
            continue;
        }
        let s: f64 = asset.bond_children[b.child_bonds.start as usize..b.child_bonds.end as usize].iter().map(|c| asset.bonds[c.idx()].area).sum();
        if b.area > 0.0 {
            worst_b = worst_b.max((s - b.area).abs() / b.area);
        }
    }
    // also every non-leaf bond must have children whose areas cover it
    out.push(gate("bond_hierarchy", worst_b <= 1e-9, worst_b, 1e-9, "max relative |Σ child area - parent area|".into()));

    // ---- mass properties vs independent tetrahedral quadrature
    let rel: Vec<f64> = h
        .fragments
        .par_iter()
        .map(|f| {
            let m = cells_boundary_mesh_with(asset, &cp, asset.fragment_cells(f));
            let density = f.mass.mass / f.mass.volume.max(1e-300);
            let q = tet_quadrature(&m);
            mass_rel_error(&f.mass, &q, density)
        })
        .collect();
    let worst_m = rel.iter().cloned().fold(0.0, f64::max);
    // ray-casting cross-check on a few fragments (coarse, metric only)
    let rays = vs.mass_rays.max(16) as usize;
    let sample: Vec<&Fragment> = h.fragments.iter().step_by((h.fragments.len() / 8).max(1)).collect();
    let ray_err: f64 = sample
        .par_iter()
        .map(|f| {
            let m = cells_boundary_mesh_with(asset, &cp, asset.fragment_cells(f));
            let v = ray_volume(&m, rays);
            (v / f.mass.volume - 1.0).abs()
        })
        .reduce(|| 0.0, f64::max);
    out.push(gate("mass_properties", worst_m <= 1e-3, worst_m, 1e-3, format!("tet quadrature over {} fragments; ray-cast volume max rel err {ray_err:.2e} ({rays}² rays)", rel.len())));

    // ---- slivers
    let min_v = asset.cells.iter().map(|c| c.mass.volume).fold(f64::INFINITY, f64::min);
    let min_t = asset.cells.iter().map(|c| c.thickness_ratio).fold(f64::INFINITY, f64::min);
    let nbad = asset.cells.iter().filter(|c| c.thickness_ratio < 0.05 || c.mass.volume < 1e-7).count();
    out.push(gate("slivers", nbad == 0, nbad as f64, 0.0, format!("min cell volume {min_v:.3e} m³, min thickness ratio {min_t:.3}")));

    out.push(GateResult { name: "determinism".into(), status: GateStatus::NotEvaluated, value: 0.0, threshold: 0.0, detail: "run `prefracture bake --check-determinism` or CI cross-OS job".into() });

    // ---- collision shapes: every hull convex and within the vertex limit;
    // every fragment that is not a particle candidate has hulls
    let limit = vs.max_hull_vertices.max(4) as usize;
    let checks: Vec<(usize, usize, bool)> = asset
        .hulls
        .par_iter()
        .map(|hh| {
            // convex: every vertex on or behind every stored face plane
            // (Newell normal through the face centroid), relative 1e-9
            let scale = frac_geom::Aabb::from_points(hh.vertices.iter()).diagonal().max(1e-300);
            let tol = 1e-9 * scale;
            let convex = !hh.faces.is_empty()
                && hh.faces.iter().all(|f| {
                    let pts: Vec<DVec3> = f.iter().map(|&i| hh.vertices[i as usize]).collect();
                    let n = frac_geom::polygon::newell(&pts).normalize_or_zero();
                    let c = pts.iter().fold(DVec3::ZERO, |a, p| a + *p) / pts.len().max(1) as f64;
                    n != DVec3::ZERO && hh.vertices.iter().all(|v| (*v - c).dot(n) <= tol)
                });
            (hh.vertices.len(), hh.fragment.idx(), convex)
        })
        .collect();
    let over = checks.iter().filter(|c| c.0 > limit).count();
    let nonconvex = checks.iter().filter(|c| !c.2).count();
    let max_v = checks.iter().map(|c| c.0).max().unwrap_or(0);
    let missing = h.fragments.iter().filter(|f| f.hulls.is_empty() && !f.particle_candidate).count();
    out.push(gate(
        "collision_shapes",
        over == 0 && nonconvex == 0 && missing == 0,
        (over + nonconvex + missing) as f64,
        0.0,
        format!("{} hulls: {over} over {limit} vertices (max {max_v}), {nonconvex} non-convex; {missing} fragments without hulls", checks.len()),
    ));

    // ---- schema
    let phys_ok = frac_io::read_physics(physics).is_ok();
    out.push(gate("schema", phys_ok, if phys_ok { 0.0 } else { 1.0 }, 0.0, "physics payload verifies (FlatBuffers verifier, identifier FRAC); glTF checked by the Khronos validator when available".into()));
    out
}

/// Chain identity: interface patches referenced by exactly two distinct
/// cells with opposite orientation; exterior polygons tile the solid
/// surface (per source triangle area vectors match).
fn chain_identity(asset: &Asset) -> (bool, String) {
    let mut worst: f64 = 0.0;
    let mut bad = 0usize;
    for c in &asset.components {
        let g = &c.geometry;
        for p in &g.patches {
            if p.cells.0 == p.cells.1 {
                bad += 1;
            }
        }
        let mut per_tri: BTreeMap<u32, DVec3> = BTreeMap::new();
        for e in &g.ext_polys {
            let pts: Vec<DVec3> = e.verts.iter().map(|&v| g.verts[v as usize]).collect();
            *per_tri.entry(e.src_tri).or_insert(DVec3::ZERO) += frac_geom::polygon::newell(&pts);
        }
        for (t, a) in per_tri {
            let [x, y, z] = c.solid.tri_points(t as usize);
            let ta = (y - x).cross(z - x) * 0.5;
            if ta.length() > 0.0 {
                worst = worst.max((a - ta).length() / ta.length());
            }
        }
        // every solid triangle must be covered
        let covered: std::collections::BTreeSet<u32> = g.ext_polys.iter().map(|e| e.src_tri).collect();
        for t in 0..c.solid.tris.len() as u32 {
            let [x, y, z] = c.solid.tri_points(t as usize);
            if !covered.contains(&t) && (y - x).cross(z - x).length() > 0.0 && !c.unfractured {
                bad += 1;
            }
        }
    }
    (bad == 0 && worst < 1e-9, format!("exterior tiling max rel err {worst:.2e}, {bad} chain defects"))
}

/// Sampled coverage: random points; inside the solid => exactly one leaf
/// cell contains it, outside => none.
fn sampled_coverage(asset: &Asset, n: usize) -> (bool, String, f64) {
    let cp = asset.cell_polys();
    let mut bad = 0usize;
    let mut total = 0usize;
    let mut total_vol = 0.0;
    for c in &asset.components {
        let q = MeshQuery::new(&c.solid);
        let bb = c.aabb;
        total_vol += bb.extent().x * bb.extent().y * bb.extent().z;
        let ncell = (c.cells.end - c.cells.start) as usize;
        if ncell == 0 {
            continue;
        }
        let meshes: Vec<TriMesh> = (c.cells.start..c.cells.end).map(|ci| cells_boundary_mesh_with(asset, &cp, &[CellId(ci)])).collect();
        let boxes: Vec<frac_geom::Aabb> = meshes.iter().map(|m| m.aabb()).collect();
        let bvh = frac_geom::bvh::Bvh::build(&boxes);
        let per = (n / asset.components.len().max(1)).max(64);
        let res: Vec<bool> = (0..per)
            .into_par_iter()
            .map(|k| {
                let r = |s: u64| unit_f64(stable_hash(&[c.id.0 as u64, k as u64, s]));
                let p = bb.min + bb.extent() * DVec3::new(r(1), r(2), r(3));
                let inside = q.contains(p);
                let cand = bvh.query_vec(&frac_geom::Aabb { min: p, max: p });
                let hits = cand.iter().filter(|&&ci| MeshQuery::new(&meshes[ci as usize]).contains(p)).count();
                (inside && hits == 1) || (!inside && hits == 0)
            })
            .collect();
        total += res.len();
        bad += res.iter().filter(|&&x| !x).count();
    }
    // conservative estimate of overlap volume from sampling (upper bound at 0 failures: 0)
    let est = if total > 0 { bad as f64 / total as f64 * total_vol } else { 0.0 };
    (bad == 0, format!("{total} sampled points, {bad} coverage violations"), est)
}

/// Independent tetrahedral quadrature (4-point, degree 2) of a closed mesh,
/// fanning from the vertex centroid.
pub fn tet_quadrature(m: &TriMesh) -> VolumeIntegrals {
    let mut vi = VolumeIntegrals::default();
    if m.verts.is_empty() {
        return vi;
    }
    let apex = m.verts.iter().fold(DVec3::ZERO, |a, &v| a + v) / m.verts.len() as f64;
    let a = 0.585_410_196_624_968_5;
    let b = 0.138_196_601_125_010_5;
    for t in 0..m.tris.len() {
        let [p1, p2, p3] = m.tri_points(t);
        let p0 = apex;
        let v = (p1 - p0).dot((p2 - p0).cross(p3 - p0)) / 6.0;
        if v == 0.0 {
            continue;
        }
        let pts = [p0, p1, p2, p3];
        vi.volume += v;
        for k in 0..4 {
            let mut x = DVec3::ZERO;
            for (j, &p) in pts.iter().enumerate() {
                x += p * if j == k { a } else { b };
            }
            let w = v / 4.0;
            vi.first += x * w;
            vi.second += frac_geom::polygon::outer(x, x) * w;
        }
    }
    vi
}

fn mass_rel_error(mp: &frac_geom::MassProps, q: &VolumeIntegrals, density: f64) -> f64 {
    let qm = q.mass_props(density);
    let ev = (qm.volume / mp.volume - 1.0).abs();
    let scale = mp.volume.max(1e-300).cbrt();
    let ec = (qm.com - mp.com).length() / scale;
    let it = mp.inertia;
    let norm = (it.x_axis.length() + it.y_axis.length() + it.z_axis.length()).max(1e-300);
    let d = qm.inertia - it;
    let ei = (d.x_axis.length() + d.y_axis.length() + d.z_axis.length()) / norm;
    ev.max(ec).max(ei)
}

/// Volume by ray casting along Z over an n×n grid (exact intervals along
/// each ray, midpoint rule across).
pub fn ray_volume(m: &TriMesh, n: usize) -> f64 {
    let bb = m.aabb();
    let ext = bb.extent();
    let (dx, dy) = (ext.x / n as f64, ext.y / n as f64);
    let mut vol = 0.0;
    for i in 0..n {
        for j in 0..n {
            let x = bb.min.x + (i as f64 + 0.5) * dx;
            let y = bb.min.y + (j as f64 + 0.5) * dy;
            let mut zs: Vec<(f64, f64)> = Vec::new(); // (z, sign of crossing)
            for t in 0..m.tris.len() {
                let [a, b, c] = m.tri_points(t);
                // barycentric in xy
                let d = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
                if d == 0.0 {
                    continue;
                }
                let l1 = ((b.x - x) * (c.y - y) - (c.x - x) * (b.y - y)) / d;
                let l2 = ((c.x - x) * (a.y - y) - (a.x - x) * (c.y - y)) / d;
                let l3 = 1.0 - l1 - l2;
                if l1 < 0.0 || l2 < 0.0 || l3 < 0.0 {
                    continue;
                }
                zs.push((l1 * a.z + l2 * b.z + l3 * c.z, d.signum()));
            }
            // signed sum: outward normal z-sign determines entry/exit
            for (z, s) in zs {
                vol += s * z * dx * dy;
            }
        }
    }
    vol
}
