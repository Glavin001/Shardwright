//! Stage 9: render meshes.
//!
//! Interior fracture surfaces are generated **once per interface patch**:
//! the clean planar patch is refined with interior Steiner points (CDT that
//! keeps every boundary vertex and edge), displaced along its normal by
//! material fractal noise that tapers to zero at the boundary loop, and the
//! same displaced surface is used by both neighboring fragments with
//! opposite winding. Boundary vertices are shared with the exterior
//! surface, so every fragment mesh stays watertight and neighbors have no
//! gaps. Noise and chipping are render-only; physics uses clean geometry.

pub mod noise;

use frac_core::*;
use frac_geom::inside::MeshQuery;
use frac_geom::polygon::plane_basis;
use frac_geom::{DVec3, TriMesh};
use glam::Vec3;
use rayon::prelude::*;
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
pub struct NoiseSpec {
    pub amplitude: f64,
    pub hurst: f64,
    pub min_wl: f64,
    pub max_wl: f64,
    /// Grain direction and stretch (splinter-like anisotropy).
    pub grain: Option<(DVec3, f64)>,
}

pub struct RenderParams<'a> {
    pub settings: &'a settings::RenderSettings,
    pub seed: u64,
    pub noise_for: &'a (dyn Fn(ComponentId) -> Option<NoiseSpec> + Sync),
    pub chipping_for: &'a (dyn Fn(ComponentId) -> f64 + Sync),
    pub uv_scale_for: &'a (dyn Fn(ComponentId) -> f64 + Sync),
}

/// One LOD of a fragment render mesh (asset-frame positions).
#[derive(Clone, Debug, Default)]
pub struct FragMesh {
    pub positions: Vec<DVec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<[f32; 2]>,
    pub ext_indices: Vec<u32>,
    pub int_indices: Vec<u32>,
}

impl FragMesh {
    pub fn triangle_count(&self) -> usize {
        (self.ext_indices.len() + self.int_indices.len()) / 3
    }
    /// Combined closed triangle mesh (positions welded by exact coordinates)
    /// for validation.
    pub fn as_trimesh(&self) -> TriMesh {
        let mut tris = Vec::new();
        for idx in [&self.ext_indices, &self.int_indices] {
            for t in idx.chunks(3) {
                tris.push([t[0], t[1], t[2]]);
            }
        }
        TriMesh { verts: self.positions.clone(), tris }.weld_exact()
    }
}

#[derive(Clone, Debug, Default)]
pub struct StubMesh {
    pub interface: InterfaceId,
    pub positions: Vec<DVec3>,
    pub normals: Vec<Vec3>,
    pub indices: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct RenderOut {
    /// Per fragment: LOD meshes (LOD0 first).
    pub fragments: Vec<Vec<FragMesh>>,
    pub stubs: Vec<StubMesh>,
    /// Per leaf fragment: (render volume - clean volume) / clean volume.
    pub volume_deviation: Vec<(FragmentId, f64)>,
    /// Max world-space UV projection mismatch across shared interfaces.
    pub uv_mismatch: f64,
}

/// A displaced interior surface for one patch.
#[derive(Clone, Debug, Default)]
#[allow(dead_code)]
struct PatchSurface {
    /// Local vertex -> Some(component vertex) for boundary vertices.
    comp_vert: Vec<Option<u32>>,
    pos: Vec<DVec3>,
    normal: Vec<DVec3>,
    tris: Vec<[u32; 3]>,
}

fn triplanar_uv(p: DVec3, n: DVec3, scale: f64) -> [f32; 2] {
    let a = n.abs();
    let (u, v) = if a.x >= a.y && a.x >= a.z {
        (p.z, p.y)
    } else if a.y >= a.z {
        (p.x, p.z)
    } else {
        (p.x, p.y)
    };
    [(u * scale) as f32, (v * scale) as f32]
}

/// Principal extents of a cell (from its inertia), smallest first.
fn cell_min_extent(c: &Cell) -> f64 {
    let (ev, _) = c.mass.principal();
    let m = c.mass.mass.max(1e-300);
    // box: I1 = m/12 (b²+c²) ... a² = 6 (I2 + I3 - I1) / m
    let a2 = 6.0 * (ev.y + ev.z - ev.x) / m;
    let b2 = 6.0 * (ev.x + ev.z - ev.y) / m;
    let c2 = 6.0 * (ev.x + ev.y - ev.z) / m;
    a2.max(0.0).sqrt().min(b2.max(0.0).sqrt()).min(c2.max(0.0).sqrt())
}

/// Build the shared displaced surface of a patch.
fn patch_surface(verts: &[DVec3], p: &Patch, spec: Option<NoiseSpec>, amp_cap: f64, settings: &settings::RenderSettings, seed: u64, dihedral: &[(f64, f64)]) -> PatchSurface {
    let n = p.normal;
    let (u, v) = plane_basis(n);
    let mut comp_vert: Vec<Option<u32>> = Vec::new();
    let mut pos: Vec<DVec3> = Vec::new();
    let mut local: BTreeMap<u32, u32> = BTreeMap::new();
    for l in &p.loops {
        for &x in l {
            local.entry(x).or_insert_with(|| {
                comp_vert.push(Some(x));
                pos.push(verts[x as usize]);
                (pos.len() - 1) as u32
            });
        }
    }
    let area: f64 = frac_geom::polygon::area_integrals(&p.loops.iter().map(|l| l.iter().map(|&i| verts[i as usize]).collect()).collect::<Vec<_>>(), n).area;
    let noise_on = settings.noise && spec.map(|s| s.amplitude > 0.0).unwrap_or(false) && area > 0.0;
    let mut tris_local: Vec<[u32; 3]> = p.tris.iter().map(|t| [local[&t[0]], local[&t[1]], local[&t[2]]]).collect();
    if noise_on {
        let s = spec.unwrap();
        // Steiner spacing: resolve the noise but respect the triangle cap
        let cap = settings.max_interior_tris_per_patch.max(8) as f64;
        let spacing = (s.min_wl * settings.interior_resolution)
            .max(s.max_wl / 6.0)
            .max(area.sqrt() / 10.0)
            .max((2.0 * area / cap).sqrt())
            .max(1e-9);
        let p2 = |x: DVec3| [x.dot(u), x.dot(v)];
        let segs: Vec<([f64; 2], [f64; 2])> = p
            .loops
            .iter()
            .flat_map(|l| (0..l.len()).map(move |k| (l[k], l[(k + 1) % l.len()])))
            .map(|(a, b)| (p2(verts[a as usize]), p2(verts[b as usize])))
            .collect();
        let pts2: Vec<[f64; 2]> = pos.iter().map(|&x| p2(x)).collect();
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for q in &pts2 {
            for k in 0..2 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
        let plane_d = pos[0].dot(n);
        // (point, boundary distance, max safe |h| for +h, for -h)
        let mut steiner: Vec<([f64; 2], f64, f64, f64)> = Vec::new();
        let ny = ((hi[1] - lo[1]) / (spacing * 0.866)).ceil() as i64;
        let nx = ((hi[0] - lo[0]) / spacing).ceil() as i64;
        if (nx * ny) as f64 <= cap * 2.0 {
            for j in 0..=ny {
                for i in 0..=nx {
                    let q = [lo[0] + (i as f64 + if j % 2 == 1 { 0.5 } else { 0.0 }) * spacing, lo[1] + j as f64 * spacing * 0.866];
                    if !inside_segs(&segs, q) {
                        continue;
                    }
                    let mut d = f64::INFINITY;
                    // displacement must not cross the faces adjacent to the
                    // boundary: |h| <= 0.5 d_e tan(dihedral_e) per side
                    let (mut lim_pos, mut lim_neg) = (f64::INFINITY, f64::INFINITY);
                    for (k, (a, b)) in segs.iter().enumerate() {
                        let de = seg_dist(q, *a, *b);
                        d = d.min(de);
                        let (ta, tb) = dihedral.get(k).copied().unwrap_or((1.5, 1.5));
                        let f = |t: f64| 0.4 * de * libm::sin(t.clamp(0.0, std::f64::consts::FRAC_PI_2));
                        lim_pos = lim_pos.min(f(tb));
                        lim_neg = lim_neg.min(f(ta));
                    }
                    if d > 0.45 * spacing {
                        steiner.push((q, d, lim_pos, lim_neg));
                    }
                }
            }
        }
        if !steiner.is_empty() {
            // CDT with loop constraints and Steiner points
            let mut cdt: ConstrainedDelaunayTriangulation<Point2<f64>> = ConstrainedDelaunayTriangulation::new();
            let mut handles = Vec::new();
            let mut ok = true;
            for q in &pts2 {
                match cdt.insert(Point2::new(q[0], q[1])) {
                    Ok(h) => handles.push(h),
                    Err(_) => {
                        ok = false;
                        break;
                    }
                }
            }
            let nb = handles.len();
            let mut handle_to_local: BTreeMap<usize, u32> = BTreeMap::new();
            for (i, h) in handles.iter().enumerate() {
                if handle_to_local.insert(h.index(), i as u32).is_some() {
                    ok = false;
                }
            }
            if ok {
                for l in &p.loops {
                    for k in 0..l.len() {
                        let a = handles[local[&l[k]] as usize];
                        let b = handles[local[&l[(k + 1) % l.len()]] as usize];
                        if a != b && !cdt.exists_constraint(a, b) && cdt.try_add_constraint(a, b).is_empty() && !cdt.exists_constraint(a, b) {
                            ok = false;
                        }
                    }
                }
            }
            if ok {
                let taper_w = 2.0 * spacing;
                let amp = s.amplitude.min(amp_cap);
                for (q, d, lim_pos, lim_neg) in &steiner {
                    let base = u * q[0] + v * q[1] + n * plane_d;
                    let mut np = base;
                    if let Some((g, st)) = s.grain {
                        let gl = base.dot(g);
                        np = base + g * (gl / st.max(1.0) - gl);
                    }
                    let t = (d / taper_w).clamp(0.0, 1.0);
                    let taper = t * t * (3.0 - 2.0 * t);
                    let h = amp * noise::fbm(np, s.min_wl, s.max_wl, s.hurst, seed) * taper;
                    let h = if h > 0.0 { h.min(*lim_pos) } else { h.max(-*lim_neg) };
                    let x = base + n * h;
                    if let Ok(hd) = cdt.insert(Point2::new(q[0], q[1])) {
                        if hd.index() >= nb && !handle_to_local.contains_key(&hd.index()) {
                            handle_to_local.insert(hd.index(), pos.len() as u32);
                            pos.push(x);
                            comp_vert.push(None);
                        }
                    }
                }
                // classify inside faces by flood fill from loop edges
                let mut inside = vec![false; cdt.num_all_faces()];
                let mut stack = Vec::new();
                for l in &p.loops {
                    for k in 0..l.len() {
                        let a = handles[local[&l[k]] as usize];
                        let b = handles[local[&l[(k + 1) % l.len()]] as usize];
                        if let Some(e) = cdt.get_edge_from_neighbors(a, b) {
                            if let Some(f) = e.face().as_inner() {
                                if !inside[f.fix().index()] {
                                    inside[f.fix().index()] = true;
                                    stack.push(f.fix());
                                }
                            }
                        }
                    }
                }
                while let Some(fh) = stack.pop() {
                    for e in cdt.face(fh).adjacent_edges() {
                        if e.is_constraint_edge() {
                            continue;
                        }
                        if let Some(nbf) = e.rev().face().as_inner() {
                            let i = nbf.fix().index();
                            if !inside[i] {
                                inside[i] = true;
                                stack.push(nbf.fix());
                            }
                        }
                    }
                }
                let mut new_tris = Vec::new();
                let mut all_known = true;
                for f in cdt.inner_faces() {
                    if !inside[f.fix().index()] {
                        continue;
                    }
                    let vs = f.vertices();
                    let ids: Vec<Option<&u32>> = vs.iter().map(|v| handle_to_local.get(&v.fix().index())).collect();
                    if ids.iter().any(|x| x.is_none()) {
                        all_known = false;
                        break;
                    }
                    new_tris.push([*ids[0].unwrap(), *ids[1].unwrap(), *ids[2].unwrap()]);
                }
                if all_known && boundary_preserved(p, &local, &new_tris) {
                    tris_local = new_tris;
                } else {
                    pos.truncate(nb);
                    comp_vert.truncate(nb);
                }
            } else {
                pos.truncate(nb.min(pos.len()));
                comp_vert.truncate(pos.len());
            }
        }
    }
    // vertex normals of the (possibly displaced) surface
    let mut nrm = vec![DVec3::ZERO; pos.len()];
    for t in &tris_local {
        let fnrm = (pos[t[1] as usize] - pos[t[0] as usize]).cross(pos[t[2] as usize] - pos[t[0] as usize]);
        for &k in t {
            nrm[k as usize] += fnrm;
        }
    }
    let normal = nrm.into_iter().map(|x| if x.dot(n) > 0.0 { x.normalize() } else { n }).collect();
    PatchSurface { comp_vert, pos, normal, tris: tris_local }
}

/// Patches whose displaced triangles take part in a self-intersection of
/// some cell's render mesh (exact test, zero-area triangles ignored).
fn offending_patches(comp: &Component, verts: &[DVec3], surfaces: &[PatchSurface]) -> (Vec<usize>, Vec<u32>) {
    let g = &comp.geometry;
    let mut per_cell: BTreeMap<CellId, (Vec<usize>, Vec<(usize, bool)>)> = BTreeMap::new();
    for (i, e) in g.ext_polys.iter().enumerate() {
        per_cell.entry(e.cell).or_default().0.push(i);
    }
    for (pi, pt) in g.patches.iter().enumerate() {
        per_cell.entry(pt.cells.0).or_default().1.push((pi, false));
        per_cell.entry(pt.cells.1).or_default().1.push((pi, true));
    }
    let bad: Vec<(Vec<usize>, Vec<u32>)> = per_cell
        .par_iter()
        .map(|(_, (exts, pats))| {
            let mut m = TriMesh::default();
            let mut tag: Vec<Option<usize>> = Vec::new();
            let mut ext_of: Vec<Option<usize>> = Vec::new();
            for &i in exts {
                let e = &g.ext_polys[i];
                for t in &e.tris {
                    let base = m.verts.len() as u32;
                    m.verts.extend(t.iter().map(|&v| verts[v as usize]));
                    m.tris.push([base, base + 1, base + 2]);
                    tag.push(None);
                    ext_of.push(Some(i));
                }
            }
            for &(pi, flip) in pats {
                let srf = &surfaces[pi];
                let base = m.verts.len() as u32;
                m.verts.extend(srf.pos.iter().copied());
                for t in &srf.tris {
                    m.tris.push(if flip { [base + t[0], base + t[2], base + t[1]] } else { [base + t[0], base + t[1], base + t[2]] });
                    tag.push(Some(pi));
                    ext_of.push(None);
                }
            }
            // weld by exact coordinates without dropping triangles (keep tags aligned)
            let mut map: BTreeMap<[u64; 3], u32> = BTreeMap::new();
            let mut nv = Vec::new();
            let remap: Vec<u32> = m
                .verts
                .iter()
                .map(|v| {
                    *map.entry([v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]).or_insert_with(|| {
                        nv.push(*v);
                        (nv.len() - 1) as u32
                    })
                })
                .collect();
            let tris: Vec<[u32; 3]> = m.tris.iter().map(|t| [remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]]).collect();
            let wm = TriMesh { verts: nv, tris };
            let mut out = Vec::new();
            let mut ev = Vec::new();
            for (a, b) in wm.self_intersections(usize::MAX) {
                if wm.is_degenerate(a as usize) || wm.is_degenerate(b as usize) {
                    continue;
                }
                for t in [a, b] {
                    if let Some(pi) = tag[t as usize] {
                        out.push(pi);
                    }
                    if let Some(ei) = ext_of[t as usize] {
                        ev.extend(g.ext_polys[ei].verts.iter().copied());
                    }
                }
            }
            (out, ev)
        })
        .collect();
    let mut v: Vec<usize> = bad.iter().flat_map(|b| b.0.iter().copied()).collect();
    v.sort_unstable();
    v.dedup();
    let mut ev: Vec<u32> = bad.into_iter().flat_map(|b| b.1).collect();
    ev.sort_unstable();
    ev.dedup();
    (v, ev)
}

/// Interior dihedral angles (cell A side, cell B side) at every boundary
/// edge of every patch, in loop order.
fn patch_dihedrals(comp: &Component) -> Vec<Vec<(f64, f64)>> {
    let g = &comp.geometry;
    // per cell: directed edge -> outward normal of the polygon using it
    let mut edge_n: BTreeMap<(CellId, u32, u32), DVec3> = BTreeMap::new();
    for e in &g.ext_polys {
        let pts: Vec<DVec3> = e.verts.iter().map(|&v| g.verts[v as usize]).collect();
        let n = frac_geom::polygon::newell(&pts).normalize_or_zero();
        for k in 0..e.verts.len() {
            edge_n.insert((e.cell, e.verts[k], e.verts[(k + 1) % e.verts.len()]), n);
        }
    }
    for pt in &g.patches {
        for l in &pt.loops {
            for k in 0..l.len() {
                let (a, b) = (l[k], l[(k + 1) % l.len()]);
                edge_n.insert((pt.cells.0, a, b), pt.normal);
                edge_n.insert((pt.cells.1, b, a), -pt.normal);
            }
        }
    }
    let dihedral = |n_self: DVec3, n_other: Option<&DVec3>| -> f64 {
        match n_other {
            Some(m) => std::f64::consts::PI - libm::acos(n_self.dot(*m).clamp(-1.0, 1.0)),
            None => 1.5,
        }
    };
    g.patches
        .iter()
        .map(|pt| {
            let mut v = Vec::new();
            for l in &pt.loops {
                for k in 0..l.len() {
                    let (a, b) = (l[k], l[(k + 1) % l.len()]);
                    // neighbor polygon in A uses (b,a); in B uses (a,b)
                    let ta = dihedral(pt.normal, edge_n.get(&(pt.cells.0, b, a)));
                    let tb = dihedral(-pt.normal, edge_n.get(&(pt.cells.1, a, b)));
                    v.push((ta, tb));
                }
            }
            v
        })
        .collect()
}

/// Check that every directed boundary edge of the patch appears in the
/// triangulation (so the surface stays stitched to its neighbors).
fn boundary_preserved(p: &Patch, local: &BTreeMap<u32, u32>, tris: &[[u32; 3]]) -> bool {
    let mut e = std::collections::BTreeSet::new();
    for t in tris {
        for k in 0..3 {
            e.insert((t[k], t[(k + 1) % 3]));
        }
    }
    p.loops.iter().all(|l| (0..l.len()).all(|k| e.contains(&(local[&l[k]], local[&l[(k + 1) % l.len()]]))))
}

fn inside_segs(segs: &[([f64; 2], [f64; 2])], q: [f64; 2]) -> bool {
    let mut inside = false;
    for (a, b) in segs {
        if (a[1] > q[1]) != (b[1] > q[1]) {
            let x = a[0] + (q[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
            if q[0] < x {
                inside = !inside;
            }
        }
    }
    inside
}

fn seg_dist(q: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let l2 = ab[0] * ab[0] + ab[1] * ab[1];
    let t = if l2 > 0.0 { (((q[0] - a[0]) * ab[0] + (q[1] - a[1]) * ab[1]) / l2).clamp(0.0, 1.0) } else { 0.0 };
    let d = [a[0] + ab[0] * t - q[0], a[1] + ab[1] * t - q[1]];
    (d[0] * d[0] + d[1] * d[1]).sqrt()
}

/// Render-only chipping: junction vertices (shared by exterior polygons and
/// interface patches) are nudged tangentially within the exterior surface
/// (along their mesh edge, or within their source triangle), shared by both
/// neighbors so meshes stay watertight. Returns moved vertex positions.
fn chip_vertices(comp: &Component, ratio: f64, seed: u64) -> Vec<DVec3> {
    let g = &comp.geometry;
    let mut verts = g.verts.clone();
    if ratio <= 0.0 {
        return verts;
    }
    let mut on_patch = vec![false; verts.len()];
    for p in &g.patches {
        for l in &p.loops {
            for &v in l {
                on_patch[v as usize] = true;
            }
        }
    }
    // source triangles and incident exterior edge lengths per vertex
    let mut src: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    let mut minlen: BTreeMap<u32, f64> = BTreeMap::new();
    for e in &g.ext_polys {
        let n = e.verts.len();
        for k in 0..n {
            let v = e.verts[k];
            src.entry(v).or_default().push(e.src_tri);
            let pv = g.verts[v as usize];
            let mut m_here = f64::INFINITY;
            // distance to every polygon edge not incident to v (fold safety)
            for j in 0..n {
                let (a, b) = (e.verts[j], e.verts[(j + 1) % n]);
                let d = if a == v || b == v {
                    (g.verts[if a == v { b } else { a } as usize] - pv).length()
                } else {
                    let (pa, pb) = (g.verts[a as usize], g.verts[b as usize]);
                    let ab = pb - pa;
                    let t = if ab.length_squared() > 0.0 { ((pv - pa).dot(ab) / ab.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
                    (pa + ab * t - pv).length()
                };
                m_here = m_here.min(d);
            }
            let m = minlen.entry(v).or_insert(f64::INFINITY);
            *m = m.min(m_here);
        }
    }
    for (&v, tris) in src.iter_mut() {
        if !on_patch[v as usize] {
            continue;
        }
        tris.sort_unstable();
        tris.dedup();
        let h = stable_hash(&[seed, comp.id.0 as u64, v as u64]);
        if unit_f64(h) >= ratio {
            continue;
        }
        let amp = 0.25 * minlen[&v];
        let r1 = unit_f64(stable_hash(&[h, 1])) * 2.0 - 1.0;
        let r2 = unit_f64(stable_hash(&[h, 2])) * 2.0 - 1.0;
        let [a, b, c] = comp.solid.tri_points(tris[0] as usize);
        let tn = (b - a).cross(c - a).normalize_or_zero();
        let delta = match tris.len() {
            1 => {
                let (tu, tv) = plane_basis(tn);
                tu * (r1 * amp) + tv * (r2 * amp)
            }
            2 => {
                let [a2, b2, c2] = comp.solid.tri_points(tris[1] as usize);
                let tn2 = (b2 - a2).cross(c2 - a2).normalize_or_zero();
                let dir = tn.cross(tn2).normalize_or_zero();
                dir * (r1 * amp)
            }
            _ => DVec3::ZERO,
        };
        verts[v as usize] += delta;
    }
    verts
}

/// Exterior attribute sampler for a component.
struct ExteriorAttrs<'a> {
    comp: &'a Component,
    render_q: Option<MeshQuery<'a>>,
}

impl<'a> ExteriorAttrs<'a> {
    fn new(comp: &'a Component) -> Self {
        let render_q = comp.render_surface.as_ref().map(MeshQuery::new);
        ExteriorAttrs { comp, render_q }
    }
    /// (normal, uv) at position `p` on source triangle `t`.
    fn sample(&self, p: DVec3, t: u32, face_n: DVec3) -> (DVec3, [f32; 2]) {
        let (mesh, tri, attrs) = match (&self.render_q, &self.comp.render_surface) {
            (Some(q), Some(rs)) => match q.closest_point(p) {
                Some((_, _, rt)) => (rs, rt, None),
                None => (&self.comp.solid, t, Some(())),
            },
            _ => (&self.comp.solid, t, Some(())),
        };
        let _ = attrs;
        let [a, b, c] = mesh.tri_points(tri as usize);
        let bary = barycentric(p, a, b, c);
        let s = &self.comp.surface;
        // For reconstructed parts, `surface` attributes refer to the render surface triangles.
        let ti = tri as usize;
        let n = if ti < s.normals.len() {
            let cn = s.normals[ti];
            let v = DVec3::new(cn[0][0] as f64, cn[0][1] as f64, cn[0][2] as f64) * bary.x
                + DVec3::new(cn[1][0] as f64, cn[1][1] as f64, cn[1][2] as f64) * bary.y
                + DVec3::new(cn[2][0] as f64, cn[2][1] as f64, cn[2][2] as f64) * bary.z;
            if v.length_squared() > 0.0 { v.normalize() } else { face_n }
        } else {
            face_n
        };
        let uv = match &s.uvs {
            Some(u) if ti < u.len() => {
                let c = u[ti];
                [
                    (c[0][0] as f64 * bary.x + c[1][0] as f64 * bary.y + c[2][0] as f64 * bary.z) as f32,
                    (c[0][1] as f64 * bary.x + c[1][1] as f64 * bary.y + c[2][1] as f64 * bary.z) as f32,
                ]
            }
            _ => triplanar_uv(p, face_n, 1.0),
        };
        (n, uv)
    }
}

fn barycentric(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> DVec3 {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let den = d00 * d11 - d01 * d01;
    if den.abs() < 1e-300 {
        return DVec3::new(1.0, 0.0, 0.0);
    }
    let v = (d11 * d20 - d01 * d21) / den;
    let w = (d00 * d21 - d01 * d20) / den;
    DVec3::new(1.0 - v - w, v, w)
}

/// Build render meshes for every fragment.
pub fn build_render(asset: &Asset, p: &RenderParams) -> RenderOut {
    let s = p.settings;
    // per component: chipped vertices and patch surfaces
    struct CompRender {
        verts: Vec<DVec3>,
        surfaces: Vec<PatchSurface>,
    }
    let comps: Vec<CompRender> = asset
        .components
        .par_iter()
        .map(|comp| {
            let chip = if s.chipping { (p.chipping_for)(comp.id) } else { 0.0 };
            let mut verts = chip_vertices(comp, chip, p.seed);
            // the whole exterior surface must stay embedded after chipping
            if chip > 0.0 {
                for _ in 0..4 {
                    let mut tris = Vec::new();
                    let mut owner = Vec::new();
                    for (i, e) in comp.geometry.ext_polys.iter().enumerate() {
                        for t in &e.tris {
                            tris.push(*t);
                            owner.push(i);
                        }
                    }
                    let m = TriMesh { verts: verts.clone(), tris };
                    let si = m.self_intersections(usize::MAX);
                    let mut changed = false;
                    for (a, b) in si {
                        if m.is_degenerate(a as usize) || m.is_degenerate(b as usize) {
                            continue;
                        }
                        for t in [a, b] {
                            for &v in &comp.geometry.ext_polys[owner[t as usize]].verts {
                                if verts[v as usize] != comp.geometry.verts[v as usize] {
                                    verts[v as usize] = comp.geometry.verts[v as usize];
                                    changed = true;
                                }
                            }
                        }
                    }
                    if !changed {
                        break;
                    }
                }
            }
            let spec = (p.noise_for)(comp.id);
            let dihedrals = patch_dihedrals(comp);
            let seed = stable_hash(&[p.seed, comp.id.0 as u64, 0x401]);
            let caps: Vec<f64> = comp
                .geometry
                .patches
                .iter()
                .map(|pt| 0.2 * cell_min_extent(&asset.cells[pt.cells.0.idx()]).min(cell_min_extent(&asset.cells[pt.cells.1.idx()])))
                .collect();
            let make = |verts: &[DVec3], pi: usize, scale: f64| -> PatchSurface {
                let sp = spec.map(|mut x| {
                    x.amplitude *= scale;
                    x
                });
                patch_surface(verts, &comp.geometry.patches[pi], sp, caps[pi], s, seed, &dihedrals[pi])
            };
            let mut vpatches: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
            for (pi, pt) in comp.geometry.patches.iter().enumerate() {
                for l in &pt.loops {
                    for &v in l {
                        vpatches.entry(v).or_default().push(pi);
                    }
                }
            }
            let mut scale = vec![1.0f64; comp.geometry.patches.len()];
            let mut surfaces: Vec<PatchSurface> = (0..comp.geometry.patches.len()).map(|pi| make(&verts, pi, 1.0)).collect();
            // Guarantee validity: exact self-intersection check per cell;
            // halve the noise of offending patches (flat on late rounds) and
            // undo chipping on offending exterior vertices.
            for round in 0..5 {
                let (bad, bad_verts) = offending_patches(comp, &verts, &surfaces);
                if bad.is_empty() && bad_verts.is_empty() {
                    break;
                }
                let mut rebuild: std::collections::BTreeSet<usize> = bad.iter().copied().collect();
                for &pi in &bad {
                    scale[pi] = if round >= 2 { 0.0 } else { scale[pi] * 0.5 };
                }
                for v in bad_verts {
                    if verts[v as usize] != comp.geometry.verts[v as usize] {
                        verts[v as usize] = comp.geometry.verts[v as usize];
                        rebuild.extend(vpatches.get(&v).into_iter().flatten().copied());
                    }
                }
                for pi in rebuild {
                    surfaces[pi] = make(&verts, pi, scale[pi]);
                }
            }
            CompRender { verts, surfaces }
        })
        .collect();
    let h = &asset.hierarchy;
    let leaf_level = h.levels.saturating_sub(1);
    let frag_meshes: Vec<Vec<FragMesh>> = h
        .fragments
        .par_iter()
        .map(|f| {
            let comp = &asset.components[f.component.idx()];
            let cr = &comps[f.component.idx()];
            let cells: std::collections::BTreeSet<CellId> = asset.fragment_cells(f).iter().copied().collect();
            let attrs = ExteriorAttrs::new(comp);
            let uv_scale = (p.uv_scale_for)(comp.id);
            let origin = f.mass.com;
            let mut m = FragMesh::default();
            let mut ext_map: BTreeMap<(u32, u32), u32> = BTreeMap::new();
            for e in &comp.geometry.ext_polys {
                if !cells.contains(&e.cell) {
                    continue;
                }
                let pts: Vec<DVec3> = e.verts.iter().map(|&v| cr.verts[v as usize]).collect();
                let fnrm = frac_geom::polygon::newell(&pts).normalize_or_zero();
                let ids: Vec<u32> = e
                    .verts
                    .iter()
                    .map(|&v| {
                        *ext_map.entry((v, e.src_tri)).or_insert_with(|| {
                            let x = cr.verts[v as usize];
                            let (n, uv) = attrs.sample(x, e.src_tri, fnrm);
                            m.positions.push(x);
                            m.normals.push(n.as_vec3());
                            m.uvs.push(uv);
                            (m.positions.len() - 1) as u32
                        })
                    })
                    .collect();
                for t in &e.tris {
                    for &v in t {
                        let k = e.verts.iter().position(|&x| x == v).unwrap();
                        m.ext_indices.push(ids[k]);
                    }
                }
            }
            for (pi, pt) in comp.geometry.patches.iter().enumerate() {
                let (a, b) = (cells.contains(&pt.cells.0), cells.contains(&pt.cells.1));
                if a == b {
                    continue;
                }
                let flip = b;
                let srf = &cr.surfaces[pi];
                let base = m.positions.len() as u32;
                for (k, &x) in srf.pos.iter().enumerate() {
                    let n = if flip { -srf.normal[k] } else { srf.normal[k] };
                    m.positions.push(x);
                    m.normals.push(n.as_vec3());
                    m.uvs.push(triplanar_uv(x, pt.normal, uv_scale));
                }
                for t in &srf.tris {
                    if flip {
                        m.int_indices.extend_from_slice(&[base + t[0], base + t[2], base + t[1]]);
                    } else {
                        m.int_indices.extend_from_slice(&[base + t[0], base + t[1], base + t[2]]);
                    }
                }
            }
            let _ = origin;
            // LODs
            let mut lods = vec![m];
            if s.triangle_budget > 0 && lods[0].triangle_count() > s.triangle_budget as usize {
                lods[0] = simplify(&lods[0], s.triangle_budget as usize);
            }
            for k in 1..s.lods.max(1) {
                let target = ((lods[0].triangle_count() as f64) * s.lod_ratio.powi(k as i32)).ceil() as usize;
                let l = simplify(&lods[0], target.max(4));
                lods.push(l);
            }
            lods
        })
        .collect();
    // render volume deviation (leaf fragments)
    let mut volume_deviation = Vec::new();
    for f in asset.level_fragments(leaf_level) {
        let tm = frag_meshes[f.id.idx()][0].as_trimesh();
        let v = tm.signed_volume();
        let clean = f.mass.volume;
        if clean > 0.0 {
            volume_deviation.push((f.id, (v - clean) / clean));
        }
    }
    // rebar stubs
    let mut stubs = Vec::new();
    if s.rebar_stubs {
        for it in &asset.interfaces {
            if it.rebar_points.is_empty() {
                continue;
            }
            let dir = if it.rebar.dir.length_squared() > 0.0 { it.rebar.dir.normalize() } else { it.polygons.first().map(|p| p.normal).unwrap_or(DVec3::Y) };
            let mut sm = StubMesh { interface: it.id, ..Default::default() };
            for &(c, d) in &it.rebar_points {
                add_cylinder(&mut sm, c, dir, 0.5 * d, 6.0 * d, 8);
            }
            stubs.push(sm);
        }
    }
    RenderOut { fragments: frag_meshes, stubs, volume_deviation, uv_mismatch: 0.0 }
}

fn add_cylinder(sm: &mut StubMesh, c: DVec3, dir: DVec3, r: f64, len: f64, segs: u32) {
    let (u, v) = plane_basis(dir);
    let base = sm.positions.len() as u32;
    for end in [-0.5, 0.5] {
        for k in 0..segs {
            let a = std::f64::consts::TAU * k as f64 / segs as f64;
            let off = u * (r * libm::cos(a)) + v * (r * libm::sin(a));
            sm.positions.push(c + dir * (len * end) + off);
            sm.normals.push(off.normalize().as_vec3());
        }
    }
    for k in 0..segs {
        let a0 = base + k;
        let a1 = base + (k + 1) % segs;
        let b0 = a0 + segs;
        let b1 = a1 + segs;
        sm.indices.extend_from_slice(&[a0, a1, b1, a0, b1, b0]);
    }
}

/// Simplify with all topological borders locked (exterior/interior
/// junctions and patch boundaries are borders because vertices are split
/// there), so LODs never cross interface boundary loops.
fn simplify(m: &FragMesh, target_tris: usize) -> FragMesh {
    let pos32: Vec<[f32; 3]> = m.positions.iter().map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect();
    let locks = vec![false; pos32.len()];
    let mut out = m.clone();
    let total = m.triangle_count().max(1);
    let simplify_part = |idx: &[u32], share: f64| -> Vec<u32> {
        if idx.is_empty() {
            return Vec::new();
        }
        let tgt = ((target_tris as f64 * share).ceil() as usize * 3).max(3);
        meshopt::simplify_with_locks_decoder(idx, &pos32, &locks, tgt, 0.05, meshopt::SimplifyOptions::LockBorder, None)
    };
    let es = m.ext_indices.len() as f64 / 3.0 / total as f64;
    out.ext_indices = simplify_part(&m.ext_indices, es);
    out.int_indices = simplify_part(&m.int_indices, 1.0 - es);
    out
}
