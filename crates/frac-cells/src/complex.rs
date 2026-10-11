//! Convex cell complexes with explicit shared faces, built either as the
//! dual of an exact Delaunay tetrahedralization (Voronoi) or from an
//! axis-aligned box partition (masonry and other explicit layouts).
//!
//! Consistency contract (relied on by [`crate::clip`]):
//! * every face is stored once, with its plane, its two cells and a loop of
//!   complex vertices oriented counter-clockwise about the plane gradient;
//! * every complex edge lies on a canonical line and knows its two end
//!   planes (interior satisfies `sign * f < 0` for both);
//! * face loops contain every complex vertex lying on their boundary.

use crate::delaunay::{Delaunay, NONE};
use crate::planes::{BoxPlanes, LineKey, P3, PlaneId, PlaneSystem, VoronoiPlanes, vor_id};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct CEdge {
    pub line: LineKey,
    pub v: [u32; 2],
    /// End constraints `(plane, sign)`: interior points satisfy `sign*f < 0`.
    pub end: [(PlaneId, i8); 2],
}

#[derive(Clone, Debug)]
pub struct CFace {
    pub plane: PlaneId,
    /// (negative side, positive side) cell; `NONE` = outside the complex
    /// region of interest.
    pub cells: [u32; 2],
    pub loop_verts: Vec<u32>,
    /// `loop_edges[k]` joins `loop_verts[k]` and `loop_verts[k+1]`.
    pub loop_edges: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct CCell {
    /// Half-spaces `(plane, sign)`: inside iff `sign * f <= 0`.
    pub halfspaces: Vec<(PlaneId, i8)>,
    pub faces: Vec<u32>,
    /// For facet planes split into several faces: the interior face-edge
    /// segments, as (other plane of the line, end constraints).
    pub facet_subdiv: Vec<(PlaneId, Vec<(PlaneId, [(PlaneId, i8); 2])>)>,
    pub aabb_min: P3,
    pub aabb_max: P3,
}

#[derive(Clone, Debug)]
pub struct Complex {
    pub planes: PlaneSystem,
    pub verts: Vec<P3>,
    pub edges: Vec<CEdge>,
    pub faces: Vec<CFace>,
    pub cells: Vec<CCell>,
    /// Seed position per cell (Voronoi) or box center (boxes), for labels.
    pub sites: Vec<P3>,
}

impl Complex {
    fn finish_aabbs(&mut self) {
        for c in self.cells.iter_mut() {
            let mut mn = [f64::INFINITY; 3];
            let mut mx = [f64::NEG_INFINITY; 3];
            for &f in &c.faces {
                for &v in &self.faces[f as usize].loop_verts {
                    let p = self.verts[v as usize];
                    for k in 0..3 {
                        mn[k] = mn[k].min(p[k]);
                        mx[k] = mx[k].max(p[k]);
                    }
                }
            }
            c.aabb_min = mn;
            c.aabb_max = mx;
        }
    }

    /// Voronoi complex of `seeds` (cells `0..seeds.len()`), bounded by
    /// far-away guard points so every seed cell is bounded. The region of
    /// interest is the box `[lo, hi]` (guards never influence it).
    pub fn voronoi(seeds: &[P3], lo: P3, hi: P3) -> Result<Complex, String> {
        let n = seeds.len();
        let center = [
            (lo[0] + hi[0]) * 0.5,
            (lo[1] + hi[1]) * 0.5,
            (lo[2] + hi[2]) * 0.5,
        ];
        let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2))
            .sqrt()
            .max(1e-9);
        // guard points: Fibonacci sphere at 3x diagonal (generic position)
        let ng = 48;
        let mut guards = Vec::with_capacity(ng);
        for k in 0..ng {
            let z = 1.0 - 2.0 * (k as f64 + 0.5) / ng as f64;
            let r = (1.0 - z * z).sqrt();
            let a = k as f64 * 2.399963229728653 + 0.1234;
            let rad = 3.0 * diag;
            guards.push([
                center[0] + rad * r * libm::cos(a),
                center[1] + rad * r * libm::sin(a),
                center[2] + rad * z,
            ]);
        }
        // insertion order: guards first, then seeds in Morton order
        let bb = frac_geom::Aabb {
            min: glam::DVec3::from_array(lo),
            max: glam::DVec3::from_array(hi),
        };
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| {
            (
                frac_geom::morton3(glam::DVec3::from_array(seeds[i]), &bb),
                i,
            )
        });
        let mut pts: Vec<P3> = guards.clone();
        pts.extend(order.iter().map(|&i| seeds[i]));
        let dt = Delaunay::build(&pts, center, diag * 4.0)?;
        // point index in dt -> cell id
        let base = dt.n_super + ng;
        let mut cell_of = vec![NONE; dt.points.len()];
        for (k, &i) in order.iter().enumerate() {
            cell_of[base + k] = i as u32;
        }
        // Plane system indexed by dt point indices (ranks = dt indices).
        let planes = PlaneSystem::Voronoi(VoronoiPlanes {
            seeds: dt.points.clone(),
        });
        let vt = dt.vertex_tet();
        // complex vertices = tets used; map tet -> vertex id
        let mut tet_vert: BTreeMap<u32, u32> = BTreeMap::new();
        let mut verts: Vec<P3> = Vec::new();
        let mut edges: Vec<CEdge> = Vec::new();
        let mut edge_of: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        let mut faces: Vec<CFace> = Vec::new();
        let mut face_of: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        let mut cells: Vec<CCell> = vec![CCell::default(); n];
        let mut sites = vec![[0.0; 3]; n];
        for (k, &i) in order.iter().enumerate() {
            sites[i] = dt.points[base + k];
        }
        // iterate cells in cell-id order for determinism
        let mut dt_index_of_cell = vec![0u32; n];
        for (k, &i) in order.iter().enumerate() {
            dt_index_of_cell[i] = (base + k) as u32;
        }
        for c in 0..n {
            let pi = dt_index_of_cell[c];
            let star = dt.star(pi, &vt);
            // neighbors
            let mut nbrs: Vec<u32> = Vec::new();
            for &t in &star {
                for &v in &dt.tets[t as usize] {
                    if v != pi {
                        nbrs.push(v);
                    }
                }
            }
            nbrs.sort_unstable();
            nbrs.dedup();
            for &pj in &nbrs {
                if (pj as usize) < dt.n_super {
                    return Err("voronoi: seed adjacent to super vertex (region too large?)".into());
                }
                let plane = vor_id(pi, pj);
                let sign: i8 = if pi < pj { 1 } else { -1 };
                cells[c].halfspaces.push((plane, sign));
                let (a, b) = (pi.min(pj), pi.max(pj));
                if let Some(&f) = face_of.get(&(a, b)) {
                    cells[c].faces.push(f);
                    continue;
                }
                // ring of tets around edge (a, b), CCW about s_b - s_a
                let t0 = *star
                    .iter()
                    .find(|&&t| dt.tets[t as usize].contains(&pj))
                    .unwrap();
                let mut ring: Vec<u32> = Vec::new();
                let mut t = t0;
                loop {
                    ring.push(t);
                    let tv = dt.tets[t as usize];
                    // find k,l such that (a,b,k,l) is an even permutation of tv
                    let pos_a = tv.iter().position(|&x| x == a).unwrap();
                    let pos_b = tv.iter().position(|&x| x == b).unwrap();
                    let others: Vec<usize> = (0..4).filter(|&s| s != pos_a && s != pos_b).collect();
                    let perm = [pos_a, pos_b, others[0], others[1]];
                    let (sk, _sl) = if perm_parity(perm) {
                        (others[0], others[1])
                    } else {
                        (others[1], others[0])
                    };
                    // next tet: across face opposite k
                    let nt = dt.neigh[t as usize][sk];
                    if nt == NONE {
                        return Err("voronoi: open ring (unbounded cell)".into());
                    }
                    t = nt;
                    if t == t0 {
                        break;
                    }
                    if ring.len() > 10_000 {
                        return Err("voronoi: ring did not close".into());
                    }
                }
                let mut loop_verts = Vec::with_capacity(ring.len());
                for &t in &ring {
                    let id = *tet_vert.entry(t).or_insert_with(|| {
                        verts.push(dt.circumcenter(t as usize));
                        (verts.len() - 1) as u32
                    });
                    loop_verts.push(id);
                }
                // edges between consecutive ring tets: shared face (a,b,l)
                let mut loop_edges = Vec::with_capacity(ring.len());
                for k in 0..ring.len() {
                    let ta = ring[k];
                    let tb = ring[(k + 1) % ring.len()];
                    let va = loop_verts[k];
                    let vb = loop_verts[(k + 1) % ring.len()];
                    let key = (va.min(vb), va.max(vb));
                    let eid = if let Some(&e) = edge_of.get(&key) {
                        e
                    } else {
                        let sa = dt.tets[ta as usize];
                        let sb = dt.tets[tb as usize];
                        // shared triangle
                        let shared: Vec<u32> =
                            sa.iter().copied().filter(|x| sb.contains(x)).collect();
                        if shared.len() != 3 {
                            return Err("voronoi: ring tets do not share a face".into());
                        }
                        let ka = sa.iter().copied().find(|x| !shared.contains(x)).unwrap();
                        let kb = sb.iter().copied().find(|x| !shared.contains(x)).unwrap();
                        let mut tri = [shared[0], shared[1], shared[2]];
                        tri.sort_unstable();
                        let line = LineKey(vor_id(tri[0], tri[1]), vor_id(tri[0], tri[2]));
                        let i0 = tri[0];
                        let end_a = (vor_id(i0, ka), if i0 < ka { 1 } else { -1 });
                        let end_b = (vor_id(i0, kb), if i0 < kb { 1 } else { -1 });
                        let (v, end) = if va <= vb {
                            ([va, vb], [end_a, end_b])
                        } else {
                            ([vb, va], [end_b, end_a])
                        };
                        edges.push(CEdge { line, v, end });
                        edge_of.insert(key, (edges.len() - 1) as u32);
                        (edges.len() - 1) as u32
                    };
                    loop_edges.push(eid);
                }
                let ca = cell_of[a as usize];
                let cb = cell_of[b as usize];
                faces.push(CFace {
                    plane: vor_id(a, b),
                    cells: [ca, cb],
                    loop_verts,
                    loop_edges,
                });
                let f = (faces.len() - 1) as u32;
                face_of.insert((a, b), f);
                cells[c].faces.push(f);
            }
        }
        let mut cx = Complex {
            planes,
            verts,
            edges,
            faces,
            cells,
            sites,
        };
        cx.finish_aabbs();
        Ok(cx)
    }

    /// Complex from an axis-aligned box partition. Boxes must tile their
    /// union exactly (shared faces with bit-identical coordinates) and the
    /// union must contain the region of interest. Boxes touching the union
    /// boundary have outside faces with `NONE` neighbors.
    pub fn boxes(boxes: &[(P3, P3)]) -> Result<Complex, String> {
        let mut bp = BoxPlanes::default();
        let mut plane_of: BTreeMap<(u8, u64), PlaneId> = BTreeMap::new();
        let mut get_plane = |ax: u8, off: f64, bp: &mut BoxPlanes| -> PlaneId {
            *plane_of.entry((ax, off.to_bits())).or_insert_with(|| {
                bp.planes.push((ax, off));
                (bp.planes.len() - 1) as PlaneId
            })
        };
        let n = boxes.len();
        let mut cells: Vec<CCell> = vec![CCell::default(); n];
        let mut sites = Vec::with_capacity(n);
        for (i, (lo, hi)) in boxes.iter().enumerate() {
            for ax in 0..3u8 {
                if !(lo[ax as usize] < hi[ax as usize]) {
                    return Err(format!("box {i} is empty"));
                }
                let pl = get_plane(ax, lo[ax as usize], &mut bp);
                let ph = get_plane(ax, hi[ax as usize], &mut bp);
                cells[i].halfspaces.push((pl, -1));
                cells[i].halfspaces.push((ph, 1));
            }
            sites.push([
                (lo[0] + hi[0]) * 0.5,
                (lo[1] + hi[1]) * 0.5,
                (lo[2] + hi[2]) * 0.5,
            ]);
        }
        // faces: pairs touching on a plane. Index boxes by (axis, lo) and (axis, hi).
        struct RawFace {
            ax: u8,
            off: f64,
            lo2: [f64; 2],
            hi2: [f64; 2],
            cells: [u32; 2],
        }
        let mut raw: Vec<RawFace> = Vec::new();
        for ax in 0..3usize {
            let (u, w) = ((ax + 1) % 3, (ax + 2) % 3);
            let mut by_hi: BTreeMap<u64, Vec<u32>> = BTreeMap::new();
            let mut by_lo: BTreeMap<u64, Vec<u32>> = BTreeMap::new();
            for (i, (lo, hi)) in boxes.iter().enumerate() {
                by_hi.entry(hi[ax].to_bits()).or_default().push(i as u32);
                by_lo.entry(lo[ax].to_bits()).or_default().push(i as u32);
            }
            // interior faces
            for (k, lows) in by_hi.iter() {
                let highs = match by_lo.get(k) {
                    Some(h) => h,
                    None => continue,
                };
                let off = f64::from_bits(*k);
                for &a in lows {
                    for &b in highs {
                        let (la, ha) = boxes[a as usize];
                        let (lb, hb) = boxes[b as usize];
                        let l2 = [la[u].max(lb[u]), la[w].max(lb[w])];
                        let h2 = [ha[u].min(hb[u]), ha[w].min(hb[w])];
                        if l2[0] < h2[0] && l2[1] < h2[1] {
                            raw.push(RawFace {
                                ax: ax as u8,
                                off,
                                lo2: l2,
                                hi2: h2,
                                cells: [a, b],
                            });
                        }
                    }
                }
            }
            // outside faces: parts of box facets not covered by neighbors are
            // only needed for completeness of cell boundaries far from the
            // solid; we add whole facets with NONE when no neighbor covers
            // any part of them.
            for (i, (lo, hi)) in boxes.iter().enumerate() {
                for (side, off) in [(0usize, lo[ax]), (1usize, hi[ax])] {
                    let covered = raw.iter().any(|f| {
                        f.ax as usize == ax && f.off == off && f.cells[1 - side] == i as u32
                    });
                    if !covered {
                        let cellsp = if side == 0 {
                            [NONE, i as u32]
                        } else {
                            [i as u32, NONE]
                        };
                        raw.push(RawFace {
                            ax: ax as u8,
                            off,
                            lo2: [lo[u], lo[w]],
                            hi2: [hi[u], hi[w]],
                            cells: cellsp,
                        });
                    }
                }
            }
        }
        raw.sort_by(|a, b| {
            (
                a.ax,
                a.off.to_bits(),
                a.cells,
                a.lo2[0].to_bits(),
                a.lo2[1].to_bits(),
            )
                .cmp(&(
                    b.ax,
                    b.off.to_bits(),
                    b.cells,
                    b.lo2[0].to_bits(),
                    b.lo2[1].to_bits(),
                ))
        });
        // complex vertices: all face corners
        let mut vert_of: BTreeMap<[u64; 3], u32> = BTreeMap::new();
        let mut verts: Vec<P3> = Vec::new();
        let corner = |f: &RawFace, cu: f64, cw: f64| -> P3 {
            let ax = f.ax as usize;
            let mut p = [0.0; 3];
            p[ax] = f.off;
            p[(ax + 1) % 3] = cu;
            p[(ax + 2) % 3] = cw;
            p
        };
        for f in &raw {
            for (cu, cw) in [
                (f.lo2[0], f.lo2[1]),
                (f.hi2[0], f.lo2[1]),
                (f.hi2[0], f.hi2[1]),
                (f.lo2[0], f.hi2[1]),
            ] {
                let p = corner(f, cu, cw);
                let k = [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
                vert_of.entry(k).or_insert_with(|| {
                    verts.push(p);
                    (verts.len() - 1) as u32
                });
            }
        }
        // vertices by axis-parallel line: key (axis of line, the two fixed coords)
        let mut on_line: BTreeMap<(u8, u64, u64), Vec<(f64, u32)>> = BTreeMap::new();
        for (i, p) in verts.iter().enumerate() {
            for ax in 0..3usize {
                let (u, w) = ((ax + 1) % 3, (ax + 2) % 3);
                on_line
                    .entry((ax as u8, p[u].to_bits(), p[w].to_bits()))
                    .or_default()
                    .push((p[ax], i as u32));
            }
        }
        for v in on_line.values_mut() {
            v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }
        let mut edges: Vec<CEdge> = Vec::new();
        let mut edge_of: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        let mut faces: Vec<CFace> = Vec::new();
        for f in &raw {
            let ax = f.ax as usize;
            let (u, w) = ((ax + 1) % 3, (ax + 2) % 3);
            let fplane = get_plane(f.ax, f.off, &mut bp);
            // CCW about +axis in (u, w) coordinates: (lo,lo)->(hi,lo)->(hi,hi)->(lo,hi)
            let cs = [
                (f.lo2[0], f.lo2[1]),
                (f.hi2[0], f.lo2[1]),
                (f.hi2[0], f.hi2[1]),
                (f.lo2[0], f.hi2[1]),
            ];
            let mut loop_verts = Vec::new();
            let mut loop_edges = Vec::new();
            for k in 0..4 {
                let a = corner(f, cs[k].0, cs[k].1);
                let b = corner(f, cs[(k + 1) % 4].0, cs[(k + 1) % 4].1);
                // the line axis is the coordinate that changes
                let lax = if a[u] != b[u] { u } else { w };
                let (fu, fw) = ((lax + 1) % 3, (lax + 2) % 3);
                let list = &on_line[&(lax as u8, a[fu].to_bits(), a[fw].to_bits())];
                let (t0, t1) = (a[lax], b[lax]);
                let mut seg: Vec<(f64, u32)> = list
                    .iter()
                    .copied()
                    .filter(|&(t, _)| t >= t0.min(t1) && t <= t0.max(t1))
                    .collect();
                if t0 > t1 {
                    seg.reverse();
                }
                // other plane through this edge line (perpendicular to face, constant coord)
                let other_ax = if lax == u { w } else { u };
                let other_plane = get_plane(other_ax as u8, a[other_ax], &mut bp);
                let line = if fplane < other_plane {
                    LineKey(fplane, other_plane)
                } else {
                    LineKey(other_plane, fplane)
                };
                for s in 0..seg.len() - 1 {
                    let (va, vb) = (seg[s].1, seg[s + 1].1);
                    loop_verts.push(va);
                    let key = (va.min(vb), va.max(vb));
                    let eid = *edge_of.entry(key).or_insert_with(|| {
                        let (pa, pb) = (verts[key.0 as usize], verts[key.1 as usize]);
                        let (lo_v, hi_v) = if pa[lax] < pb[lax] {
                            (key.0, key.1)
                        } else {
                            (key.1, key.0)
                        };
                        let plo = verts[lo_v as usize][lax];
                        let phi = verts[hi_v as usize][lax];
                        let e_lo = (get_plane(lax as u8, plo, &mut bp), -1i8);
                        let e_hi = (get_plane(lax as u8, phi, &mut bp), 1i8);
                        let (v, end) = if key.0 == lo_v {
                            ([key.0, key.1], [e_lo, e_hi])
                        } else {
                            ([key.0, key.1], [e_hi, e_lo])
                        };
                        edges.push(CEdge { line, v, end });
                        (edges.len() - 1) as u32
                    });
                    loop_edges.push(eid);
                }
            }
            faces.push(CFace {
                plane: fplane,
                cells: f.cells,
                loop_verts,
                loop_edges,
            });
        }
        for (fi, f) in faces.iter().enumerate() {
            for &c in &f.cells {
                if c != NONE {
                    cells[c as usize].faces.push(fi as u32);
                }
            }
        }
        // facet subdivisions: lines of face edges that run through the
        // interior of a cell's facet (i.e. not along the cell's own planes)
        for ci in 0..n {
            let own: std::collections::BTreeSet<PlaneId> =
                cells[ci].halfspaces.iter().map(|h| h.0).collect();
            let mut by_plane: BTreeMap<PlaneId, Vec<(PlaneId, [(PlaneId, i8); 2])>> =
                BTreeMap::new();
            for &f in &cells[ci].faces {
                let fc = &faces[f as usize];
                for &e in &fc.loop_edges {
                    let ce = &edges[e as usize];
                    let l = ce.line;
                    let other = if l.0 == fc.plane { l.1 } else { l.0 };
                    if !own.contains(&other) {
                        by_plane.entry(fc.plane).or_default().push((other, ce.end));
                    }
                }
            }
            for (p, mut qs) in by_plane {
                qs.sort_unstable();
                qs.dedup();
                cells[ci].facet_subdiv.push((p, qs));
            }
        }
        let _ = boxes;
        let mut cx = Complex {
            planes: PlaneSystem::Boxes(bp),
            verts,
            edges,
            faces,
            cells,
            sites,
        };
        cx.finish_aabbs();
        Ok(cx)
    }
}

/// True if the permutation (of 0..4) is even.
fn perm_parity(p: [usize; 4]) -> bool {
    let mut inv = 0;
    for i in 0..4 {
        for j in i + 1..4 {
            if p[i] > p[j] {
                inv += 1;
            }
        }
    }
    inv % 2 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face_area(cx: &Complex, f: &CFace) -> glam::DVec3 {
        let pts: Vec<glam::DVec3> = f
            .loop_verts
            .iter()
            .map(|&v| glam::DVec3::from_array(cx.verts[v as usize]))
            .collect();
        frac_geom::polygon::newell(&pts)
    }

    #[test]
    fn voronoi_cells_partition_box_volume() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
        let seeds: Vec<P3> = (0..200)
            .map(|_| {
                [
                    rng.gen_range(0.0..1.0),
                    rng.gen_range(0.0..1.0),
                    rng.gen_range(0.0..1.0),
                ]
            })
            .collect();
        let cx = Complex::voronoi(&seeds, [0.0; 3], [1.0; 3]).unwrap();
        // every face loop is CCW about its plane normal
        for f in &cx.faces {
            let n = glam::DVec3::from_array(cx.planes.normal(f.plane));
            let a = face_area(&cx, f);
            assert!(a.dot(n) >= -1e-12, "face orientation");
        }
        // cells are closed: sum of outward area vectors = 0
        for (ci, c) in cx.cells.iter().enumerate() {
            let mut s = glam::DVec3::ZERO;
            for &f in &c.faces {
                let fa = &cx.faces[f as usize];
                let a = face_area(&cx, fa);
                s += if fa.cells[0] == ci as u32 { a } else { -a };
            }
            assert!(s.length() < 1e-9, "cell {ci} not closed: {s:?}");
        }
    }

    #[test]
    fn box_partition_with_t_junctions() {
        // two bricks below, one straddling above
        let b = vec![
            ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            ([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]),
            ([0.0, 1.0, 0.0], [0.5, 2.0, 1.0]),
            ([0.5, 1.0, 0.0], [1.5, 2.0, 1.0]),
            ([1.5, 1.0, 0.0], [2.0, 2.0, 1.0]),
        ];
        let cx = Complex::boxes(&b).unwrap();
        // the lower-left brick top facet is split in two faces
        let c0 = &cx.cells[0];
        assert!(!c0.facet_subdiv.is_empty());
        for (ci, c) in cx.cells.iter().enumerate() {
            let mut s = glam::DVec3::ZERO;
            for &f in &c.faces {
                let fa = &cx.faces[f as usize];
                let a = face_area(&cx, fa);
                s += if fa.cells[0] == ci as u32 { a } else { -a };
            }
            assert!(s.length() < 1e-12, "cell {ci}");
        }
    }
}
