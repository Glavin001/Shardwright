//! Material recipes (spec Stage 2): build a convex complex in a local frame,
//! clip the component solid, and post-process into nested cells.
//!
//! Nesting is exact by construction: the *fine* cells form one complex and
//! every analysis cell is a (connected) union of fine cells.

use crate::cellset::{CellSet, CellSetParams};
use crate::clip::{Clipper, SiteGrid, VKey};
use crate::complex::Complex;
use crate::seeding;
use frac_core::determinism::{RngStage, rng_for, stable_hash, unit_f64};
use frac_geom::inside::MeshQuery;
use frac_geom::integrals::sym_eigen3;
use frac_geom::polygon::outer;
use frac_geom::{DMat3, DVec3, TriMesh};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BondPattern {
    Stretcher,
    Stack,
    English,
    Flemish,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct MasonryLayout {
    /// Brick length, height, depth (m), excluding mortar.
    pub brick: [f64; 3],
    pub bond: BondPattern,
    /// Mortar joint thickness (m); joints are interfaces, not volumes.
    pub mortar: f64,
    /// Course alignment offset along (length, height) in the wall frame.
    pub origin: [f64; 2],
    /// Fraction of bricks split in half (sub-brick fine cells).
    pub half_split_fraction: f64,
    /// Fraction of bricks with a corner chip cell.
    pub chip_fraction: f64,
    /// Optional explicit wall length axis (world); default: principal axis.
    pub length_axis: Option<[f64; 3]>,
}

impl Default for MasonryLayout {
    fn default() -> Self {
        MasonryLayout {
            brick: [0.215, 0.065, 0.1025],
            bond: BondPattern::Stretcher,
            mortar: 0.01,
            origin: [0.0, 0.0],
            half_split_fraction: 0.25,
            chip_fraction: 0.1,
            length_axis: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum Recipe {
    /// Concrete, ceramic, generic brittle: clustered Voronoi.
    ClusteredVoronoi,
    Masonry(MasonryLayout),
    /// Wood: anisotropic Voronoi stretched along the grain.
    Wood {
        stretch: f64,
        fine_stretch: f64,
    },
    /// Annealed glass: radial + concentric pattern around an impact center.
    GlassAnnealed {
        impact: Option<[f64; 3]>,
        rings: u32,
        spokes: u32,
    },
    /// Tempered glass: fine near-uniform dicing.
    GlassTempered {
        cell_size: Option<f64>,
    },
    /// Drywall/plaster panels: panel-scale Voronoi with ragged sub-cells.
    Panel,
    /// Steel: one cell per member.
    Steel,
    /// Stone: bedding-plane slabs with Voronoi inside.
    Stone {
        bedding: [f64; 3],
        layer: f64,
        flatten: f64,
    },
    /// User-provided seed points (world frame).
    Custom {
        seeds: Vec<[f64; 3]>,
    },
}

impl Recipe {
    pub fn from_name(name: &str) -> Option<Recipe> {
        Some(match name {
            "concrete_clustered_voronoi" | "clustered_voronoi" | "voronoi" | "ceramic" => {
                Recipe::ClusteredVoronoi
            }
            "masonry" | "masonry_bricks" => Recipe::Masonry(MasonryLayout::default()),
            "wood_anisotropic_voronoi" | "wood" => Recipe::Wood {
                stretch: 6.0,
                fine_stretch: 9.0,
            },
            "glass_annealed" | "glass_radial" => Recipe::GlassAnnealed {
                impact: None,
                rings: 6,
                spokes: 14,
            },
            "glass_tempered" | "glass_dicing" => Recipe::GlassTempered { cell_size: None },
            "drywall" | "plaster" | "panel" => Recipe::Panel,
            "steel" | "steel_member" => Recipe::Steel,
            "stone" | "stone_bedding" => Recipe::Stone {
                bedding: [0.0, 1.0, 0.0],
                layer: 0.15,
                flatten: 2.5,
            },
            _ => return None,
        })
    }
    pub fn name(&self) -> &'static str {
        match self {
            Recipe::ClusteredVoronoi => "clustered_voronoi",
            Recipe::Masonry(_) => "masonry",
            Recipe::Wood { .. } => "wood",
            Recipe::GlassAnnealed { .. } => "glass_annealed",
            Recipe::GlassTempered { .. } => "glass_tempered",
            Recipe::Panel => "panel",
            Recipe::Steel => "steel",
            Recipe::Stone { .. } => "stone",
            Recipe::Custom { .. } => "custom",
        }
    }
}

/// Inputs to cell construction for one component.
pub struct CellParams<'a> {
    pub recipe: Recipe,
    pub analysis_target: usize,
    pub fine_per_analysis: usize,
    pub max_fine: usize,
    pub asset_seed: u64,
    pub component: u32,
    pub variant: u32,
    pub grain: Option<DVec3>,
    pub min_cell_volume: f64,
    pub min_thickness_ratio: f64,
    /// Relative spacing field (1 = nominal, <1 denser), world frame.
    pub spacing: Option<&'a (dyn Fn(DVec3) -> f64 + Sync)>,
}

/// Local frame: `local = scale * (rot * (x - origin))`.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub rot: DMat3,
    pub origin: DVec3,
    pub scale: DVec3,
}

impl Frame {
    pub fn identity() -> Frame {
        Frame {
            rot: DMat3::IDENTITY,
            origin: DVec3::ZERO,
            scale: DVec3::ONE,
        }
    }
    pub fn is_identity(&self) -> bool {
        self.rot == DMat3::IDENTITY && self.origin == DVec3::ZERO && self.scale == DVec3::ONE
    }
    #[inline]
    pub fn to_local(&self, x: DVec3) -> DVec3 {
        (self.rot * (x - self.origin)) * self.scale
    }
    #[inline]
    pub fn to_world(&self, y: DVec3) -> DVec3 {
        self.rot.transpose() * (y / self.scale) + self.origin
    }
    /// Frame whose rows are the given orthonormal axes (u, v, w).
    pub fn from_axes(u: DVec3, v: DVec3, w: DVec3, origin: DVec3) -> Frame {
        Frame {
            rot: DMat3::from_cols(u, v, w).transpose(),
            origin,
            scale: DVec3::ONE,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CellBuild {
    pub cells: CellSet,
    pub n_clusters: u32,
    pub recipe: String,
    /// Fine cells requested / produced (diagnostics).
    pub fine_seeds: usize,
    pub analysis_seeds: usize,
}

/// Principal axes of a solid (eigenvectors of the volume covariance,
/// ascending variance) and the covariance eigenvalues.
pub fn principal_axes(m: &TriMesh) -> (DVec3, DMat3, DVec3) {
    let vi = m.volume_integrals();
    let c = vi.com();
    let cov = (vi.second - outer(c, c) * vi.volume) / vi.volume.max(1e-300);
    let (ev, vecs) = sym_eigen3(&cov);
    (ev, vecs, c)
}

fn hash_unit(seed: u64, k: u64, salt: u64) -> f64 {
    unit_f64(stable_hash(&[seed, k, salt]))
}

/// Build the fine cell complex, clip the solid, and post-process.
pub fn build_cells(solid: &TriMesh, p: &CellParams) -> Result<CellBuild, String> {
    let mut rng = rng_for(p.asset_seed, p.component, RngStage::Seeding, 3, p.variant);
    let (ev, axes, com) = principal_axes(solid);
    let comp_min_extent = (12.0 * ev.x.max(0.0)).sqrt();
    let n_a = p.analysis_target.max(1);
    let n_f = (n_a * p.fine_per_analysis.max(1))
        .min(p.max_fine.max(1))
        .max(n_a);
    let jitter_seed = stable_hash(&[p.asset_seed, p.component as u64, p.variant as u64, 77]);

    // ---- choose frame and build complex in the local frame
    let (frame, complex, cluster_of, unit_of, na_used, nf_used) = match &p.recipe {
        Recipe::Masonry(layout) => {
            let up = DVec3::Y;
            let mut len_axis = match layout.length_axis {
                Some(a) => DVec3::from_array(a),
                None => {
                    // horizontal principal direction of largest variance
                    let mut best = DVec3::X;
                    let mut bv = -1.0;
                    for k in 0..3 {
                        let a = axes.col(k);
                        let h = DVec3::new(a.x, 0.0, a.z);
                        if h.length() < 1e-6 {
                            continue;
                        }
                        let var = ev[k] * (h.length_squared());
                        if var > bv {
                            bv = var;
                            best = h.normalize();
                        }
                    }
                    best
                }
            };
            len_axis = DVec3::new(len_axis.x, 0.0, len_axis.z).normalize();
            // canonical sign
            if len_axis.x < 0.0 || (len_axis.x == 0.0 && len_axis.z < 0.0) {
                len_axis = -len_axis;
            }
            let thick = len_axis.cross(up);
            // exact axis-aligned frame when possible
            let snap = |v: DVec3| -> DVec3 {
                let r = DVec3::new(v.x.round(), v.y.round(), v.z.round());
                if (r - v).length() < 1e-12 && r.length() == 1.0 {
                    r
                } else {
                    v
                }
            };
            let frame = Frame::from_axes(snap(len_axis), up, snap(thick), DVec3::ZERO);
            let local = solid.transformed(|x| frame.to_local(x));
            let bb = local.aabb();
            let boxes = masonry_boxes(layout, bb.min, bb.max, p.asset_seed ^ jitter_seed);
            let cx = Complex::boxes(&boxes.iter().map(|b| (b.0, b.1)).collect::<Vec<_>>())?;
            let unit: Vec<u32> = boxes.iter().map(|b| b.2).collect();
            let nbricks = unit.iter().copied().max().map(|x| x + 1).unwrap_or(0) as usize;
            (frame, cx, unit.clone(), unit, nbricks, boxes.len())
        }
        recipe => {
            // Voronoi family
            let (frame, fine_scale_z, cluster_scale_z, planar): (Frame, f64, f64, bool) =
                match recipe {
                    Recipe::Wood {
                        stretch,
                        fine_stretch,
                    } => {
                        let g = p.grain.unwrap_or(axes.col(2)).normalize();
                        let (u, v) = frac_geom::polygon::plane_basis(g);
                        let mut f = Frame::from_axes(u, v, g, com);
                        // grain = local z, compressed so cells elongate along it
                        f.scale = DVec3::new(1.0, 1.0, 1.0 / fine_stretch.max(1.0));
                        (f, *fine_stretch, *stretch, false)
                    }
                    Recipe::Stone {
                        bedding, flatten, ..
                    } => {
                        let b = DVec3::from_array(*bedding).normalize();
                        let (u, v) = frac_geom::polygon::plane_basis(b);
                        let mut f = Frame::from_axes(u, v, b, com);
                        f.scale = DVec3::new(1.0, 1.0, flatten.max(1.0));
                        (f, 1.0, 1.0, false)
                    }
                    Recipe::GlassAnnealed { .. } | Recipe::GlassTempered { .. } | Recipe::Panel => {
                        // pane frame: w = thinnest principal axis
                        let w = axes.col(0);
                        let u = axes.col(2);
                        let v = w.cross(u);
                        (Frame::from_axes(u, v, w, com), 1.0, 1.0, true)
                    }
                    _ => (Frame::identity(), 1.0, 1.0, false),
                };
            let local = if frame.is_identity() {
                solid.clone()
            } else {
                solid.transformed(|x| frame.to_local(x))
            };
            let q = MeshQuery::new(&local);
            let bb = local.aabb();
            let inside = |x: DVec3| q.contains(x);
            let spacing_local = |x: DVec3| -> f64 {
                match p.spacing {
                    Some(f) => f(frame.to_world(x)),
                    None => 1.0,
                }
            };
            let mid_w = 0.5 * (bb.min.z + bb.max.z);
            let flat = if planar { Some((2usize, mid_w)) } else { None };
            let (mut fine, analysis): (Vec<DVec3>, Vec<DVec3>) = match recipe {
                Recipe::Steel => (vec![bb.center()], vec![bb.center()]),
                Recipe::Custom { seeds } => {
                    let f: Vec<DVec3> = seeds
                        .iter()
                        .map(|s| frame.to_local(DVec3::from_array(*s)))
                        .collect();
                    (f.clone(), f)
                }
                Recipe::GlassAnnealed {
                    impact,
                    rings,
                    spokes,
                } => {
                    let c = impact
                        .map(|i| frame.to_local(DVec3::from_array(i)))
                        .unwrap_or(DVec3::new(0.0, 0.0, mid_w));
                    let rmax = [bb.min, bb.max]
                        .iter()
                        .flat_map(|a| [bb.min, bb.max].map(move |b| DVec3::new(a.x, b.y, mid_w)))
                        .map(|corner| (corner - c).length())
                        .fold(0.0, f64::max);
                    let mut pts = vec![DVec3::new(c.x, c.y, mid_w)];
                    let rings = (*rings).max(2) as usize;
                    // scale ring/spoke counts towards the fine target
                    let base = (*spokes).max(6) as usize;
                    let per_ring_scale = ((n_f as f64) / (rings * base) as f64).sqrt().max(1.0);
                    let rings_n = ((rings as f64) * per_ring_scale).round() as usize;
                    for k in 1..=rings_n {
                        let t = k as f64 / rings_n as f64;
                        let r = rmax * t.powf(1.4);
                        let m = ((base as f64 * per_ring_scale) * (0.6 + 0.4 * t))
                            .round()
                            .max(6.0) as usize;
                        let phase = hash_unit(jitter_seed, k as u64, 1) * std::f64::consts::TAU;
                        for j in 0..m {
                            let a = phase
                                + std::f64::consts::TAU
                                    * (j as f64
                                        + 0.35
                                            * (hash_unit(jitter_seed, (k * 1000 + j) as u64, 2)
                                                - 0.5))
                                    / m as f64;
                            let rr = r
                                * (1.0
                                    + 0.12
                                        * (hash_unit(jitter_seed, (k * 1000 + j) as u64, 3) - 0.5));
                            pts.push(DVec3::new(
                                c.x + rr * libm::cos(a),
                                c.y + rr * libm::sin(a),
                                mid_w,
                            ));
                        }
                    }
                    // keep seeds whose column hits the pane
                    let keep: Vec<DVec3> = pts.into_iter().filter(|s| inside(*s)).collect();
                    let cand =
                        seeding::candidates_in(bb.min, bb.max, &inside, n_a * 32, flat, &mut rng);
                    let an = seeding::eliminate(&cand, &spacing_local, n_a, &mut rng);
                    (keep, an)
                }
                _ => {
                    let target_f = match recipe {
                        Recipe::GlassTempered { cell_size } => {
                            let thick = bb.extent().z;
                            let cs = cell_size.unwrap_or(2.5 * thick).max(1e-4);
                            let area = (local.signed_volume() / thick.max(1e-9)).max(1e-9);
                            ((area / (cs * cs)) as usize).clamp(1, p.max_fine.max(1))
                        }
                        _ => n_f,
                    };
                    let cand = seeding::candidates_in(
                        bb.min,
                        bb.max,
                        &inside,
                        (target_f * 8).max(64),
                        flat,
                        &mut rng,
                    );
                    let fine = seeding::eliminate(&cand, &spacing_local, target_f, &mut rng);
                    let an_cand = if cand.len() >= n_a {
                        cand.clone()
                    } else {
                        fine.clone()
                    };
                    let an = seeding::eliminate(&an_cand, &spacing_local, n_a, &mut rng);
                    (fine, an)
                }
            };
            if fine.is_empty() {
                fine.push(bb.center());
            }
            // jitter to break exact symmetries (relative to spacing)
            let sp = (bb.extent().x.max(bb.extent().y).max(bb.extent().z))
                / (fine.len() as f64).cbrt().max(1.0);
            for (i, s) in fine.iter_mut().enumerate() {
                let j = DVec3::new(
                    hash_unit(jitter_seed, i as u64, 11) - 0.5,
                    hash_unit(jitter_seed, i as u64, 12) - 0.5,
                    if planar {
                        0.0
                    } else {
                        hash_unit(jitter_seed, i as u64, 13) - 0.5
                    },
                );
                *s += j * (sp * 1e-6);
            }
            let fine_arr: Vec<[f64; 3]> = fine.iter().map(|s| s.to_array()).collect();
            let pad = bb.extent() * 0.01 + DVec3::splat(1e-9);
            let cx = Complex::voronoi(
                &fine_arr,
                (bb.min - pad).to_array(),
                (bb.max + pad).to_array(),
            )?;
            // clustering: nearest analysis seed in the analysis metric
            // local z = world_z / fine_stretch; analysis metric wants world_z / stretch
            let zs = fine_scale_z / cluster_scale_z.max(1e-9);
            let scale_pt = |x: DVec3| [x.x, x.y, x.z * zs];
            let analysis_pts: Vec<[f64; 3]> = analysis.iter().map(|a| scale_pt(*a)).collect();
            let grid = SiteGrid::new(&analysis_pts);
            let mut cluster: Vec<u32> = cx
                .sites
                .iter()
                .map(|s| grid.nearest(scale_pt(DVec3::from_array(*s))).unwrap_or(0))
                .collect();
            if let Recipe::Stone { layer, flatten, .. } = recipe {
                // slabs: combine bedding layer index with lateral clusters
                let lz = layer * flatten.max(1.0);
                let nl = analysis.len().max(1) as u32;
                for (i, s) in cx.sites.iter().enumerate() {
                    let layer_idx = ((s[2] - bb.min.z) / lz.max(1e-9)).floor() as u32;
                    cluster[i] = layer_idx * nl + cluster[i];
                }
            }
            let unit: Vec<u32> = (0..cx.cells.len() as u32).collect();
            let na = analysis.len();
            let nf = fine.len();
            (frame, cx, cluster, unit, na, nf)
        }
    };

    // ---- clip in the local frame
    let local = if frame.is_identity() {
        solid.clone()
    } else {
        solid.transformed(|x| frame.to_local(x))
    };
    let mut out = Clipper::new(&local, &complex).run()?;
    if !frame.is_identity() {
        for (i, k) in out.keys.iter().enumerate() {
            out.verts[i] = match k {
                VKey::Orig(v) => solid.verts[*v as usize],
                _ => frame.to_world(out.verts[i]),
            };
        }
        // patch normals back to world (covector transform for scaled frames)
        for pt in out.patches.iter_mut() {
            let nl = pt.normal * frame.scale; // gradient transforms with the scale
            pt.normal = (frame.rot.transpose() * nl).normalize();
        }
    }
    let params = CellSetParams {
        min_cell_volume: p.min_cell_volume,
        min_thickness_ratio: p.min_thickness_ratio,
        component_min_extent: comp_min_extent,
    };
    let mut cells = CellSet::from_clip(out, &unit_of, &cluster_of, &params);
    // recompute patch areas in the world frame
    for pt in cells.patches.iter_mut() {
        let loops: Vec<Vec<DVec3>> = pt
            .loops
            .iter()
            .map(|l| l.iter().map(|&v| cells.verts[v as usize]).collect())
            .collect();
        pt.area = frac_geom::polygon::area_integrals(&loops, pt.normal).area;
    }
    let n_clusters = cells.finalize_clusters();
    Ok(CellBuild {
        cells,
        n_clusters,
        recipe: p.recipe.name().to_string(),
        fine_seeds: nf_used,
        analysis_seeds: na_used,
    })
}

/// Axis-aligned brick boxes `(lo, hi, brick_id)` tiling an extended box
/// around `[lo, hi]` in the wall frame (x = length, y = height, z = depth).
pub fn masonry_boxes(
    l: &MasonryLayout,
    lo: DVec3,
    hi: DVec3,
    seed: u64,
) -> Vec<([f64; 3], [f64; 3], u32)> {
    let ext = hi - lo;
    let big = ext.length().max(1.0);
    let (bl, bh, bd) = (
        l.brick[0] + l.mortar,
        l.brick[1] + l.mortar,
        l.brick[2] + l.mortar,
    );
    let thick = ext.z;
    let nw = ((thick / bd).round() as usize).max(1);
    let mut zc: Vec<f64> = vec![lo.z - big];
    for k in 1..nw {
        zc.push(lo.z + thick * k as f64 / nw as f64);
    }
    zc.push(hi.z + big);
    // course cuts
    let y0 = lo.y + l.origin[1];
    let mut yc = vec![lo.y - big];
    let mut k = 1;
    while y0 + k as f64 * bh < hi.y - 1e-9 * bh {
        yc.push(y0 + k as f64 * bh);
        k += 1;
    }
    yc.push(hi.y + big);
    let mut boxes = Vec::new();
    let mut brick = 0u32;
    for c in 0..yc.len() - 1 {
        // brick pattern along x for this course: (width, spans_all_wythes)
        let header_course = matches!(l.bond, BondPattern::English) && c % 2 == 1;
        let (pattern, offset): (Vec<(f64, bool)>, f64) = match l.bond {
            BondPattern::Stack => (vec![(bl, false)], 0.0),
            BondPattern::Stretcher => (vec![(bl, false)], if c % 2 == 1 { bl * 0.5 } else { 0.0 }),
            BondPattern::English => {
                if header_course {
                    (vec![(bd, true)], bd * 0.5)
                } else {
                    (vec![(bl, false)], 0.0)
                }
            }
            BondPattern::Flemish => (
                vec![(bl, false), (bd, true)],
                if c % 2 == 1 { (bl + bd) * 0.5 } else { 0.0 },
            ),
        };
        let period: f64 = pattern.iter().map(|p| p.0).sum();
        let x_start = lo.x + l.origin[0] + offset;
        // walk from well before lo.x
        let n_back = (((x_start - lo.x) / period).ceil().max(0.0) as usize) + 1;
        let mut x = x_start - n_back as f64 * period;
        let mut pi = 0usize;
        let mut cuts: Vec<(f64, bool)> = Vec::new(); // (cut, span flag of segment after cut)
        let mut first_flag = pattern[0].1;
        while x < hi.x {
            let (w, f) = pattern[pi % pattern.len()];
            let x1 = x + w;
            if x1 <= lo.x {
                first_flag = pattern[(pi + 1) % pattern.len()].1;
            } else if x1 < hi.x {
                cuts.push((x1, pattern[(pi + 1) % pattern.len()].1));
            }
            let _ = f;
            x = x1;
            pi += 1;
        }
        let mut segs: Vec<(f64, f64, bool)> = Vec::new();
        let mut prev = (lo.x - big, first_flag);
        for (cx_, f) in cuts {
            segs.push((prev.0, cx_, prev.1));
            prev = (cx_, f);
        }
        segs.push((prev.0, hi.x + big, prev.1));
        for (x0, x1, span) in segs {
            let wy: Vec<(f64, f64)> = if span || nw == 1 {
                vec![(zc[0], *zc.last().unwrap())]
            } else {
                zc.windows(2).map(|w| (w[0], w[1])).collect()
            };
            for (z0, z1) in wy {
                let id = brick;
                brick += 1;
                let (yl, yh) = (yc[c], yc[c + 1]);
                let h = stable_hash(&[seed, id as u64, 5]);
                let u = unit_f64(h);
                let interior =
                    x0 > lo.x - big && x1 < hi.x + big && yl > lo.y - big && yh < hi.y + big;
                if interior && u < l.half_split_fraction {
                    let xm = 0.5 * (x0 + x1);
                    boxes.push(([x0, yl, z0], [xm, yh, z1], id));
                    boxes.push(([xm, yl, z0], [x1, yh, z1], id));
                } else if interior && u < l.half_split_fraction + l.chip_fraction {
                    let corner = (h >> 20) & 3;
                    let cw = 0.25 * (x1 - x0);
                    let chh = 0.4 * (yh - yl);
                    let (xa, xb) = if corner & 1 == 0 {
                        (x0, x0 + cw)
                    } else {
                        (x1 - cw, x1)
                    };
                    let (ya, yb) = if corner & 2 == 0 {
                        (yl, yl + chh)
                    } else {
                        (yh - chh, yh)
                    };
                    // chip, column remainder, rest
                    boxes.push(([xa, ya, z0], [xb, yb, z1], id));
                    let (ry0, ry1) = if corner & 2 == 0 { (yb, yh) } else { (yl, ya) };
                    boxes.push(([xa, ry0, z0], [xb, ry1, z1], id));
                    let (rx0, rx1) = if corner & 1 == 0 { (xb, x1) } else { (x0, xa) };
                    boxes.push(([rx0, yl, z0], [rx1, yh, z1], id));
                } else {
                    boxes.push(([x0, yl, z0], [x1, yh, z1], id));
                }
            }
        }
    }
    boxes
}
