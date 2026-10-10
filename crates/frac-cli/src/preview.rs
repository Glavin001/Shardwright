//! `prefracture preview`: software-rasterized PNG previews of a baked asset,
//! one distinct colour per fragment, optionally exploded, for any hierarchy
//! level (or all levels side by side). Uses the exact clean fragment
//! boundaries from the asset JSON; deterministic.

use frac_core::Asset;
use glam::DVec3;
use rayon::prelude::*;
use std::path::Path;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BondColour {
    /// Joint type (cold joint, bearing, mortar, adhesive, monolithic, anchor…).
    Kind,
    /// Tensile capacity (interface/material strength × Weibull scale), log heat map.
    Strength,
}

pub struct PreviewOptions {
    /// Draw the level's bonds (their contact surfaces) instead of fragments.
    pub bonds: Option<BondColour>,
    /// Cutaway: drop geometry whose centroid has z above this value.
    pub clip_z: Option<f64>,
    pub level: Option<u8>,
    pub all_levels: bool,
    pub width: u32,
    pub height: u32,
    pub explode: f64,
    pub azimuth_deg: f64,
    pub elevation_deg: f64,
}

struct Tri {
    p: [DVec3; 3],
    frag: u32,
    shade: f32,
    /// Fixed colour (bond views); fragment colour from `frag` otherwise.
    rgb: Option<[f32; 3]>,
}

/// Distinct, saturated colour per fragment (golden-ratio hue walk with
/// varying saturation/value bands).
fn colour(id: u32) -> [f32; 3] {
    let h = (id as f64 * 0.618_033_988_75).fract();
    let s = [0.65, 0.85, 0.5][(id % 3) as usize];
    let v = [0.95, 0.8, 0.9][((id / 3) % 3) as usize];
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s));
    let (r, g, b) = match i as i32 % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [r as f32, g as f32, b as f32]
}

fn kind_colour(k: frac_core::InterfaceKind) -> [f32; 3] {
    use frac_core::InterfaceKind as K;
    match k {
        K::ColdJoint => [0.20, 0.45, 0.85],
        K::Bearing => [0.95, 0.55, 0.10],
        K::MortarJoint => [0.65, 0.25, 0.20],
        K::Adhesive => [0.20, 0.70, 0.35],
        K::Weld | K::Bolted => [0.90, 0.80, 0.15],
        K::GrainBoundary => [0.55, 0.30, 0.70],
        K::Anchor => [0.05, 0.05, 0.05],
        K::Monolithic => [0.70, 0.70, 0.72],
        _ => [0.20, 0.65, 0.70],
    }
}

/// Perceptual-ish heat map (dark blue → teal → yellow) for t ∈ [0, 1].
fn heat(t: f64) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    let stops = [[0.15, 0.10, 0.45], [0.10, 0.45, 0.60], [0.25, 0.70, 0.40], [0.95, 0.85, 0.15]];
    let x = t * 3.0;
    let i = (x.floor() as usize).min(2);
    let f = x - i as f64;
    let (a, b) = (stops[i], stops[i + 1]);
    [(a[0] + (b[0] - a[0]) * f) as f32, (a[1] + (b[1] - a[1]) * f) as f32, (a[2] + (b[2] - a[2]) * f) as f32]
}

/// Bond contact surfaces of a level, coloured by kind or tensile capacity.
fn bond_tris(asset: &Asset, lib: &frac_material::MaterialLibrary, level: u8, mode: BondColour, clip_z: Option<f64>, light: DVec3) -> (Vec<Tri>, String) {
    let cap = |b: &frac_core::Bond| -> f64 {
        let comp = b.composition.first();
        let mat_t = |f: frac_core::FragmentId| {
            let m = asset.fragment(f).material_mix.first().map(|x| x.0).unwrap_or(frac_core::MaterialId(0));
            lib.material(m).tensile_strength.unwrap_or(1e6)
        };
        let side = match b.b {
            frac_core::FragmentOrWorld::Fragment(f) => mat_t(b.a).min(mat_t(f)),
            frac_core::FragmentOrWorld::World => mat_t(b.a),
        };
        let ft = comp.and_then(|c| c.interface_material).and_then(|m| lib.interface_material(m)).and_then(|im| im.tensile_strength).unwrap_or(side);
        ft * b.strength_scale as f64
    };
    let bonds: Vec<&frac_core::Bond> = asset.level_bonds(level).filter(|b| clip_z.is_none_or(|z| b.centroid.z <= z)).collect();
    let caps: Vec<f64> = bonds.iter().map(|b| cap(b).max(1.0)).collect();
    let (lo, hi) = caps.iter().fold((f64::INFINITY, 0.0f64), |(l, h), &c| (l.min(c), h.max(c)));
    let mut tris = Vec::new();
    for (k, b) in bonds.iter().enumerate() {
        let kind = if b.anchor { frac_core::InterfaceKind::Anchor } else { b.composition.first().map(|c| c.kind).unwrap_or(frac_core::InterfaceKind::Monolithic) };
        let rgb = match mode {
            BondColour::Kind => kind_colour(kind),
            BondColour::Strength if b.anchor => [0.05, 0.05, 0.05],
            BondColour::Strength => heat(if hi > lo { (caps[k].ln() - lo.ln()) / (hi.ln() - lo.ln()) } else { 0.5 }),
        };
        let shade = (0.55 + 0.45 * b.normal.dot(light).abs()) as f32;
        for &i in &b.interfaces {
            for poly in &asset.interfaces[i.idx()].polygons {
                let Some(outer) = poly.loops.first() else { continue };
                for t in 1..outer.len().saturating_sub(1) {
                    tris.push(Tri { p: [outer[0], outer[t], outer[t + 1]], frag: k as u32, shade, rgb: Some(rgb) });
                }
            }
        }
    }
    let legend = match mode {
        BondColour::Kind => "colour = joint kind: cold joint blue, bearing orange, mortar red-brown, adhesive green, weld/bolted yellow, grain boundary purple, monolithic grey, anchor black".to_string(),
        BondColour::Strength => format!("colour = tensile capacity, log scale {:.2} MPa (dark blue) → {:.2} MPa (yellow); anchors black", lo / 1e6, hi / 1e6),
    };
    (tris, legend)
}

fn level_tris(asset: &Asset, level: u8, explode: f64, clip_z: Option<f64>, light: DVec3) -> Vec<Tri> {
    let frs: Vec<&frac_core::Fragment> = asset.level_fragments(level).iter().filter(|f| clip_z.is_none_or(|z| f.mass.com.z <= z)).collect();
    let centre = {
        let (mut s, mut m) = (DVec3::ZERO, 0.0);
        for f in &frs {
            s += f.mass.com * f.mass.volume;
            m += f.mass.volume;
        }
        s / m.max(1e-300)
    };
    frs.par_iter()
        .flat_map_iter(|&f| {
            let mesh = frac_collision::cells_boundary_mesh(asset, asset.fragment_cells(f));
            let off = (f.mass.com - centre) * explode;
            let id = f.id.0;
            (0..mesh.tris.len())
                .filter_map(move |t| {
                    let p = mesh.tri_points(t).map(|x| x + off);
                    let n = (p[1] - p[0]).cross(p[2] - p[0]);
                    if n.length_squared() == 0.0 {
                        return None;
                    }
                    let lam = n.normalize().dot(light).max(0.0);
                    Some(Tri { p, frag: id, shade: (0.35 + 0.65 * lam) as f32, rgb: None })
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Render one panel into an RGB buffer (supersampled ×2, then box-filtered).
fn render_panel(tris: &[Tri], w: u32, h: u32, view: DVec3, up: DVec3) -> Vec<[u8; 3]> {
    let ss = 2u32;
    let (sw, sh) = (w * ss, h * ss);
    let right = up.cross(view).normalize();
    let up2 = view.cross(right).normalize();
    let proj = |p: DVec3| DVec3::new(p.dot(right), p.dot(up2), p.dot(view));
    let (mut lo, mut hi) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
    for t in tris {
        for p in t.p {
            let q = proj(p);
            lo = lo.min(q);
            hi = hi.max(q);
        }
    }
    let ext = (hi - lo).max(DVec3::splat(1e-9));
    let scale = 0.92 * (sw as f64 / ext.x).min(sh as f64 / ext.y);
    let cx = 0.5 * (lo.x + hi.x);
    let cy = 0.5 * (lo.y + hi.y);
    let to_px = |q: DVec3| DVec3::new(0.5 * sw as f64 + (q.x - cx) * scale, 0.5 * sh as f64 - (q.y - cy) * scale, q.z);
    let n = (sw * sh) as usize;
    let mut depth = vec![f64::NEG_INFINITY; n];
    let mut frag = vec![u32::MAX; n];
    let mut shade = vec![0f32; n];
    let mut rgbs = vec![[0f32; 3]; n];
    for t in tris {
        let a = to_px(proj(t.p[0]));
        let b = to_px(proj(t.p[1]));
        let c = to_px(proj(t.p[2]));
        let area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        if area.abs() < 1e-12 {
            continue;
        }
        let x0 = a.x.min(b.x).min(c.x).floor().max(0.0) as i64;
        let x1 = a.x.max(b.x).max(c.x).ceil().min(sw as f64 - 1.0) as i64;
        let y0 = a.y.min(b.y).min(c.y).floor().max(0.0) as i64;
        let y1 = a.y.max(b.y).max(c.y).ceil().min(sh as f64 - 1.0) as i64;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                let w0 = ((b.x - px) * (c.y - py) - (b.y - py) * (c.x - px)) / area;
                let w1 = ((c.x - px) * (a.y - py) - (c.y - py) * (a.x - px)) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let z = w0 * a.z + w1 * b.z + w2 * c.z;
                let i = (y as u32 * sw + x as u32) as usize;
                // larger projected z = closer to the camera (view points at it)
                if z > depth[i] {
                    depth[i] = z;
                    frag[i] = t.frag;
                    shade[i] = t.shade;
                    rgbs[i] = t.rgb.unwrap_or_else(|| colour(t.frag));
                }
            }
        }
    }
    // fragment outlines: darken pixels whose 4-neighbourhood holds another fragment
    let mut rgb = vec![[0f32; 3]; n];
    for y in 0..sh {
        for x in 0..sw {
            let i = (y * sw + x) as usize;
            let f = frag[i];
            if f == u32::MAX {
                rgb[i] = [0.97, 0.97, 0.98];
                continue;
            }
            let c = rgbs[i];
            let mut edge = false;
            for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if nx >= 0 && ny >= 0 && (nx as u32) < sw && (ny as u32) < sh {
                    let j = (ny as u32 * sw + nx as u32) as usize;
                    if frag[j] != f {
                        edge = true;
                    }
                }
            }
            let k = if edge { 0.25 } else { shade[i] };
            rgb[i] = [c[0] * k, c[1] * k, c[2] * k];
        }
    }
    let mut out = vec![[0u8; 3]; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0f32; 3];
            for sy in 0..ss {
                for sx in 0..ss {
                    let p = rgb[((y * ss + sy) * sw + x * ss + sx) as usize];
                    for k in 0..3 {
                        acc[k] += p[k];
                    }
                }
            }
            let d = (ss * ss) as f32;
            out[(y * w + x) as usize] = acc.map(|v| ((v / d).clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    out
}

pub fn preview(asset: &Asset, lib: &frac_material::MaterialLibrary, out: &Path, o: &PreviewOptions) -> Result<String, String> {
    let (az, el) = (o.azimuth_deg.to_radians(), o.elevation_deg.to_radians());
    // camera looks along -view; `view` points from the scene towards the camera
    let view = DVec3::new(az.sin() * el.cos(), el.sin(), az.cos() * el.cos()).normalize();
    let up = DVec3::Y;
    let light = (view + DVec3::new(0.3, 0.6, 0.2)).normalize();
    let leaf = asset.hierarchy.levels.saturating_sub(1);
    let levels: Vec<u8> = if o.all_levels { (0..asset.hierarchy.levels).collect() } else { vec![o.level.unwrap_or(leaf).min(leaf)] };
    let mut legend = String::new();
    let panels: Vec<Vec<[u8; 3]>> = levels
        .iter()
        .map(|&l| {
            let tris = match o.bonds {
                Some(mode) => {
                    let (t, lg) = bond_tris(asset, lib, l, mode, o.clip_z, light);
                    legend = lg;
                    t
                }
                None => level_tris(asset, l, o.explode, o.clip_z, light),
            };
            render_panel(&tris, o.width, o.height, view, up)
        })
        .collect();
    let total_w = o.width * panels.len() as u32;
    let mut img = vec![0u8; (total_w * o.height * 3) as usize];
    for (k, p) in panels.iter().enumerate() {
        for y in 0..o.height {
            for x in 0..o.width {
                let src = p[(y * o.width + x) as usize];
                let dst = ((y * total_w + k as u32 * o.width + x) * 3) as usize;
                img[dst..dst + 3].copy_from_slice(&src);
            }
        }
    }
    let f = std::fs::File::create(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), total_w, o.height);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header().map_err(|e| e.to_string())?;
    wr.write_image_data(&img).map_err(|e| e.to_string())?;
    let counts: Vec<String> = levels
        .iter()
        .map(|&l| if o.bonds.is_some() { format!("L{l}: {} bonds", asset.level_bonds(l).count()) } else { format!("L{l}: {} fragments", asset.level_fragments(l).len()) })
        .collect();
    Ok(format!("wrote {} ({}×{}; {}){}", out.display(), total_w, o.height, counts.join(", "), if legend.is_empty() { String::new() } else { format!("\n{legend}") }))
}
