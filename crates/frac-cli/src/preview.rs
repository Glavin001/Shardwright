//! `prefracture preview`: software-rasterized PNG previews of a baked asset,
//! one distinct colour per fragment, optionally exploded, for any hierarchy
//! level (or all levels side by side), or of an input mesh before baking.
//! Uses the exact clean fragment boundaries from the asset JSON;
//! deterministic. Every image carries panel titles and a legend (what the
//! colours mean; for the strength view a colour bar with its range and
//! units), drawn with an 8×8 bitmap font.

use font8x8::UnicodeFonts;
use frac_core::input::InputScene;
use frac_core::{Asset, InterfaceKind};
use glam::DVec3;
use rayon::prelude::*;
use std::path::Path;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BondColour {
    /// Joint type (cold joint, bearing, mortar, adhesive, monolithic, anchor…).
    Kind,
    /// Tensile capacity (interface/material strength × Weibull scale), log heat map.
    Strength,
    /// Where breaks are favoured: bonds on coarser boundaries (L1 cuts, L0
    /// joints, anchors) by class, and their capacity relative to the bonds
    /// inside the coarser fragments.
    Weakening,
}

pub struct PreviewOptions {
    /// Draw the cosmetic debris pieces instead of fragments.
    pub debris: bool,
    /// Draw the level's bonds (their contact surfaces) instead of fragments.
    pub bonds: Option<BondColour>,
    /// Cutaway: drop geometry whose centroid has z above this value.
    pub clip_z: Option<f64>,
    pub level: Option<u8>,
    pub all_levels: bool,
    /// With `all_levels`: draw levels 0..=max_level only.
    pub max_level: Option<u8>,
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

fn kind_label(k: InterfaceKind) -> &'static str {
    use frac_core::InterfaceKind as K;
    match k {
        K::ColdJoint => "cold joint",
        K::Bearing => "bearing",
        K::MortarJoint => "mortar joint",
        K::Adhesive => "adhesive",
        K::Weld => "weld",
        K::Bolted => "bolted",
        K::GrainBoundary => "grain boundary",
        K::Anchor => "anchor (to ground)",
        K::Monolithic => "monolithic (same material)",
        K::ComponentConnection => "component connection",
    }
}

/// Perceptual-ish heat map (dark blue → teal → yellow) for t ∈ [0, 1].
fn heat(t: f64) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    let stops = [
        [0.15, 0.10, 0.45],
        [0.10, 0.45, 0.60],
        [0.25, 0.70, 0.40],
        [0.95, 0.85, 0.15],
    ];
    let x = t * 3.0;
    let i = (x.floor() as usize).min(2);
    let f = x - i as f64;
    let (a, b) = (stops[i], stops[i + 1]);
    [
        (a[0] + (b[0] - a[0]) * f) as f32,
        (a[1] + (b[1] - a[1]) * f) as f32,
        (a[2] + (b[2] - a[2]) * f) as f32,
    ]
}

/// Bonds of a level (after the cutaway) with their tensile capacity (Pa):
/// the interface material's tensile strength, else the weaker side's
/// material strength, times the bond's Weibull strength scale.
fn level_bonds<'a>(
    asset: &'a Asset,
    lib: &frac_material::MaterialLibrary,
    level: u8,
    clip_z: Option<f64>,
) -> Vec<(&'a frac_core::Bond, f64)> {
    let cap = |b: &frac_core::Bond| -> f64 {
        let comp = b.composition.first();
        let mat_t = |f: frac_core::FragmentId| {
            let m = asset
                .fragment(f)
                .material_mix
                .first()
                .map(|x| x.0)
                .unwrap_or(frac_core::MaterialId(0));
            lib.material(m).tensile_strength.unwrap_or(1e6)
        };
        let side = match b.b {
            frac_core::FragmentOrWorld::Fragment(f) => mat_t(b.a).min(mat_t(f)),
            frac_core::FragmentOrWorld::World => mat_t(b.a),
        };
        let ft = comp
            .and_then(|c| c.interface_material)
            .and_then(|m| lib.interface_material(m))
            .and_then(|im| im.tensile_strength)
            .unwrap_or(side);
        ft * b.strength_scale as f64
    };
    asset
        .level_bonds(level)
        .filter(|b| clip_z.is_none_or(|z| b.centroid.z <= z))
        .map(|b| (b, cap(b).max(0.0)))
        .collect()
}

/// Bonds below this capacity (Pa) carry no tension (e.g. bearing joints):
/// drawn grey, outside the colour scale.
const NO_TENSION: f64 = 1.0;
const NO_TENSION_RGB: [f32; 3] = [0.55, 0.55, 0.58];

fn bond_kind(b: &frac_core::Bond) -> InterfaceKind {
    if b.anchor {
        InterfaceKind::Anchor
    } else {
        b.composition
            .first()
            .map(|c| c.kind)
            .unwrap_or(InterfaceKind::Monolithic)
    }
}

/// Position of a capacity on the log colour scale [lo, hi].
fn log_t(c: f64, lo: f64, hi: f64) -> f64 {
    if hi > lo * (1.0 + 1e-9) {
        (c.ln() - lo.ln()) / (hi.ln() - lo.ln())
    } else {
        0.5
    }
}

/// Bond contact surfaces, coloured by kind or by tensile capacity on the
/// log scale [lo, hi] (Pa; shared by all panels of one image).
fn bond_tris(
    bonds: &[(&frac_core::Bond, f64)],
    asset: &Asset,
    mode: BondColour,
    (lo, hi): (f64, f64),
    light: DVec3,
) -> Vec<Tri> {
    let mut tris = Vec::new();
    for (k, &(b, c)) in bonds.iter().enumerate() {
        let rgb = match mode {
            BondColour::Kind => kind_colour(bond_kind(b)),
            BondColour::Strength if b.anchor => [0.05, 0.05, 0.05],
            BondColour::Strength if c < NO_TENSION => NO_TENSION_RGB,
            BondColour::Strength | BondColour::Weakening => heat(log_t(c, lo, hi)),
        };
        let shade = (0.55 + 0.45 * b.normal.dot(light).abs()) as f32;
        for &i in &b.interfaces {
            for poly in &asset.interfaces[i.idx()].polygons {
                let Some(outer) = poly.loops.first() else {
                    continue;
                };
                for t in 1..outer.len().saturating_sub(1) {
                    tris.push(Tri {
                        p: [outer[0], outer[t], outer[t + 1]],
                        frag: k as u32,
                        shade,
                        rgb: Some(rgb),
                    });
                }
            }
        }
    }
    tris
}

fn level_tris(
    asset: &Asset,
    level: u8,
    explode: f64,
    clip_z: Option<f64>,
    light: DVec3,
) -> Vec<Tri> {
    let frs: Vec<&frac_core::Fragment> = asset
        .level_fragments(level)
        .iter()
        .filter(|f| clip_z.is_none_or(|z| f.mass.com.z <= z))
        .collect();
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
                    Some(Tri {
                        p,
                        frag: id,
                        shade: (0.35 + 0.65 * lam) as f32,
                        rgb: None,
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Triangles of the cosmetic debris pieces (fan-triangulated hull faces),
/// exploded from the volume-weighted centre of the pieces.
fn debris_tris(asset: &Asset, explode: f64, clip_z: Option<f64>, light: DVec3) -> Vec<Tri> {
    let cent = |h: &frac_core::Hull| {
        h.vertices.iter().copied().sum::<DVec3>() / h.vertices.len().max(1) as f64
    };
    let centre = asset.debris.iter().map(cent).sum::<DVec3>() / asset.debris.len().max(1) as f64;
    let mut tris = Vec::new();
    for (k, h) in asset.debris.iter().enumerate() {
        let c = cent(h);
        if clip_z.is_some_and(|z| c.z > z) {
            continue;
        }
        let off = (c - centre) * explode;
        for f in &h.faces {
            for t in 1..f.len().saturating_sub(1) {
                let p = [f[0], f[t], f[t + 1]].map(|i| h.vertices[i as usize] + off);
                let n = (p[1] - p[0]).cross(p[2] - p[0]);
                if n.length_squared() == 0.0 {
                    continue;
                }
                let lam = n.normalize().dot(light).abs();
                tris.push(Tri {
                    p,
                    frag: k as u32,
                    shade: (0.35 + 0.65 * lam) as f32,
                    rgb: None,
                });
            }
        }
    }
    tris
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
    let to_px = |q: DVec3| {
        DVec3::new(
            0.5 * sw as f64 + (q.x - cx) * scale,
            0.5 * sh as f64 - (q.y - cy) * scale,
            q.z,
        )
    };
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
            out[(y * w + x) as usize] =
                acc.map(|v| ((v / d).clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    out
}

/// RGB image with simple drawing primitives and bitmap text.
struct Canvas {
    w: u32,
    h: u32,
    px: Vec<[u8; 3]>,
}

const INK: [u8; 3] = [30, 30, 36];
const DIM: [u8; 3] = [95, 95, 105];

fn rgb8(c: [f32; 3]) -> [u8; 3] {
    c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
}

impl Canvas {
    fn new(w: u32, h: u32) -> Canvas {
        Canvas {
            w,
            h,
            px: vec![[247, 247, 250]; (w * h) as usize],
        }
    }
    fn rect(&mut self, x: i64, y: i64, w: i64, h: i64, c: [u8; 3]) {
        for yy in y.max(0)..(y + h).min(self.h as i64) {
            for xx in x.max(0)..(x + w).min(self.w as i64) {
                self.px[(yy as u32 * self.w + xx as u32) as usize] = c;
            }
        }
    }
    /// Text with its top-left corner at (x, y); glyphs are 8×8 × `scale`.
    /// Returns the width drawn.
    fn text(&mut self, x: i64, y: i64, scale: i64, s: &str, c: [u8; 3]) -> i64 {
        let mut cx = x;
        for ch in s.chars() {
            if let Some(g) = font8x8::BASIC_FONTS.get(ch) {
                for (row, bits) in g.iter().enumerate() {
                    for col in 0..8 {
                        if bits >> col & 1 == 1 {
                            self.rect(cx + col * scale, y + row as i64 * scale, scale, scale, c);
                        }
                    }
                }
            }
            cx += 8 * scale;
        }
        cx - x
    }
    fn blit(&mut self, src: &[[u8; 3]], x0: u32, y0: u32, w: u32, h: u32) {
        for y in 0..h {
            for x in 0..w {
                self.px[((y0 + y) * self.w + x0 + x) as usize] = src[(y * w + x) as usize];
            }
        }
    }
    fn save(&self, out: &Path) -> Result<(), String> {
        let f = std::fs::File::create(out).map_err(|e| format!("{}: {e}", out.display()))?;
        let mut enc = png::Encoder::new(std::io::BufWriter::new(f), self.w, self.h);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut wr = enc.write_header().map_err(|e| e.to_string())?;
        wr.write_image_data(&self.px.concat())
            .map_err(|e| e.to_string())
    }
}

fn text_w(s: &str, scale: i64) -> i64 {
    s.chars().count() as i64 * 8 * scale
}

/// Greedy word wrap of `s` to lines at most `max_w` pixels wide.
fn wrap(s: &str, scale: i64, max_w: i64) -> Vec<String> {
    let max_chars = (max_w / (8 * scale)).max(8) as usize;
    let mut lines = Vec::new();
    let mut cur = String::new();
    for word in s.split_whitespace() {
        if !cur.is_empty() && cur.chars().count() + 1 + word.chars().count() > max_chars {
            lines.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}

const WEAKENING_NOTES: [(&str, i64); 3] = [
    (
        "Left: bonds on coarser boundaries, by class (bonds inside L1 fragments are hidden).",
        2,
    ),
    (
        "Right: the same bonds coloured by tensile capacity / median capacity of interior bonds.",
        2,
    ),
    (
        "Weibull weakest-link size effect: a bond on a coarser boundary takes the strength of that whole surface, (A_surface / A_ref)^(-1/m); joints also carry their joint material's strength.",
        1,
    ),
];

/// What the colours of an image mean.
enum Legend {
    /// One arbitrary colour per fragment.
    Fragments,
    /// One arbitrary colour per input part.
    Parts,
    /// One arbitrary colour per cosmetic debris piece.
    Debris,
    /// Joint kinds present, with bond counts.
    Kind(Vec<(InterfaceKind, usize)>),
    /// Bond classes with (count, median capacity Pa, median ratio to the
    /// interior median), and the ratio colour bar range.
    Weakening {
        rows: Vec<(String, [f32; 3], usize, f64, f64)>,
        lo: f64,
    },
    /// Log colour bar over [lo, hi] Pa, and the numbers of anchor bonds and
    /// of bonds without tensile capacity.
    Strength {
        lo: f64,
        hi: f64,
        anchors: usize,
        no_tension: usize,
    },
}

impl Legend {
    fn height(&self, w: u32) -> u32 {
        match self {
            Legend::Fragments | Legend::Parts | Legend::Debris => 44,
            Legend::Kind(k) => 52 + 30 * k.len().div_ceil(3) as u32,
            Legend::Weakening { rows, .. } => {
                let notes: i64 = WEAKENING_NOTES
                    .iter()
                    .map(|&(t, sc)| wrap(t, sc, w as i64 - 48).len() as i64 * (10 * sc + 2))
                    .sum();
                (notes + 24 + 28 * (rows.len() as i64 + 1) + 70) as u32
            }
            Legend::Strength { .. } => 130,
        }
    }
}

/// Capacity in MPa with 2–3 significant digits.
fn fmt_mpa(pa: f64) -> String {
    let v = pa / 1e6;
    if v >= 100.0 {
        format!("{v:.0}")
    } else if v >= 10.0 {
        format!("{v:.1}")
    } else if v >= 1.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.3}")
    }
}

fn draw_legend(c: &mut Canvas, y0: i64, lg: &Legend) {
    let x0 = 24i64;
    match lg {
        Legend::Fragments => {
            c.text(
                x0,
                y0 + 14,
                2,
                "Colour = fragment identity (arbitrary, no scale)",
                INK,
            );
        }
        Legend::Debris => {
            c.text(
                x0,
                y0 + 14,
                2,
                "Cosmetic debris (no bonds). Colour = piece (arbitrary, no scale)",
                INK,
            );
        }
        Legend::Parts => {
            c.text(
                x0,
                y0 + 14,
                2,
                "Input mesh as imported. Colour = part (arbitrary, no scale)",
                INK,
            );
        }
        Legend::Weakening { rows, lo } => {
            let max_w = c.w as i64 - 2 * x0;
            let mut y = y0 + 10;
            for &(t, sc) in &WEAKENING_NOTES {
                for line in wrap(t, sc, max_w) {
                    c.text(x0, y, sc, &line, if sc == 2 { INK } else { DIM });
                    y += 10 * sc + 2;
                }
            }
            y += 12;
            // table columns sized to their content
            let name_w = rows
                .iter()
                .map(|r| text_w(&r.0, 2))
                .max()
                .unwrap_or(0)
                .max(text_w("class", 2));
            let cols = [
                x0 + 34,
                x0 + 34 + name_w + 32,
                x0 + 34 + name_w + 32 + 140,
                x0 + 34 + name_w + 32 + 140 + 200,
            ];
            c.text(cols[0], y, 2, "class", DIM);
            c.text(cols[1], y, 2, "bonds", DIM);
            c.text(cols[2], y, 2, "median MPa", DIM);
            c.text(cols[3], y, 2, "x interior", DIM);
            y += 26;
            for (name, col, n, cap, ratio) in rows {
                c.rect(x0, y, 22, 20, rgb8(*col));
                c.text(cols[0], y + 2, 2, name, INK);
                c.text(cols[1], y + 2, 2, &format!("{n}"), INK);
                c.text(cols[2], y + 2, 2, &fmt_mpa(*cap), INK);
                c.text(cols[3], y + 2, 2, &format!("{ratio:.2}"), INK);
                y += 28;
            }
            // ratio colour bar (log), lo .. 1
            let bw = (max_w - 220).clamp(200, 900);
            let (bx, by) = (x0 + 30, y + 6);
            for i in 0..bw {
                c.rect(bx + i, by, 1, 18, rgb8(heat(i as f64 / (bw - 1) as f64)));
            }
            for v in [*lo, 0.2, 0.3, 0.5, 0.7, 1.0] {
                if v < *lo {
                    continue;
                }
                let x = bx + (log_t(v, *lo, 1.0) * (bw - 1) as f64).round() as i64;
                c.rect(x, by + 18, 2, 6, INK);
                let label = format!("{v:.1}");
                c.text(x - text_w(&label, 1) / 2, by + 28, 1, &label, INK);
            }
            c.text(bx + bw + 12, by + 4, 1, "capacity ratio (log)", DIM);
        }
        Legend::Kind(kinds) => {
            c.text(
                x0,
                y0 + 12,
                2,
                "Bond colour = joint kind (number of bonds)",
                INK,
            );
            let col_w = ((c.w as i64 - 2 * x0) / 3).max(200);
            for (i, &(k, n)) in kinds.iter().enumerate() {
                let x = x0 + (i % 3) as i64 * col_w;
                let y = y0 + 44 + (i / 3) as i64 * 30;
                c.rect(x, y, 22, 22, rgb8(kind_colour(k)));
                c.rect(x, y, 22, 1, DIM);
                c.text(x + 32, y + 3, 2, &format!("{} ({n})", kind_label(k)), INK);
            }
        }
        &Legend::Strength {
            lo,
            hi,
            anchors,
            no_tension,
        } => {
            c.text(
                x0,
                y0 + 10,
                2,
                "Bond tensile capacity, MPa (log colour scale)",
                INK,
            );
            c.text(
                x0,
                y0 + 32,
                1,
                "capacity = tensile strength of the joint (or of the weaker side's material) x the bond's Weibull strength scale",
                DIM,
            );
            let reserve = if anchors > 0 || no_tension > 0 {
                330
            } else {
                0
            };
            let bw = (c.w as i64 - 2 * x0 - reserve - 60).clamp(200, 1400);
            let (bx, by, bh) = (x0 + 30, y0 + 52, 26i64);
            for i in 0..bw {
                c.rect(bx + i, by, 1, bh, rgb8(heat(i as f64 / (bw - 1) as f64)));
            }
            // ticks: the range ends plus round values in between (denser
            // steps for narrow ranges; labels that would collide are dropped)
            let mut ticks = vec![lo];
            if hi > lo * (1.0 + 1e-9) {
                let steps: &[f64] = match hi / lo {
                    r if r < 3.0 => &[1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0],
                    r if r < 30.0 => &[1.0, 1.5, 2.0, 3.0, 5.0, 7.0],
                    _ => &[1.0, 2.0, 5.0],
                };
                let mut d = 10f64.powf((lo / 1e6).log10().floor()) * 1e6;
                while d <= hi {
                    for &m in steps {
                        let v = d * m;
                        if v > lo && v < hi {
                            ticks.push(v);
                        }
                    }
                    d *= 10.0;
                }
                ticks.push(hi);
            }
            let mut last_right = i64::MIN;
            let n = ticks.len();
            for (k, &v) in ticks.iter().enumerate() {
                let x = bx + (log_t(v, lo, hi) * (bw - 1) as f64).round() as i64;
                let label = match k {
                    0 => format!("min {}", fmt_mpa(v)),
                    _ if k == n - 1 => format!("max {}", fmt_mpa(v)),
                    _ => fmt_mpa(v),
                };
                let w = text_w(&label, 2);
                let lx = if k == 0 {
                    x
                } else if k == n - 1 {
                    x - w
                } else {
                    x - w / 2
                };
                // keep both ends; drop middle labels that would collide
                let is_end = k == 0 || k == n - 1;
                if !is_end
                    && (lx < last_right + 12 || lx + w > bx + bw - text_w("max 000.0", 2) - 12)
                {
                    continue;
                }
                c.rect(x, by + bh, 2, 8, INK);
                c.text(lx, by + bh + 12, 2, &label, INK);
                last_right = lx + w;
            }
            let ax = bx + bw + 40;
            let mut ay = by;
            if anchors > 0 {
                c.rect(ax, ay, 26, 20, [13, 13, 13]);
                c.text(ax + 36, ay + 2, 2, &format!("anchor ({anchors})"), INK);
                ay += 30;
            }
            if no_tension > 0 {
                c.rect(ax, ay, 26, 20, rgb8(NO_TENSION_RGB));
                c.text(
                    ax + 36,
                    ay + 2,
                    2,
                    &format!("no tension ({no_tension})"),
                    INK,
                );
                c.text(
                    ax + 36,
                    ay + 22,
                    1,
                    "compression-only joint, e.g. bearing",
                    DIM,
                );
            }
        }
    }
}

/// Compose panels (titled) above a legend and write the PNG.
fn compose(
    panels: &[Vec<[u8; 3]>],
    titles: &[String],
    o: &PreviewOptions,
    lg: &Legend,
    out: &Path,
) -> Result<(u32, u32), String> {
    let head = 40u32;
    let total_w = o.width * panels.len() as u32;
    let total_h = head + o.height + lg.height(total_w);
    let mut c = Canvas::new(total_w, total_h);
    for (k, (p, t)) in panels.iter().zip(titles).enumerate() {
        c.blit(p, k as u32 * o.width, head, o.width, o.height);
        // titles shrink to fit their panel
        let sc = if text_w(t, 2) <= o.width as i64 - 24 {
            2
        } else {
            1
        };
        c.text(
            k as i64 * o.width as i64 + 16,
            if sc == 2 { 12 } else { 16 },
            sc,
            t,
            INK,
        );
        if k > 0 {
            c.rect(
                k as i64 * o.width as i64,
                0,
                1,
                (head + o.height) as i64,
                [200, 200, 206],
            );
        }
    }
    c.rect(
        0,
        (head + o.height) as i64,
        total_w as i64,
        1,
        [200, 200, 206],
    );
    draw_legend(&mut c, (head + o.height) as i64, lg);
    c.save(out)?;
    Ok((total_w, total_h))
}

fn view_light(o: &PreviewOptions) -> (DVec3, DVec3) {
    let (az, el) = (o.azimuth_deg.to_radians(), o.elevation_deg.to_radians());
    // camera looks along -view; `view` points from the scene towards the camera
    let view = DVec3::new(az.sin() * el.cos(), el.sin(), az.cos() * el.cos()).normalize();
    let light = (view + DVec3::new(0.3, 0.6, 0.2)).normalize();
    (view, light)
}

pub fn preview(
    asset: &Asset,
    lib: &frac_material::MaterialLibrary,
    out: &Path,
    o: &PreviewOptions,
) -> Result<String, String> {
    let (view, light) = view_light(o);
    let leaf = asset.hierarchy.levels.saturating_sub(1);
    let levels: Vec<u8> = if o.all_levels {
        let top = o.max_level.map_or(leaf, |m| m.min(leaf));
        (0..=top).collect()
    } else {
        vec![o.level.unwrap_or(leaf).min(leaf)]
    };
    let mut titles = Vec::new();
    if o.debris {
        let tris = debris_tris(asset, o.explode, o.clip_z, light);
        let chunks: std::collections::BTreeSet<u32> =
            asset.debris.iter().map(|h| h.fragment.0).collect();
        titles.push(format!(
            "debris: {} pieces from {} chunks",
            asset.debris.len(),
            chunks.len()
        ));
        let panel = render_panel(&tris, o.width, o.height, view, DVec3::Y);
        let (w, h) = compose(&[panel], &titles, o, &Legend::Debris, out)?;
        return Ok(format!("wrote {} ({w}x{h}; {})", out.display(), titles[0]));
    }
    if o.bonds == Some(BondColour::Weakening) {
        let l = levels[levels.len() - 1];
        let bonds = level_bonds(asset, lib, l, o.clip_z);
        // class of a bond: the level of the top of its parent chain
        let class_of = |b: &frac_core::Bond| -> (u8, &'static str, [f32; 3]) {
            if b.anchor {
                return (0, "anchor (to ground)", [0.05, 0.05, 0.05]);
            }
            let mut top = b;
            while let Some(p) = top.parent_bond {
                top = &asset.bonds[p.idx()];
            }
            match top.level {
                lv if lv == b.level => (3, "interior", [0.78, 0.78, 0.80]),
                0 => (1, "L0 joint (between parts)", [0.85, 0.15, 0.15]),
                _ => (2, "L1 boundary", [0.98, 0.55, 0.05]),
            }
        };
        let interior: Vec<f64> = bonds
            .iter()
            .filter(|(b, _)| class_of(b).0 == 3)
            .map(|&(_, c)| c)
            .collect();
        let median = |mut v: Vec<f64>| -> f64 {
            if v.is_empty() {
                return f64::NAN;
            }
            v.sort_by(|x, y| x.total_cmp(y));
            v[v.len() / 2]
        };
        let ref_cap = median(interior).max(1.0);
        let lo = 0.1;
        let shown: Vec<&(&frac_core::Bond, f64)> =
            bonds.iter().filter(|(b, _)| class_of(b).0 != 3).collect();
        let mk = |colour: &dyn Fn(&frac_core::Bond, f64) -> [f32; 3]| -> Vec<Tri> {
            let mut tris = Vec::new();
            for (k, &&(b, cap)) in shown.iter().enumerate() {
                let rgb = colour(b, cap);
                let shade = (0.55 + 0.45 * b.normal.dot(light).abs()) as f32;
                for &i in &b.interfaces {
                    for poly in &asset.interfaces[i.idx()].polygons {
                        let Some(outer) = poly.loops.first() else {
                            continue;
                        };
                        for t in 1..outer.len().saturating_sub(1) {
                            tris.push(Tri {
                                p: [outer[0], outer[t], outer[t + 1]],
                                frag: k as u32,
                                shade,
                                rgb: Some(rgb),
                            });
                        }
                    }
                }
            }
            tris
        };
        let left = mk(&|b, _| class_of(b).2);
        let right = mk(&|b, cap| {
            if b.anchor {
                [0.05, 0.05, 0.05]
            } else {
                heat(log_t((cap / ref_cap).clamp(lo, 1.0), lo, 1.0))
            }
        });
        let panels = vec![
            render_panel(&left, o.width, o.height, view, DVec3::Y),
            render_panel(&right, o.width, o.height, view, DVec3::Y),
        ];
        titles.push(format!(
            "L{l} bonds on coarser boundaries ({} of {})",
            shown.len(),
            bonds.len()
        ));
        titles.push("capacity / interior median".to_string());
        let mut rows = Vec::new();
        for cls in [3u8, 2, 1, 0] {
            let sel: Vec<f64> = bonds
                .iter()
                .filter(|(b, _)| class_of(b).0 == cls)
                .map(|&(_, c)| c)
                .collect();
            if sel.is_empty() {
                continue;
            }
            let (_, name, col) =
                class_of(bonds.iter().find(|(b, _)| class_of(b).0 == cls).unwrap().0);
            let n = sel.len();
            let m = median(sel);
            rows.push((name.to_string(), col, n, m, m / ref_cap));
        }
        let legend = Legend::Weakening { rows, lo };
        let (w, h) = compose(&panels, &titles, o, &legend, out)?;
        return Ok(format!(
            "wrote {} ({w}x{h}; {})",
            out.display(),
            titles.join(", ")
        ));
    }
    let (panels, legend): (Vec<Vec<[u8; 3]>>, Legend) = match o.bonds {
        Some(mode) => {
            let per: Vec<Vec<(&frac_core::Bond, f64)>> = levels
                .iter()
                .map(|&l| level_bonds(asset, lib, l, o.clip_z))
                .collect();
            // one colour scale for all panels (non-anchor bonds)
            let (lo, hi) = per
                .iter()
                .flatten()
                .filter(|(b, c)| !b.anchor && *c >= NO_TENSION)
                .fold((f64::INFINITY, 0.0f64), |(l, h), &(_, c)| {
                    (l.min(c), h.max(c))
                });
            let (lo, hi) = if lo.is_finite() { (lo, hi) } else { (1.0, 1.0) };
            let mut kinds: std::collections::BTreeMap<u8, (InterfaceKind, usize)> =
                Default::default();
            let (mut anchors, mut no_tension) = (0, 0);
            for &(b, c) in per.iter().flatten() {
                let k = bond_kind(b);
                kinds.entry(k as u8).or_insert((k, 0)).1 += 1;
                anchors += b.anchor as usize;
                no_tension += (!b.anchor && c < NO_TENSION) as usize;
            }
            let panels = levels
                .iter()
                .zip(&per)
                .map(|(&l, bs)| {
                    titles.push(format!("L{l}: {} bonds", bs.len()));
                    render_panel(
                        &bond_tris(bs, asset, mode, (lo, hi), light),
                        o.width,
                        o.height,
                        view,
                        DVec3::Y,
                    )
                })
                .collect();
            let legend = match mode {
                BondColour::Kind => Legend::Kind(kinds.into_values().collect()),
                BondColour::Strength | BondColour::Weakening => Legend::Strength {
                    lo,
                    hi,
                    anchors,
                    no_tension,
                },
            };
            (panels, legend)
        }
        None => {
            let panels = levels
                .iter()
                .map(|&l| {
                    titles.push(format!(
                        "L{l}: {} fragments",
                        asset.level_fragments(l).len()
                    ));
                    render_panel(
                        &level_tris(asset, l, o.explode, o.clip_z, light),
                        o.width,
                        o.height,
                        view,
                        DVec3::Y,
                    )
                })
                .collect();
            (panels, Legend::Fragments)
        }
    };
    let (w, h) = compose(&panels, &titles, o, &legend, out)?;
    let detail = match &legend {
        Legend::Strength { lo, hi, .. } => format!(
            "\ncolour = bond tensile capacity, log scale {} MPa (dark blue) to {} MPa (yellow); anchors black, bonds without tensile capacity grey",
            fmt_mpa(*lo),
            fmt_mpa(*hi)
        ),
        _ => String::new(),
    };
    Ok(format!(
        "wrote {} ({w}x{h}; {}){detail}",
        out.display(),
        titles.join(", ")
    ))
}

/// Preview of an input scene before baking: one colour per part.
pub fn preview_input(scene: &InputScene, out: &Path, o: &PreviewOptions) -> Result<String, String> {
    let (view, light) = view_light(o);
    let centre_of = |m: &frac_geom::TriMesh| {
        let n = m.verts.len().max(1) as f64;
        m.verts.iter().copied().sum::<DVec3>() / n
    };
    let all: Vec<DVec3> = scene.parts.iter().map(|p| centre_of(&p.mesh)).collect();
    let centre = all.iter().copied().sum::<DVec3>() / all.len().max(1) as f64;
    let mut tris = Vec::new();
    let mut ntri = 0usize;
    for (pi, part) in scene.parts.iter().enumerate() {
        let off = (all[pi] - centre) * o.explode;
        for t in &part.mesh.tris {
            let p = t.map(|i| part.mesh.verts[i as usize] + off);
            if o.clip_z
                .is_some_and(|z| (p[0].z + p[1].z + p[2].z) / 3.0 > z)
            {
                continue;
            }
            let n = (p[1] - p[0]).cross(p[2] - p[0]);
            if n.length_squared() == 0.0 {
                continue;
            }
            ntri += 1;
            // two-sided: input soups may be inconsistently oriented
            let lam = n.normalize().dot(light).abs();
            tris.push(Tri {
                p,
                frag: pi as u32,
                shade: (0.35 + 0.65 * lam) as f32,
                rgb: None,
            });
        }
    }
    let title = format!("input: {} parts, {ntri} triangles", scene.parts.len());
    let panel = render_panel(&tris, o.width, o.height, view, DVec3::Y);
    let (w, h) = compose(
        &[panel],
        std::slice::from_ref(&title),
        o,
        &Legend::Parts,
        out,
    )?;
    Ok(format!("wrote {} ({w}x{h}; {title})", out.display()))
}
