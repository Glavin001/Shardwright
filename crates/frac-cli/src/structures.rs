//! Procedural structures whose failure modes come from how they are built
//! (ported from the vibe-land town kit and calibration scenarios): a
//! voussoir arch on Heyman's limit, a timber Pratt truss footbridge, a
//! brick-veneer timber-frame house, a multi-span RC viaduct and a
//! fracturable tree.
//!
//! Every joint is a pair of coplanar, opposed faces (that is how contacts
//! become bonds), so sloped members end in horizontal seats or plumb cuts
//! instead of meeting at arbitrary angles.

use crate::buildings::BuildingPart;
use crate::meshgen::{box_at, extrude, reorient};
use frac_core::input::{ConnectionSpec, PartMeta, RebarSpec};
use frac_geom::{DVec3, TriMesh};
use std::f64::consts::PI;

fn bpart(name: impl Into<String>, mesh: TriMesh, material: &str, role: &str) -> BuildingPart {
    BuildingPart {
        name: name.into(),
        mesh,
        material: material.into(),
        meta: PartMeta {
            material: Some(material.into()),
            component_role: Some(role.into()),
            ..Default::default()
        },
    }
}

fn anchored(mut p: BuildingPart) -> BuildingPart {
    p.meta.anchor_below = Some(p.mesh.aabb().min.y);
    p
}

fn grain(mut p: BuildingPart, d: DVec3) -> BuildingPart {
    p.meta.grain = Some(d.normalize().to_array());
    p
}

fn conn(a: &str, b: &str, kind: &str, imat: Option<&str>) -> ConnectionSpec {
    ConnectionSpec {
        a: a.into(),
        b: b.into(),
        kind: kind.into(),
        interface_material: imat.map(Into::into),
    }
}

/// Connection specs for every pair of parts whose bounding boxes touch or
/// overlap, typed by `rule` (None: leave the pair to the automatic contact
/// typing). Specs only take effect where the parts really share a face.
fn auto_conns(
    parts: &[BuildingPart],
    rule: impl Fn(&BuildingPart, &BuildingPart) -> Option<(&'static str, Option<&'static str>)>,
) -> Vec<ConnectionSpec> {
    let boxes: Vec<_> = parts.iter().map(|p| p.mesh.aabb()).collect();
    let mut out = Vec::new();
    for i in 0..parts.len() {
        for j in i + 1..parts.len() {
            let (a, b) = (&boxes[i], &boxes[j]);
            let eps = 1e-6;
            if (0..3).any(|k| a.min[k] > b.max[k] + eps || b.min[k] > a.max[k] + eps) {
                continue;
            }
            if let Some((kind, imat)) = rule(&parts[i], &parts[j]) {
                out.push(conn(&parts[i].name, &parts[j].name, kind, imat));
            }
        }
    }
    out
}

fn signed_area(poly: &[[f64; 2]]) -> f64 {
    (0..poly.len())
        .map(|i| {
            let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
            p[0] * q[1] - q[0] * p[1]
        })
        .sum::<f64>()
        * 0.5
}

fn ccw(poly: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut p = poly.to_vec();
    if signed_area(&p) < 0.0 {
        p.reverse();
    }
    p
}

/// Simple polygon in (x, y) extruded along z.
fn prism(poly: &[[f64; 2]], z0: f64, z1: f64) -> TriMesh {
    extrude(&[ccw(poly)], z0, z1)
}

/// Simple polygon in (z, y) extruded along x.
fn prism_zy(poly: &[[f64; 2]], x0: f64, x1: f64) -> TriMesh {
    reorient(&prism(poly, x0, x1), |p| DVec3::new(p.z, p.y, p.x), true)
}

/// Regular polygon in the plane through `c` with normal `n`; `u` fixes the
/// phase (it is projected into the plane).
fn ring(c: DVec3, n: DVec3, r: f64, u: DVec3, sides: usize) -> Vec<DVec3> {
    let n = n.normalize();
    let u = (u - n * u.dot(n)).normalize();
    let v = n.cross(u);
    (0..sides)
        .map(|i| {
            let a = std::f64::consts::TAU * i as f64 / sides as f64;
            c + (u * a.cos() + v * a.sin()) * r
        })
        .collect()
}

/// Closed solid between two planar rings with matching vertex order, so
/// consecutive segments share their joint polygon exactly.
fn frustum(a: &[DVec3], b: &[DVec3]) -> TriMesh {
    let n = a.len() as u32;
    let verts: Vec<DVec3> = a.iter().chain(b).copied().collect();
    let mut tris = Vec::new();
    for i in 1..n - 1 {
        tris.push([0, i + 1, i]);
        tris.push([n, n + i, n + i + 1]);
    }
    for i in 0..n {
        let (a, b) = (i, (i + 1) % n);
        tris.push([a, b, b + n]);
        tris.push([a, b + n, a + n]);
    }
    let m = TriMesh { verts, tris };
    if m.signed_volume() < 0.0 {
        m.flipped()
    } else {
        m
    }
}

// ---------------------------------------------------------------------------
// 1. Voussoir arch

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(not(test), allow(dead_code))] // the semicircle is the failing control
pub enum ArchShape {
    Semicircle,
    /// Uniform catenary with the same span and rise: its own line of thrust
    /// under self weight.
    Catenary,
}

/// Voussoir arch on two piers, joined by lime mortar with no tensile
/// strength, so every joint is a potential hinge. Heyman's limit analysis
/// puts the minimum thickness of a semicircular arch at t/R ≈ 0.1075; the
/// elastic self-weight gate is stricter still and fails thin semicircles at
/// the crown. The catenary carries its weight along its centreline, so the
/// same voussoirs at the same thickness stand.
#[derive(Clone, Debug)]
pub struct MasonryArch {
    pub shape: ArchShape,
    /// Half span and rise (m).
    pub radius: f64,
    /// Ring thickness over radius.
    pub t_over_r: f64,
    pub voussoirs: usize,
    pub depth: f64,
    pub pier_height: f64,
}

impl Default for MasonryArch {
    fn default() -> Self {
        MasonryArch {
            shape: ArchShape::Catenary,
            radius: 2.0,
            t_over_r: 0.1,
            voussoirs: 21,
            depth: 0.6,
            pier_height: 1.0,
        }
    }
}

impl MasonryArch {
    /// Joint lines (intrados point, extrados point) from one springing to
    /// the other, springings horizontal at y = 0.
    fn joints(&self) -> Vec<[[f64; 2]; 2]> {
        let (r, n) = (self.radius, self.voussoirs);
        let t = self.t_over_r * r;
        match self.shape {
            ArchShape::Semicircle => (0..=n)
                .map(|k| {
                    let a = PI * k as f64 / n as f64;
                    let at = |rr: f64| [rr * a.cos(), rr * a.sin()];
                    [at(r - 0.5 * t), at(r + 0.5 * t)]
                })
                .collect(),
            ArchShape::Catenary => {
                // y = f + a - a cosh(x / a) through (±r, 0) with rise f = r
                let (mut lo, mut hi) = (0.01 * r, 100.0 * r);
                for _ in 0..200 {
                    let a = 0.5 * (lo + hi);
                    if a * ((r / a).cosh() - 1.0) > r {
                        lo = a;
                    } else {
                        hi = a;
                    }
                }
                let a = 0.5 * (lo + hi);
                let f = r;
                let off = |x: f64, side: f64| {
                    let (c, s) = ((x / a).cosh(), (x / a).sinh());
                    let y = f + a - a * c;
                    [x + side * 0.5 * t * s / c, y + side * 0.5 * t / c]
                };
                // offset point on the springing line y = 0 (bisection on x)
                let springing = |sign: f64, side: f64| {
                    let (mut lo, mut hi) = (0.5 * r, 2.0 * r);
                    for _ in 0..200 {
                        let m = 0.5 * (lo + hi);
                        if off(sign * m, side)[1] > 0.0 {
                            lo = m;
                        } else {
                            hi = m;
                        }
                    }
                    off(sign * 0.5 * (lo + hi), side)
                };
                let len = 2.0 * a * (r / a).sinh();
                (0..=n)
                    .map(|k| {
                        if k == 0 {
                            [springing(-1.0, -1.0), springing(-1.0, 1.0)]
                        } else if k == n {
                            [springing(1.0, -1.0), springing(1.0, 1.0)]
                        } else {
                            let s = -0.5 * len + len * k as f64 / n as f64;
                            let x = a * (s / a).asinh();
                            [off(x, -1.0), off(x, 1.0)]
                        }
                    })
                    .collect()
            }
        }
    }

    pub fn build(&self) -> (Vec<BuildingPart>, Vec<ConnectionSpec>) {
        let h = self.pier_height;
        let joints: Vec<[[f64; 2]; 2]> = self
            .joints()
            .into_iter()
            .map(|j| j.map(|p| [p[0], p[1] + h]))
            .collect();
        let n = self.voussoirs;
        let mut parts = Vec::new();
        let mut conns = Vec::new();
        let lime = Some("lime_mortar");
        for (name, j) in [("pier_a", joints[0]), ("pier_b", joints[n])] {
            let (x0, x1) = (j[0][0].min(j[1][0]) - 0.15, j[0][0].max(j[1][0]) + 0.15);
            parts.push(anchored(bpart(
                name,
                box_at(DVec3::new(x0, 0.0, 0.0), DVec3::new(x1, h, self.depth)),
                "sandstone",
                "column",
            )));
        }
        let name = |k: usize| format!("voussoir_{k:02}");
        for k in 0..n {
            let (p, q) = (joints[k], joints[k + 1]);
            parts.push(bpart(
                name(k),
                prism(&[p[0], p[1], q[1], q[0]], 0.0, self.depth),
                "sandstone",
                "generic",
            ));
            let prev = if k == 0 {
                "pier_a".to_string()
            } else {
                name(k - 1)
            };
            conns.push(conn(&prev, &name(k), "mortar_joint", lime));
        }
        conns.push(conn(&name(n - 1), "pier_b", "mortar_joint", lime));
        (parts, conns)
    }
}

// ---------------------------------------------------------------------------
// 2. Timber Pratt truss footbridge

/// Through-truss footbridge: two glulam Pratt trusses (chords split per
/// panel, verticals, diagonals in tension under gravity), floor beams hung
/// under the bottom chords, a plank deck nailed to the floor beams, and
/// concrete abutments the end floor beams are bolted down to.
#[derive(Clone, Debug)]
pub struct PrattTruss {
    pub panels: usize,
    pub panel: f64,
    pub height: f64,
    pub width: f64,
    /// Chord depth.
    pub chord: f64,
    /// Member width along the span (verticals) and truss thickness.
    pub member: f64,
    /// Horizontal width of the diagonals' end cuts.
    pub diagonal: f64,
}

impl Default for PrattTruss {
    fn default() -> Self {
        PrattTruss {
            panels: 4,
            panel: 2.0,
            height: 1.6,
            width: 1.8,
            chord: 0.2,
            member: 0.14,
            diagonal: 0.22,
        }
    }
}

impl PrattTruss {
    pub fn build(&self) -> (Vec<BuildingPart>, Vec<ConnectionSpec>) {
        let (n, p, w, hc) = (self.panels, self.panel, self.member, self.chord);
        let len = n as f64 * p;
        let (fb, abut) = (0.2, 0.8); // floor beam depth, abutment height
        let y0 = abut + fb; // underside of the bottom chords
        let (yb, yt) = (y0 + hc, y0 + self.height - hc);
        let mut parts = Vec::new();
        for (side, z0) in [("n", 0.0), ("s", self.width - w)] {
            let z1 = z0 + w;
            for i in 0..n {
                let xa = if i == 0 { -0.5 * w } else { i as f64 * p };
                let xb = if i == n - 1 {
                    len + 0.5 * w
                } else {
                    (i + 1) as f64 * p
                };
                for (chord, ylo) in [("bot", y0), ("top", yt)] {
                    parts.push(grain(
                        bpart(
                            format!("{side}_{chord}_{i}"),
                            box_at(DVec3::new(xa, ylo, z0), DVec3::new(xb, ylo + hc, z1)),
                            "pine",
                            "beam",
                        ),
                        DVec3::X,
                    ));
                }
                // Pratt: diagonals fall from the top chord towards midspan
                let d = self.diagonal;
                let (xl, xr) = (i as f64 * p + 0.5 * w, (i + 1) as f64 * p - 0.5 * w);
                let poly = if 2 * i < n {
                    [[xr - d, yb], [xr, yb], [xl + d, yt], [xl, yt]]
                } else {
                    [[xl, yb], [xl + d, yb], [xr, yt], [xr - d, yt]]
                };
                let dir = DVec3::new(poly[2][0] - poly[1][0], yt - yb, 0.0);
                parts.push(grain(
                    bpart(
                        format!("{side}_diag_{i}"),
                        prism(&poly, z0, z1),
                        "pine",
                        "generic",
                    ),
                    dir,
                ));
            }
            for i in 0..=n {
                let x = i as f64 * p;
                parts.push(grain(
                    bpart(
                        format!("{side}_vert_{i}"),
                        box_at(
                            DVec3::new(x - 0.5 * w, yb, z0),
                            DVec3::new(x + 0.5 * w, yt, z1),
                        ),
                        "pine",
                        "column",
                    ),
                    DVec3::Y,
                ));
            }
        }
        for i in 0..=n {
            let x = i as f64 * p;
            parts.push(grain(
                bpart(
                    format!("floor_beam_{i}"),
                    box_at(
                        DVec3::new(x - 0.5 * w, abut, 0.0),
                        DVec3::new(x + 0.5 * w, y0, self.width),
                    ),
                    "pine",
                    "beam",
                ),
                DVec3::Z,
            ));
        }
        // deck planks bear on the floor beams, clear of the chords
        let gap = 0.02;
        for i in 0..n {
            parts.push(grain(
                bpart(
                    format!("deck_{i}"),
                    box_at(
                        DVec3::new(i as f64 * p, y0, w + gap),
                        DVec3::new((i + 1) as f64 * p, y0 + 0.05, self.width - w - gap),
                    ),
                    "pine",
                    "slab",
                ),
                DVec3::X,
            ));
        }
        for (name, xa, xb) in [
            ("abutment_w", -0.8, 0.5 * w),
            ("abutment_e", len - 0.5 * w, len + 0.8),
        ] {
            parts.push(anchored(bpart(
                name,
                box_at(
                    DVec3::new(xa, 0.0, -0.1),
                    DVec3::new(xb, abut, self.width + 0.1),
                ),
                "concrete_c30",
                "column",
            )));
        }
        // end floor beams are anchor-bolted to the abutments: a free
        // bearing would read the trusses' end rotation as overturning
        let conns = auto_conns(&parts, |a, b| {
            (a.material == "pine" || b.material == "pine").then_some(("bolted", None))
        });
        (parts, conns)
    }
}

// ---------------------------------------------------------------------------
// 3. Brick-veneer timber-frame house

/// Rough opening in a wall, in the wall's own (u, v) coordinates; a door
/// has `sill == 0`.
#[derive(Clone, Copy, Debug)]
pub struct Opening {
    pub a: f64,
    pub b: f64,
    pub sill: f64,
    pub head: f64,
}

/// Single-storey house built the way real ones are: a pine stud frame
/// (plates, studs at 600 mm centres, king/jack studs, headers, sills,
/// cripples, ceiling joists, rafters on seat cuts, ridge board, roof
/// sheathing) on a concrete footing; a 90 mm brick skin carried by the
/// footing and held to the frame only by weak wall ties, with precast
/// concrete lintels over its openings; drywall lining glued inside; glazing
/// in the windows. Removing studs drops the roof while the skin and the
/// lining just peel.
#[derive(Clone, Debug)]
pub struct VeneerHouse {
    /// Footprint along x.
    pub length: f64,
    /// Footprint along z.
    pub depth: f64,
    pub wall_height: f64,
    pub pitch_deg: f64,
    pub stud_spacing: f64,
    /// Openings per wall: front (z = 0), back, west (x = 0), east; u runs
    /// along +x for front/back and +z for the side walls, in house
    /// coordinates.
    pub openings: [Vec<Opening>; 4],
}

impl Default for VeneerHouse {
    fn default() -> Self {
        let win = |a: f64, b: f64, sill: f64| Opening {
            a,
            b,
            sill,
            head: 2.1,
        };
        VeneerHouse {
            length: 6.0,
            depth: 4.0,
            wall_height: 2.4,
            pitch_deg: 30.0,
            stud_spacing: 0.6,
            openings: [
                vec![win(1.2, 2.1, 0.0), win(3.6, 4.8, 0.9)],
                vec![win(1.5, 2.7, 0.9), win(4.2, 5.0, 1.1)],
                vec![win(1.4, 2.6, 0.9)],
                vec![],
            ],
        }
    }
}

const STUD: f64 = 0.045; // stud thickness along the wall
const FRAME: f64 = 0.09; // stud depth through the wall
const PLATE: f64 = 0.045;
const BRICK: f64 = 0.09;
const LINING: f64 = 0.0125;
const HEADER: f64 = 0.15;
const LINING_GAP: f64 = 0.01;
const SET_BACK: f64 = 0.005; // horizontal frame members behind the cavity face
const LINTEL: [f64; 2] = [0.15, 0.15]; // bearing length each side, depth

/// Wall outline in (u, v): doors are notches from the bottom edge, windows
/// holes; with `lintel` the openings get a slot above the head for the
/// lintel to bear in.
fn wall_loops(u0: f64, u1: f64, h: f64, openings: &[Opening], lintel: bool) -> Vec<Vec<[f64; 2]>> {
    let (e, lt) = if lintel {
        (LINTEL[0], LINTEL[1])
    } else {
        (0.0, 0.0)
    };
    let head_shape = |o: &Opening| -> Vec<[f64; 2]> {
        // left to right across the top of the opening
        if lintel {
            vec![
                [o.a, o.head],
                [o.a - e, o.head],
                [o.a - e, o.head + lt],
                [o.b + e, o.head + lt],
                [o.b + e, o.head],
                [o.b, o.head],
            ]
        } else {
            vec![[o.a, o.head], [o.b, o.head]]
        }
    };
    let mut sorted = openings.to_vec();
    sorted.sort_by(|p, q| p.a.total_cmp(&q.a));
    let mut outer = vec![[u0, 0.0]];
    let mut holes = Vec::new();
    for o in &sorted {
        if o.sill <= 0.0 {
            outer.push([o.a, 0.0]);
            outer.extend(head_shape(o));
            outer.push([o.b, 0.0]);
        } else {
            let mut hole = vec![[o.a, o.sill], [o.b, o.sill]];
            hole.extend(head_shape(o).into_iter().rev());
            hole.reverse(); // CCW traversal reversed: holes are CW
            holes.push(hole);
        }
    }
    outer.extend([[u1, 0.0], [u1, h], [u0, h]]);
    let mut loops = vec![outer];
    loops.extend(holes);
    loops
}

/// Maps wall-local boxes and panels to house coordinates.
#[derive(Clone, Copy)]
struct WallFrame {
    /// Wall runs along x (front/back) or along z (sides).
    along_x: bool,
}

impl WallFrame {
    fn pt(&self, u: f64, v: f64, d: f64) -> DVec3 {
        if self.along_x {
            DVec3::new(u, v, d)
        } else {
            DVec3::new(d, v, u)
        }
    }

    fn cuboid(&self, u: [f64; 2], v: [f64; 2], d: [f64; 2]) -> TriMesh {
        let (a, b) = (self.pt(u[0], v[0], d[0]), self.pt(u[1], v[1], d[1]));
        box_at(a.min(b), a.max(b))
    }

    fn panel(&self, loops: &[Vec<[f64; 2]>], d: [f64; 2]) -> TriMesh {
        let m = extrude(loops, d[0], d[1]);
        if self.along_x {
            m
        } else {
            reorient(&m, |p| DVec3::new(p.z, p.y, p.x), true)
        }
    }

    fn u_axis(&self) -> DVec3 {
        if self.along_x { DVec3::X } else { DVec3::Z }
    }
}

impl VeneerHouse {
    /// Stud wall members as (name, role, u, v, grain along u).
    fn frame_members(
        &self,
        u0: f64,
        u1: f64,
        openings: &[Opening],
    ) -> Vec<(String, &'static str, [f64; 2], [f64; 2], bool)> {
        let s = STUD;
        let h = self.wall_height;
        let top = h - 2.0 * PLATE; // underside of the doubled top plate
        let mut m = Vec::new();
        // bottom plate, cut at doors
        let mut cuts: Vec<[f64; 2]> = openings
            .iter()
            .filter(|o| o.sill <= 0.0)
            .map(|o| [o.a, o.b])
            .collect();
        cuts.sort_by(|p, q| p[0].total_cmp(&q[0]));
        let mut start = u0;
        for (k, c) in cuts.iter().chain([[u1, u1]].iter()).enumerate() {
            m.push((
                format!("plate_bot_{k}"),
                "beam",
                [start, c[0]],
                [0.0, PLATE],
                true,
            ));
            start = c[1];
        }
        m.push((
            "plate_top_0".into(),
            "beam",
            [u0, u1],
            [top, top + PLATE],
            true,
        ));
        m.push((
            "plate_top_1".into(),
            "beam",
            [u0, u1],
            [top + PLATE, h],
            true,
        ));
        // common studs at regular centres, skipping the framed openings
        let clear = |u: f64| {
            openings
                .iter()
                .all(|o| u + s <= o.a - 2.0 * s + 1e-9 || u >= o.b + 2.0 * s - 1e-9)
        };
        let mut studs = Vec::new();
        let mut u = u0;
        while u + 2.0 * s <= u1 {
            studs.push(u);
            u += self.stud_spacing;
        }
        studs.push(u1 - s);
        for (k, &u) in studs.iter().enumerate() {
            if clear(u) {
                m.push((
                    format!("stud_{k}"),
                    "column",
                    [u, u + s],
                    [PLATE, top],
                    false,
                ));
            }
        }
        for (k, o) in openings.iter().enumerate() {
            let (a, b) = (o.a, o.b);
            m.push((
                format!("o{k}_king_a"),
                "column",
                [a - 2.0 * s, a - s],
                [PLATE, top],
                false,
            ));
            m.push((
                format!("o{k}_king_b"),
                "column",
                [b + s, b + 2.0 * s],
                [PLATE, top],
                false,
            ));
            m.push((
                format!("o{k}_jack_a"),
                "column",
                [a - s, a],
                [PLATE, o.head],
                false,
            ));
            m.push((
                format!("o{k}_jack_b"),
                "column",
                [b, b + s],
                [PLATE, o.head],
                false,
            ));
            m.push((
                format!("o{k}_header"),
                "beam",
                [a - s, b + s],
                [o.head, o.head + HEADER],
                true,
            ));
            // cripples on the stud grid between the jacks
            let mut grid: Vec<f64> = studs
                .iter()
                .copied()
                .filter(|&u| u >= a + 1e-9 && u + s <= b - 1e-9)
                .collect();
            if grid.is_empty() {
                grid.push(0.5 * (a + b - s));
            }
            if o.head + HEADER < top - 1e-9 {
                for (j, &u) in grid.iter().enumerate() {
                    m.push((
                        format!("o{k}_cripple_top_{j}"),
                        "column",
                        [u, u + s],
                        [o.head + HEADER, top],
                        false,
                    ));
                }
            }
            if o.sill > 0.0 {
                m.push((
                    format!("o{k}_sill"),
                    "beam",
                    [a, b],
                    [o.sill - PLATE, o.sill],
                    true,
                ));
                for (j, &u) in grid.iter().enumerate() {
                    m.push((
                        format!("o{k}_cripple_bot_{j}"),
                        "column",
                        [u, u + s],
                        [PLATE, o.sill - PLATE],
                        false,
                    ));
                }
            }
        }
        m
    }

    pub fn build(&self) -> (Vec<BuildingPart>, Vec<ConnectionSpec>) {
        let (lx, lz, h) = (self.length, self.depth, self.wall_height);
        let mut parts = Vec::new();
        parts.push(anchored(bpart(
            "footing",
            box_at(
                DVec3::new(-0.1, -0.3, -0.1),
                DVec3::new(lx + 0.1, 0.0, lz + 0.1),
            ),
            "concrete_c30",
            "slab",
        )));
        // roof geometry: joist depth, then the rafter line, which the gables
        // follow
        let jd = 0.19;
        let th = self.pitch_deg.to_radians();
        let (yj, z0, seat) = (h + jd, BRICK + SET_BACK, 0.15);
        let ridge = 0.045;
        let zr = 0.5 * lz - 0.5 * ridge;
        let hv = 0.14 / th.cos(); // vertical depth of a 140 mm rafter
        let h0 = hv - seat * th.tan();
        let y_bot = |z: f64| yj + (z - z0 - seat) * th.tan();
        let y_top = |z: f64| yj + h0 + (z - z0) * th.tan();
        // roof line over the side walls (the underside of the sheathing)
        let roof_line = |u: f64| y_top(u.min(lz - u).min(zr));
        // (name, along x, outward normal sign, wall plane offset)
        let walls = [
            ("front", true, 0.0, 1.0),
            ("back", true, lz, -1.0),
            ("west", false, 0.0, 1.0),
            ("east", false, lx, -1.0),
        ];
        for (wi, &(wname, along_x, base, inward)) in walls.iter().enumerate() {
            let wf = WallFrame { along_x };
            let span = if along_x { lx } else { lz };
            // the front/back walls run through the corners, the side walls
            // fit between them
            let inset = if along_x { 0.0 } else { BRICK };
            let layer = |d0: f64, d1: f64| {
                let (a, b) = (base + inward * d0, base + inward * d1);
                [a.min(b), a.max(b)]
            };
            let d_brick = layer(0.0, BRICK);
            let d_frame = layer(BRICK, BRICK + FRAME);
            let d_lining = layer(BRICK + FRAME, BRICK + FRAME + LINING);
            let ops = &self.openings[wi];
            // brick skin with lintel slots, and the precast lintels
            let (bu0, bu1) = (inset, span - inset);
            let mut loops = wall_loops(bu0, bu1, h, ops, true);
            if !along_x {
                // gable: the side skins rise to the roof line under the
                // sheathing (outer loop ends [.., (bu1, h), (bu0, h)])
                let outer = &mut loops[0];
                outer.truncate(outer.len() - 2);
                outer.extend([
                    [bu1, roof_line(bu1)],
                    [lz - zr, roof_line(zr)],
                    [zr, roof_line(zr)],
                    [bu0, roof_line(bu0)],
                ]);
            }
            let mut skin = bpart(
                format!("{wname}_brick"),
                wf.panel(&loops, d_brick),
                "brick_clay",
                "wall",
            );
            skin.meta.masonry_layout = Some(serde_json::json!({
                "brick": [0.215, 0.065, 0.1025],
                "bond": "stretcher",
                "mortar": 0.01,
                "half_split_fraction": 0.2,
                "chip_fraction": 0.05
            }));
            parts.push(skin);
            for (k, o) in ops.iter().enumerate() {
                parts.push(bpart(
                    format!("{wname}_lintel_{k}"),
                    wf.cuboid(
                        [o.a - LINTEL[0], o.b + LINTEL[0]],
                        [o.head, o.head + LINTEL[1]],
                        d_brick,
                    ),
                    "concrete_c30",
                    "wall",
                ));
                let mid = 0.5 * (d_frame[0] + d_frame[1]);
                if o.sill > 0.0 {
                    parts.push(bpart(
                        format!("{wname}_glass_{k}"),
                        wf.cuboid([o.a, o.b], [o.sill, o.head], [mid - 0.003, mid + 0.003]),
                        "glass_annealed",
                        "glazing",
                    ));
                } else {
                    // door leaf hung on the hinge-side jack, clear of the
                    // floor, head and latch jack
                    parts.push(grain(
                        bpart(
                            format!("{wname}_door_{k}"),
                            wf.cuboid(
                                [o.a, o.b - 0.01],
                                [0.01, o.head - 0.005],
                                [mid - 0.02, mid + 0.02],
                            ),
                            "pine",
                            "prop",
                        ),
                        DVec3::Y,
                    ));
                }
            }
            // stud frame
            let (fu0, fu1) = if along_x {
                (BRICK, span - BRICK)
            } else {
                (BRICK + FRAME, span - BRICK - FRAME)
            };
            // ties are fixed to the studs: horizontal members (plates,
            // headers, sills) sit back from the cavity face, so the skin
            // only touches vertical members
            let d_set_back = layer(BRICK + SET_BACK, BRICK + FRAME);
            for (name, role, u, v, along_u) in self.frame_members(fu0, fu1, ops) {
                let (g, d) = if along_u {
                    (wf.u_axis(), d_set_back)
                } else {
                    (DVec3::Y, d_frame)
                };
                parts.push(grain(
                    bpart(format!("{wname}_{name}"), wf.cuboid(u, v, d), "pine", role),
                    g,
                ));
            }
            // drywall lining (no lintel slots)
            let (lu0, lu1) = if along_x {
                (BRICK + FRAME, span - BRICK - FRAME)
            } else {
                (BRICK + FRAME + LINING, span - BRICK - FRAME - LINING)
            };
            // hung on the frame 10 mm clear of the footing, as drywall is
            let mut loops = wall_loops(lu0, lu1, h, ops, false);
            for p in loops.iter_mut().flatten() {
                if p[1] == 0.0 {
                    p[1] = LINING_GAP;
                }
            }
            parts.push(bpart(
                format!("{wname}_lining"),
                wf.panel(&loops, d_lining),
                "drywall",
                "cosmetic",
            ));
        }
        // ceiling joists across the short span, bearing on the front and
        // back top plates
        let mut xs = Vec::new();
        let mut x = 0.3;
        while x + STUD <= lx - 0.3 + 1e-9 {
            xs.push(x);
            x += self.stud_spacing;
        }
        for (k, &x) in xs.iter().enumerate() {
            parts.push(grain(
                bpart(
                    format!("joist_{k}"),
                    // ends flush with the set-back plate edge: ingest finds
                    // contacts from vertices, so a joist merely crossing
                    // the plate would not register as bearing on it
                    box_at(
                        DVec3::new(x, h, BRICK + SET_BACK),
                        DVec3::new(x + STUD, h + jd, lz - BRICK - SET_BACK),
                    ),
                    "pine",
                    "beam",
                ),
                DVec3::Z,
            ));
        }
        // rafters seated on the joists, plumb-cut against the ridge board
        let rafter = [
            [z0, yj],
            [z0 + seat, yj],
            [zr, y_bot(zr)],
            [zr, y_top(zr)],
            [z0, y_top(z0)],
        ];
        let mirror = |poly: &[[f64; 2]]| -> Vec<[f64; 2]> {
            poly.iter().map(|p| [lz - p[0], p[1]]).collect()
        };
        for (k, &x) in xs.iter().enumerate() {
            parts.push(grain(
                bpart(
                    format!("rafter_f_{k}"),
                    prism_zy(&rafter, x, x + STUD),
                    "pine",
                    "beam",
                ),
                DVec3::new(0.0, th.sin(), th.cos()),
            ));
            parts.push(grain(
                bpart(
                    format!("rafter_b_{k}"),
                    prism_zy(&mirror(&rafter), x, x + STUD),
                    "pine",
                    "beam",
                ),
                DVec3::new(0.0, th.sin(), -th.cos()),
            ));
        }
        let (xr0, xr1) = (xs[0], xs[xs.len() - 1] + STUD);
        parts.push(grain(
            bpart(
                "ridge",
                box_at(
                    DVec3::new(xr0, y_bot(zr) - 0.02, zr),
                    DVec3::new(xr1, y_top(zr), zr + ridge),
                ),
                "pine",
                "beam",
            ),
            DVec3::X,
        ));
        // roof sheathing on the rafter backs
        let tv = 0.018 / th.cos();
        // with eaves past the walls and verges past the gables
        let (eave, verge) = (-0.35, 0.3);
        let sheet = [
            [eave, y_top(eave)],
            [zr, y_top(zr)],
            [zr, y_top(zr) + tv],
            [eave, y_top(eave) + tv],
        ];
        for (name, poly) in [("roof_f", sheet.to_vec()), ("roof_b", mirror(&sheet))] {
            parts.push(grain(
                bpart(name, prism_zy(&poly, -verge, lx + verge), "pine", "slab"),
                DVec3::X,
            ));
        }
        // plasterboard ceiling screwed to the joists, sheets jointed on the
        // joist centre lines (a sheet edge on a joist is also what lets
        // ingest find the contact: see the joist ends above)
        let c0 = BRICK + FRAME + LINING;
        let mut edges = vec![c0];
        edges.extend(xs.iter().map(|x| x + 0.5 * STUD));
        edges.push(lx - c0);
        for (k, e) in edges.windows(2).enumerate() {
            parts.push(bpart(
                format!("ceiling_{k}"),
                box_at(
                    DVec3::new(e[0], h - LINING, c0),
                    DVec3::new(e[1], h, lz - c0),
                ),
                "drywall",
                "cosmetic",
            ));
        }
        let conns = auto_conns(&parts, |a, b| {
            let mut m = [a.material.as_str(), b.material.as_str()];
            m.sort();
            match m {
                ["pine", "pine"] => Some(("bolted", None)),
                ["brick_clay", "pine"] => Some(("component_connection", Some("wall_tie"))),
                ["drywall", "pine"] | ["drywall", "drywall"] => Some(("adhesive", None)),
                ["brick_clay", "brick_clay"] => Some(("mortar_joint", Some("mortar_type_n"))),
                // anchor-bolted sole plates; brick bedded on the footing
                // and on the lintels
                ["concrete_c30", "pine"] => Some(("bolted", None)),
                ["brick_clay", "concrete_c30"] => Some(("mortar_joint", Some("mortar_type_n"))),
                // glazing set in silicone
                ["glass_annealed", "pine"] => Some(("adhesive", None)),
                _ => None,
            }
        });
        (parts, conns)
    }
}

// ---------------------------------------------------------------------------
// 4. RC viaduct

/// Multi-span reinforced-concrete slab viaduct: deck spans cast onto
/// two-column bents with crossheads (cold joints; expansion gaps between
/// spans), on anchored pile caps, with end abutments and parapets. Free
/// bearings would let each span rotate on its support, which the
/// compression-only bearing check reads as overturning.
#[derive(Clone, Debug)]
pub struct RcViaduct {
    pub spans: usize,
    pub span: f64,
    pub width: f64,
    pub deck: f64,
    pub column_height: f64,
}

impl Default for RcViaduct {
    fn default() -> Self {
        RcViaduct {
            spans: 4,
            span: 6.0,
            width: 4.0,
            deck: 0.5,
            column_height: 4.4,
        }
    }
}

fn rebar(points: [[f64; 3]; 2], diameter: f64) -> RebarSpec {
    RebarSpec {
        points: points.to_vec(),
        diameter,
    }
}

impl RcViaduct {
    pub fn build(&self) -> (Vec<BuildingPart>, Vec<ConnectionSpec>) {
        let (n, l, b) = (self.spans, self.span, self.width);
        let (cap, xh, col) = (0.6, 0.6, 0.7);
        let y_col = cap;
        let y_xh = y_col + self.column_height;
        let y_deck = y_xh + xh;
        let y_top = y_deck + self.deck;
        let gap = 0.02;
        let mut parts = Vec::new();
        let concrete = "concrete_c30";
        for i in 1..n {
            let x = i as f64 * l;
            parts.push(anchored(bpart(
                format!("pile_cap_{i}"),
                box_at(
                    DVec3::new(x - 0.8, 0.0, -0.2),
                    DVec3::new(x + 0.8, cap, b + 0.2),
                ),
                concrete,
                "generic",
            )));
            for (k, zc) in [0.2 * b, 0.8 * b].into_iter().enumerate() {
                let (x0, z0) = (x - 0.5 * col, zc - 0.5 * col);
                let mut c = bpart(
                    format!("column_{i}_{k}"),
                    box_at(
                        DVec3::new(x0, y_col, z0),
                        DVec3::new(x0 + col, y_xh, z0 + col),
                    ),
                    concrete,
                    "column",
                );
                let c0 = 0.07;
                for (dx, dz) in [
                    (c0, c0),
                    (col - c0, c0),
                    (col - c0, col - c0),
                    (c0, col - c0),
                ] {
                    c.meta.rebar.push(rebar(
                        [
                            [x0 + dx, y_col + 0.05, z0 + dz],
                            [x0 + dx, y_xh - 0.05, z0 + dz],
                        ],
                        0.025,
                    ));
                }
                parts.push(c);
            }
            let mut c = bpart(
                format!("crosshead_{i}"),
                box_at(
                    DVec3::new(x - 0.5, y_xh, 0.0),
                    DVec3::new(x + 0.5, y_deck, b),
                ),
                concrete,
                "beam",
            );
            for (dx, y) in [
                (-0.4, y_xh + 0.06),
                (0.4, y_xh + 0.06),
                (-0.4, y_deck - 0.06),
                (0.4, y_deck - 0.06),
            ] {
                c.meta
                    .rebar
                    .push(rebar([[x + dx, y, 0.05], [x + dx, y, b - 0.05]], 0.025));
            }
            parts.push(c);
        }
        let end = n as f64 * l;
        for (name, xa, xb) in [
            ("abutment_w", -1.2, 0.5),
            ("abutment_e", end - 0.5, end + 1.2),
        ] {
            parts.push(anchored(bpart(
                name,
                box_at(DVec3::new(xa, 0.0, -0.2), DVec3::new(xb, y_deck, b + 0.2)),
                concrete,
                "column",
            )));
        }
        for i in 0..n {
            let (x0, x1) = (i as f64 * l + gap, (i + 1) as f64 * l - gap);
            let name = format!("deck_{i}");
            let mut d = bpart(
                &name,
                box_at(DVec3::new(x0, y_deck, 0.0), DVec3::new(x1, y_top, b)),
                concrete,
                "slab",
            );
            for k in 0..6 {
                let z = 0.3 + (b - 0.6) * k as f64 / 5.0;
                d.meta.rebar.push(rebar(
                    [[x0 + 0.05, y_deck + 0.05, z], [x1 - 0.05, y_deck + 0.05, z]],
                    0.025,
                ));
            }
            parts.push(d);
            for (side, z0) in [("n", 0.0), ("s", b - 0.25)] {
                parts.push(bpart(
                    format!("parapet_{side}_{i}"),
                    box_at(
                        DVec3::new(x0, y_top, z0),
                        DVec3::new(x1, y_top + 0.9, z0 + 0.25),
                    ),
                    concrete,
                    "wall",
                ));
            }
        }
        // the solid abutments and pile caps are most of the volume but fail
        // least: coarse cells there, finer in the deck, crossheads and
        // columns (the suite default would give over a million bonds)
        for p in &mut parts {
            let massive = p.name.starts_with("abutment") || p.name.starts_with("pile_cap");
            p.meta.analysis_cells_per_m3 = Some(if massive { 8.0 } else { 40.0 });
        }
        (parts, Vec::new())
    }
}

// ---------------------------------------------------------------------------
// 5. Trees

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TreeFamily {
    /// Short trunk, a forking leader and forking limbs from its upper
    /// segments.
    Broadleaf,
    /// Tall tapering trunk with whorls of short drooping branches that
    /// shorten towards the top.
    Conifer,
}

/// Fracturable tree. The trunk is a stack of tapered octagonal segments;
/// branches grow out of the trunk's flat side faces (the joint polygon lies
/// in the face), curve, and fork on their end faces, every segment pine
/// with the grain along it. Where a limb meets its parent the bond is a
/// branch union, well below clear wood, so a crown sheds at its forks and
/// a trunk cut drops everything above.
#[derive(Clone, Debug)]
pub struct Tree {
    pub family: TreeFamily,
    pub seed: u64,
}

impl Default for Tree {
    fn default() -> Self {
        Tree {
            family: TreeFamily::Broadleaf,
            seed: 7,
        }
    }
}

const SIDES: usize = 8;
const MIN_TWIG: f64 = 0.015;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in [-1, 1).
    fn signed(&mut self) -> f64 {
        2.0 * self.next() - 1.0
    }
}

/// How a branch grows from its base ring.
#[derive(Clone, Copy)]
struct Growth {
    length: f64,
    segments: usize,
    /// Upward bend per segment (negative droops).
    bend: f64,
    taper: f64,
    /// Fork generations left.
    forks: usize,
    /// Sideways splay of the two fork children.
    splay: f64,
}

struct TreeBuilder {
    parts: Vec<BuildingPart>,
    conns: Vec<ConnectionSpec>,
    rng: Lcg,
}

impl TreeBuilder {
    fn wood(&mut self, name: String, a: &[DVec3], b: &[DVec3], axis: DVec3) {
        self.parts
            .push(grain(bpart(name, frustum(a, b), "pine", "generic"), axis));
    }

    fn union(&mut self, parent: &str, child: &str) {
        self.conns
            .push(conn(parent, child, "grain_boundary", Some("branch_union")));
    }

    fn solid(&mut self, a: &str, b: &str) {
        self.conns.push(conn(a, b, "monolithic", None));
    }

    /// Grow a branch from `base` (a ring in its parent's face, centre `c`,
    /// face normal `n`, radius `r`) along `dir`, then fork.
    #[allow(clippy::too_many_arguments)]
    fn branch(
        &mut self,
        name: &str,
        parent: &str,
        base: Vec<DVec3>,
        mut c: DVec3,
        n: DVec3,
        mut r: f64,
        mut dir: DVec3,
        g: Growth,
    ) {
        let seg_len = g.length / g.segments as f64;
        let mut ring_a = base;
        let mut prev = parent.to_string();
        let mut plane = n;
        for s in 0..g.segments {
            let jitter = DVec3::new(self.rng.signed(), 0.0, self.rng.signed()) * 0.12;
            let next = (dir + DVec3::Y * g.bend + jitter).normalize();
            let c1 = c + dir * seg_len;
            let last = s + 1 == g.segments;
            // joint plane bisects the two segment axes; the tip is square
            let n1 = if last { dir } else { (dir + next).normalize() };
            // tips no finer than 15 mm: thinner slivers make degenerate cells
            let r1 = (r * g.taper).max(MIN_TWIG.min(r));
            let ring_b = ring(c1, n1, r1, ring_a[0] - c, SIDES);
            let part = format!("{name}_{s}");
            self.wood(part.clone(), &ring_a, &ring_b, dir);
            if s == 0 {
                self.union(&prev, &part);
            } else {
                self.solid(&prev, &part);
            }
            prev = part;
            ring_a = ring_b;
            c = c1;
            r = r1;
            plane = n1;
            if !last {
                dir = next;
            }
        }
        // fork on the tip face: two children side by side inside it
        let rc = 0.42 * r;
        if g.forks == 0 || rc < 0.02 {
            return;
        }
        let spin = self.rng.next() * std::f64::consts::TAU;
        let (u, v) = {
            let u = (DVec3::Y - plane * plane.y).normalize_or(DVec3::X);
            (u, plane.cross(u))
        };
        let w = u * spin.cos() + v * spin.sin();
        let child = Growth {
            length: g.length * 0.7,
            segments: g.segments.max(2) - usize::from(g.segments > 2),
            forks: g.forks - 1,
            ..g
        };
        for (k, side) in [1.0, -1.0].into_iter().enumerate() {
            let cc = c + w * side * 0.48 * r;
            let base = ring(cc, plane, rc, ring_a[0] - c, SIDES);
            let d = (dir + w * side * g.splay).normalize();
            self.branch(
                &format!("{name}{}", ["a", "b"][k]),
                &prev,
                base,
                cc,
                plane,
                rc,
                d,
                child,
            );
        }
    }
}

impl Tree {
    pub fn build(&self) -> (Vec<BuildingPart>, Vec<ConnectionSpec>) {
        let mut t = TreeBuilder {
            parts: Vec::new(),
            conns: Vec::new(),
            rng: Lcg(self.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1),
        };
        let conifer = self.family == TreeFamily::Conifer;
        // (segments, segment height, base radius, top radius)
        let (ns, sh, r0, r1) = if conifer {
            (8, 0.8, 0.3, 0.08)
        } else {
            (3, 1.0, 0.34, 0.26)
        };
        let rad = |i: usize| r0 + (r1 - r0) * i as f64 / ns as f64;
        let root_h = 0.4;
        let hring = |y: f64, r: f64| ring(DVec3::new(0.0, y, 0.0), DVec3::Y, r, DVec3::X, SIDES);
        let root = anchored(grain(
            bpart(
                "root",
                frustum(&hring(0.0, 1.5 * r0), &hring(root_h, r0)),
                "pine",
                "generic",
            ),
            DVec3::Y,
        ));
        t.parts.push(root);
        let mut below = "root".to_string();
        let phase = t.rng.next();
        for i in 0..ns {
            let (ya, yb) = (root_h + i as f64 * sh, root_h + (i + 1) as f64 * sh);
            let (ra, rb) = (hring(ya, rad(i)), hring(yb, rad(i + 1)));
            let name = format!("trunk_{i}");
            t.wood(name.clone(), &ra, &rb, DVec3::Y);
            t.solid(&below, &name);
            below = name.clone();
            // side branches from the trunk faces, in a spiral
            let faces: &[usize] = if conifer {
                if i == 0 { &[] } else { &[0, 3, 5] }
            } else if i == 0 {
                &[]
            } else {
                &[0, 3, 5]
            };
            for (j, &f0) in faces.iter().enumerate() {
                let k = (f0 + 3 * i + (phase * 8.0) as usize) % SIDES;
                let (p, q) = (ra[k], ra[(k + 1) % SIDES]);
                let (p2, q2) = (rb[k], rb[(k + 1) % SIDES]);
                let fc = (p + q + p2 + q2) * 0.25;
                let mut fnrm = (q - p).cross(p2 - p).normalize();
                if fnrm.dot(DVec3::new(fc.x, 0.0, fc.z)) < 0.0 {
                    fnrm = -fnrm;
                }
                let rb_ = 0.4 * (q2 - p2).length();
                if rb_ < 0.02 {
                    continue;
                }
                let base = ring(fc, fnrm, rb_, DVec3::Y, SIDES);
                let height = i as f64 / ns as f64;
                let g = if conifer {
                    Growth {
                        length: 0.5 + 2.2 * (1.0 - height),
                        segments: 2,
                        bend: -0.1,
                        taper: 0.65,
                        forks: 0,
                        splay: 0.0,
                    }
                } else {
                    Growth {
                        length: 1.6,
                        segments: 2,
                        bend: 0.45,
                        taper: 0.85,
                        forks: 1,
                        splay: 0.6,
                    }
                };
                let up = if conifer { 0.15 } else { 0.5 };
                let dir = (fnrm + DVec3::Y * up).normalize();
                t.branch(&format!("b{i}{j}"), &name, base, fc, fnrm, rb_, dir, g);
            }
        }
        // leader from the trunk top
        let y_top = root_h + ns as f64 * sh;
        let rl = 0.75 * r1;
        let c = DVec3::new(0.0, y_top, 0.0);
        let base = ring(c, DVec3::Y, rl, DVec3::X, SIDES);
        let g = if conifer {
            Growth {
                length: 1.2,
                segments: 2,
                bend: 0.0,
                taper: 0.5,
                forks: 0,
                splay: 0.0,
            }
        } else {
            Growth {
                length: 1.6,
                segments: 2,
                bend: 0.3,
                taper: 0.85,
                forks: 2,
                splay: 0.7,
            }
        };
        t.branch("leader", &below, base, c, DVec3::Y, rl, DVec3::Y, g);
        (t.parts, t.conns)
    }
}

// ---------------------------------------------------------------------------
// 6. Steel-frame office

/// Three-storey steel-frame office: rolled H columns, I beams nested into
/// them (along one axis onto the column flanges, along the other between
/// the flanges onto the web), continuous concrete floor slabs on the beams,
/// a ground slab, and a glass curtain wall hung on mullions off the slab
/// edges all round. Steel sections are slightly stocky (thicker plates than
/// rolled tables) so their cells stay well shaped.
#[derive(Clone, Debug)]
pub struct SteelOffice {
    pub bays_x: usize,
    pub bays_z: usize,
    pub storeys: usize,
    pub span: f64,
    pub storey: f64,
    pub slab: f64,
    /// Glass panes per bay along the facade.
    pub panes_per_bay: usize,
}

impl Default for SteelOffice {
    fn default() -> Self {
        SteelOffice {
            bays_x: 2,
            bays_z: 2,
            storeys: 3,
            span: 6.0,
            storey: 3.6,
            slab: 0.2,
            panes_per_bay: 3,
        }
    }
}

/// I/H section outline in (s, t): depth s in [0, d], width t in
/// [-b/2, b/2], flanges tf, web tw.
fn i_section(d: f64, b: f64, tf: f64, tw: f64) -> Vec<[f64; 2]> {
    let (hb, hw) = (0.5 * b, 0.5 * tw);
    vec![
        [0.0, -hb],
        [tf, -hb],
        [tf, -hw],
        [d - tf, -hw],
        [d - tf, -hb],
        [d, -hb],
        [d, hb],
        [d - tf, hb],
        [d - tf, hw],
        [tf, hw],
        [tf, hb],
        [0.0, hb],
    ]
}

/// Simple polygon in (x, z) extruded along y.
fn prism_xz(poly: &[[f64; 2]], y0: f64, y1: f64) -> TriMesh {
    reorient(&prism(poly, y0, y1), |p| DVec3::new(p.x, p.z, p.y), true)
}

impl SteelOffice {
    pub fn build(&self) -> (Vec<BuildingPart>, Vec<ConnectionSpec>) {
        let (nx, nz, s, hs) = (self.bays_x, self.bays_z, self.span, self.storey);
        // column H 300 (flanges normal to x), beam I 400 x 180
        let (c, ctf, ctw) = (0.3, 0.03, 0.02);
        let (bd, bb, btf, btw) = (0.4, 0.18, 0.025, 0.016);
        let (lx, lz) = (nx as f64 * s + c, nz as f64 * s + c);
        let steel = "steel_s355";
        let mut parts = Vec::new();
        let mut ground = anchored(bpart(
            "ground_slab",
            box_at(
                DVec3::new(-0.6, -0.4, -0.6),
                DVec3::new(lx + 0.6, 0.0, lz + 0.6),
            ),
            "concrete_c30",
            "slab",
        ));
        ground.meta.analysis_cells_per_m3 = Some(5.0);
        parts.push(ground);
        let column = i_section(c, c, ctf, ctw);
        let beam = i_section(bd, bb, btf, btw);
        for f in 0..self.storeys {
            let y0 = f as f64 * hs;
            let yt = y0 + hs - self.slab; // column top, beam top, slab soffit
            for i in 0..=nx {
                for j in 0..=nz {
                    let (x, z) = (i as f64 * s, j as f64 * s);
                    let poly: Vec<[f64; 2]> = column
                        .iter()
                        .map(|p| [x + p[0], z + 0.5 * c + p[1]])
                        .collect();
                    parts.push(grain(
                        bpart(
                            format!("f{f}_col_{i}_{j}"),
                            prism_xz(&poly, y0, yt),
                            steel,
                            "column",
                        ),
                        DVec3::Y,
                    ));
                }
            }
            // beams along x onto the column flanges
            for j in 0..=nz {
                let zc = j as f64 * s + 0.5 * c;
                let poly: Vec<[f64; 2]> =
                    beam.iter().map(|p| [zc + p[1], yt - bd + p[0]]).collect();
                for i in 0..nx {
                    let (x0, x1) = (i as f64 * s + c, (i + 1) as f64 * s);
                    parts.push(grain(
                        bpart(
                            format!("f{f}_beam_x_{i}_{j}"),
                            prism_zy(&poly, x0, x1),
                            steel,
                            "beam",
                        ),
                        DVec3::X,
                    ));
                }
            }
            // beams along z between the flanges onto the column webs
            for i in 0..=nx {
                let xc = i as f64 * s + 0.5 * c;
                let poly: Vec<[f64; 2]> =
                    beam.iter().map(|p| [xc + p[1], yt - bd + p[0]]).collect();
                for j in 0..nz {
                    let z0 = j as f64 * s + 0.5 * c + 0.5 * ctw;
                    let z1 = (j + 1) as f64 * s + 0.5 * c - 0.5 * ctw;
                    parts.push(grain(
                        bpart(
                            format!("f{f}_beam_z_{i}_{j}"),
                            prism(&poly, z0, z1),
                            steel,
                            "beam",
                        ),
                        DVec3::Z,
                    ));
                }
            }
            // floor slab, continuous over the beams (a composite deck):
            // panels jointed on the beam lines would each bear on half a
            // flange, eccentrically
            let mut p = bpart(
                format!("f{f}_slab"),
                box_at(DVec3::new(0.0, yt, 0.0), DVec3::new(lx, y0 + hs, lz)),
                "concrete_c30",
                "slab",
            );
            p.meta.analysis_cells_per_m3 = Some(40.0);
            parts.push(p);
        }
        // curtain wall: mullions and panes per storey, outside the slab
        // edges; front/back run through the corners, the sides between
        let (md, mw, gt) = (0.15, 0.08, 0.016); // 8+8 laminated glass
        let facades = [
            ("south", true, 0.0, -1.0, nx),
            ("north", true, lz, 1.0, nx),
            ("west", false, 0.0, -1.0, nz),
            ("east", false, lx, 1.0, nz),
        ];
        for &(name, along_x, base, out, bays) in &facades {
            let wf = WallFrame { along_x };
            let layer = |d0: f64, d1: f64| {
                let (a, b) = (base + out * d0, base + out * d1);
                [a.min(b), a.max(b)]
            };
            let (u0, u1) = if along_x { (-md, lx + md) } else { (0.0, lz) };
            let n = bays * self.panes_per_bay;
            let pitch = (u1 - u0 - mw) / n as f64;
            let mid = 0.5 * md;
            for f in 0..self.storeys {
                let v = [f as f64 * hs, (f + 1) as f64 * hs];
                for k in 0..=n {
                    let a = u0 + k as f64 * pitch;
                    parts.push(grain(
                        bpart(
                            format!("{name}_f{f}_mullion_{k}"),
                            wf.cuboid([a, a + mw], v, layer(0.0, md)),
                            steel,
                            "wall",
                        ),
                        DVec3::Y,
                    ));
                    if k < n {
                        parts.push(bpart(
                            format!("{name}_f{f}_pane_{k}"),
                            wf.cuboid(
                                [a + mw, a + pitch],
                                v,
                                layer(mid - 0.5 * gt, mid + 0.5 * gt),
                            ),
                            "glass_annealed",
                            "glazing",
                        ));
                    }
                }
            }
        }
        let conns = auto_conns(&parts, |a, b| {
            let mut m = [a.material.as_str(), b.material.as_str()];
            m.sort();
            match m {
                ["steel_s355", "steel_s355"] | ["concrete_c30", "steel_s355"] => {
                    Some(("bolted", None))
                }
                ["concrete_c30", "glass_annealed"]
                | ["glass_annealed", "glass_annealed"]
                | ["glass_annealed", "steel_s355"] => Some(("adhesive", None)),
                _ => None,
            }
        });
        (parts, conns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(parts: &[BuildingPart], conns: &[ConnectionSpec]) {
        let mut names = std::collections::BTreeSet::new();
        for p in parts {
            assert!(names.insert(p.name.clone()), "duplicate part {}", p.name);
            let t = p.mesh.topology();
            assert!(t.is_closed_manifold(), "{} is not closed: {t:?}", p.name);
            assert!(
                p.mesh.signed_volume() > 0.0,
                "{} has volume {}",
                p.name,
                p.mesh.signed_volume()
            );
        }
        for c in conns {
            assert!(
                names.contains(&c.a) && names.contains(&c.b),
                "{} - {}",
                c.a,
                c.b
            );
        }
        // no two parts overlap by more than a touching sliver: volume of the
        // AABB intersection is only a cheap necessary check for boxes
        for (i, a) in parts.iter().enumerate() {
            for b in &parts[i + 1..] {
                let (x, y) = (a.mesh.aabb(), b.mesh.aabb());
                let ov: f64 = (0..3)
                    .map(|k| (x.max[k].min(y.max[k]) - x.min[k].max(y.min[k])).max(0.0))
                    .product();
                let small = a.mesh.signed_volume().min(b.mesh.signed_volume());
                // sloped and oblique parts have loose boxes; only flag gross overlaps
                assert!(
                    ov < 0.5 * small || !is_box(&a.mesh) || !is_box(&b.mesh),
                    "{} overlaps {}",
                    a.name,
                    b.name
                );
            }
        }
    }

    fn is_box(m: &TriMesh) -> bool {
        let bb = m.aabb();
        let v = (bb.max - bb.min).x * (bb.max - bb.min).y * (bb.max - bb.min).z;
        (m.signed_volume() - v).abs() < 1e-9 * v.max(1e-12)
    }

    /// Bake parts at building-starter resolution; gate verdicts plus the
    /// self-weight joint checks sorted by utilization.
    pub(super) fn bake(
        parts: Vec<BuildingPart>,
        conns: Vec<ConnectionSpec>,
        ground: f64,
    ) -> (
        std::collections::BTreeMap<String, (frac_validate::GateStatus, String)>,
        Vec<frac_validate::structure::JointCheck>,
    ) {
        use frac_core::input::{AuthoringMeta, InputPart, InputScene};
        let mut meta = AuthoringMeta {
            connections: conns,
            ground_height: Some(ground),
            ..Default::default()
        };
        let mut scene = Vec::new();
        for p in parts {
            let n = p.mesh.tris.len();
            scene.push(InputPart {
                name: p.name.clone(),
                mesh: p.mesh,
                material_slot: vec![0; n],
                material_names: vec![p.material],
                extras: serde_json::Value::Null,
                ..Default::default()
            });
            meta.parts.insert(p.name, p.meta);
        }
        let input = frac_pipeline::InputSpec {
            name: "structure".into(),
            scene: InputScene {
                source: "test".into(),
                parts: scene,
            },
            meta,
            variant: 0,
        };
        let mut s = frac_core::Settings::default();
        s.cells.target_analysis_cells_per_m3 = 20.0;
        s.cells.fine_per_analysis = 4;
        s.modes.enabled = false;
        s.render.lods = 1;
        s.render.noise = false;
        s.render.chipping = false;
        let lib = frac_material::MaterialLibrary::builtin();
        let out = frac_pipeline::run(&input, &s, &lib).expect("pipeline");
        if std::env::var_os("SHOW_WARNINGS").is_some() {
            for w in &out.report.warnings {
                println!("warning: {w}");
            }
        }
        let mut joints = frac_validate::structure::self_weight(&out.asset, &lib);
        joints.sort_by(|a, b| b.utilization.total_cmp(&a.utilization));
        let gates = out
            .report
            .scorecard
            .gates
            .iter()
            .map(|g| (g.name.clone(), (g.status.clone(), g.detail.clone())))
            .collect();
        (gates, joints)
    }

    #[test]
    fn catenary_arch_stands_where_the_semicircle_hinges() {
        use frac_validate::GateStatus;
        let arch = |shape| {
            let (p, c) = MasonryArch {
                shape,
                ..Default::default()
            }
            .build();
            bake(p, c, 0.0)
        };
        // same voussoirs, same t/R = 0.1, no-tension lime mortar
        let (g, j) = arch(ArchShape::Catenary);
        for gate in ["structural_support", "self_weight"] {
            assert_eq!(g[gate].0, GateStatus::Pass, "{gate}: {}", g[gate].1);
        }
        assert!(j[0].utilization < 1.0);
        let (g, j) = arch(ArchShape::Semicircle);
        assert_eq!(g["structural_support"].0, GateStatus::Pass);
        assert_eq!(
            g["self_weight"].0,
            GateStatus::Fail,
            "{}",
            g["self_weight"].1
        );
        // it hinges at the crown, where the thrust line leaves the ring
        let n = MasonryArch::default().voussoirs;
        let crown = [
            format!("voussoir_{:02}", n / 2 - 1),
            format!("voussoir_{:02}", n / 2),
            format!("voussoir_{:02}", n / 2 + 1),
        ];
        assert_eq!(j[0].mode, "overturning");
        assert!(
            crown.contains(&j[0].a) && crown.contains(&j[0].b),
            "{} - {}",
            j[0].a,
            j[0].b
        );
    }

    #[test]
    fn built_structures_pass_self_weight() {
        use frac_validate::GateStatus;
        for (name, (p, c), ground) in [
            ("truss", PrattTruss::default().build(), 0.0),
            ("house", VeneerHouse::default().build(), -0.3),
            ("viaduct", RcViaduct::default().build(), 0.0),
            ("tree", Tree::default().build(), 0.0),
            ("office", SteelOffice::default().build(), -0.4),
            (
                "conifer",
                Tree {
                    family: TreeFamily::Conifer,
                    ..Default::default()
                }
                .build(),
                0.0,
            ),
        ] {
            let (g, _) = bake(p, c, ground);
            for gate in ["structural_support", "self_weight"] {
                assert_eq!(g[gate].0, GateStatus::Pass, "{name} {gate}: {}", g[gate].1);
            }
        }
    }

    /// `cargo test -p frac-cli --release -- --ignored --nocapture joints`
    #[test]
    #[ignore]
    fn joints() {
        let which = std::env::var("STRUCT").unwrap_or_default();
        let t_r: f64 = std::env::var("T_R")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.1);
        let (parts, conns, g) = match which.as_str() {
            "truss" => {
                let (p, c) = PrattTruss::default().build();
                (p, c, 0.0)
            }
            "house" => {
                let (p, c) = VeneerHouse::default().build();
                (p, c, -0.3)
            }
            "viaduct" => {
                let (p, c) = RcViaduct::default().build();
                (p, c, 0.0)
            }
            "office" => {
                let small = std::env::var_os("SMALL").is_some();
                let (p, c) = SteelOffice {
                    bays_x: if small { 1 } else { 2 },
                    bays_z: if small { 1 } else { 2 },
                    storeys: if small { 1 } else { 3 },
                    ..Default::default()
                }
                .build();
                (p, c, -0.4)
            }
            "conifer" => {
                let (p, c) = Tree {
                    family: TreeFamily::Conifer,
                    seed: std::env::var("SEED")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(7),
                }
                .build();
                (p, c, 0.0)
            }
            "tree" => {
                let (p, c) = Tree::default().build();
                (p, c, 0.0)
            }
            "semicircle" => {
                let (p, c) = MasonryArch {
                    shape: ArchShape::Semicircle,
                    t_over_r: t_r,
                    ..Default::default()
                }
                .build();
                (p, c, 0.0)
            }
            _ => {
                let (p, c) = MasonryArch {
                    t_over_r: t_r,
                    ..Default::default()
                }
                .build();
                (p, c, 0.0)
            }
        };
        let (gates, joints) = bake(parts, conns, g);
        for (k, v) in &gates {
            if v.0 != frac_validate::GateStatus::Pass {
                println!("{k}: {:?} {}", v.0, v.1);
            }
        }
        for j in joints.iter().take(25) {
            println!(
                "{:8.3} {:12} {:?} {} - {}",
                j.utilization, j.mode, j.kind, j.a, j.b
            );
        }
    }

    #[test]
    fn structures_are_closed_and_named_uniquely() {
        let all = [
            MasonryArch::default().build(),
            PrattTruss::default().build(),
            VeneerHouse::default().build(),
            RcViaduct::default().build(),
            Tree::default().build(),
            Tree {
                family: TreeFamily::Conifer,
                ..Default::default()
            }
            .build(),
            SteelOffice::default().build(),
        ];
        for (parts, conns) in &all {
            check(parts, conns);
        }
    }
}
