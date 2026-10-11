//! Procedural buildings made only of cuboids, with structural roles,
//! materials per part, explicit connections and ground anchors. The first
//! rung of a ladder of increasingly realistic buildings: every part is a
//! primitive, the load path is explicit (columns → beams → slabs), and the
//! non-structural parts (infill walls, glazing, partitions, parapets) are
//! carried by the frame.

use crate::meshgen::box_at;
use frac_core::input::{ConnectionSpec, PartMeta};
use frac_geom::{DVec3, TriMesh};

pub struct BuildingPart {
    pub name: String,
    pub mesh: TriMesh,
    pub material: String,
    pub meta: PartMeta,
}

/// Parameters of a cuboid frame building (metres, y up).
#[derive(Clone, Debug)]
pub struct FrameBuilding {
    pub bays_x: usize,
    pub bays_z: usize,
    pub floors: usize,
    /// Column grid spacing.
    pub span: f64,
    /// Floor-to-floor height.
    pub storey: f64,
    pub slab: f64,
    pub column: f64,
    pub beam_depth: f64,
    pub wall: f64,
    pub window: [f64; 2],
    pub sill: f64,
    pub partition: f64,
    pub parapet: [f64; 2],
    /// Brick parts fracture brick-by-brick (masonry recipe) instead of as
    /// Voronoi solids.
    pub masonry: bool,
}

impl Default for FrameBuilding {
    fn default() -> Self {
        FrameBuilding {
            bays_x: 2,
            bays_z: 2,
            floors: 2,
            span: 4.0,
            storey: 3.0,
            slab: 0.2,
            column: 0.3,
            beam_depth: 0.4,
            wall: 0.2,
            window: [1.2, 1.2],
            sill: 0.9,
            partition: 0.1,
            parapet: [0.9, 0.15],
            masonry: false,
        }
    }
}

fn part(
    name: String,
    lo: DVec3,
    hi: DVec3,
    material: &str,
    role: &str,
    anchored: bool,
) -> BuildingPart {
    let meta = PartMeta {
        material: Some(material.into()),
        component_role: Some(role.into()),
        anchor_below: anchored.then_some(lo.y),
        ..Default::default()
    };
    BuildingPart {
        name,
        mesh: box_at(lo, hi),
        material: material.into(),
        meta,
    }
}

fn conn(a: &str, b: &str, kind: &str) -> ConnectionSpec {
    ConnectionSpec {
        a: a.into(),
        b: b.into(),
        kind: kind.into(),
        interface_material: None,
    }
}

impl FrameBuilding {
    fn brick(&self, mut p: BuildingPart) -> BuildingPart {
        if !self.masonry {
            p.meta.recipe = Some("clustered_voronoi".into());
        }
        p
    }

    /// All parts and the connections that differ from the automatic contact
    /// typing (concrete–concrete cold joints and horizontal bearing joints
    /// are inferred).
    pub fn build(&self) -> (Vec<BuildingPart>, Vec<ConnectionSpec>) {
        let (s, c) = (self.span, self.column);
        let col_h = self.storey - self.slab;
        let (nx, nz) = (self.bays_x, self.bays_z);
        let (xmax, zmax) = (nx as f64 * s + c, nz as f64 * s + c);
        let mut parts = Vec::new();
        let mut conns = Vec::new();
        for f in 0..self.floors {
            let y0 = f as f64 * self.storey;
            let (yb, yt) = (y0 + col_h - self.beam_depth, y0 + col_h);
            let ground = f == 0;
            // columns (load bearing; ground floor anchored)
            for i in 0..=nx {
                for j in 0..=nz {
                    let (x, z) = (i as f64 * s, j as f64 * s);
                    parts.push(part(
                        format!("f{f}_col_{i}_{j}"),
                        DVec3::new(x, y0, z),
                        DVec3::new(x + c, yt, z + c),
                        "concrete_c30",
                        "column",
                        ground,
                    ));
                }
            }
            // beams framing into the column faces, along x and z
            for j in 0..=nz {
                for i in 0..nx {
                    let (x, z) = (i as f64 * s, j as f64 * s);
                    parts.push(part(
                        format!("f{f}_beamx_{i}_{j}"),
                        DVec3::new(x + c, yb, z),
                        DVec3::new(x + s, yt, z + c),
                        "concrete_c30",
                        "beam",
                        false,
                    ));
                }
            }
            for i in 0..=nx {
                for j in 0..nz {
                    let (x, z) = (i as f64 * s, j as f64 * s);
                    parts.push(part(
                        format!("f{f}_beamz_{i}_{j}"),
                        DVec3::new(x, yb, z + c),
                        DVec3::new(x + c, yt, z + s),
                        "concrete_c30",
                        "beam",
                        false,
                    ));
                }
            }
            // slab on columns and beams
            parts.push(part(
                format!("f{f}_slab"),
                DVec3::new(0.0, yt, 0.0),
                DVec3::new(xmax, y0 + self.storey, zmax),
                "concrete_c30",
                "slab",
                false,
            ));
            // perimeter infill walls (non-load-bearing brick) with a window,
            // split into four cuboids around the opening plus a glass pane
            let off = 0.5 * (c - self.wall);
            let facade = |name: String,
                          along_x: bool,
                          line: f64,
                          a0: f64,
                          a1: f64,
                          parts: &mut Vec<BuildingPart>,
                          conns: &mut Vec<ConnectionSpec>| {
                let mid = 0.5 * (a0 + a1);
                let (wl, wr) = (mid - 0.5 * self.window[0], mid + 0.5 * self.window[0]);
                let (ws, wh) = (y0 + self.sill, y0 + self.sill + self.window[1]);
                let (t0, t1) = (line + off, line + off + self.wall);
                let bx = |a: f64, b: f64, y_lo: f64, y_hi: f64, u0: f64, u1: f64| {
                    if along_x {
                        (DVec3::new(a, y_lo, u0), DVec3::new(b, y_hi, u1))
                    } else {
                        (DVec3::new(u0, y_lo, a), DVec3::new(u1, y_hi, b))
                    }
                };
                let pieces = [
                    ("left", bx(a0, wl, y0, yb, t0, t1)),
                    ("right", bx(wr, a1, y0, yb, t0, t1)),
                    ("sill", bx(wl, wr, y0, ws, t0, t1)),
                    ("head", bx(wl, wr, wh, yb, t0, t1)),
                ];
                let names: Vec<String> =
                    pieces.iter().map(|(k, _)| format!("{name}_{k}")).collect();
                for ((k, (lo, hi)), n) in pieces.iter().zip(&names) {
                    let anchored = ground && (*k == "left" || *k == "right" || *k == "sill");
                    parts.push(self.brick(part(
                        n.clone(),
                        *lo,
                        *hi,
                        "brick_clay",
                        "wall",
                        anchored,
                    )));
                }
                let gm = 0.5 * (t0 + t1);
                let (glo, ghi) = bx(wl, wr, ws, wh, gm - 0.005, gm + 0.005);
                let gname = format!("{name}_glass");
                parts.push(part(
                    gname.clone(),
                    glo,
                    ghi,
                    "glass_annealed",
                    "glazing",
                    false,
                ));
                for n in &names {
                    conns.push(conn(&gname, n, "adhesive"));
                    for m in &names {
                        if n < m {
                            conns.push(conn(n, m, "mortar_joint"));
                        }
                    }
                }
                names
            };
            for j in [0, nz] {
                for i in 0..nx {
                    let line = j as f64 * s;
                    let names = facade(
                        format!("f{f}_wallx_{i}_{j}"),
                        true,
                        line,
                        i as f64 * s + c,
                        (i + 1) as f64 * s,
                        &mut parts,
                        &mut conns,
                    );
                    conns.push(conn(
                        &names[0],
                        &format!("f{f}_col_{i}_{j}"),
                        "mortar_joint",
                    ));
                    conns.push(conn(
                        &names[1],
                        &format!("f{f}_col_{}_{j}", i + 1),
                        "mortar_joint",
                    ));
                }
            }
            for i in [0, nx] {
                for j in 0..nz {
                    let line = i as f64 * s;
                    let names = facade(
                        format!("f{f}_wallz_{i}_{j}"),
                        false,
                        line,
                        j as f64 * s + c,
                        (j + 1) as f64 * s,
                        &mut parts,
                        &mut conns,
                    );
                    conns.push(conn(
                        &names[0],
                        &format!("f{f}_col_{i}_{j}"),
                        "mortar_joint",
                    ));
                    conns.push(conn(
                        &names[1],
                        &format!("f{f}_col_{i}_{}", j + 1),
                        "mortar_joint",
                    ));
                }
            }
            // interior drywall partitions along the inner column lines (x)
            for j in 1..nz {
                for i in 0..nx {
                    let (x, z) = (i as f64 * s, j as f64 * s);
                    let zo = z + 0.5 * (c - self.partition);
                    let n = format!("f{f}_partition_{i}_{j}");
                    parts.push(part(
                        n.clone(),
                        DVec3::new(x + c, y0, zo),
                        DVec3::new(x + s, yb, zo + self.partition),
                        "drywall",
                        "cosmetic",
                        ground,
                    ));
                    conns.push(conn(&n, &format!("f{f}_col_{i}_{j}"), "adhesive"));
                    conns.push(conn(&n, &format!("f{f}_col_{}_{j}", i + 1), "adhesive"));
                }
            }
        }
        // roof parapet (cosmetic brick) on the roof slab perimeter
        let ytop = self.floors as f64 * self.storey;
        let [ph, pt] = self.parapet;
        let edges = [
            (
                "parapet_s",
                DVec3::new(0.0, ytop, 0.0),
                DVec3::new(xmax, ytop + ph, pt),
            ),
            (
                "parapet_n",
                DVec3::new(0.0, ytop, zmax - pt),
                DVec3::new(xmax, ytop + ph, zmax),
            ),
            (
                "parapet_w",
                DVec3::new(0.0, ytop, pt),
                DVec3::new(pt, ytop + ph, zmax - pt),
            ),
            (
                "parapet_e",
                DVec3::new(xmax - pt, ytop, pt),
                DVec3::new(xmax, ytop + ph, zmax - pt),
            ),
        ];
        for (n, lo, hi) in edges {
            parts.push(self.brick(part(n.into(), lo, hi, "brick_clay", "cosmetic", false)));
        }
        for (a, b) in [
            ("parapet_w", "parapet_s"),
            ("parapet_w", "parapet_n"),
            ("parapet_e", "parapet_s"),
            ("parapet_e", "parapet_n"),
        ] {
            conns.push(conn(a, b, "mortar_joint"));
        }
        (parts, conns)
    }
}
