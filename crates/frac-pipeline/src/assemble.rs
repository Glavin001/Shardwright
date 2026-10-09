//! Assembly of per-component cell builds into the asset (global stable IDs,
//! typed interfaces, rebar crossings, contact and anchor interfaces).

use frac_bonds::contacts::{anchor_polygons, cell_faces, contact_polygons};
use frac_bonds::{patch_polygon, polygon_integrals, rebar_crossings};
use frac_cells::cellset::CellSet;
use frac_core::input::PartMeta;
use frac_core::*;
use frac_geom::{morton3, Aabb, DVec3, MassProps};
use frac_material::MaterialLibrary;
use std::collections::BTreeMap;

pub struct BuiltComponent {
    pub name: String,
    pub meta: PartMeta,
    pub role: ComponentRole,
    pub material: MaterialId,
    pub solid: frac_geom::TriMesh,
    pub surface: SurfaceAttributes,
    pub reconstructed: bool,
    pub render_surface: Option<frac_geom::TriMesh>,
    pub cells: CellSet,
    pub grain: Option<DVec3>,
    pub unfractured: bool,
    pub joint: Option<(InterfaceKind, Option<MaterialId>)>,
    pub wood: bool,
}

/// Per-component analysis-cell info kept for later stages.
pub struct ComponentCells {
    pub cell_range: std::ops::Range<u32>,
    /// Local analysis index per local cell (after Morton renumbering).
    pub cell_analysis: Vec<u32>,
    pub n_analysis: u32,
}

pub fn assemble(asset: &mut Asset, built: Vec<BuiltComponent>, lib: &MaterialLibrary) -> Vec<ComponentCells> {
    let mut infos = Vec::new();
    for (ci, mut b) in built.into_iter().enumerate() {
        let comp_id = ComponentId(ci as u32);
        let density = lib.material(b.material).density;
        let bbox = b.solid.aabb();
        // stable local cell order: Morton code of centroid within the component
        let mut order: Vec<u32> = (0..b.cells.cells.len() as u32).collect();
        order.sort_by_key(|&c| {
            let cc = &b.cells.cells[c as usize];
            (morton3(cc.vi.com(), &bbox), c)
        });
        b.cells.reorder(&order);
        // analysis clusters: renumber by Morton of cluster centroid
        let ncl = b.cells.cells.iter().map(|c| c.cluster).max().map(|x| x + 1).unwrap_or(0) as usize;
        let mut cl_vi = vec![frac_geom::VolumeIntegrals::default(); ncl];
        for c in &b.cells.cells {
            cl_vi[c.cluster as usize].add(&c.vi);
        }
        let mut cl_order: Vec<u32> = (0..ncl as u32).collect();
        cl_order.sort_by_key(|&k| (morton3(cl_vi[k as usize].com(), &bbox), k));
        let mut cl_map = vec![0u32; ncl];
        for (new, &old) in cl_order.iter().enumerate() {
            cl_map[old as usize] = new as u32;
        }
        let base = asset.cells.len() as u32;
        let abase = asset.analysis_cells.len() as u32;
        let mut cell_analysis = Vec::new();
        for (k, c) in b.cells.cells.iter().enumerate() {
            let a = cl_map[c.cluster as usize];
            cell_analysis.push(a);
            let mp = c.vi.mass_props(density);
            asset.cells.push(Cell {
                id: CellId(base + k as u32),
                component: comp_id,
                analysis_cell: abase + a,
                material: b.material,
                mass: mp,
                aabb: c.aabb,
                thickness_ratio: c.thickness_ratio,
            });
        }
        for a in 0..ncl as u32 {
            let members: Vec<CellId> = cell_analysis.iter().enumerate().filter(|(_, x)| **x == a).map(|(k, _)| CellId(base + k as u32)).collect();
            let mp = MassProps::combine(&members.iter().map(|c| asset.cells[c.idx()].mass).collect::<Vec<_>>());
            asset.analysis_cells.push(AnalysisCell { id: abase + a, component: comp_id, cells: members, mass: mp });
        }
        // geometry with global cell ids and interfaces per cell pair
        let mut geom = ComponentGeometry { verts: b.cells.verts.clone(), ext_polys: Vec::new(), patches: Vec::new() };
        for e in &b.cells.ext {
            geom.ext_polys.push(ExtPoly { verts: e.verts.clone(), cell: CellId(base + e.cell), src_tri: e.src_tri });
        }
        let mut by_pair: BTreeMap<(u32, u32), Vec<usize>> = BTreeMap::new();
        for (pi, p) in b.cells.patches.iter().enumerate() {
            let (x, y) = (p.cells[0], p.cells[1]);
            by_pair.entry((x.min(y), x.max(y))).or_default().push(pi);
        }
        for ((lo, hi), pis) in by_pair {
            let iid = InterfaceId(asset.interfaces.len() as u32);
            let mut polys = Vec::new();
            let mut patch_ids = Vec::new();
            for &pi in &pis {
                let p = &b.cells.patches[pi];
                // orient from lo to hi
                let flip = p.cells[0] != lo;
                let pid = geom.patches.len() as u32;
                let tris = if flip { p.tris.iter().map(|t| [t[0], t[2], t[1]]).collect() } else { p.tris.clone() };
                let loops: Vec<Vec<u32>> = if flip { p.loops.iter().map(|l| l.iter().rev().copied().collect()).collect() } else { p.loops.clone() };
                let patch = Patch { loops, tris, normal: if flip { -p.normal } else { p.normal }, cells: (CellId(base + lo), CellId(base + hi)), interface: iid };
                polys.push(patch_polygon(&geom.verts, &patch, false));
                geom.patches.push(patch);
                patch_ids.push(pid);
            }
            let area: f64 = polys.iter().map(|q| polygon_integrals(q).area).sum();
            let (ua, ub) = (b.cells.cells[lo as usize].unit, b.cells.cells[hi as usize].unit);
            let (kind, imat) = if ua != ub {
                b.joint.unwrap_or((InterfaceKind::Monolithic, None))
            } else if b.wood {
                let n = polys.first().map(|q| q.normal).unwrap_or(DVec3::Y);
                let g = b.grain.unwrap_or(DVec3::X);
                if n.dot(g).abs() < 0.5 { (InterfaceKind::GrainBoundary, None) } else { (InterfaceKind::Monolithic, None) }
            } else {
                (InterfaceKind::Monolithic, None)
            };
            let (rebar, rebar_points) = rebar_crossings(&polys, &b.meta.rebar);
            asset.interfaces.push(Interface {
                id: iid,
                cells: (CellId(base + lo), CellOrWorld::Cell(CellId(base + hi))),
                polygons: polys,
                kind,
                interface_material: imat,
                rebar,
                area,
                patches: patch_ids,
                rebar_points,
            });
        }
        let vol = b.cells.cells.iter().map(|c| c.vi.volume).sum::<f64>();
        asset.components.push(Component {
            id: comp_id,
            name: b.name.clone(),
            role: b.role,
            material: b.material,
            solid: b.solid,
            surface: b.surface,
            reconstructed: b.reconstructed,
            render_surface: b.render_surface,
            geometry: geom,
            cells: base..base + b.cells.cells.len() as u32,
            analysis_cells: abase..abase + ncl as u32,
            volume: vol,
            aabb: bbox,
            grain: b.grain,
            unfractured: b.unfractured,
        });
        infos.push(ComponentCells { cell_range: base..base + b.cells.cells.len() as u32, cell_analysis, n_analysis: ncl as u32 });
        let _ = b.meta;
    }
    infos
}

/// Typed contact interfaces between components in contact.
pub fn add_contacts(
    asset: &mut Asset,
    contacts: &[(usize, usize)],
    connections: &[frac_core::input::ConnectionSpec],
    lib: &MaterialLibrary,
    tol: f64,
    cos: f64,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let faces: Vec<_> = asset.components.iter().map(|c| cell_faces(&c.geometry)).collect();
    for &(a, b) in contacts {
        let (na, nb) = (asset.components[a].name.clone(), asset.components[b].name.clone());
        let spec = connections.iter().find(|c| (c.a == na && c.b == nb) || (c.a == nb && c.b == na));
        let polys = contact_polygons(&faces[a], &faces[b], tol, cos);
        if polys.is_empty() {
            warnings.push(format!("components '{na}' and '{nb}' are close but share no contact face"));
            continue;
        }
        for (ca, cb, poly) in polys {
            let (kind, imat) = match spec {
                Some(s) => (
                    InterfaceKind::from_name(&s.kind).unwrap_or(InterfaceKind::ComponentConnection),
                    s.interface_material.as_ref().and_then(|m| lib.interface_material_id(m)).or_else(|| lib.interface_material_id(&s.kind)),
                ),
                None => {
                    let ma = lib.material(asset.components[a].material).id.clone();
                    let mb = lib.material(asset.components[b].material).id.clone();
                    if ma.starts_with("concrete") && mb.starts_with("concrete") {
                        (InterfaceKind::ColdJoint, lib.interface_material_id("cold_joint"))
                    } else if poly.normal.y.abs() > 0.7 {
                        (InterfaceKind::Bearing, lib.interface_material_id("bearing"))
                    } else {
                        (InterfaceKind::ComponentConnection, None)
                    }
                }
            };
            // orient low cell id -> high
            let (lo, hi, poly) = if ca < cb {
                (ca, cb, poly)
            } else {
                let mut q = poly;
                q.normal = -q.normal;
                for l in q.loops.iter_mut() {
                    l.reverse();
                }
                (cb, ca, q)
            };
            let area = polygon_integrals(&poly).area;
            let id = InterfaceId(asset.interfaces.len() as u32);
            asset.interfaces.push(Interface {
                id,
                cells: (lo, CellOrWorld::Cell(hi)),
                polygons: vec![poly],
                kind,
                interface_material: imat,
                rebar: RebarCrossing::default(),
                area,
                patches: Vec::new(),
                rebar_points: Vec::new(),
            });
        }
    }
    warnings
}

/// Anchor interfaces (cell <-> world).
pub fn add_anchors(asset: &mut Asset, metas: &[PartMeta], ground: Option<f64>, tol: f64) {
    for ci in 0..asset.components.len() {
        let meta = &metas[ci];
        let comp = &asset.components[ci];
        let height = match (meta.anchor_below, meta.anchor, ground) {
            (Some(h), _, _) => Some(h),
            (None, true, _) => Some(comp.aabb.min.y),
            (None, false, Some(g)) if comp.aabb.min.y <= g + tol => Some(g),
            _ => None,
        };
        let Some(h) = height else { continue };
        let faces = cell_faces(&comp.geometry);
        for (cell, poly) in anchor_polygons(&faces, h, tol) {
            let area = polygon_integrals(&poly).area;
            let id = InterfaceId(asset.interfaces.len() as u32);
            asset.interfaces.push(Interface {
                id,
                cells: (cell, CellOrWorld::World),
                polygons: vec![poly],
                kind: InterfaceKind::Anchor,
                interface_material: None,
                rebar: RebarCrossing::default(),
                area,
                patches: Vec::new(),
                rebar_points: Vec::new(),
            });
        }
    }
}

/// Asset bounding box.
pub fn asset_bbox(asset: &Asset) -> Aabb {
    asset.components.iter().fold(Aabb::EMPTY, |a, c| a.union(&c.aabb))
}
