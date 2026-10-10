//! The asset data model (spec §9). All positions are in the asset frame
//! (meters, right-handed, +Y up).

use crate::ids::*;
use frac_geom::{Aabb, MassProps, TriMesh};
use glam::{DMat3, DVec3};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::ops::Range;

/// Interface type between two cells (or a cell and the world).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum InterfaceKind {
    Monolithic = 0,
    MortarJoint = 1,
    Weld = 2,
    Bolted = 3,
    ColdJoint = 4,
    GrainBoundary = 5,
    Adhesive = 6,
    ComponentConnection = 7,
    Anchor = 8,
    Bearing = 9,
}

impl InterfaceKind {
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "monolithic" => Self::Monolithic,
            "mortar_joint" | "mortar" => Self::MortarJoint,
            "weld" => Self::Weld,
            "bolted" => Self::Bolted,
            "cold_joint" => Self::ColdJoint,
            "grain_boundary" => Self::GrainBoundary,
            "adhesive" => Self::Adhesive,
            "component_connection" => Self::ComponentConnection,
            "anchor" => Self::Anchor,
            "bearing" => Self::Bearing,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum ComponentRole {
    #[default]
    Generic = 0,
    Column = 1,
    Beam = 2,
    Slab = 3,
    Wall = 4,
    Connection = 5,
    Cosmetic = 6,
    Prop = 7,
    Glazing = 8,
}

impl ComponentRole {
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "generic" => Self::Generic,
            "column" => Self::Column,
            "beam" => Self::Beam,
            "slab" => Self::Slab,
            "wall" => Self::Wall,
            "connection" => Self::Connection,
            "cosmetic" => Self::Cosmetic,
            "prop" => Self::Prop,
            "glazing" => Self::Glazing,
            _ => return None,
        })
    }
}

/// Rebar crossing an interface.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct RebarCrossing {
    pub count: u32,
    pub steel_area: f64,
    /// Area-weighted (unnormalized) bar direction; normalized on export.
    pub dir: DVec3,
}

impl Default for RebarCrossing {
    fn default() -> Self {
        RebarCrossing { count: 0, steel_area: 0.0, dir: DVec3::ZERO }
    }
}

impl RebarCrossing {
    pub fn add(&mut self, o: &RebarCrossing) {
        self.count += o.count;
        self.steel_area += o.steel_area;
        self.dir += o.dir;
    }
}

/// A planar polygon with holes in 3D: `loops[0]` is the outer loop (CCW
/// about `normal`), the rest are holes (CW).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Polygon3 {
    pub loops: Vec<Vec<DVec3>>,
    pub normal: DVec3,
}

/// In-plane oriented bounding rectangle in a bond frame.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Obb2 {
    pub center: DVec3,
    pub axis_u: DVec3,
    pub axis_v: DVec3,
    pub half: [f64; 2],
}

impl Default for Obb2 {
    fn default() -> Self {
        Obb2 { center: DVec3::ZERO, axis_u: DVec3::X, axis_v: DVec3::Y, half: [0.0; 2] }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct BondComposition {
    pub kind: InterfaceKind,
    pub interface_material: Option<MaterialId>,
    pub fraction: f32,
}

/// Per-corner surface attributes of a component's solid (indexed by solid
/// triangle), used to carry UVs/normals onto exterior fragment faces.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SurfaceAttributes {
    pub normals: Vec<[[f32; 3]; 3]>,
    pub uvs: Option<Vec<[[f32; 2]; 3]>>,
    pub tangents: Option<Vec<[[f32; 4]; 3]>>,
    /// Source material slot per triangle (for multi-slot parts).
    pub material_slot: Option<Vec<u16>>,
}

/// An exterior polygon: part of the component's solid surface inside one cell.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtPoly {
    pub verts: Vec<u32>,
    pub cell: CellId,
    pub src_tri: u32,
    /// Non-degenerate triangulation (CDT; T-junction vertices on polygon
    /// edges are kept without zero-area triangles).
    pub tris: Vec<[u32; 3]>,
}

/// A connected planar interface patch shared by two cells of a component.
/// Stored once; both cells reference it (with opposite orientation).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Patch {
    /// Outer loop first (CCW about `normal`), then holes.
    pub loops: Vec<Vec<u32>>,
    /// Triangulation of the patch (CCW about `normal`), boundary vertices only.
    pub tris: Vec<[u32; 3]>,
    /// Unit normal, pointing from `cells.0` into `cells.1`.
    pub normal: DVec3,
    pub cells: (CellId, CellId),
    pub interface: InterfaceId,
}

/// Clean cell-complex geometry of one component (shared vertex table).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ComponentGeometry {
    pub verts: Vec<DVec3>,
    pub ext_polys: Vec<ExtPoly>,
    pub patches: Vec<Patch>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Component {
    pub id: ComponentId,
    pub name: String,
    pub role: ComponentRole,
    pub material: MaterialId,
    /// Clean manifold solid (asset frame).
    pub solid: TriMesh,
    pub surface: SurfaceAttributes,
    /// True when the solid was reconstructed (non-watertight input).
    pub reconstructed: bool,
    /// The original render surface when it differs from `solid`.
    pub render_surface: Option<TriMesh>,
    pub geometry: ComponentGeometry,
    /// Cells of this component (global id range).
    pub cells: Range<u32>,
    /// Analysis cells of this component (global id range).
    pub analysis_cells: Range<u32>,
    pub volume: f64,
    pub aabb: Aabb,
    /// Grain direction (unit) for anisotropic materials.
    pub grain: Option<DVec3>,
    /// True if the component was passed through unfractured (failure).
    pub unfractured: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cell {
    pub id: CellId,
    pub component: ComponentId,
    /// Global analysis-cell index this fine cell belongs to.
    pub analysis_cell: u32,
    pub material: MaterialId,
    pub mass: MassProps,
    pub aabb: Aabb,
    /// Thickness ratio (min/max principal extent) for sliver checks.
    pub thickness_ratio: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnalysisCell {
    pub id: u32,
    pub component: ComponentId,
    pub cells: Vec<CellId>,
    pub mass: MassProps,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Interface {
    pub id: InterfaceId,
    /// Ordered (low, high); `World` for anchor interfaces.
    pub cells: (CellId, CellOrWorld),
    /// Clean polygons, normals oriented from `cells.0` towards `cells.1`.
    pub polygons: Vec<Polygon3>,
    pub kind: InterfaceKind,
    pub interface_material: Option<MaterialId>,
    pub rebar: RebarCrossing,
    pub area: f64,
    /// Component-local patch ids (empty for cross-component/anchor interfaces).
    pub patches: Vec<u32>,
    /// Rebar crossing points and bar diameters (for render stubs).
    pub rebar_points: Vec<(DVec3, f64)>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RenderRefs {
    pub gltf_node: i32,
    pub lod_meshes: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Fragment {
    pub id: FragmentId,
    pub level: u8,
    pub component: ComponentId,
    pub parent: Option<FragmentId>,
    pub children: Range<u32>,
    /// Range into `Hierarchy::cell_order`.
    pub cells: Range<u32>,
    pub material_mix: SmallVec<[(MaterialId, f32); 4]>,
    pub mass: MassProps,
    pub hulls: Range<u32>,
    pub render: RenderRefs,
    pub particle_candidate: bool,
    pub role: ComponentRole,
    /// Interior (fracture) surface area of this fragment, for debris budgets.
    pub interior_area: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Hierarchy {
    pub levels: u8,
    /// Sorted by (level, id); `fragments[i].id == i`.
    pub fragments: Vec<Fragment>,
    /// Fragment index range per level.
    pub level_ranges: Vec<Range<u32>>,
    /// Cells ordered so every fragment's cells are contiguous at every level.
    pub cell_order: Vec<CellId>,
    /// For each cell, its fragment at each level: `cell_fragment[level][cell]`.
    pub cell_fragment: Vec<Vec<FragmentId>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bond {
    pub id: BondId,
    pub level: u8,
    pub a: FragmentId,
    pub b: FragmentOrWorld,
    pub area: f64,
    pub centroid: DVec3,
    /// Oriented from fragment A to fragment B.
    pub normal: DVec3,
    pub planarity: f32,
    pub frame_u: DVec3,
    pub frame_v: DVec3,
    pub i_uu: f64,
    pub i_vv: f64,
    pub i_uv: f64,
    pub j: f64,
    pub extent: Obb2,
    pub dist_a: f64,
    pub dist_b: f64,
    pub composition: SmallVec<[BondComposition; 4]>,
    pub reinforcement: RebarCrossing,
    pub strength_scale: f32,
    pub parent_bond: Option<BondId>,
    pub child_bonds: Range<u32>,
    pub boundary_loops: Range<u32>,
    pub spawn: Range<u32>,
    pub anchor: bool,
    /// Interfaces aggregated into this bond.
    pub interfaces: Vec<InterfaceId>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Hull {
    pub fragment: FragmentId,
    pub vertices: Vec<DVec3>,
    /// Polygon faces (CCW outward), indices into `vertices`.
    pub faces: Vec<Vec<u32>>,
}

/// A crack spawn sample: position and normal on an interface.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpawnPoint {
    pub p: DVec3,
    pub n: DVec3,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AssetMeta {
    pub name: String,
    pub seed: u64,
    pub variant: u32,
    pub tool_version: String,
    pub settings_hash: String,
    pub material_library_id: String,
    pub material_library_version: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Asset {
    pub meta: AssetMeta,
    pub components: Vec<Component>,
    pub cells: Vec<Cell>,
    pub analysis_cells: Vec<AnalysisCell>,
    pub interfaces: Vec<Interface>,
    pub hierarchy: Hierarchy,
    /// All levels, sorted by (level, a, b).
    pub bonds: Vec<Bond>,
    pub hulls: Vec<Hull>,
    /// `Bond::child_bonds` ranges index into this list.
    pub bond_children: Vec<BondId>,
    pub loops: Vec<Vec<DVec3>>,
    pub spawn: Vec<SpawnPoint>,
    /// Per-level structural (L1) cut-interface diagnostics.
    pub diagnostics: AssetDiagnostics,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AssetDiagnostics {
    /// Per component: method used for level 1 ("modes", "agglomeration", "single").
    pub level1_method: Vec<String>,
    /// Per analysis-cell-pair max jump (when modes ran): (component, a, b, jump).
    pub mode_jumps: Vec<(u32, u32, u32, f64)>,
    pub warnings: Vec<String>,
}

impl Asset {
    pub fn fragment(&self, id: FragmentId) -> &Fragment {
        &self.hierarchy.fragments[id.idx()]
    }
    pub fn level_fragments(&self, level: u8) -> &[Fragment] {
        let r = self.hierarchy.level_ranges[level as usize].clone();
        &self.hierarchy.fragments[r.start as usize..r.end as usize]
    }
    pub fn level_bonds(&self, level: u8) -> impl Iterator<Item = &Bond> {
        self.bonds.iter().filter(move |b| b.level == level)
    }
    pub fn fragment_cells(&self, f: &Fragment) -> &[CellId] {
        &self.hierarchy.cell_order[f.cells.start as usize..f.cells.end as usize]
    }

    /// Per-cell polygon index (so the boundary of a few cells never scans
    /// a whole component).
    pub fn cell_polys(&self) -> CellPolys {
        let n = self.cells.len();
        let mut ext = vec![Vec::new(); n];
        let mut patch = vec![Vec::new(); n];
        for comp in &self.components {
            let g = &comp.geometry;
            for (i, e) in g.ext_polys.iter().enumerate() {
                ext[e.cell.idx()].push(i as u32);
            }
            for (i, p) in g.patches.iter().enumerate() {
                patch[p.cells.0.idx()].push(i as u32);
                if p.cells.1 != p.cells.0 {
                    patch[p.cells.1.idx()].push(i as u32);
                }
            }
        }
        CellPolys { ext, patch }
    }
}

/// Indices into the owning component's `ext_polys` and `patches`, per cell.
#[derive(Clone, Debug, Default)]
pub struct CellPolys {
    pub ext: Vec<Vec<u32>>,
    pub patch: Vec<Vec<u32>>,
}

impl CellPolys {
    /// Exterior polygons of a cell set and its boundary patches
    /// (`(patch, flipped)`: flipped when the set holds the patch's second
    /// cell), both sorted. `inside` must answer membership in the set.
    pub fn boundary_of(&self, cells: &[CellId], inside: impl Fn(CellId) -> bool, geom: &ComponentGeometry) -> (Vec<usize>, Vec<(usize, bool)>) {
        let mut exts = Vec::new();
        let mut pats = Vec::new();
        for &c in cells {
            exts.extend(self.ext[c.idx()].iter().map(|&i| i as usize));
            for &pi in &self.patch[c.idx()] {
                let pt = &geom.patches[pi as usize];
                let (a, b) = (inside(pt.cells.0), inside(pt.cells.1));
                if a != b {
                    pats.push((pi as usize, b));
                }
            }
        }
        exts.sort_unstable();
        exts.dedup();
        pats.sort_unstable();
        pats.dedup();
        (exts, pats)
    }
}

/// Inertia tensor helper: zero matrix (glam's default is identity!).
pub const ZERO_MAT3: DMat3 = DMat3::ZERO;
