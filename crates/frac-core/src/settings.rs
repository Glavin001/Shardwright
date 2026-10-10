//! Bake settings (`bake.toml`, spec §15). Every field has a default so a
//! partial file is valid.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub seed: u64,
    pub variants: u32,
    /// Number of hierarchy levels (L0..L{levels-1}); 4 = component,
    /// structural, analysis, fine.
    pub levels: u8,
    /// Ground plane height (Y up). Faces at or below it become anchors.
    pub ground_height: Option<f64>,
    pub ingest: IngestSettings,
    pub cells: CellSettings,
    pub modes: ModeSettings,
    pub hierarchy: HierarchySettings,
    pub bonds: BondSettings,
    pub collision: CollisionSettings,
    pub render: RenderSettings,
    pub validation: ValidationSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            seed: 1337,
            variants: 1,
            levels: 4,
            ground_height: None,
            ingest: IngestSettings::default(),
            cells: CellSettings::default(),
            modes: ModeSettings::default(),
            hierarchy: HierarchySettings::default(),
            bonds: BondSettings::default(),
            collision: CollisionSettings::default(),
            render: RenderSettings::default(),
            validation: ValidationSettings::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct IngestSettings {
    /// Unit scale applied on import (to meters).
    pub unit_scale: f64,
    /// Convert Z-up sources to Y-up.
    pub z_up: bool,
    /// Contact detection tolerance between components (m).
    pub contact_tolerance: f64,
    /// Max cos angle for contact faces to be considered opposed (normals
    /// dot product must be below `-contact_cos`).
    pub contact_cos: f64,
    /// Grid resolution (cells along the longest axis) for winding-number
    /// solidification of non-watertight parts.
    pub solidify_resolution: u32,
    /// Weld tolerance relative to the part diagonal.
    pub weld_tolerance: f64,
}

impl Default for IngestSettings {
    fn default() -> Self {
        IngestSettings {
            unit_scale: 1.0,
            z_up: false,
            contact_tolerance: 1e-3,
            contact_cos: 0.95,
            solidify_resolution: 96,
            weld_tolerance: 1e-9,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct CellSettings {
    /// Analysis-cell density (per m³); per-material overrides allowed in the
    /// material library (`analysis_cells_per_m3`).
    pub target_analysis_cells_per_m3: f64,
    /// Fine cells per analysis cell.
    pub fine_per_analysis: u32,
    /// Lower and upper caps on the number of analysis cells per component.
    pub min_analysis_cells: u32,
    pub max_analysis_cells: u32,
    /// Hard cap on fine cells per component.
    pub max_fine_cells: u32,
    pub min_cell_volume: f64,
    pub min_thickness_ratio: f64,
    /// Relative seed jitter used to break exact symmetries (fraction of
    /// the local spacing).
    pub seed_jitter: f64,
}

impl Default for CellSettings {
    fn default() -> Self {
        CellSettings {
            target_analysis_cells_per_m3: 200.0,
            fine_per_analysis: 8,
            min_analysis_cells: 4,
            max_analysis_cells: 2000,
            max_fine_cells: 200_000,
            min_cell_volume: 1.0e-7,
            min_thickness_ratio: 0.05,
            seed_jitter: 1e-6,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ModeSettings {
    pub enabled: bool,
    pub k: usize,
    pub omega: f64,
    /// Multiply the geometric interface weight by the fracture-energy weight
    /// `sqrt(G_f / G_ref)` of the interface material.
    pub material_aware: bool,
    /// Fracture-mode model (Sellán et al., "Breaking Good", ACM TOG 2023):
    /// "translational" (default; the paper's §3.6 model: one displacement
    /// per analysis cell, translations only, ICCM started from
    /// vector-Laplacian eigenvectors), "p1" (linear-elastic P1 cell-exploded
    /// field: full space for small problems, cell-affine reduction above
    /// `large_problem_dofs`), "full" or "cell-p1" (force either P1 variant).
    pub discretization: String,
    /// Geometric interface weight `sqrt(A_g / mean A)`, which makes the cut
    /// cost proportional to the interface area times the jump (the
    /// discontinuity measure of the authors' reference implementation).
    /// `false`: unit geometric weights (the spec's `w_g = 1`).
    pub area_weighting: bool,
    /// ICCM multi-start over degenerate initial eigenspaces (keeps the
    /// lowest-energy mode; matters for symmetric parts).
    pub multi_start: bool,
    pub target_level1_fragments: u32,
    pub max_iccm_iters: usize,
    pub iccm_tolerance: f64,
    /// Modes problems with more unknowns than this (e.g. masonry walls with
    /// ~1000+ analysis cells) are solved by ADMM alone, without the
    /// interior-point confirmation solves whose cost grows superlinearly,
    /// and ICCM stops at `max(iccm_tolerance, large_iccm_tolerance)` with
    /// subproblems certified to a tenth of that.
    pub large_problem_dofs: usize,
    /// ICCM tolerance for problems above `large_problem_dofs`.
    pub large_iccm_tolerance: f64,
    pub tet_edge_ratio: f64,
    /// Upper bound on analysis tets per component (resolution is coarsened
    /// to respect it).
    pub max_tets: usize,
    /// Conic solver: "auto", "clarabel" or "admm".
    pub solver: String,
    /// Optional external fTetWild binary for the analysis mesh.
    pub ftetwild: Option<String>,
}

impl Default for ModeSettings {
    fn default() -> Self {
        ModeSettings {
            enabled: true,
            k: 10,
            omega: 1.0e-3,
            material_aware: true,
            discretization: "translational".into(),
            area_weighting: true,
            multi_start: true,
            target_level1_fragments: 12,
            max_iccm_iters: 50,
            iccm_tolerance: 1e-4,
            large_problem_dofs: 3000,
            large_iccm_tolerance: 1e-3,
            tet_edge_ratio: 0.33,
            max_tets: 20_000,
            solver: "auto".into(),
            ftetwild: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct HierarchySettings {
    /// Optional agglomeration target for L2 (0 = keep analysis cells).
    pub level2_target: u32,
    /// Compactness penalty weight in agglomeration (method B).
    pub compactness: f64,
}

impl Default for HierarchySettings {
    fn default() -> Self {
        HierarchySettings {
            level2_target: 0,
            compactness: 0.5,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct BondSettings {
    /// Douglas–Peucker tolerance for boundary loops, relative to sqrt(area).
    pub loop_simplify: f64,
    /// Crack spawn samples per m² of interface (capped per bond).
    pub spawn_density: f64,
    pub max_spawn_per_bond: u32,
}

impl Default for BondSettings {
    fn default() -> Self {
        BondSettings {
            loop_simplify: 0.02,
            spawn_density: 400.0,
            max_spawn_per_bond: 64,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct CollisionSettings {
    /// "coacd" (built-in collision-aware decomposition, CoACD-style) or
    /// "cell" (one hull per cell, then merge).
    pub method: String,
    /// Concavity threshold as a fraction of fragment diameter.
    pub concavity: f64,
    pub max_hulls_per_fragment: u32,
    pub margin: f64,
    pub min_rigid_size: f64,
    /// Levels that get collision hulls (empty = all).
    pub levels: Vec<u8>,
    /// CoACD tree-search effort for cutting non-convex cells. Upstream
    /// defaults are 150 iterations, depth 3, 20 planes per axis and 2000
    /// samples per unit normalized area; the bake defaults (12 iterations,
    /// depth 1) are ~20x cheaper with no measurable loss after merging (see
    /// `crates/frac-collision/README.md`).
    pub mcts_iterations: u32,
    pub mcts_depth: u32,
    pub mcts_nodes: u32,
    pub resolution: u32,
    /// Maximum vertices per convex hull (physics-engine limit); larger hulls
    /// are reduced (inner approximation by a volume-greedy vertex subset).
    pub max_hull_vertices: u32,
}

impl Default for CollisionSettings {
    fn default() -> Self {
        CollisionSettings {
            method: "coacd".into(),
            concavity: 0.02,
            max_hulls_per_fragment: 8,
            margin: 0.0005,
            min_rigid_size: 0.01,
            levels: Vec::new(),
            mcts_iterations: 12,
            mcts_depth: 1,
            mcts_nodes: 20,
            resolution: 2000,
            max_hull_vertices: 64,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct RenderSettings {
    pub lods: u32,
    pub noise: bool,
    pub chipping: bool,
    pub interior_uv: String,
    /// Target interior sample spacing relative to noise min wavelength.
    pub interior_resolution: f64,
    /// Max interior triangles per interface patch.
    pub max_interior_tris_per_patch: u32,
    /// Max triangles per fragment per LOD (0 = unlimited); LODn target is
    /// LOD0 * lod_ratio^n.
    pub triangle_budget: u32,
    pub lod_ratio: f64,
    pub meshopt_compression: bool,
    pub rebar_stubs: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        RenderSettings {
            lods: 3,
            noise: true,
            chipping: true,
            interior_uv: "triplanar_asset_space".into(),
            interior_resolution: 4.0,
            max_interior_tris_per_patch: 128,
            triangle_budget: 0,
            lod_ratio: 0.5,
            meshopt_compression: true,
            rebar_stubs: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ValidationSettings {
    pub gates: String,
    pub metrics: Vec<String>,
    /// Rays per axis for independent mass-property integration.
    pub mass_rays: u32,
    /// Hard limit on vertices per collision hull (physics-engine friendly
    /// compound convex shapes); enforced by the `collision_shapes` gate.
    pub max_hull_vertices: u32,
}

impl Default for ValidationSettings {
    fn default() -> Self {
        ValidationSettings {
            gates: "all".into(),
            metrics: vec![
                "bond_fidelity".into(),
                "distributions".into(),
                "collision".into(),
                "render".into(),
            ],
            mass_rays: 160,
            max_hull_vertices: 64,
        }
    }
}

impl Settings {
    pub fn from_toml(s: &str) -> Result<Settings, crate::FracError> {
        toml::from_str(s)
            .map_err(|e| crate::FracError::new(crate::Stage::Config, "bake.toml", e.to_string()))
    }
    /// Canonical hash of the settings (JSON with sorted keys).
    pub fn hash(&self) -> String {
        let v = serde_json::to_value(self).expect("settings serialize");
        crate::stable_hash_hex(serde_json::to_string(&v).unwrap().as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_spec_excerpt() {
        let s = r#"
seed = 1337
variants = 1
levels = 4

[cells]
target_analysis_cells_per_m3 = 200
fine_per_analysis = 8
min_cell_volume = 1.0e-7
min_thickness_ratio = 0.05

[modes]
enabled = true
k = 10
omega = 1.0e-3
material_aware = true
target_level1_fragments = 12
max_iccm_iters = 50
tet_edge_ratio = 0.33

[collision]
method = "coacd"
concavity = 0.02
max_hulls_per_fragment = 8
margin = 0.0005
min_rigid_size = 0.01

[render]
lods = 3
noise = true
chipping = true
interior_uv = "triplanar_asset_space"

[validation]
gates = "all"
metrics = ["bond_fidelity", "crack_fscore", "distributions", "collision", "render"]
"#;
        let st = Settings::from_toml(s).unwrap();
        assert_eq!(st.modes.k, 10);
        assert_eq!(st.cells.fine_per_analysis, 8);
        assert_eq!(st.hash(), Settings::from_toml(s).unwrap().hash());
    }
}
