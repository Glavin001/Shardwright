//! Imported input scene (produced by `frac-io`, consumed by `frac-ingest`)
//! and authoring metadata (sidecar JSON/TOML or glTF extras, spec §5.3).

use frac_geom::TriMesh;
use glam::DVec3;
use serde::{Deserialize, Serialize};

/// One imported part, already converted to the asset frame (meters, +Y up).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InputPart {
    pub name: String,
    /// Raw triangle soup or indexed mesh (not necessarily welded/closed).
    pub mesh: TriMesh,
    /// Per-triangle-corner attributes (index = triangle).
    pub normals: Option<Vec<[[f32; 3]; 3]>>,
    pub uvs: Option<Vec<[[f32; 2]; 3]>>,
    pub tangents: Option<Vec<[[f32; 4]; 3]>>,
    /// Material slot per triangle (index into `material_names`).
    pub material_slot: Vec<u16>,
    pub material_names: Vec<String>,
    /// Optional painted scalar per mesh vertex (density paint).
    pub vertex_paint: Option<Vec<f32>>,
    /// Source extras (glTF node + mesh extras merged), used for metadata.
    pub extras: serde_json::Value,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InputScene {
    pub source: String,
    pub parts: Vec<InputPart>,
}

/// Axis-aligned or oriented box volume (forbidden zones, region materials).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BoxVolume {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl BoxVolume {
    pub fn contains(&self, p: DVec3) -> bool {
        (0..3).all(|k| p[k] >= self.min[k] && p[k] <= self.max[k])
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RebarSpec {
    /// Polyline points (asset frame).
    pub points: Vec<[f64; 3]>,
    /// Bar diameter (m).
    pub diameter: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConnectionSpec {
    /// Part names.
    pub a: String,
    pub b: String,
    /// weld, bolted, mortar_joint, cold_joint, adhesive, bearing
    pub kind: String,
    #[serde(default)]
    pub interface_material: Option<String>,
}

/// Per-part authoring metadata.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PartMeta {
    pub component_role: Option<String>,
    /// Material ID in the material library.
    pub material: Option<String>,
    /// Map of source material slot name -> library material ID.
    pub material_map: std::collections::BTreeMap<String, String>,
    /// Anchored to the world (whole part, or faces below `anchor_below`).
    pub anchor: bool,
    pub anchor_below: Option<f64>,
    /// Constant grain direction (asset frame).
    pub grain: Option<[f64; 3]>,
    /// Density multiplier boxes: (box, factor) — factor > 1 => finer.
    pub density_boxes: Vec<(BoxVolume, f64)>,
    pub forbidden_zones: Vec<BoxVolume>,
    pub rebar: Vec<RebarSpec>,
    /// Masonry layout (parsed by the masonry recipe).
    pub masonry_layout: Option<serde_json::Value>,
    /// Override recipe name.
    pub recipe: Option<String>,
    /// Group with other parts into one component.
    pub group: Option<String>,
    /// Annealed glass impact center.
    pub impact_center: Option<[f64; 3]>,
}

/// Authoring sidecar (spec §5.3): per-part metadata plus connections.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AuthoringMeta {
    /// Keyed by part name; the key "*" applies to all parts.
    pub parts: std::collections::BTreeMap<String, PartMeta>,
    pub connections: Vec<ConnectionSpec>,
    /// Ground plane height for automatic anchors.
    pub ground_height: Option<f64>,
}

impl AuthoringMeta {
    /// Effective metadata for a part: "*" defaults overlaid by the part's
    /// entry and by glTF extras (`extras.prefracture`).
    pub fn for_part(&self, name: &str, extras: &serde_json::Value) -> PartMeta {
        let mut base = self.parts.get("*").cloned().unwrap_or_default();
        if let Some(e) = extras.get("prefracture") {
            if let Ok(m) = serde_json::from_value::<PartMeta>(e.clone()) {
                base = merge(base, m);
            }
        }
        if let Some(m) = self.parts.get(name) {
            base = merge(base, m.clone());
        }
        base
    }
    pub fn from_json(s: &str) -> Result<Self, crate::FracError> {
        serde_json::from_str(s).map_err(|e| crate::FracError::new(crate::Stage::Config, "metadata", e.to_string()))
    }
    pub fn from_toml(s: &str) -> Result<Self, crate::FracError> {
        toml::from_str(s).map_err(|e| crate::FracError::new(crate::Stage::Config, "metadata", e.to_string()))
    }
}

fn merge(mut a: PartMeta, b: PartMeta) -> PartMeta {
    macro_rules! take {
        ($f:ident) => {
            if b.$f.is_some() {
                a.$f = b.$f;
            }
        };
    }
    take!(component_role);
    take!(material);
    take!(anchor_below);
    take!(grain);
    take!(masonry_layout);
    take!(recipe);
    take!(group);
    take!(impact_center);
    a.anchor |= b.anchor;
    a.material_map.extend(b.material_map);
    a.density_boxes.extend(b.density_boxes);
    a.forbidden_zones.extend(b.forbidden_zones);
    a.rebar.extend(b.rebar);
    a
}
