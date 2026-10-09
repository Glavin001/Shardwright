//! Material library (spec §11): a versioned TOML file shared by this library
//! and the destruction solver. Materials, interface materials (mortar,
//! adhesives, welds) and noise profiles.

use frac_core::{FracError, MaterialId, Stage};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Anisotropy {
    /// Only "grain" is supported: transversely isotropic about the grain.
    #[serde(default = "default_axis")]
    pub axis: String,
    #[serde(rename = "E_long")]
    pub e_long: f64,
    #[serde(rename = "E_trans")]
    pub e_trans: f64,
    #[serde(rename = "G")]
    pub g: f64,
    #[serde(default = "default_nu_trans")]
    pub nu_trans: f64,
    #[serde(default = "default_nu_long")]
    pub nu_long: f64,
}
fn default_axis() -> String {
    "grain".into()
}
fn default_nu_trans() -> f64 {
    0.4
}
fn default_nu_long() -> f64 {
    0.3
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Material {
    pub id: String,
    pub density: f64,
    #[serde(default)]
    pub youngs_modulus: Option<f64>,
    #[serde(default = "default_poisson")]
    pub poisson: f64,
    #[serde(default)]
    pub anisotropy: Option<Anisotropy>,
    #[serde(default)]
    pub tensile_strength: Option<f64>,
    #[serde(default)]
    pub cohesion: Option<f64>,
    #[serde(default)]
    pub friction_angle_deg: Option<f64>,
    #[serde(default)]
    pub compressive_strength: Option<f64>,
    /// Fracture energy G_f (J/m²), isotropic materials.
    #[serde(default)]
    pub fracture_energy: Option<f64>,
    #[serde(default)]
    pub fracture_energy_along_grain: Option<f64>,
    #[serde(default)]
    pub fracture_energy_across_grain: Option<f64>,
    #[serde(default = "default_weibull")]
    pub weibull_modulus: f64,
    #[serde(default)]
    pub dif_curve: Option<String>,
    #[serde(default)]
    pub recipe: Option<String>,
    #[serde(default)]
    pub grain_stretch: Option<f64>,
    #[serde(default)]
    pub noise_profile: Option<String>,
    #[serde(default)]
    pub chipping_ratio: Option<f64>,
    #[serde(default)]
    pub interior_render_material: Option<String>,
    /// Interior render base color (linear RGBA) used when exporting.
    #[serde(default)]
    pub interior_color: Option<[f32; 4]>,
    /// Exterior color fallback when the source has no material.
    #[serde(default)]
    pub exterior_color: Option<[f32; 4]>,
    /// Analysis-cell density override (per m³).
    #[serde(default)]
    pub analysis_cells_per_m3: Option<f64>,
    /// Fine cells per analysis cell override.
    #[serde(default)]
    pub fine_per_analysis: Option<u32>,
    /// Triplanar interior UV scale (UV units per meter).
    #[serde(default)]
    pub interior_uv_scale: Option<f64>,
    /// Default interface kind between units of this material (e.g. masonry
    /// joints) and its interface material.
    #[serde(default)]
    pub joint_interface_material: Option<String>,
    /// Masonry layout (for the masonry recipe).
    #[serde(default)]
    pub masonry_layout: Option<serde_json::Value>,
}
fn default_poisson() -> f64 {
    0.2
}
fn default_weibull() -> f64 {
    8.0
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InterfaceMaterial {
    pub id: String,
    #[serde(default)]
    pub youngs_modulus: Option<f64>,
    #[serde(default)]
    pub shear_modulus: Option<f64>,
    #[serde(default)]
    pub tensile_strength: Option<f64>,
    #[serde(default)]
    pub cohesion: Option<f64>,
    #[serde(default)]
    pub friction_angle_deg: Option<f64>,
    #[serde(default)]
    pub fracture_energy: Option<f64>,
    #[serde(default)]
    pub thickness: Option<f64>,
    #[serde(default = "default_weibull")]
    pub weibull_modulus: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NoiseProfile {
    pub id: String,
    pub amplitude: f64,
    #[serde(default = "default_hurst")]
    pub hurst_exponent: f64,
    pub min_wavelength: f64,
    pub max_wavelength: f64,
    /// Anisotropic stretch along the grain (wood splinters).
    #[serde(default)]
    pub grain_stretch: Option<f64>,
}
fn default_hurst() -> f64 {
    0.8
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MaterialLibrary {
    #[serde(default = "default_lib_id")]
    pub library_id: String,
    pub library_version: String,
    #[serde(default, rename = "material")]
    pub materials: Vec<Material>,
    #[serde(default, rename = "interface_material")]
    pub interface_materials: Vec<InterfaceMaterial>,
    #[serde(default, rename = "noise_profile")]
    pub noise_profiles: Vec<NoiseProfile>,
    /// Reference fracture energy for the material-aware weighting
    /// `w_g = sqrt(G_f / G_ref)`.
    #[serde(default = "default_gref")]
    pub reference_fracture_energy: f64,
}
fn default_lib_id() -> String {
    "default".into()
}
fn default_gref() -> f64 {
    100.0
}

/// Elastic properties used by analysis and validation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Elastic {
    pub youngs: f64,
    pub poisson: f64,
    pub shear: f64,
    pub density: f64,
}

impl MaterialLibrary {
    pub fn from_toml(s: &str) -> Result<Self, FracError> {
        let lib: MaterialLibrary = toml::from_str(s).map_err(|e| FracError::new(Stage::Config, "materials.toml", e.to_string()))?;
        lib.validate()?;
        Ok(lib)
    }

    pub fn validate(&self) -> Result<(), FracError> {
        let err = |m: String| Err(FracError::new(Stage::Config, "materials.toml", m));
        let parts: Vec<&str> = self.library_version.split('.').collect();
        if parts.len() != 3 || parts.iter().any(|p| p.parse::<u64>().is_err()) {
            return err(format!("library_version '{}' is not semver", self.library_version));
        }
        let mut seen = BTreeMap::new();
        for (i, m) in self.materials.iter().enumerate() {
            if seen.insert(m.id.clone(), i).is_some() {
                return err(format!("duplicate material id '{}'", m.id));
            }
            if !(m.density > 0.0) {
                return err(format!("material '{}': density must be > 0", m.id));
            }
            if m.youngs_modulus.is_none() && m.anisotropy.is_none() {
                return err(format!("material '{}': needs youngs_modulus or anisotropy", m.id));
            }
            if !(m.poisson > -1.0 && m.poisson < 0.5) {
                return err(format!("material '{}': poisson out of range", m.id));
            }
            if let Some(np) = &m.noise_profile {
                if self.noise(np).is_none() {
                    return err(format!("material '{}': unknown noise profile '{np}'", m.id));
                }
            }
            if let Some(j) = &m.joint_interface_material {
                if self.interface_index(j).is_none() {
                    return err(format!("material '{}': unknown interface material '{j}'", m.id));
                }
            }
        }
        if self.materials.len() > u16::MAX as usize {
            return err("too many materials".into());
        }
        let mut seen = BTreeMap::new();
        for m in &self.interface_materials {
            if seen.insert(m.id.clone(), ()).is_some() {
                return err(format!("duplicate interface material id '{}'", m.id));
            }
        }
        for n in &self.noise_profiles {
            if !(n.min_wavelength > 0.0 && n.max_wavelength >= n.min_wavelength) {
                return err(format!("noise profile '{}': bad wavelengths", n.id));
            }
        }
        Ok(())
    }

    pub fn material_id(&self, id: &str) -> Option<MaterialId> {
        self.materials.iter().position(|m| m.id == id).map(|i| MaterialId(i as u16))
    }
    pub fn material(&self, id: MaterialId) -> &Material {
        &self.materials[id.idx()]
    }
    /// Interface materials share the id space after materials:
    /// `MaterialId(materials.len() + k)`.
    pub fn interface_index(&self, id: &str) -> Option<usize> {
        self.interface_materials.iter().position(|m| m.id == id)
    }
    pub fn interface_material_id(&self, id: &str) -> Option<MaterialId> {
        self.interface_index(id).map(|k| MaterialId((self.materials.len() + k) as u16))
    }
    pub fn interface_material(&self, id: MaterialId) -> Option<&InterfaceMaterial> {
        id.idx().checked_sub(self.materials.len()).and_then(|k| self.interface_materials.get(k))
    }
    pub fn noise(&self, id: &str) -> Option<&NoiseProfile> {
        self.noise_profiles.iter().find(|n| n.id == id)
    }

    /// Isotropic-equivalent elastic constants (for anisotropic materials the
    /// longitudinal modulus is used for E).
    pub fn elastic(&self, id: MaterialId) -> Elastic {
        let m = self.material(id);
        let (e, g) = match (&m.anisotropy, m.youngs_modulus) {
            (Some(a), _) => (a.e_long, a.g),
            (None, Some(e)) => (e, e / (2.0 * (1.0 + m.poisson))),
            _ => (1e9, 1e9 / 2.4),
        };
        Elastic { youngs: e, poisson: m.poisson, shear: g, density: m.density }
    }

    /// Fracture energy for a crack plane with unit normal `n` in a material
    /// with grain direction `grain` (spec §4.2): along-grain cracks have
    /// normals perpendicular to the grain.
    pub fn fracture_energy(&self, id: MaterialId, n: Option<[f64; 3]>, grain: Option<[f64; 3]>) -> f64 {
        let m = self.material(id);
        match (m.fracture_energy_along_grain, m.fracture_energy_across_grain, n, grain) {
            (Some(ga), Some(gx), Some(n), Some(g)) => {
                let c = (n[0] * g[0] + n[1] * g[1] + n[2] * g[2]).abs().min(1.0);
                // c = 1: crack plane across the grain; c = 0: along the grain
                ga + (gx - ga) * c * c
            }
            (Some(ga), Some(gx), _, _) => 0.5 * (ga + gx),
            _ => m.fracture_energy.unwrap_or(self.reference_fracture_energy),
        }
    }

    pub fn interface_fracture_energy(&self, id: MaterialId) -> f64 {
        self.interface_material(id).and_then(|m| m.fracture_energy).unwrap_or(self.reference_fracture_energy)
    }

    /// Weibull modulus for strength scaling.
    pub fn weibull(&self, id: MaterialId) -> f64 {
        if let Some(im) = self.interface_material(id) {
            return im.weibull_modulus;
        }
        self.materials.get(id.idx()).map(|m| m.weibull_modulus).unwrap_or(8.0)
    }

    /// Canonical identity hash of the library content.
    pub fn content_hash(&self) -> String {
        frac_core::stable_hash_hex(serde_json::to_string(self).unwrap().as_bytes())
    }

    /// The default library shipped with the tool (illustrative values; the
    /// content team owns calibration).
    pub fn builtin() -> Self {
        Self::from_toml(BUILTIN).expect("builtin material library")
    }
}

pub const BUILTIN: &str = include_str!("../materials.toml");

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn builtin_parses_and_validates() {
        let l = MaterialLibrary::builtin();
        let c = l.material_id("concrete_c30").unwrap();
        assert_eq!(l.material(c).density, 2400.0);
        let pine = l.material_id("pine").unwrap();
        let along = l.fracture_energy(pine, Some([0.0, 1.0, 0.0]), Some([1.0, 0.0, 0.0]));
        let across = l.fracture_energy(pine, Some([1.0, 0.0, 0.0]), Some([1.0, 0.0, 0.0]));
        assert!(across > 10.0 * along);
        assert!(l.interface_material_id("mortar_type_n").is_some());
    }
    #[test]
    fn spec_example_parses() {
        let s = r#"
library_version = "1.0.0"

[[material]]
id = "concrete_c30"
density = 2400.0
youngs_modulus = 30.0e9
poisson = 0.2
tensile_strength = 2.9e6
cohesion = 3.5e6
friction_angle_deg = 37.0
compressive_strength = 30.0e6
fracture_energy = 120.0
weibull_modulus = 8.0
dif_curve = "concrete_default"
recipe = "concrete_clustered_voronoi"
noise_profile = "concrete_rough"
chipping_ratio = 0.35
interior_render_material = "concrete_interior"

[[material]]
id = "pine"
density = 500.0
anisotropy = { axis = "grain", E_long = 11.0e9, E_trans = 0.6e9, G = 0.7e9 }
fracture_energy_along_grain = 300.0
fracture_energy_across_grain = 8000.0
recipe = "wood_anisotropic_voronoi"
grain_stretch = 6.0
noise_profile = "wood_splinter"

[[interface_material]]
id = "mortar_type_n"
youngs_modulus = 5.0e9
tensile_strength = 0.3e6
cohesion = 0.4e6
friction_angle_deg = 35.0
fracture_energy = 10.0
thickness = 0.01

[[noise_profile]]
id = "concrete_rough"
amplitude = 0.004
hurst_exponent = 0.8
min_wavelength = 0.002
max_wavelength = 0.08

[[noise_profile]]
id = "wood_splinter"
amplitude = 0.003
min_wavelength = 0.002
max_wavelength = 0.05
"#;
        let l = MaterialLibrary::from_toml(s).unwrap();
        assert_eq!(l.materials.len(), 2);
    }
    #[test]
    fn rejects_bad() {
        assert!(MaterialLibrary::from_toml("library_version = \"1.0\"").is_err());
    }
}
