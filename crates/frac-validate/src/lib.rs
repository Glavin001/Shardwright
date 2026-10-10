//! Validation (spec §13): hard gates, continuous metrics and the reference
//! bond-network solver used for bond-fidelity comparisons against FEM.

pub mod gates;
pub mod metrics;
pub mod network;

use frac_core::settings::ValidationSettings;
use frac_core::Asset;
use frac_material::MaterialLibrary;
use frac_render::RenderOut;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum GateStatus {
    Pass,
    Fail,
    NotEvaluated,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateResult {
    pub name: String,
    pub status: GateStatus,
    /// Worst observed value (gate-specific units) and the threshold.
    pub value: f64,
    pub threshold: f64,
    pub detail: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Scorecard {
    pub gates: Vec<GateResult>,
    pub metrics: serde_json::Value,
}

impl Scorecard {
    pub fn all_pass(&self) -> bool {
        self.gates.iter().all(|g| g.status != GateStatus::Fail)
    }
    pub fn failures(&self) -> Vec<&GateResult> {
        self.gates.iter().filter(|g| g.status == GateStatus::Fail).collect()
    }
    pub fn to_markdown(&self) -> String {
        let mut s = String::from("## Hard gates\n\n| Gate | Status | Worst value | Threshold | Detail |\n|---|---|---|---|---|\n");
        for g in &self.gates {
            let st = match g.status {
                GateStatus::Pass => "PASS",
                GateStatus::Fail => "**FAIL**",
                GateStatus::NotEvaluated => "n/a",
            };
            s += &format!("| {} | {} | {:.3e} | {:.1e} | {} |\n", g.name, st, g.value, g.threshold, g.detail);
        }
        s += "\n## Metrics\n\n```json\n";
        s += &serde_json::to_string_pretty(&self.metrics).unwrap();
        s += "\n```\n";
        s
    }
}

/// Run hard gates and metrics on a baked asset.
pub fn validate(asset: &Asset, render: &RenderOut, lib: &MaterialLibrary, vs: &ValidationSettings, physics: &[u8]) -> Scorecard {
    let mut gates = gates::run_gates(asset, render, vs, physics);
    gates.sort_by_key(|g| gates::order(&g.name));
    let metrics = metrics::compute(asset, render, lib);
    gates::log_step("metrics");
    Scorecard { gates, metrics }
}
