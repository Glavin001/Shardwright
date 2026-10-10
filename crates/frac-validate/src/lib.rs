//! Validation (spec §13): hard gates, continuous metrics and the reference
//! bond-network solver used for bond-fidelity comparisons against FEM.

pub mod gates;
pub mod metrics;
pub mod network;
pub mod structure;

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
    gates.extend(structure_gates(asset, lib));
    gates.sort_by_key(|g| gates::order(&g.name));
    let metrics = metrics::compute(asset, render, lib);
    gates::log_step("metrics");
    Scorecard { gates, metrics }
}

/// `structural_support` and `self_weight` (see [`structure`]); not evaluated
/// for assets without ground anchors.
fn structure_gates(asset: &Asset, lib: &MaterialLibrary) -> Vec<GateResult> {
    let na = |name: &str| GateResult { name: name.into(), status: GateStatus::NotEvaluated, value: 0.0, threshold: 0.0, detail: "no ground anchors".into() };
    let sup = structure::support(asset);
    if !sup.applicable {
        return vec![na("structural_support"), na("self_weight")];
    }
    let bad = sup.unsupported_structural.len() + sup.unsupported_other.len();
    let list = |v: &[String]| v.iter().take(5).cloned().collect::<Vec<_>>().join(", ");
    let support = GateResult {
        name: "structural_support".into(),
        status: if bad == 0 { GateStatus::Pass } else { GateStatus::Fail },
        value: bad as f64,
        threshold: 0.0,
        detail: format!(
            "{} structural components without a structural load path to ground [{}]; {} other components unsupported [{}]",
            sup.unsupported_structural.len(),
            list(&sup.unsupported_structural),
            sup.unsupported_other.len(),
            list(&sup.unsupported_other)
        ),
    };
    gates::log_step("structural_support");
    let joints = structure::self_weight(asset, lib);
    let worst = joints.iter().max_by(|a, b| a.utilization.total_cmp(&b.utilization));
    let over = joints.iter().filter(|j| j.utilization > 1.0).count();
    let umax = worst.map(|w| w.utilization).unwrap_or(0.0);
    let sw = GateResult {
        name: "self_weight".into(),
        status: if over == 0 { GateStatus::Pass } else { GateStatus::Fail },
        value: umax,
        threshold: 1.0,
        detail: match worst {
            Some(w) => format!("{} joints; max utilization {:.3} ({} {:?} joint {} – {}); {over} joints over capacity", joints.len(), w.utilization, w.mode, w.kind, w.a, w.b),
            None => "no inter-component joints".into(),
        },
    };
    gates::log_step("self_weight");
    vec![support, sw]
}
