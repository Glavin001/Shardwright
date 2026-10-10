//! Bake report (JSON + Markdown).

use frac_core::*;
use frac_render::RenderOut;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StageTiming {
    pub stage: String,
    pub ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LevelStats {
    pub level: u8,
    pub fragments: usize,
    pub bonds: usize,
    pub anchor_bonds: usize,
    pub hulls: usize,
    pub tris_lod0: usize,
    pub volume_p10: f64,
    pub volume_p50: f64,
    pub volume_p90: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub name: String,
    pub tool_version: String,
    pub settings_hash: String,
    pub seed: u64,
    pub variant: u32,
    pub components: usize,
    pub cells: usize,
    pub analysis_cells: usize,
    pub interfaces: usize,
    pub levels: Vec<LevelStats>,
    pub level1_methods: Vec<String>,
    pub timings: Vec<StageTiming>,
    pub total_ms: f64,
    pub payload_bytes: (usize, usize),
    pub warnings: Vec<String>,
    pub scorecard: frac_validate::Scorecard,
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}

impl Report {
    pub fn new(
        asset: &Asset,
        render: &RenderOut,
        timings: Vec<StageTiming>,
        warnings: Vec<String>,
        scorecard: frac_validate::Scorecard,
    ) -> Report {
        let h = &asset.hierarchy;
        let levels = (0..h.levels)
            .map(|l| {
                let fr = asset.level_fragments(l);
                let mut vols: Vec<f64> = fr.iter().map(|f| f.mass.volume).collect();
                LevelStats {
                    level: l,
                    fragments: fr.len(),
                    bonds: asset.level_bonds(l).count(),
                    anchor_bonds: asset.level_bonds(l).filter(|b| b.anchor).count(),
                    hulls: fr.iter().map(|f| f.hulls.len()).sum(),
                    tris_lod0: fr
                        .iter()
                        .map(|f| {
                            render.fragments[f.id.idx()]
                                .first()
                                .map(|m| m.triangle_count())
                                .unwrap_or(0)
                        })
                        .sum(),
                    volume_p10: pct(&mut vols, 0.1),
                    volume_p50: pct(&mut vols, 0.5),
                    volume_p90: pct(&mut vols, 0.9),
                }
            })
            .collect();
        let mut warnings = warnings;
        warnings.extend(asset.diagnostics.warnings.iter().cloned());
        Report {
            name: asset.meta.name.clone(),
            tool_version: asset.meta.tool_version.clone(),
            settings_hash: asset.meta.settings_hash.clone(),
            seed: asset.meta.seed,
            variant: asset.meta.variant,
            components: asset.components.len(),
            cells: asset.cells.len(),
            analysis_cells: asset.analysis_cells.len(),
            interfaces: asset.interfaces.len(),
            levels,
            level1_methods: asset.diagnostics.level1_method.clone(),
            timings,
            total_ms: 0.0,
            payload_bytes: (0, 0),
            warnings,
            scorecard,
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap()
    }

    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        s += &format!("# Pre-fracture report: {}\n\n", self.name);
        s += &format!(
            "tool {} · seed {} · variant {} · settings `{}`\n\n",
            self.tool_version,
            self.seed,
            self.variant,
            &self.settings_hash[..16.min(self.settings_hash.len())]
        );
        s += &format!(
            "Components {} · cells {} · analysis cells {} · interfaces {} · total {:.1} s · glb {} KiB · fracphys {} KiB\n\n",
            self.components,
            self.cells,
            self.analysis_cells,
            self.interfaces,
            self.total_ms / 1e3,
            self.payload_bytes.0 / 1024,
            self.payload_bytes.1 / 1024
        );
        s += "## Levels\n\n| Level | Fragments | Bonds | Anchor bonds | Hulls | Tris (LOD0) | Volume p10 / p50 / p90 (m³) |\n|---|---|---|---|---|---|---|\n";
        for l in &self.levels {
            s += &format!(
                "| L{} | {} | {} | {} | {} | {} | {:.3e} / {:.3e} / {:.3e} |\n",
                l.level,
                l.fragments,
                l.bonds,
                l.anchor_bonds,
                l.hulls,
                l.tris_lod0,
                l.volume_p10,
                l.volume_p50,
                l.volume_p90
            );
        }
        s += &format!(
            "\nLevel-1 method per component: {}\n\n",
            self.level1_methods.join(", ")
        );
        s += &self.scorecard.to_markdown();
        s += "\n## Timings\n\n| Stage | ms |\n|---|---|\n";
        for t in &self.timings {
            s += &format!("| {} | {:.1} |\n", t.stage, t.ms);
        }
        if !self.warnings.is_empty() {
            s += "\n## Warnings\n\n";
            for w in &self.warnings {
                s += &format!("- {w}\n");
            }
        }
        s
    }
}
