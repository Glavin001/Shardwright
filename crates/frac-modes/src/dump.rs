//! Exact text dump of a fracture-modes problem (diagnostics and benchmarks).
//!
//! When the environment variable `FRAC_MODES_DUMP` names a directory, the
//! pipeline writes one file per component with everything needed to rerun
//! [`crate::compute_modes`] and the Level-1 segmentation offline (see the
//! `modes_bench` example). Floats are stored as IEEE bit patterns, so a
//! reloaded problem is bit-identical.

use crate::{Discretization, ModesParams, Solver};
use frac_fem::{ElasticMaterial, TetMesh, TransverseIsotropic};
use std::fmt::Write as _;

/// A self-contained fracture-modes problem plus Level-1 segmentation inputs.
#[derive(Clone, Debug)]
pub struct ModesDump {
    pub mesh: TetMesh,
    pub tet_material: Vec<ElasticMaterial>,
    pub tet_cell: Vec<u32>,
    /// Analysis-cell adjacency `(a < b, shared area, w_g)`.
    pub adjacency: Vec<(u32, u32, f64, f64)>,
    pub anchored_vertices: Vec<u32>,
    pub params: ModesParams,
    pub n_analysis: u32,
    pub cell_volume: Vec<f64>,
    pub target: u32,
    pub min_volume: f64,
}

fn f(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

fn disc_name(d: Option<Discretization>) -> String {
    match d {
        None => "auto".into(),
        Some(Discretization::Full) => "full".into(),
        Some(Discretization::CellPolynomial(k)) => format!("p{k}"),
    }
}

fn parse_disc(t: &str) -> Result<Option<Discretization>, String> {
    Ok(match t {
        "auto" => None,
        "full" => Some(Discretization::Full),
        _ => Some(Discretization::CellPolynomial(
            t.strip_prefix('p').and_then(|d| d.parse().ok()).ok_or_else(|| format!("bad discretization {t}"))?,
        )),
    })
}

fn solver_name(s: Solver) -> &'static str {
    match s {
        Solver::Auto => "auto",
        Solver::Clarabel => "clarabel",
        Solver::Admm => "admm",
    }
}

impl ModesDump {
    /// Weight lookup matching the pipeline (`1.0` for pairs not in the adjacency).
    pub fn weight_fn(&self) -> impl Fn(u32, u32) -> f64 + Sync + '_ {
        let map: std::collections::BTreeMap<(u32, u32), f64> =
            self.adjacency.iter().map(|&(a, b, _, w)| ((a, b), w)).collect();
        move |a: u32, b: u32| *map.get(&(a.min(b), a.max(b))).unwrap_or(&1.0)
    }

    pub fn to_text(&self) -> String {
        let mut s = String::new();
        let p = &self.params;
        writeln!(s, "frac-modes-dump 1").unwrap();
        writeln!(
            s,
            "params {} {} {} {} {} {} {} {} {} {} {}",
            p.k,
            f(p.omega),
            f(p.eps),
            p.max_iters,
            solver_name(p.solver),
            p.seed,
            p.large_dofs,
            f(p.eps_large),
            disc_name(p.discretization),
            u8::from(p.area_weighted),
            u8::from(p.multi_start)
        )
        .unwrap();
        writeln!(s, "seg {} {} {}", self.n_analysis, self.target, f(self.min_volume)).unwrap();
        writeln!(s, "verts {}", self.mesh.verts.len()).unwrap();
        for v in &self.mesh.verts {
            writeln!(s, "{} {} {}", f(v[0]), f(v[1]), f(v[2])).unwrap();
        }
        writeln!(s, "tets {}", self.mesh.tets.len()).unwrap();
        for (t, tt) in self.mesh.tets.iter().enumerate() {
            writeln!(s, "{} {} {} {} {}", tt[0], tt[1], tt[2], tt[3], self.tet_cell[t]).unwrap();
        }
        writeln!(s, "materials {}", self.tet_material.len()).unwrap();
        for m in &self.tet_material {
            write!(s, "{} {} {}", f(m.youngs), f(m.poisson), f(m.density)).unwrap();
            if let Some(t) = &m.transverse {
                write!(
                    s,
                    " {} {} {} {} {} {} {} {}",
                    f(t.axis[0]),
                    f(t.axis[1]),
                    f(t.axis[2]),
                    f(t.e_long),
                    f(t.e_trans),
                    f(t.g_long),
                    f(t.nu_trans),
                    f(t.nu_long)
                )
                .unwrap();
            }
            s.push('\n');
        }
        writeln!(s, "adjacency {}", self.adjacency.len()).unwrap();
        for &(a, b, ar, w) in &self.adjacency {
            writeln!(s, "{a} {b} {} {}", f(ar), f(w)).unwrap();
        }
        writeln!(s, "anchored {}", self.anchored_vertices.len()).unwrap();
        for a in &self.anchored_vertices {
            writeln!(s, "{a}").unwrap();
        }
        writeln!(s, "volumes {}", self.cell_volume.len()).unwrap();
        for v in &self.cell_volume {
            writeln!(s, "{}", f(*v)).unwrap();
        }
        s
    }

    pub fn from_text(text: &str) -> Result<ModesDump, String> {
        let mut lines = text.lines();
        let mut next = || lines.next().ok_or_else(|| "truncated dump".to_string());
        let hf = |t: &str| -> Result<f64, String> {
            u64::from_str_radix(t, 16).map(f64::from_bits).map_err(|e| format!("bad float {t}: {e}"))
        };
        let pu = |t: &str| -> Result<u64, String> { t.parse::<u64>().map_err(|e| format!("bad int {t}: {e}")) };
        let header = |line: &str, key: &str| -> Result<usize, String> {
            let mut it = line.split_whitespace();
            if it.next() != Some(key) {
                return Err(format!("expected '{key}', got '{line}'"));
            }
            it.next().ok_or("missing count")?.parse::<usize>().map_err(|e| e.to_string())
        };
        if next()? != "frac-modes-dump 1" {
            return Err("not a frac-modes dump".into());
        }
        let l = next()?;
        let t: Vec<&str> = l.split_whitespace().collect();
        if !(t.len() == 7 || t.len() == 9 || t.len() == 12) || t[0] != "params" {
            return Err("bad params line".into());
        }
        let solver = match t[5] {
            "clarabel" => Solver::Clarabel,
            "admm" => Solver::Admm,
            _ => Solver::Auto,
        };
        let params = ModesParams {
            k: pu(t[1])? as usize,
            omega: hf(t[2])?,
            eps: hf(t[3])?,
            max_iters: pu(t[4])? as usize,
            solver,
            seed: pu(t[6])?,
            large_dofs: if t.len() >= 9 { pu(t[7])? as usize } else { ModesParams::default().large_dofs },
            eps_large: if t.len() >= 9 { hf(t[8])? } else { ModesParams::default().eps_large },
            // dumps before these fields were written by the linear-elastic P1 model
            discretization: if t.len() == 12 { parse_disc(t[9])? } else { None },
            area_weighted: t.len() == 12 && t[10] == "1",
            multi_start: t.len() == 12 && t[11] == "1",
        };
        let l = next()?;
        let t: Vec<&str> = l.split_whitespace().collect();
        if t.len() != 4 || t[0] != "seg" {
            return Err("bad seg line".into());
        }
        let (n_analysis, target, min_volume) = (pu(t[1])? as u32, pu(t[2])? as u32, hf(t[3])?);
        let nv = header(next()?, "verts")?;
        let mut verts = Vec::with_capacity(nv);
        for _ in 0..nv {
            let l = next()?;
            let t: Vec<&str> = l.split_whitespace().collect();
            verts.push([hf(t[0])?, hf(t[1])?, hf(t[2])?]);
        }
        let nt = header(next()?, "tets")?;
        let mut tets = Vec::with_capacity(nt);
        let mut tet_cell = Vec::with_capacity(nt);
        for _ in 0..nt {
            let l = next()?;
            let t: Vec<u32> = l.split_whitespace().map(|x| x.parse::<u32>().map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
            tets.push([t[0], t[1], t[2], t[3]]);
            tet_cell.push(t[4]);
        }
        let nm = header(next()?, "materials")?;
        let mut tet_material = Vec::with_capacity(nm);
        for _ in 0..nm {
            let l = next()?;
            let t: Vec<f64> = l.split_whitespace().map(hf).collect::<Result<_, _>>()?;
            let transverse = if t.len() == 11 {
                Some(TransverseIsotropic {
                    axis: [t[3], t[4], t[5]],
                    e_long: t[6],
                    e_trans: t[7],
                    g_long: t[8],
                    nu_trans: t[9],
                    nu_long: t[10],
                })
            } else {
                None
            };
            tet_material.push(ElasticMaterial { youngs: t[0], poisson: t[1], density: t[2], transverse });
        }
        let na = header(next()?, "adjacency")?;
        let mut adjacency = Vec::with_capacity(na);
        for _ in 0..na {
            let l = next()?;
            let t: Vec<&str> = l.split_whitespace().collect();
            adjacency.push((pu(t[0])? as u32, pu(t[1])? as u32, hf(t[2])?, hf(t[3])?));
        }
        let nanc = header(next()?, "anchored")?;
        let mut anchored_vertices = Vec::with_capacity(nanc);
        for _ in 0..nanc {
            anchored_vertices.push(pu(next()?.trim())? as u32);
        }
        let nvol = header(next()?, "volumes")?;
        let mut cell_volume = Vec::with_capacity(nvol);
        for _ in 0..nvol {
            cell_volume.push(hf(next()?.trim())?);
        }
        Ok(ModesDump {
            mesh: TetMesh { verts, tets },
            tet_material,
            tet_cell,
            adjacency,
            anchored_vertices,
            params,
            n_analysis,
            cell_volume,
            target,
            min_volume,
        })
    }
}
