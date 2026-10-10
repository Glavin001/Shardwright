//! Runs the fracture-mode analysis on a given tetrahedral mesh and writes the
//! result as JSON. Used by the reference-comparison harness
//! (`tools/harness/fracture_modes_ref.py`, see docs/VALIDATION.md).
//!
//! ```sh
//! cargo run --release -p frac-modes --example modes_from_mesh -- in.json out.json
//! ```
//!
//! Input (all keys but the mesh are optional):
//!
//! ```json
//! { "verts": [[x, y, z], ...], "tets": [[a, b, c, d], ...],
//!   "tet_cell": [0, 1, ...],          // default: every tet its own cell
//!   "anchors": [vertex, ...],         // default: free body
//!   "group_weight": "uniform",        // or "sqrt_area": w_g = sqrt(A_g / mean A)
//!   "material": {"youngs": 2e11, "poisson": 0.3, "density": 7850},
//!   "discretizations": ["full", "cell-p1"],
//!   "level1_targets": [2, 3],
//!   "params": {"k": 4, "omega": 1e-3, "eps": 1e-4, "max_iters": 50,
//!              "solver": "auto" | "clarabel" | "admm", "seed": 1592652460} }
//! ```
//!
//! Instead of `verts`/`tets` the input may give a closed surface
//! `"solid": {"verts": [...], "tris": [...]}` with `"h"` (target edge) and
//! optional `"max_tets"`; it is tetrahedralized with
//! `frac_fem::tetrahedralize` and the mesh is echoed in the output, so the
//! caller can feed exactly the same `(V, T)` to another implementation.
//! `"modes": false` skips the analysis (mesh generation only).
//!
//! Output: `{"mesh": {...}?, "runs": [{"discretization", "groups",
//! "group_area", "jumps", "energies", "eigenvalues", "iterations",
//! "converged", "n_dofs", "solver_used", "timings_ms", "level1": [...]}]}`.

#![allow(unexpected_cfgs)]

use frac_fem::{ElasticMaterial, TetMesh};
use frac_geom::{DVec3, TriMesh};
use frac_modes::{compute_modes_with, segment_level1, Discretization, ModesInput, ModesOutput, ModesParams, Solver};
use std::collections::BTreeMap;
use std::fmt::Write as _;

// ---------------------------------------------------------------- JSON ----

#[derive(Clone, Debug)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    fn get(&self, k: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.get(k),
            _ => None,
        }
    }
    fn num(&self) -> Result<f64, String> {
        match self {
            Json::Num(x) => Ok(*x),
            _ => Err(format!("expected a number, got {self:?}")),
        }
    }
    fn arr(&self) -> Result<&[Json], String> {
        match self {
            Json::Arr(a) => Ok(a),
            _ => Err("expected an array".into()),
        }
    }
    fn str(&self) -> Result<&str, String> {
        match self {
            Json::Str(s) => Ok(s),
            _ => Err("expected a string".into()),
        }
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn expect(&mut self, c: u8) -> Result<(), String> {
        self.ws();
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", c as char, self.i))
        }
    }
    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.s.get(self.i) {
            None => Err("unexpected end of input".into()),
            Some(b'{') => {
                self.i += 1;
                let mut m = BTreeMap::new();
                self.ws();
                if self.s.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(m));
                }
                loop {
                    self.ws();
                    let k = match self.value()? {
                        Json::Str(s) => s,
                        _ => return Err(format!("object key expected at byte {}", self.i)),
                    };
                    self.expect(b':')?;
                    let v = self.value()?;
                    m.insert(k, v);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(m));
                        }
                        _ => return Err(format!("',' or '}}' expected at byte {}", self.i)),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Arr(a));
                }
                loop {
                    a.push(self.value()?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(a));
                        }
                        _ => return Err(format!("',' or ']' expected at byte {}", self.i)),
                    }
                }
            }
            Some(b'"') => {
                self.i += 1;
                let mut out = String::new();
                loop {
                    match self.s.get(self.i) {
                        None => return Err("unterminated string".into()),
                        Some(b'"') => {
                            self.i += 1;
                            return Ok(Json::Str(out));
                        }
                        Some(b'\\') => {
                            let c = *self.s.get(self.i + 1).ok_or("bad escape")?;
                            self.i += 2;
                            match c {
                                b'n' => out.push('\n'),
                                b't' => out.push('\t'),
                                b'u' => {
                                    let h = std::str::from_utf8(&self.s[self.i..self.i + 4]).map_err(|e| e.to_string())?;
                                    let cp = u32::from_str_radix(h, 16).map_err(|e| e.to_string())?;
                                    out.push(char::from_u32(cp).unwrap_or('?'));
                                    self.i += 4;
                                }
                                c => out.push(c as char),
                            }
                        }
                        Some(_) => {
                            let start = self.i;
                            while self.i < self.s.len() && self.s[self.i] != b'"' && self.s[self.i] != b'\\' {
                                self.i += 1;
                            }
                            out.push_str(std::str::from_utf8(&self.s[start..self.i]).map_err(|e| e.to_string())?);
                        }
                    }
                }
            }
            Some(b't') if self.s[self.i..].starts_with(b"true") => {
                self.i += 4;
                Ok(Json::Bool(true))
            }
            Some(b'f') if self.s[self.i..].starts_with(b"false") => {
                self.i += 5;
                Ok(Json::Bool(false))
            }
            Some(b'n') if self.s[self.i..].starts_with(b"null") => {
                self.i += 4;
                Ok(Json::Null)
            }
            Some(_) => {
                let start = self.i;
                while self.i < self.s.len() && matches!(self.s[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
                    self.i += 1;
                }
                let t = std::str::from_utf8(&self.s[start..self.i]).map_err(|e| e.to_string())?;
                t.parse::<f64>().map(Json::Num).map_err(|_| format!("bad number '{t}' at byte {start}"))
            }
        }
    }
}

fn parse_json(text: &str) -> Result<Json, String> {
    let mut p = Parser { s: text.as_bytes(), i: 0 };
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("trailing characters at byte {}", p.i));
    }
    Ok(v)
}

/// Minimal JSON writer.
struct Out(String);

impl Out {
    fn num(&mut self, x: f64) {
        if x.is_finite() {
            // shortest round-trip representation
            let _ = write!(self.0, "{x:?}");
        } else {
            self.0.push_str("null");
        }
    }
    fn nums(&mut self, xs: impl IntoIterator<Item = f64>) {
        self.0.push('[');
        for (i, x) in xs.into_iter().enumerate() {
            if i > 0 {
                self.0.push(',');
            }
            self.num(x);
        }
        self.0.push(']');
    }
    fn key(&mut self, k: &str, first: bool) {
        if !first {
            self.0.push(',');
        }
        let _ = write!(self.0, "\"{k}\":");
    }
}

// ------------------------------------------------------------- helpers ----

fn usize_list(j: &Json) -> Result<Vec<u32>, String> {
    j.arr()?.iter().map(|v| v.num().map(|x| x as u32)).collect()
}

fn vec3_list(j: &Json) -> Result<Vec<[f64; 3]>, String> {
    j.arr()?
        .iter()
        .map(|p| {
            let a = p.arr()?;
            if a.len() != 3 {
                return Err("expected [x, y, z]".into());
            }
            Ok([a[0].num()?, a[1].num()?, a[2].num()?])
        })
        .collect()
}

fn idx_list<const N: usize>(j: &Json) -> Result<Vec<[u32; N]>, String> {
    j.arr()?
        .iter()
        .map(|p| {
            let a = p.arr()?;
            if a.len() != N {
                return Err(format!("expected {N} indices"));
            }
            let mut r = [0u32; N];
            for k in 0..N {
                r[k] = a[k].num()? as u32;
            }
            Ok(r)
        })
        .collect()
}

/// Fault-face area per analysis-cell pair `(a < b)` (same faces as the
/// library's groups).
fn cell_pair_areas(mesh: &TetMesh, cells: &[u32]) -> BTreeMap<(u32, u32), f64> {
    let mut acc = BTreeMap::new();
    let f = mesh.sorted_faces();
    let mut i = 0;
    while i < f.len() {
        let mut j = i + 1;
        while j < f.len() && f[j].0 == f[i].0 {
            j += 1;
        }
        if j - i == 2 {
            let (a, b) = (cells[f[i].1 as usize], cells[f[i + 1].1 as usize]);
            if a != b {
                let p = f[i].0.map(|v| mesh.verts[v as usize]);
                let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
                let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
                let c = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
                let ar = 0.5 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                *acc.entry((a.min(b), a.max(b))).or_insert(0.0) += ar;
            }
        }
        i = j;
    }
    acc
}

fn write_run(o: &mut Out, name: &str, out: &ModesOutput, n_cells: u32, targets: &[u32]) {
    o.0.push('{');
    o.key("discretization", true);
    let _ = write!(o.0, "\"{name}\"");
    o.key("groups", false);
    o.0.push('[');
    for (i, &(a, b)) in out.groups.iter().enumerate() {
        if i > 0 {
            o.0.push(',');
        }
        let _ = write!(o.0, "[{a},{b}]");
    }
    o.0.push(']');
    o.key("group_area", false);
    o.nums(out.group_area.iter().copied());
    o.key("jumps", false);
    o.0.push('[');
    for (i, j) in out.jumps.iter().enumerate() {
        if i > 0 {
            o.0.push(',');
        }
        o.nums(j.iter().copied());
    }
    o.0.push(']');
    o.key("energies", false);
    o.nums(out.energies.iter().copied());
    o.key("eigenvalues", false);
    o.nums(out.eigenvalues.iter().copied());
    o.key("iterations", false);
    o.nums(out.iterations.iter().map(|&x| x as f64));
    o.key("converged", false);
    let _ = write!(o.0, "[{}]", out.converged.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(","));
    o.key("n_dofs", false);
    let _ = write!(o.0, "{}", out.n_dofs);
    o.key("solver_used", false);
    let _ = write!(o.0, "\"{}\"", out.solver_used);
    o.key("timings_ms", false);
    o.0.push('{');
    for (i, (k, v)) in out.timings_ms.iter().enumerate() {
        o.key(k, i == 0);
        o.num(*v);
    }
    o.0.push('}');
    o.key("level1", false);
    o.0.push('[');
    let mj = out.max_jump();
    for (i, &t) in targets.iter().enumerate() {
        if i > 0 {
            o.0.push(',');
        }
        let l1 = segment_level1(n_cells, &out.groups, &mj, t);
        let _ = write!(o.0, "{{\"target\":{t},\"n_fragments\":{},\"hit_target\":{},\"sigma\":", l1.n_fragments, l1.hit_target);
        o.num(l1.sigma);
        o.key("labels", false);
        o.nums(l1.labels.iter().map(|&x| x as f64));
        o.0.push('}');
    }
    o.0.push(']');
    write_fields(o, out);
    o.0.push('}');
}

/// Mode displacement fields per exploded node. Only available in builds of
/// the library that expose them (the harness builds a patched copy with
/// `--cfg frac_modes_fields`, see tools/harness/patches/).
#[cfg(frac_modes_fields)]
fn write_fields(o: &mut Out, out: &ModesOutput) {
    o.key("nodes", false);
    o.0.push('[');
    for (i, &(v, c)) in out.nodes.iter().enumerate() {
        if i > 0 {
            o.0.push(',');
        }
        let _ = write!(o.0, "[{v},{c}]");
    }
    o.0.push(']');
    o.key("mode_fields", false);
    o.0.push('[');
    for (i, f) in out.mode_fields.iter().enumerate() {
        if i > 0 {
            o.0.push(',');
        }
        o.nums(f.iter().flat_map(|p| p.iter().copied()));
    }
    o.0.push(']');
}

#[cfg(not(frac_modes_fields))]
fn write_fields(_o: &mut Out, _out: &ModesOutput) {}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err(format!("usage: {} IN.json OUT.json", args[0]));
    }
    let text = std::fs::read_to_string(&args[1]).map_err(|e| format!("{}: {e}", args[1]))?;
    let inp = parse_json(&text)?;
    let mut o = Out(String::from("{"));
    let mut first = true;

    let mesh = if let Some(solid) = inp.get("solid") {
        let verts: Vec<DVec3> = vec3_list(solid.get("verts").ok_or("solid.verts missing")?)?
            .into_iter()
            .map(DVec3::from_array)
            .collect();
        let tris = idx_list::<3>(solid.get("tris").ok_or("solid.tris missing")?)?;
        let h = inp.get("h").ok_or("'h' is required with 'solid'")?.num()?;
        let max_tets = inp.get("max_tets").map(|v| v.num()).transpose()?.unwrap_or(0.0) as usize;
        let m = frac_fem::tetrahedralize(&TriMesh { verts, tris }, h, max_tets);
        if m.tets.is_empty() {
            return Err("tetrahedralization produced no tets".into());
        }
        o.key("mesh", first);
        first = false;
        o.0.push_str("{\"verts\":[");
        for (i, p) in m.verts.iter().enumerate() {
            if i > 0 {
                o.0.push(',');
            }
            o.nums(p.iter().copied());
        }
        o.0.push_str("],\"tets\":[");
        for (i, t) in m.tets.iter().enumerate() {
            if i > 0 {
                o.0.push(',');
            }
            let _ = write!(o.0, "[{},{},{},{}]", t[0], t[1], t[2], t[3]);
        }
        o.0.push_str("]}");
        m
    } else {
        TetMesh {
            verts: vec3_list(inp.get("verts").ok_or("verts missing")?)?,
            tets: idx_list::<4>(inp.get("tets").ok_or("tets missing")?)?,
        }
    };
    let nt = mesh.tets.len();

    let run_modes = !matches!(inp.get("modes"), Some(Json::Bool(false)));
    if run_modes {
        let tet_cell = match inp.get("tet_cell") {
            Some(j) => usize_list(j)?,
            None => (0..nt as u32).collect(),
        };
        if tet_cell.len() != nt {
            return Err("tet_cell must have one entry per tet".into());
        }
        let n_cells = tet_cell.iter().copied().max().unwrap_or(0) + 1;
        let anchors = match inp.get("anchors") {
            Some(j) => usize_list(j)?,
            None => Vec::new(),
        };
        let mat = match inp.get("material") {
            Some(m) => ElasticMaterial::isotropic(
                m.get("youngs").map(|v| v.num()).transpose()?.unwrap_or(200e9),
                m.get("poisson").map(|v| v.num()).transpose()?.unwrap_or(0.3),
                m.get("density").map(|v| v.num()).transpose()?.unwrap_or(7850.0),
            ),
            None => ElasticMaterial::isotropic(200e9, 0.3, 7850.0),
        };
        let mats = vec![mat; nt];

        let mut params = ModesParams::default();
        if let Some(p) = inp.get("params") {
            if let Some(v) = p.get("k") {
                params.k = v.num()? as usize;
            }
            if let Some(v) = p.get("omega") {
                params.omega = v.num()?;
            }
            if let Some(v) = p.get("eps") {
                params.eps = v.num()?;
            }
            if let Some(v) = p.get("max_iters") {
                params.max_iters = v.num()? as usize;
            }
            if let Some(v) = p.get("seed") {
                params.seed = v.num()? as u64;
            }
            if let Some(v) = p.get("solver") {
                params.solver = match v.str()? {
                    "auto" => Solver::Auto,
                    "clarabel" => Solver::Clarabel,
                    "admm" => Solver::Admm,
                    s => return Err(format!("unknown solver '{s}'")),
                };
            }
        }

        // geometric group weights (no material): uniform (spec w_g = 1) or
        // sqrt(A_g / mean A), which turns the per-group penalty
        // ω·w_g·‖B̂_g u‖ = ω·w_g·√(A_g/L²)·rms into one proportional to A_g·rms
        let mode = inp.get("group_weight").map(|v| v.str()).transpose()?.unwrap_or("uniform").to_string();
        let areas = cell_pair_areas(&mesh, &tet_cell);
        let mean_area = areas.values().sum::<f64>() / areas.len().max(1) as f64;
        let weight = move |a: u32, b: u32| -> f64 {
            match mode.as_str() {
                "sqrt_area" => (areas.get(&(a, b)).copied().unwrap_or(0.0) / mean_area).sqrt(),
                _ => 1.0,
            }
        };
        if !matches!(inp.get("group_weight").map(|v| v.str()).transpose()?, None | Some("uniform") | Some("sqrt_area")) {
            return Err("group_weight must be \"uniform\" or \"sqrt_area\"".into());
        }

        let discs: Vec<String> = match inp.get("discretizations") {
            Some(j) => j.arr()?.iter().map(|v| v.str().map(str::to_string)).collect::<Result<_, _>>()?,
            None => vec!["full".into()],
        };
        let targets = match inp.get("level1_targets") {
            Some(j) => usize_list(j)?,
            None => vec![2],
        };

        let input = ModesInput {
            mesh: &mesh,
            tet_material: &mats,
            tet_cell: &tet_cell,
            group_weight: &weight,
            anchored_vertices: &anchors,
            params,
        };
        o.key("runs", first);
        o.0.push('[');
        for (i, name) in discs.iter().enumerate() {
            let disc = match name.as_str() {
                "full" => Discretization::Full,
                // per-cell translations (the paper's §3.6 space); builds
                // without degree-0 support clamp it to degree 1, which shows
                // in `solver_used`
                "cell-p0" => Discretization::CellPolynomial(0),
                "cell-p1" => Discretization::CellPolynomial(1),
                "cell-p2" => Discretization::CellPolynomial(2),
                s => return Err(format!("unknown discretization '{s}'")),
            };
            let out = compute_modes_with(&input, disc)?;
            eprintln!(
                "[{name}] tets {nt} cells {n_cells} groups {} dofs {} solver {} iters {:?} converged {:?}",
                out.groups.len(),
                out.n_dofs,
                out.solver_used,
                out.iterations,
                out.converged
            );
            if i > 0 {
                o.0.push(',');
            }
            write_run(&mut o, name, &out, n_cells, &targets);
        }
        o.0.push(']');
    }
    o.0.push('}');
    std::fs::write(&args[2], o.0).map_err(|e| format!("{}: {e}", args[2]))?;
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
