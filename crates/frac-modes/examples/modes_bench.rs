//! Offline rerun of a dumped fracture-modes problem (see `frac_modes::dump`).
//!
//! ```text
//! FRAC_MODES_DUMP=/tmp/d prefracture bake ...           # writes component_<i>.modes.txt
//! cargo run --release -p frac-modes --example modes_bench -- /tmp/d/component_0.modes.txt \
//!     [--ref labels.txt] [--out prefix] [--k K] [--iters N] [--eps E]
//!     [--solver auto|clarabel|admm] [--disc p0|p1|full|auto] [--large-dofs N]
//!     [--multi-start 0|1] [--area-weighted 0|1]
//! ```
//!
//! Prints timings and the Level-1 segmentation; `--out` writes
//! `<prefix>.labels.txt` (analysis-cell labels) and `<prefix>.jumps.txt`;
//! `--ref` reports the adjusted Rand index against reference labels.

use frac_modes::dump::ModesDump;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut path = None;
    let mut reference = None;
    let mut out = None;
    let (mut k, mut iters, mut eps) = (None, None, None);
    let mut overrides: Vec<(String, String)> = Vec::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--ref" => {
                reference = Some(args[i + 1].clone());
                i += 1;
            }
            "--k" => {
                k = Some(args[i + 1].parse::<usize>().unwrap());
                i += 1;
            }
            "--iters" => {
                iters = Some(args[i + 1].parse::<usize>().unwrap());
                i += 1;
            }
            "--eps" => {
                eps = Some(args[i + 1].parse::<f64>().unwrap());
                i += 1;
            }
            "--solver" | "--disc" | "--large-dofs" | "--multi-start" | "--area-weighted" => {
                overrides.push((args[i].clone(), args[i + 1].clone()));
                i += 1;
            }
            "--out" => {
                out = Some(args[i + 1].clone());
                i += 1;
            }
            p => path = Some(p.to_string()),
        }
        i += 1;
    }
    let path = path.expect("usage: modes_bench <dump> [--ref labels] [--out prefix]");
    let mut d = ModesDump::from_text(&std::fs::read_to_string(&path).expect("read dump")).expect("parse dump");
    if let Some(k) = k {
        d.params.k = k;
    }
    if let Some(it) = iters {
        d.params.max_iters = it;
    }
    if let Some(e) = eps {
        d.params.eps = e;
    }
    for (key, v) in &overrides {
        match key.as_str() {
            "--solver" => {
                d.params.solver = match v.as_str() {
                    "clarabel" => frac_modes::Solver::Clarabel,
                    "admm" => frac_modes::Solver::Admm,
                    _ => frac_modes::Solver::Auto,
                }
            }
            "--disc" => {
                d.params.discretization = match v.as_str() {
                    "auto" => None,
                    "full" => Some(frac_modes::Discretization::Full),
                    p => Some(frac_modes::Discretization::CellPolynomial(p.trim_start_matches('p').parse().expect("--disc"))),
                }
            }
            "--large-dofs" => d.params.large_dofs = v.parse().expect("--large-dofs"),
            "--multi-start" => d.params.multi_start = v == "1",
            "--area-weighted" => d.params.area_weighted = v == "1",
            _ => unreachable!(),
        }
    }
    let w = d.weight_fn();
    let input = frac_modes::ModesInput {
        mesh: &d.mesh,
        tet_material: &d.tet_material,
        tet_cell: &d.tet_cell,
        group_weight: &w,
        anchored_vertices: &d.anchored_vertices,
        params: d.params,
    };
    eprintln!(
        "tets {} verts {} cells {} groups {} target {}",
        d.mesh.tets.len(),
        d.mesh.verts.len(),
        d.n_analysis,
        d.adjacency.len(),
        d.target
    );
    let t = Instant::now();
    let res = frac_modes::compute_modes(&input).expect("modes");
    let secs = t.elapsed().as_secs_f64();
    // same segmentation input as the pipeline (frac-pipeline level1.rs)
    let adj_pairs: Vec<(u32, u32)> = d.adjacency.iter().map(|&(a, b, _, _)| (a, b)).collect();
    let (seg_groups, mj) = match res.pair_max_jump(&adj_pairs) {
        Some(mj) => (adj_pairs, mj),
        None => (res.groups.clone(), res.max_jump()),
    };
    let (l1, _, mjx) = frac_modes::segment_from_jumps(d.n_analysis, &seg_groups, &mj, &d.adjacency, &d.cell_volume, d.target, d.min_volume);
    let timings: Vec<String> = res.timings_ms.iter().map(|(k, v)| format!("{k}={:.0}ms", v)).collect();
    println!(
        "time {secs:.2}s n {} solver {} iters {:?} conv {}/{} frags {} sigma {:.4e}",
        res.n_dofs,
        res.solver_used,
        res.iterations,
        res.converged.iter().filter(|&&c| c).count(),
        res.converged.len(),
        l1.n_fragments,
        l1.sigma
    );
    println!("timings {}", timings.join(" "));
    println!("energies {:?}", res.energies.iter().map(|e| format!("{e:.5e}")).collect::<Vec<_>>());
    if let Some(r) = &reference {
        let text = std::fs::read_to_string(r).expect("read ref");
        let ref_labels: Vec<u32> = text.split_whitespace().map(|x| x.parse().unwrap()).collect();
        println!("ARI vs ref {:.4}", frac_modes::adjusted_rand_index(&ref_labels, &l1.labels));
    }
    if let Some(o) = &out {
        let lt: String = l1.labels.iter().map(|l| format!("{l}\n")).collect();
        std::fs::write(format!("{o}.labels.txt"), lt).unwrap();
        let jt: String = mjx.iter().map(|j| format!("{j:.17e}\n")).collect();
        std::fs::write(format!("{o}.jumps.txt"), jt).unwrap();
        let mut et = String::new();
        for m in &res.jumps {
            et.push_str(&m.iter().map(|j| format!("{j:.6e}")).collect::<Vec<_>>().join(" "));
            et.push('\n');
        }
        std::fs::write(format!("{o}.modejumps.txt"), et).unwrap();
        let gt: String = res.groups.iter().map(|(a, b)| format!("{a} {b}\n")).collect();
        std::fs::write(format!("{o}.groups.txt"), gt).unwrap();
    }
}
