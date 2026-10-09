//! Timing benchmark (ignored by default):
//! `cargo test -p frac-modes --test benchmark -- --ignored --nocapture`
//! Set `FRAC_MODES_BENCH_CLARABEL=1` to also time the reference solver.

mod common;
use common::*;
use frac_modes::*;

#[test]
#[ignore]
fn benchmark_notched_bar_k10() {
    let solid = notched_bar();
    for (h, dims) in [(0.2, [16usize, 2, 2]), (0.15, [16, 4, 4])] {
        let t = std::time::Instant::now();
        let c = case(&solid, h, [0.0, 0.0, 0.0], [4.0, 1.0, 1.0], dims);
        let t_mesh = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("== h {h} cells {:?}: {} tets, {} verts, mesh {t_mesh:.0} ms", dims, c.mesh.tets.len(), c.mesh.verts.len());
        let mut solvers = vec![Solver::Auto];
        if std::env::var_os("FRAC_MODES_BENCH_CLARABEL").is_some() {
            solvers.push(Solver::Clarabel);
        }
        for s in solvers {
            let t = std::time::Instant::now();
            let out = run(&c, &|_, _| 1.0, &[], ModesParams { k: 10, solver: s, ..Default::default() });
            let mj = out.max_jump();
            let l1 = segment_level1(c.n_cells, &out.groups, &mj, 2);
            eprintln!(
                "   solver {} dofs {} groups {} total {:.1} s; level1 n={} sigma={:.3}",
                out.solver_used,
                out.n_dofs,
                out.groups.len(),
                t.elapsed().as_secs_f64(),
                l1.n_fragments,
                l1.sigma
            );
        }
    }
}
