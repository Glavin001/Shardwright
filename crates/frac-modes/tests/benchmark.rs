//! Timing benchmark (ignored by default):
//! `cargo test -p frac-modes --test benchmark -- --ignored --nocapture`
//! Set `FRAC_MODES_BENCH_FULL=1` to also time the full (unreduced) problem
//! with ADMM (minutes).

mod common;
use common::*;
use frac_modes::*;

#[test]
#[ignore]
fn benchmark_notched_bar_k10() {
    let solid = notched_bar();
    let cfgs: Vec<(f64, [usize; 3])> = if std::env::var_os("BENCH_BIG_ONLY").is_some() {
        vec![(0.15, [16, 4, 4])]
    } else {
        vec![(0.2, [16, 2, 2]), (0.15, [16, 4, 4])]
    };
    for (h, dims) in cfgs {
        let t = std::time::Instant::now();
        let c = case(&solid, h, [0.0, 0.0, 0.0], [4.0, 1.0, 1.0], dims);
        let t_mesh = t.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "== h {h} cells {:?}: {} tets, {} verts, mesh {t_mesh:.0} ms",
            dims,
            c.mesh.tets.len(),
            c.mesh.verts.len()
        );
        let mut variants = vec![(Solver::Auto, None)];
        if std::env::var_os("FRAC_MODES_BENCH_FULL").is_some() {
            variants.push((Solver::Admm, Some(Discretization::Full)));
        }
        for (s, disc) in variants {
            let t = std::time::Instant::now();
            let params = ModesParams {
                k: std::env::var("BENCH_K")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(10),
                solver: s,
                ..Default::default()
            };
            let out = match disc {
                Some(d) => run_disc(&c, &|_, _| 1.0, &[], params, d),
                None => run(&c, &|_, _| 1.0, &[], params),
            };
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
