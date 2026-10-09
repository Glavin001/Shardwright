//! Determinism and solver-agreement tests through the public API.

mod common;
use common::*;
use frac_geom::DVec3;
use frac_modes::*;

fn small_case() -> Case {
    let solid = frac_geom::mesh::box_mesh(DVec3::ZERO, DVec3::new(2.0, 1.0, 0.6));
    case(&solid, 0.3, [0.0, 0.0, 0.0], [2.0, 1.0, 0.6], [4, 2, 1])
}

fn strip(mut o: ModesOutput) -> ModesOutput {
    o.timings_ms.clear();
    o
}

fn bits(o: &ModesOutput) -> Vec<u64> {
    let mut v: Vec<u64> = o.jumps.iter().flatten().map(|x| x.to_bits()).collect();
    v.extend(o.energies.iter().map(|x| x.to_bits()));
    v.extend(o.eigenvalues.iter().map(|x| x.to_bits()));
    v.extend(o.group_area.iter().map(|x| x.to_bits()));
    v
}

#[test]
fn compute_modes_is_bitwise_deterministic() {
    let c = small_case();
    let w = |a: u32, b: u32| if (a + b) % 3 == 0 { 0.5 } else { 1.0 };
    for solver in [Solver::Admm, Solver::Clarabel] {
        let p = ModesParams { k: 3, solver, ..Default::default() };
        let a = strip(run(&c, &w, &[], p));
        let b = strip(run(&c, &w, &[], p));
        assert_eq!(bits(&a), bits(&b), "{solver:?}");
        assert_eq!(a, b, "{solver:?}");
    }
    // the mesher is deterministic too
    let c2 = small_case();
    assert_eq!(c.mesh, c2.mesh);
}

#[test]
fn auto_picks_clarabel_for_small_and_agrees_with_admm() {
    let c = small_case();
    let auto = run(&c, &|_, _| 1.0, &[], ModesParams { k: 2, ..Default::default() });
    assert!(auto.n_dofs < AUTO_CLARABEL_MAX_DOFS);
    assert_eq!(auto.solver_used, "clarabel");
    let fast = run(&c, &|_, _| 1.0, &[], ModesParams { k: 2, solver: Solver::Admm, ..Default::default() });
    assert!(fast.solver_used.starts_with("admm"));
    for i in 0..2 {
        let rel = (auto.energies[i] - fast.energies[i]).abs() / auto.energies[i];
        assert!(rel < 1e-3, "mode {i}: energies {} vs {} ({rel})", auto.energies[i], fast.energies[i]);
        let jmax = auto.jumps[i].iter().cloned().fold(0.0, f64::max);
        for g in 0..auto.groups.len() {
            let d = (auto.jumps[i][g] - fast.jumps[i][g]).abs();
            assert!(d < 0.02 * jmax, "mode {i} group {g}: {} vs {}", auto.jumps[i][g], fast.jumps[i][g]);
        }
    }
}

#[test]
fn jumps_are_normalized_rms() {
    // consistency of reported quantities
    let c = small_case();
    let out = run(&c, &|_, _| 1.0, &[], ModesParams { k: 2, solver: Solver::Admm, ..Default::default() });
    assert_eq!(out.jumps.len(), 2);
    assert_eq!(out.groups.len(), out.group_area.len());
    assert!(out.groups.windows(2).all(|w| w[0] < w[1]));
    assert!(out.groups.iter().all(|&(a, b)| a < b));
    // total fault area equals the sum over groups
    let total: f64 = out.group_area.iter().sum();
    // 4 x 2 x 1 cells of a 2 x 1 x 0.6 box: 3 planes of 1 x 0.6 and one of 2 x 0.6 (staircased)
    assert!(total > 0.9 * 3.0 && total < 1.6 * 3.0, "total fault area {total}");
    assert!(out.eigenvalues[0] > 0.0);
    assert!(out.energies.iter().all(|&e| e > 0.0 && e.is_finite()));
}
