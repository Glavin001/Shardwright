//! Timing test at the size of the pipeline's RC-column benchmark:
//! 0.4 x 3 x 0.4 m concrete box, 96 analysis cells, ~20k tets, k = 10,
//! anchored bottom, Solver::Auto.

mod common;
use common::*;
use frac_fem::ElasticMaterial;
use frac_geom::DVec3;
use frac_modes::*;

fn column() -> Case {
    let solid = frac_geom::mesh::box_mesh(DVec3::ZERO, DVec3::new(0.4, 3.0, 0.4));
    let mut c = case(&solid, 0.066, [0.0, 0.0, 0.0], [0.4, 3.0, 0.4], [2, 24, 2]);
    c.mats = vec![ElasticMaterial::isotropic(30e9, 0.2, 2400.0); c.mesh.tets.len()];
    c
}

fn anchors(c: &Case) -> Vec<u32> {
    (0..c.mesh.verts.len() as u32).filter(|&v| c.mesh.verts[v as usize][1] < 1e-9).collect()
}

#[test]
fn rc_column_auto_k10_timing() {
    let c = column();
    assert_eq!(c.n_cells, 96);
    assert!(c.mesh.tets.len() > 15_000, "{}", c.mesh.tets.len());
    let a = anchors(&c);
    let t = std::time::Instant::now();
    let out = run(&c, &|_, _| 1.0, &a, ModesParams { k: 10, ..Default::default() });
    let secs = t.elapsed().as_secs_f64();
    let l1 = segment_level1(c.n_cells, &out.groups, &out.max_jump(), 4);
    eprintln!("RC column: {} tets, {} unknowns, {} -> {secs:.2} s, level1 n={}", c.mesh.tets.len(), out.n_dofs, out.solver_used, l1.n_fragments);
    assert!(out.solver_used.starts_with("cell-p"));
    assert_eq!(out.jumps.len(), 10);
    assert!(secs < 60.0, "compute_modes took {secs:.1} s");
}

#[test]
#[ignore]
fn rc_column_variants() {
    let c = column();
    let a = anchors(&c);
    for (disc, solver) in [
        (Discretization::CellPolynomial(1), Solver::Admm),
        (Discretization::CellPolynomial(2), Solver::Admm),
    ] {
        let input = ModesInput {
            mesh: &c.mesh,
            tet_material: &c.mats,
            tet_cell: &c.cells,
            group_weight: &|_, _| 1.0,
            anchored_vertices: &a,
            params: ModesParams { k: 10, solver, ..Default::default() },
        };
        let t = std::time::Instant::now();
        let out = compute_modes_with(&input, disc).unwrap();
        eprintln!("{disc:?} {}: {:.2} s iters {:?} conv {:?} timings {:?}", out.solver_used, t.elapsed().as_secs_f64(), out.iterations, out.converged, out.timings_ms);
    }
}
