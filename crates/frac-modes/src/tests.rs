//! Differential tests: Clarabel (reference) vs ADMM on identical subproblems.

use super::*;
use frac_fem::{tetrahedralize, ElasticMaterial};
use frac_geom::mesh::box_mesh;
use frac_geom::DVec3;

struct Fixture {
    mesh: frac_fem::TetMesh,
    cells: Vec<u32>,
    mats: Vec<ElasticMaterial>,
}

fn fixture(h: f64, dims: [usize; 3]) -> Fixture {
    let ext = [2.0, 1.0, 0.6];
    let solid = box_mesh(DVec3::ZERO, DVec3::from_array(ext));
    let mesh = tetrahedralize(&solid, h, 0);
    let cells = (0..mesh.tets.len())
        .map(|t| {
            let c = mesh.tet_centroid(t);
            let mut id = [0usize; 3];
            for k in 0..3 {
                id[k] = ((c[k] / ext[k] * dims[k] as f64).floor().max(0.0) as usize).min(dims[k] - 1);
            }
            (id[0] + dims[0] * (id[1] + dims[1] * id[2])) as u32
        })
        .collect();
    let mats = vec![ElasticMaterial::isotropic(1e9, 0.3, 1000.0); mesh.tets.len()];
    Fixture { mesh, cells, mats }
}

fn compare_subproblem(f: &Fixture, anchors: &[u32], w: &(dyn Fn(u32, u32) -> f64 + Sync), mode: usize) {
    let input = ModesInput {
        mesh: &f.mesh,
        tet_material: &f.mats,
        tet_cell: &f.cells,
        group_weight: w,
        anchored_vertices: anchors,
        params: ModesParams { k: mode + 1, ..Default::default() },
    };
    let mut t = Vec::new();
    let pb = problem::build(&input, &mut t).unwrap();
    let n = pb.n;
    let mut c = pb.init[mode].clone();
    let rigid: Vec<Vec<f64>> = pb.rigid_rows.iter().map(|r| (0..n).map(|i| r[i] / pb.m[i]).collect()).collect();
    orthogonalize(&mut c, &rigid, &pb.m);
    let nc = mdot(&c, &c, &pb.m).sqrt();
    c.iter_mut().for_each(|x| *x /= nc);
    let cur: Vec<f64> = (0..n).map(|i| pb.m[i] * c[i]).collect();
    let persistent: Vec<&[f64]> = pb.rigid_rows.iter().map(|r| r.as_slice()).collect();
    let mut rows = persistent.clone();
    rows.push(&cur);
    let mut rhs = vec![0.0; rows.len()];
    *rhs.last_mut().unwrap() = 1.0;
    let ref_sol = clarabel_solver::solve(&pb, &rows, &rhs).unwrap();
    assert!(ref_sol.ok);
    let mut admm = admm::Admm::new(&pb).unwrap();
    admm.warm_from(&pb, &c);
    let fast = admm.solve(&pb, &persistent, &cur, &rhs, ADMM_TOL_MIN).unwrap();
    assert!(fast.ok, "ADMM did not converge");
    let e_ref = pb.objective(&ref_sol.u);
    let e_fast = pb.objective(&fast.u);
    let rel = (e_fast - e_ref).abs() / e_ref.abs();
    // both satisfy the equality constraints
    for (k, r) in rows.iter().enumerate() {
        let a: f64 = r.iter().zip(&fast.u).map(|(x, y)| x * y).sum();
        let b: f64 = r.iter().zip(&ref_sol.u).map(|(x, y)| x * y).sum();
        assert!((a - rhs[k]).abs() < 1e-8 && (b - rhs[k]).abs() < 1e-6, "constraint {k}: {a} {b}");
    }
    let j_ref = pb.group_norms(&ref_sol.u);
    let j_fast = pb.group_norms(&fast.u);
    let jmax = j_ref.iter().cloned().fold(0.0, f64::max);
    let jerr = j_ref.iter().zip(&j_fast).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let du = {
        let mut d = 0.0;
        for i in 0..n {
            d += (ref_sol.u[i] - fast.u[i]).powi(2) * pb.m[i];
        }
        d.sqrt()
    };
    eprintln!(
        "n {n} groups {} active {}: E_ref {e_ref:.8e} E_admm {e_fast:.8e} rel {rel:.2e}; max jump {jmax:.4e} err {jerr:.2e}; |du|_M {du:.2e}; admm iters {} clarabel iters {}",
        pb.groups.len(),
        pb.active.len(),
        fast.iterations,
        ref_sol.iterations
    );
    assert!(rel < 1e-3, "objective mismatch {rel}");
    assert!(jerr < 1e-2 * jmax.max(1e-12), "jump mismatch {jerr} vs {jmax}");
}

#[test]
fn clarabel_vs_admm_unanchored() {
    let f = fixture(0.25, [4, 2, 1]);
    compare_subproblem(&f, &[], &|_, _| 1.0, 0);
    compare_subproblem(&f, &[], &|_, _| 1.0, 2);
}

#[test]
fn clarabel_vs_admm_anchored_weighted() {
    let f = fixture(0.25, [4, 2, 2]);
    let anchors: Vec<u32> = (0..f.mesh.verts.len() as u32).filter(|&v| f.mesh.verts[v as usize][0] < 1e-9).collect();
    // heterogeneous weights incl. a free (0) and a forbidden (inf) interface
    let w = |a: u32, b: u32| match (a, b) {
        (0, 1) => f64::INFINITY,
        (5, 6) => 0.0,
        _ => 0.5 + 0.1 * ((a * 7 + b * 3) % 5) as f64,
    };
    compare_subproblem(&f, &anchors, &w, 0);
    compare_subproblem(&f, &anchors, &w, 1);
}

#[test]
fn full_modes_clarabel_vs_admm() {
    let f = fixture(0.3, [4, 2, 1]);
    let run = |solver| {
        let input = ModesInput {
            mesh: &f.mesh,
            tet_material: &f.mats,
            tet_cell: &f.cells,
            group_weight: &|_, _| 1.0,
            anchored_vertices: &[],
            params: ModesParams { k: 3, solver, ..Default::default() },
        };
        compute_modes(&input).unwrap()
    };
    let a = run(Solver::Clarabel);
    let b = run(Solver::Admm);
    assert_eq!(a.solver_used, "clarabel");
    assert!(b.solver_used.starts_with("admm"));
    eprintln!("energies clarabel {:?} admm {:?}", a.energies, b.energies);
    eprintln!("iters clarabel {:?} admm {:?}", a.iterations, b.iterations);
    for i in 0..3 {
        let rel = (a.energies[i] - b.energies[i]).abs() / a.energies[i];
        assert!(rel < 5e-3, "mode {i}: energy rel diff {rel}");
        let jmax = a.jumps[i].iter().cloned().fold(0.0, f64::max);
        for g in 0..a.groups.len() {
            assert!((a.jumps[i][g] - b.jumps[i][g]).abs() < 0.05 * jmax, "mode {i} group {g}");
        }
    }
}

#[test]
#[ignore]
fn rho_sweep() {
    let f = fixture(0.25, [4, 2, 1]);
    let input = ModesInput {
        mesh: &f.mesh,
        tet_material: &f.mats,
        tet_cell: &f.cells,
        group_weight: &|_, _| 1.0,
        anchored_vertices: &[],
        params: ModesParams { k: 1, ..Default::default() },
    };
    let mut t = Vec::new();
    let pb = problem::build(&input, &mut t).unwrap();
    let n = pb.n;
    let mut c = pb.init[0].clone();
    let rigid: Vec<Vec<f64>> = pb.rigid_rows.iter().map(|r| (0..n).map(|i| r[i] / pb.m[i]).collect()).collect();
    orthogonalize(&mut c, &rigid, &pb.m);
    let nc = mdot(&c, &c, &pb.m).sqrt();
    c.iter_mut().for_each(|x| *x /= nc);
    let cur: Vec<f64> = (0..n).map(|i| pb.m[i] * c[i]).collect();
    let persistent: Vec<&[f64]> = pb.rigid_rows.iter().map(|r| r.as_slice()).collect();
    let mut rhs = vec![0.0; persistent.len() + 1];
    *rhs.last_mut().unwrap() = 1.0;
    let base = admm::Admm::new(&pb).unwrap().rho;
    for alpha in [1.0, 1.6, 1.8] {
        for f in [1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0] {
            let mut a = admm::Admm::with_rho(&pb, Some(base * f)).unwrap();
            a.settings.adaptive = false;
            a.settings.alpha = alpha;
            a.warm_from(&pb, &c);
            let r = a.solve(&pb, &persistent, &cur, &rhs, 1e-6).unwrap();
            eprintln!("alpha {alpha} rho {:.3e} (x{f}) iters {} obj {:.8e}", base * f, r.iterations, pb.objective(&r.u));
        }
    }
}
