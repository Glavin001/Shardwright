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

fn modes_clarabel_vs_admm(disc: Option<Discretization>, prefix: &str) {
    let f = fixture(0.3, [4, 2, 1]);
    let run = |solver| {
        let input = ModesInput {
            mesh: &f.mesh,
            tet_material: &f.mats,
            tet_cell: &f.cells,
            group_weight: &|_, _| 1.0,
            anchored_vertices: &[],
            params: ModesParams { k: 3, solver, discretization: disc, ..Default::default() },
        };
        compute_modes(&input).unwrap()
    };
    let a = run(Solver::Clarabel);
    let b = run(Solver::Admm);
    assert_eq!(a.solver_used, format!("{prefix}clarabel"));
    assert!(b.solver_used.starts_with(&format!("{prefix}admm")), "{}", b.solver_used);
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
fn full_modes_clarabel_vs_admm() {
    // linear-elastic P1 model (size-based choice: full space here)
    modes_clarabel_vs_admm(None, "");
}

#[test]
fn translational_modes_clarabel_vs_admm() {
    // default: the paper's per-cell translation model
    modes_clarabel_vs_admm(Some(Discretization::CellPolynomial(0)), "cell-p0+");
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

#[test]
fn jump_operator_integrates_exactly() {
    // two tets sharing the face (1,2,3), in different cells
    let mut mesh = frac_fem::TetMesh {
        verts: vec![[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.3, 0.1, 0.0], [0.2, 0.9, 0.05], [0.4, 0.3, 1.0]],
        tets: vec![[0, 1, 3, 2], [4, 1, 2, 3]],
    };
    mesh.fix_orientation();
    assert!(mesh.min_volume() > 0.0);
    let mats = vec![ElasticMaterial::isotropic(1e3, 0.3, 1.0); 2];
    let input = ModesInput {
        mesh: &mesh,
        tet_material: &mats,
        tet_cell: &[0, 1],
        group_weight: &|_, _| 1.0,
        anchored_vertices: &[],
        params: ModesParams { k: 1, ..Default::default() },
    };
    let mut t = Vec::new();
    let pb = problem::build(&input, &mut t).unwrap();
    assert_eq!(pb.groups, vec![(0, 1)]);
    // nodes sorted by (vertex, cell): (0,0) (1,0) (1,1) (2,0) (2,1) (3,0) (3,1) (4,1)
    let nodes = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 1)];
    assert_eq!(pb.n, 3 * nodes.len());
    let f = |v: usize, c: usize, k: usize| ((v * 7 + c * 3 + k * 5) % 11) as f64 * 0.1 - 0.4 + c as f64 * 0.05 * k as f64;
    let mut u = vec![0.0; pb.n];
    for (i, &(v, c)) in nodes.iter().enumerate() {
        for k in 0..3 {
            u[3 * i + k] = f(v, c, k);
        }
    }
    let l2 = pb.length_scale * pb.length_scale;
    let got = pb.group_norms(&u)[0].powi(2) * l2;
    // exact: for a linear field on a triangle, ∫ f² = A/6 (f0² + f1² + f2² + f0 f1 + f1 f2 + f2 f0)
    let p = [mesh.verts[1], mesh.verts[2], mesh.verts[3]];
    let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
    let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
    let cr = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    let area = 0.5 * (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
    let mut exact = 0.0;
    for k in 0..3 {
        let j: Vec<f64> = [1, 2, 3].iter().map(|&v| f(v, 0, k) - f(v, 1, k)).collect();
        exact += area / 6.0 * (j[0] * j[0] + j[1] * j[1] + j[2] * j[2] + j[0] * j[1] + j[1] * j[2] + j[2] * j[0]);
    }
    assert!((got - exact).abs() < 1e-12 * exact.max(1e-30), "{got} vs {exact}");
    assert!((pb.group_area[0] - area).abs() < 1e-14);
    // normalization: total mass 1
    let mt: f64 = pb.m.iter().sum::<f64>() / 3.0;
    assert!((mt - 1.0).abs() < 1e-12);
}

#[test]
fn clarabel_vs_admm_reduced_subproblem() {
    let f = fixture(0.25, [4, 2, 2]);
    let input = ModesInput {
        mesh: &f.mesh,
        tet_material: &f.mats,
        tet_cell: &f.cells,
        group_weight: &|_, _| 1.0,
        anchored_vertices: &[],
        params: ModesParams { k: 2, ..Default::default() },
    };
    let mut t = Vec::new();
    let (full, info) = problem::build_full(&input, &mut t, false).unwrap();
    for degree in [1u8, 2] {
        let pb = reduce::reduce(&full, &info, degree, input.params.omega).unwrap();
        assert_eq!(pb.n, 16 * 3 * if degree == 1 { 4 } else { 10 });
        let n = pb.n;
        let c = pb.init[0].clone();
        let cur: Vec<f64> = c.clone(); // M_r = I
        let persistent: Vec<&[f64]> = pb.rigid_rows.iter().map(|r| r.as_slice()).collect();
        assert_eq!(persistent.len(), 6);
        let mut rows = persistent.clone();
        rows.push(&cur);
        let mut rhs = vec![0.0; rows.len()];
        *rhs.last_mut().unwrap() = 1.0;
        let r = clarabel_solver::solve(&pb, &rows, &rhs).unwrap();
        let mut a = admm::Admm::new(&pb).unwrap();
        a.warm_from(&pb, &c);
        let fast = a.solve(&pb, &persistent, &cur, &rhs, 1e-7).unwrap();
        let (e1, e2) = (pb.objective(&r.u), pb.objective(&fast.u));
        eprintln!("p{degree}: n {n} clarabel {e1:.8e} admm {e2:.8e}");
        assert!((e1 - e2).abs() < 1e-3 * e1);
        // same order of magnitude as the full problem (the reduced constraint
        // uses the normalized projection of c, so energies are not ordered)
        let rf = {
            let mut cf = full.init[0].clone();
            let rig: Vec<Vec<f64>> = full.rigid_rows.iter().map(|r| (0..full.n).map(|i| r[i] / full.m[i]).collect()).collect();
            orthogonalize(&mut cf, &rig, &full.m);
            let nc = mdot(&cf, &cf, &full.m).sqrt();
            cf.iter_mut().for_each(|x| *x /= nc);
            let curf: Vec<f64> = (0..full.n).map(|i| full.m[i] * cf[i]).collect();
            let mut rowsf: Vec<&[f64]> = full.rigid_rows.iter().map(|r| r.as_slice()).collect();
            rowsf.push(&curf);
            clarabel_solver::solve(&full, &rowsf, &rhs).unwrap()
        };
        let ef = full.objective(&rf.u);
        eprintln!("   full {ef:.8e}");
        assert!(e1 < 3.0 * ef && e1 > 0.3 * ef, "reduced energy far from full: {e1} vs {ef}");
    }
}
