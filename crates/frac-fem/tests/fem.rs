use frac_fem::eigen::{smallest_eigenpairs, EigenOptions};
use frac_fem::*;
use frac_geom::mesh::{box_mesh, icosphere};
use frac_geom::DVec3;

fn bar(l: f64, a: f64, h: f64) -> TetMesh {
    let solid = box_mesh(DVec3::ZERO, DVec3::new(l, a, a));
    tetrahedralize(&solid, h, 0)
}

fn verts_where(m: &TetMesh, f: impl Fn([f64; 3]) -> bool) -> Vec<u32> {
    (0..m.verts.len() as u32).filter(|&v| f(m.verts[v as usize])).collect()
}

/// Consistent nodal forces for a uniform traction on boundary faces selected by `sel`.
fn face_load(m: &TetMesh, sel: impl Fn([f64; 3]) -> bool, traction: [f64; 3]) -> (Vec<[f64; 3]>, f64) {
    let mut f = vec![[0.0; 3]; m.verts.len()];
    let mut area = 0.0;
    for t in m.boundary_faces() {
        let p = t.map(|v| m.verts[v as usize]);
        if p.iter().all(|&q| sel(q)) {
            let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
            let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
            let c = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            let ar = 0.5 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            area += ar;
            for &v in &t {
                for k in 0..3 {
                    f[v as usize][k] += traction[k] * ar / 3.0;
                }
            }
        }
    }
    (f, area)
}

#[test]
fn box_mesh_quality_and_volume() {
    let solid = box_mesh(DVec3::ZERO, DVec3::new(2.0, 1.0, 0.5));
    let m = tetrahedralize(&solid, 0.1, 0);
    assert!(m.min_volume() > 0.0);
    let v = m.volume();
    assert!((v - 1.0).abs() < 0.02, "volume {v}");
    // all boundary vertices lie on the box surface
    for f in m.boundary_faces() {
        for v in f {
            let p = m.verts[v as usize];
            let d = p[0].min(2.0 - p[0]).min(p[1]).min(1.0 - p[1]).min(p[2]).min(0.5 - p[2]);
            assert!(d.abs() < 1e-9, "boundary vertex off surface: {p:?}");
        }
    }
    assert_eq!(m.tet_components().0, 1);
    // determinism
    let m2 = tetrahedralize(&solid, 0.1, 0);
    assert_eq!(m, m2);
    // coarsening cap
    let mc = tetrahedralize(&solid, 0.02, 2000);
    assert!(mc.tets.len() <= 2000 && mc.tets.len() > 500, "{}", mc.tets.len());
}

#[test]
fn sphere_mesh_volume() {
    let s = icosphere(DVec3::new(0.1, -0.2, 0.3), 1.0, 3);
    let m = tetrahedralize(&s, 0.15, 0);
    assert!(m.min_volume() > 0.0);
    let ratio = m.volume() / s.signed_volume();
    assert!((ratio - 1.0).abs() < 0.03, "ratio {ratio}");
    let e = m.mean_edge_length();
    assert!(e < 0.15 && e > 0.08, "mean edge {e}");
}

#[test]
fn axial_bar_stiffness() {
    let (l, a) = (5.0, 1.0);
    let m = bar(l, a, 0.25);
    let e = 1.0e3;
    let mats = vec![ElasticMaterial::isotropic(e, 0.3, 1.0); m.tets.len()];
    let fixed = verts_where(&m, |p| p[0].abs() < 1e-9);
    let p_total = 1.0;
    let (f, area) = face_load(&m, |p| (p[0] - l).abs() < 1e-9, [p_total / (a * a), 0.0, 0.0]);
    assert!((area - a * a).abs() < 1e-9);
    let u = solve_static(&m, &mats, &fixed, &f).unwrap();
    let tip = verts_where(&m, |p| (p[0] - l).abs() < 1e-9);
    let mean: f64 = tip.iter().map(|&v| u[v as usize][0]).sum::<f64>() / tip.len() as f64;
    let k_fem = p_total / mean;
    let k_exact = e * a * a / l;
    let err = (k_fem / k_exact - 1.0).abs();
    eprintln!("axial: k_fem {k_fem} exact {k_exact} err {err}");
    assert!(err < 0.03, "axial stiffness error {err}");
}

#[test]
fn cantilever_tip_deflection_and_frequency() {
    let (l, a) = (8.0, 1.0);
    let m = bar(l, a, a / 6.0);
    let (e, rho) = (1.0e4, 1.0);
    let mats = vec![ElasticMaterial::isotropic(e, 0.3, rho); m.tets.len()];
    let fixed = verts_where(&m, |p| p[0].abs() < 1e-9);
    let p = 0.01;
    let (f, _) = face_load(&m, |q| (q[0] - l).abs() < 1e-9, [0.0, 0.0, -p / (a * a)]);
    let u = solve_static(&m, &mats, &fixed, &f).unwrap();
    let tip = verts_where(&m, |q| (q[0] - l).abs() < 1e-9);
    let mean: f64 = -tip.iter().map(|&v| u[v as usize][2]).sum::<f64>() / tip.len() as f64;
    let inertia = a.powi(4) / 12.0;
    let exact = p * l.powi(3) / (3.0 * e * inertia);
    let err = (mean / exact - 1.0).abs();
    eprintln!("cantilever: tets {} fem {mean} exact {exact} err {err}", m.tets.len());
    assert!(err < 0.10, "cantilever deflection error {err}");

    let freqs = natural_frequencies(&m, &mats, &fixed, 3).unwrap();
    let beta = 1.875_104_068_711_961;
    let f1 = beta * beta / (2.0 * std::f64::consts::PI) * (e * inertia / (rho * a * a * l.powi(4))).sqrt();
    let ferr = (freqs[0] / f1 - 1.0).abs();
    eprintln!("cantilever freq: fem {:?} EB {f1} err {ferr}", freqs);
    assert!(ferr < 0.10, "frequency error {ferr}");
    // square section: first two bending modes are (nearly) degenerate
    assert!((freqs[1] / freqs[0] - 1.0).abs() < 0.03);
}

#[test]
fn transverse_isotropic_axial() {
    // grain along x: axial stiffness uses E_long; grain along y: uses E_trans
    let (l, a) = (4.0, 1.0);
    let m = bar(l, a, 0.25);
    let fixed = verts_where(&m, |p| p[0].abs() < 1e-9);
    let (f, _) = face_load(&m, |p| (p[0] - l).abs() < 1e-9, [1.0, 0.0, 0.0]);
    let tip = verts_where(&m, |p| (p[0] - l).abs() < 1e-9);
    let mut ks = Vec::new();
    for axis in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
        let t = TransverseIsotropic { axis, e_long: 10.0e3, e_trans: 1.0e3, g_long: 0.8e3, nu_trans: 0.0, nu_long: 0.0 };
        let mats = vec![ElasticMaterial::transverse(t, 1.0); m.tets.len()];
        let u = solve_static(&m, &mats, &fixed, &f).unwrap();
        let mean: f64 = tip.iter().map(|&v| u[v as usize][0]).sum::<f64>() / tip.len() as f64;
        ks.push(1.0 / mean);
    }
    let along = ks[0] / (10.0e3 * a * a / l);
    let across = ks[1] / (1.0e3 * a * a / l);
    eprintln!("transverse: along {along} across {across}");
    assert!((along - 1.0).abs() < 0.03 && (across - 1.0).abs() < 0.03);
}

#[test]
fn eigensolver_matches_dense() {
    // a mesh just above the dense threshold so the iterative path is used
    let solid = box_mesh(DVec3::ZERO, DVec3::new(1.0, 0.5, 0.4));
    let m = tetrahedralize(&solid, 0.2, 0);
    let mats = vec![ElasticMaterial::isotropic(100.0, 0.25, 2.0); m.tets.len()];
    let k = assemble_stiffness(&m, &mats);
    let mm = assemble_lumped_mass(&m, &mats);
    let n = k.n_rows;
    assert!(n > 240 && n < 700, "dofs {n}");
    let t0 = std::time::Instant::now();
    let defl = rigid_modes(&m.verts);
    let res = smallest_eigenpairs(&k, &mm, &defl, &EigenOptions { n: 8, seed: 3, ..Default::default() }).unwrap();
    eprintln!("n {n} iters {} res {} t {:?}", res.iterations, res.max_residual, t0.elapsed());
    assert!(res.converged);
    // dense reference: M^{-1/2} K M^{-1/2}
    let kd = k.to_dense();
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = kd[i * n + j] / (mm[i] * mm[j]).sqrt();
        }
    }
    let (w, _) = frac_fem::dense::sym_eigen(&a, n);
    // first 6 are rigid (≈0)
    let scale = w[n - 1];
    for i in 0..6 {
        assert!(w[i].abs() < 1e-9 * scale, "rigid {i} {}", w[i]);
    }
    for i in 0..8 {
        let rel = (res.values[i] - w[6 + i]).abs() / w[6 + i];
        assert!(rel < 1e-8, "eig {i}: {} vs {} ({rel})", res.values[i], w[6 + i]);
    }
    // M-orthonormal vectors
    for i in 0..8 {
        for j in 0..8 {
            let d: f64 = (0..n).map(|q| res.vectors[i][q] * mm[q] * res.vectors[j][q]).sum();
            assert!((d - if i == j { 1.0 } else { 0.0 }).abs() < 1e-8);
        }
    }
    // anchored problem via the dense path vs iterative path on the same matrix
    let fixed: Vec<u32> = (0..m.verts.len() as u32).filter(|&v| m.verts[v as usize][0] < 1e-9).collect();
    let (vals, vecs) = frac_fem::analysis::eigenmodes(&m, &mats, &fixed, 4, 1).unwrap();
    assert_eq!(vals.len(), 4);
    assert_eq!(vecs[0].len(), n);
    assert!(vals[0] > 0.0);
}
