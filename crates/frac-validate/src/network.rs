//! Reference bond-network solver (spec §13.3): rigid fragments (6 DOF at
//! the center of mass) connected by bond springs with the reference
//! stiffness formulas of spec §6:
//!
//! * normal  k_n  = A / (d_a/E_a + d_b/E_b + t_m/E_m)
//! * shear   k_s  = A / (d_a/G_a + d_b/G_b + t_m/G_m)
//! * bending k_θu = I_uu / (d_a/E_a + d_b/E_b + t_m/E_m), same for v
//! * torsion k_t  = J / (d_a/G_a + d_b/G_b + t_m/G_m)

use faer::linalg::solvers::{DenseSolveCore, Solve};
use frac_core::*;
use frac_material::MaterialLibrary;
use glam::{DMat3, DVec3};
use serde::{Deserialize, Serialize};

/// A static load case.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoadCase {
    pub name: String,
    /// Gravity acceleration applied to every fragment (e.g. (0,-9.81,0)).
    pub gravity: DVec3,
    /// Point forces: (fragment, force) at the fragment COM.
    pub forces: Vec<(FragmentId, DVec3)>,
    /// Additional fixed fragments (besides anchor bonds).
    pub fixed: Vec<FragmentId>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BondForce {
    pub bond: BondId,
    pub normal_force: f64,
    pub shear_force: DVec3,
    pub moment: DVec3,
    /// Normal traction F_n / A.
    pub traction: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkResult {
    pub level: u8,
    /// Per fragment of the level: (translation, rotation).
    pub displacements: Vec<(DVec3, DVec3)>,
    pub bond_forces: Vec<BondForce>,
}

pub trait BondNetworkSolver {
    fn static_solve(&self, asset: &Asset, level: u8, loads: &LoadCase) -> NetworkResult;
    fn modal(&self, asset: &Asset, level: u8, n: usize) -> Vec<f64>;
}

/// Spring constants of one bond in its frame (n, u, v).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BondSprings {
    pub kn: f64,
    pub ks: f64,
    pub ktu: f64,
    pub ktv: f64,
    pub kt: f64,
}

pub struct ReferenceSolver<'a> {
    pub lib: &'a MaterialLibrary,
}

impl<'a> ReferenceSolver<'a> {
    fn mat_eg(&self, asset: &Asset, f: FragmentId) -> (f64, f64) {
        let m = asset.fragment(f).material_mix.first().map(|x| x.0).unwrap_or(MaterialId(0));
        let e = self.lib.elastic(m);
        (e.youngs, e.shear)
    }

    pub fn springs(&self, asset: &Asset, b: &Bond) -> BondSprings {
        let (ea, ga) = self.mat_eg(asset, b.a);
        let (eb, gb) = match b.b {
            FragmentOrWorld::Fragment(f) => self.mat_eg(asset, f),
            FragmentOrWorld::World => (f64::INFINITY, f64::INFINITY),
        };
        // interface layer from the dominant composition entry
        let (tm, em, gm) = b
            .composition
            .first()
            .and_then(|c| c.interface_material)
            .and_then(|m| self.lib.interface_material(m))
            .map(|im| {
                let e = im.youngs_modulus.unwrap_or(ea);
                let g = im.shear_modulus.unwrap_or(e / 2.4);
                (im.thickness.unwrap_or(0.0), e, g)
            })
            .unwrap_or((0.0, 1.0, 1.0));
        let da = b.dist_a.max(1e-9);
        let db = if matches!(b.b, FragmentOrWorld::World) { 0.0 } else { b.dist_b.max(1e-9) };
        let ce = da / ea + if eb.is_finite() { db / eb } else { 0.0 } + if tm > 0.0 { tm / em } else { 0.0 };
        let cg = da / ga + if gb.is_finite() { db / gb } else { 0.0 } + if tm > 0.0 { tm / gm } else { 0.0 };
        BondSprings { kn: b.area / ce, ks: b.area / cg, ktu: b.i_uu / ce, ktv: b.i_vv / ce, kt: b.j / cg }
    }

    /// Assemble the stiffness matrix (dense, 6N x 6N) for a level.
    fn assemble(&self, asset: &Asset, level: u8) -> (Vec<f64>, usize, Vec<(BondId, BondSprings)>) {
        let r = asset.hierarchy.level_ranges[level as usize].clone();
        let n = (r.end - r.start) as usize;
        let dim = 6 * n;
        let mut k = vec![0.0f64; dim * dim];
        let mut springs = Vec::new();
        for b in asset.level_bonds(level) {
            let s = self.springs(asset, b);
            springs.push((b.id, s));
            let ia = (b.a.0 - r.start) as usize;
            let ib = match b.b {
                FragmentOrWorld::Fragment(f) => Some((f.0 - r.start) as usize),
                FragmentOrWorld::World => None,
            };
            let c = b.centroid;
            let xa = asset.fragment(b.a).mass.com;
            // relative motion operator: δ = u_B(c) - u_A(c), Δθ = θ_B - θ_A
            // rows: [n,u,v] for δ and [u,v,n] for Δθ with stiffness diag
            let frame = [b.normal, b.frame_u, b.frame_v];
            let kd = [s.kn, s.ks, s.ks];
            let kr = [s.kt, s.ktu, s.ktv];
            // For each row vector w (in body space), gradient wrt body DOFs:
            // translation: w ; rotation: (c - x) × w
            let mut rows: Vec<(f64, [f64; 6], Option<[f64; 6]>)> = Vec::new();
            for q in 0..3 {
                let w = frame[q];
                let ra = (c - xa).cross(w);
                let ga = [-w.x, -w.y, -w.z, -ra.x, -ra.y, -ra.z];
                let gb = ib.map(|ib| {
                    let xb = asset.hierarchy.fragments[r.start as usize + ib].mass.com;
                    let rb = (c - xb).cross(w);
                    [w.x, w.y, w.z, rb.x, rb.y, rb.z]
                });
                rows.push((kd[q], ga, gb));
                let ga2 = [0.0, 0.0, 0.0, -w.x, -w.y, -w.z];
                let gb2 = ib.map(|_| [0.0, 0.0, 0.0, w.x, w.y, w.z]);
                rows.push((kr[q], ga2, gb2));
            }
            for (kk, ga, gb) in rows {
                let mut idx: Vec<(usize, f64)> = (0..6).map(|j| (6 * ia + j, ga[j])).collect();
                if let (Some(ib), Some(gb)) = (ib, gb) {
                    idx.extend((0..6).map(|j| (6 * ib + j, gb[j])));
                }
                for &(i, gi) in &idx {
                    for &(j, gj) in &idx {
                        k[i * dim + j] += kk * gi * gj;
                    }
                }
            }
        }
        (k, dim, springs)
    }

    fn mass_matrix(&self, asset: &Asset, level: u8) -> Vec<f64> {
        let r = asset.hierarchy.level_ranges[level as usize].clone();
        let n = (r.end - r.start) as usize;
        let dim = 6 * n;
        let mut m = vec![0.0f64; dim * dim];
        for i in 0..n {
            let f = &asset.hierarchy.fragments[r.start as usize + i];
            for a in 0..3 {
                m[(6 * i + a) * dim + 6 * i + a] = f.mass.mass;
            }
            let it: DMat3 = f.mass.inertia;
            for a in 0..3 {
                for bb in 0..3 {
                    m[(6 * i + 3 + a) * dim + 6 * i + 3 + bb] = it.col(bb)[a];
                }
            }
        }
        m
    }
}

fn dense_solve(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    use faer::Mat;
    let am = Mat::<f64>::from_fn(n, n, |i, j| a[i * n + j]);
    let bm = Mat::<f64>::from_fn(n, 1, |i, _| b[i]);
    let llt = am.llt(faer::Side::Lower).ok()?;
    let x = llt.solve(&bm);
    Some((0..n).map(|i| x[(i, 0)]).collect())
}

impl<'a> BondNetworkSolver for ReferenceSolver<'a> {
    fn static_solve(&self, asset: &Asset, level: u8, loads: &LoadCase) -> NetworkResult {
        let r = asset.hierarchy.level_ranges[level as usize].clone();
        let n = (r.end - r.start) as usize;
        let (mut k, dim, springs) = self.assemble(asset, level);
        let mut f = vec![0.0f64; dim];
        for i in 0..n {
            let fr = &asset.hierarchy.fragments[r.start as usize + i];
            let g = loads.gravity * fr.mass.mass;
            f[6 * i] += g.x;
            f[6 * i + 1] += g.y;
            f[6 * i + 2] += g.z;
        }
        for &(fid, force) in &loads.forces {
            let i = (fid.0 - r.start) as usize;
            f[6 * i] += force.x;
            f[6 * i + 1] += force.y;
            f[6 * i + 2] += force.z;
        }
        // fixed fragments: penalty-free elimination via large diagonal
        let big = k.iter().cloned().fold(0.0, f64::max).max(1.0) * 1e12;
        for fid in &loads.fixed {
            let i = (fid.0 - r.start) as usize;
            for j in 0..6 {
                k[(6 * i + j) * dim + 6 * i + j] += big;
            }
        }
        // tiny regularization for floating mechanisms
        let reg = k.iter().step_by(dim + 1).cloned().fold(0.0, f64::max).max(1.0) * 1e-12;
        for i in 0..dim {
            k[i * dim + i] += reg;
        }
        let u = dense_solve(&k, &f, dim).unwrap_or_else(|| vec![0.0; dim]);
        let disp: Vec<(DVec3, DVec3)> = (0..n).map(|i| (DVec3::new(u[6 * i], u[6 * i + 1], u[6 * i + 2]), DVec3::new(u[6 * i + 3], u[6 * i + 4], u[6 * i + 5]))).collect();
        let mut forces = Vec::new();
        for (bid, s) in springs {
            let b = &asset.bonds[bid.idx()];
            let ia = (b.a.0 - r.start) as usize;
            let xa = asset.fragment(b.a).mass.com;
            let ua = disp[ia].0 + disp[ia].1.cross(b.centroid - xa);
            let (ub, thb) = match b.b {
                FragmentOrWorld::Fragment(fb) => {
                    let ib = (fb.0 - r.start) as usize;
                    let xb = asset.fragment(fb).mass.com;
                    (disp[ib].0 + disp[ib].1.cross(b.centroid - xb), disp[ib].1)
                }
                FragmentOrWorld::World => (DVec3::ZERO, DVec3::ZERO),
            };
            let d = ub - ua;
            let dth = thb - disp[ia].1;
            let fnrm = s.kn * d.dot(b.normal);
            let shear = (b.frame_u * d.dot(b.frame_u) + b.frame_v * d.dot(b.frame_v)) * s.ks;
            let moment = b.frame_u * (s.ktu * dth.dot(b.frame_u)) + b.frame_v * (s.ktv * dth.dot(b.frame_v)) + b.normal * (s.kt * dth.dot(b.normal));
            forces.push(BondForce { bond: bid, normal_force: fnrm, shear_force: shear, moment, traction: fnrm / b.area.max(1e-300) });
        }
        NetworkResult { level, displacements: disp, bond_forces: forces }
    }

    fn modal(&self, asset: &Asset, level: u8, nmodes: usize) -> Vec<f64> {
        use faer::Mat;
        let (k, dim, _) = self.assemble(asset, level);
        let m = self.mass_matrix(asset, level);
        if dim == 0 || dim > 6000 {
            return Vec::new();
        }
        // generalized symmetric eigenproblem via Cholesky of M: L^-1 K L^-T
        let mm = Mat::<f64>::from_fn(dim, dim, |i, j| m[i * dim + j]);
        let km = Mat::<f64>::from_fn(dim, dim, |i, j| k[i * dim + j]);
        let Ok(llt) = mm.llt(faer::Side::Lower) else { return Vec::new() };
        let l = llt.L().to_owned();
        let linv = l.as_ref().partial_piv_lu().inverse();
        let a = &linv * &km * linv.transpose();
        let a = Mat::<f64>::from_fn(dim, dim, |i, j| 0.5 * (a[(i, j)] + a[(j, i)]));
        let Ok(evd) = a.self_adjoint_eigen(faer::Side::Lower) else { return Vec::new() };
        let s = evd.S();
        let mut w: Vec<f64> = (0..dim).map(|i| s[i]).collect();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let tol = w.last().cloned().unwrap_or(0.0).abs() * 1e-10;
        w.into_iter().filter(|&x| x > tol).take(nmodes).map(|x| x.sqrt() / std::f64::consts::TAU).collect()
    }
}
