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
    /// Point forces: (fragment, force), applied at `force_points` (or the
    /// fragment COM).
    pub forces: Vec<(FragmentId, DVec3)>,
    /// Optional application points of `forces`; empty = fragment COM.
    #[serde(default)]
    pub force_points: Vec<DVec3>,
    /// Additional external couples per fragment.
    #[serde(default)]
    pub moments: Vec<(FragmentId, DVec3)>,
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
    /// Raw traction vector F / A (force on B from A, i.e. σ·n convention
    /// with n from A to B).
    pub traction_vec: DVec3,
    /// Traction recovered from the Love–Weber stress of the two
    /// fragments: ½(σ_a + σ_b)·n.
    pub recovered_traction: DVec3,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkResult {
    pub level: u8,
    /// Per fragment of the level: (translation, rotation).
    pub displacements: Vec<(DVec3, DVec3)>,
    /// Love–Weber average stress per fragment.
    pub stress: Vec<DMat3>,
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

/// Bond stiffness model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StiffnessModel {
    /// Spec §6 reference formulas verbatim (k_s uses G).
    Spec,
    /// Isotropy-calibrated discrete model: for random (Voronoi-like) cell
    /// geometries a normal/shear spring network has
    /// `E_eff = E0 (2+3α)/(4+α)` and `ν_eff = (1-α)/(4+α)` with
    /// `α = k_s/k_n` (Eliáš 2020). Choosing `α = (1-4ν)/(1+ν)` and
    /// `E0 = E (4+α)/(2+3α)` reproduces the material's E and ν. Rotational
    /// springs integrate the same distributed normal (bending) and shear
    /// (torsion) stiffness over the facet.
    Calibrated,
    /// Isotropic volumetric–deviatoric network: deviatoric normal/shear
    /// springs `k_n = k_s = 2μA/h` (reproducing `2μ ε·n` under affine
    /// motion) plus a volumetric pressure `λ tr ε̃`, where `ε̃_i` is a
    /// least-squares affine strain fitted to neighboring fragment
    /// displacements. Spring lengths and the strain fit use
    /// [`kinematic_centers`] (normal-aligned connectors) on Voronoi-like
    /// levels. Statics transmit the pressure through the bonds as face forces
    /// `½(λ_a tr ε̃_a + λ_b tr ε̃_b) A n` (finite-volume form: uniform stress
    /// states are exact equilibria, free surfaces included; non-symmetric
    /// operator, LU solve); modal analysis uses the symmetric energy
    /// `½ λ V_i (tr ε̃_i)²`. Patch tests: `patch_test`, `patch_test_with`,
    /// `patch_test_field`.
    Tensorial,
}

/// Volumetric strain operator of one fragment: `tr ε̃_i = Σ_k g_k·(u_k − u_i)`
/// over neighbors k (`None` = world, zero displacement).
#[derive(Clone, Debug)]
struct VolOp {
    /// Least-squares operator for the full displacement gradient:
    /// `G = Σ_k (u_k − u_i) ⊗ (W_k M⁻¹ Δx_k)`, with `u` sampled at the
    /// kinematic centers.
    terms: Vec<(Option<usize>, DVec3)>,
    lambda: f64,
    mu: f64,
    volume: f64,
}

/// Kinematic reference points of the fragments of a level, used by the
/// Tensorial model for spring lengths and the strain fit.
///
/// Rigid-body-spring networks reproduce a uniform stress state exactly only
/// when every bond's connector `p_b − p_a` is parallel to the bond normal
/// (then the affine motion `u_i = ε p_i, θ_i = 0` is an equilibrium and every
/// bond carries exactly `σ·n A`). Mass centroids do not have this property;
/// Voronoi sites do. We recover such points for any cell complex (merged
/// slivers, clusters, masonry offsets included) by least squares:
///
/// `min Σ_b A_b [ |(I − n nᵀ)(p_b − p_a)|² + γ (n·(p_b − p_a) − h_b)² ] + β Σ_i a_i |p_i − x_i|²`
///
/// with `x_i` the centroid, `h_b` the centroid distance along the normal and
/// `a_i` the fragment's bonded area. The tangential term vanishes exactly at
/// the sites of an (unmerged) Voronoi complex; the weak normal term anchors
/// the scale (so the fit cannot shrink connectors to reduce the residual of
/// non-Voronoi clusters) and the centroid term fixes the translation.
pub fn kinematic_centers(asset: &Asset, level: u8) -> Vec<DVec3> {
    let r = asset.hierarchy.level_ranges[level as usize].clone();
    let n = (r.end - r.start) as usize;
    let com: Vec<DVec3> = (0..n).map(|i| asset.hierarchy.fragments[r.start as usize + i].mass.com).collect();
    let mut edges: Vec<(usize, usize, f64, DVec3, f64)> = Vec::new();
    let mut a_i = vec![0.0f64; n];
    for b in asset.level_bonds(level) {
        if let FragmentOrWorld::Fragment(f) = b.b {
            let (ia, ib) = ((b.a.0 - r.start) as usize, (f.0 - r.start) as usize);
            if ia == ib || !(b.area > 0.0) {
                continue;
            }
            edges.push((ia, ib, b.area, b.normal, b.dist_a + b.dist_b));
            a_i[ia] += b.area;
            a_i[ib] += b.area;
        }
    }
    const BETA: f64 = 1e-3;
    const GAMMA: f64 = 0.05;
    let alpha: Vec<f64> = a_i.iter().map(|&a| BETA * a.max(1e-300)).collect();
    // operator v ↦ (I − n nᵀ) v + γ n nᵀ v
    let proj = |v: DVec3, nrm: DVec3| v - nrm * ((1.0 - GAMMA) * v.dot(nrm));
    let apply = |p: &[DVec3], y: &mut [DVec3]| {
        for i in 0..n {
            y[i] = p[i] * alpha[i];
        }
        for &(a, b, w, nrm, _) in &edges {
            let t = proj(p[b] - p[a], nrm) * w;
            y[b] += t;
            y[a] -= t;
        }
    };
    // Jacobi-preconditioned conjugate gradients, warm-started at the centroids
    let diag: Vec<f64> = (0..n).map(|i| alpha[i] + a_i[i] * (2.0 / 3.0)).collect();
    let mut rhs: Vec<DVec3> = (0..n).map(|i| com[i] * alpha[i]).collect();
    for &(a, b, w, nrm, h) in &edges {
        let t = nrm * (GAMMA * h * w);
        rhs[b] += t;
        rhs[a] -= t;
    }
    let mut x = com.clone();
    let mut y = vec![DVec3::ZERO; n];
    apply(&x, &mut y);
    let mut res: Vec<DVec3> = (0..n).map(|i| rhs[i] - y[i]).collect();
    let mut z: Vec<DVec3> = (0..n).map(|i| res[i] / diag[i]).collect();
    let mut d = z.clone();
    let dot = |a: &[DVec3], b: &[DVec3]| a.iter().zip(b).map(|(u, v)| u.dot(*v)).sum::<f64>();
    let mut rz = dot(&res, &z);
    let r0 = dot(&rhs, &rhs).sqrt().max(1e-300);
    for _ in 0..(20 * n).clamp(200, 20000) {
        if dot(&res, &res).sqrt() <= 1e-13 * r0 {
            break;
        }
        apply(&d, &mut y);
        let dy = dot(&d, &y);
        if !(dy > 0.0) {
            break;
        }
        let step = rz / dy;
        for i in 0..n {
            x[i] += d[i] * step;
            res[i] -= y[i] * step;
            z[i] = res[i] / diag[i];
        }
        let rz2 = dot(&res, &z);
        let beta = rz2 / rz;
        rz = rz2;
        for i in 0..n {
            d[i] = z[i] + d[i] * beta;
        }
    }
    if x.iter().all(|p| p.is_finite()) {
        x
    } else {
        com
    }
}

pub struct ReferenceSolver<'a> {
    pub lib: &'a MaterialLibrary,
    pub model: StiffnessModel,
}

impl<'a> ReferenceSolver<'a> {
    /// Effective (normal, shear) moduli of a fragment's material for the
    /// selected stiffness model.
    fn mat_eg(&self, asset: &Asset, f: FragmentId) -> (f64, f64) {
        let m = asset.fragment(f).material_mix.first().map(|x| x.0).unwrap_or(MaterialId(0));
        let e = self.lib.elastic(m);
        match self.model {
            StiffnessModel::Spec => (e.youngs, e.shear),
            StiffnessModel::Calibrated => {
                let nu = e.poisson.clamp(-0.9, 0.24);
                let alpha = ((1.0 - 4.0 * nu) / (1.0 + nu)).clamp(0.01, 1.0);
                let e0 = e.youngs * (4.0 + alpha) / (2.0 + 3.0 * alpha);
                (e0, alpha * e0)
            }
            StiffnessModel::Tensorial => {
                let mu = e.youngs / (2.0 * (1.0 + e.poisson));
                (2.0 * mu, 2.0 * mu)
            }
        }
    }

    /// Lamé constants of a fragment's material.
    fn lame(&self, asset: &Asset, f: FragmentId) -> (f64, f64) {
        let m = asset.fragment(f).material_mix.first().map(|x| x.0).unwrap_or(MaterialId(0));
        let e = self.lib.elastic(m);
        let mu = e.youngs / (2.0 * (1.0 + e.poisson));
        let lam = e.youngs * e.poisson / ((1.0 + e.poisson) * (1.0 - 2.0 * e.poisson));
        (lam, mu)
    }

    /// Least-squares strain operators for the Tensorial model.
    fn vol_ops(&self, asset: &Asset, level: u8, centers: &[DVec3]) -> Vec<VolOp> {
        let r = asset.hierarchy.level_ranges[level as usize].clone();
        let n = (r.end - r.start) as usize;
        let mut nb: Vec<Vec<(Option<usize>, DVec3, f64)>> = vec![Vec::new(); n];
        for b in asset.level_bonds(level) {
            let ia = (b.a.0 - r.start) as usize;
            let xa = centers[ia];
            match b.b {
                FragmentOrWorld::Fragment(f) => {
                    let ib = (f.0 - r.start) as usize;
                    let xb = centers[ib];
                    let w = b.area / (xb - xa).length_squared().max(1e-300);
                    nb[ia].push((Some(ib), xb - xa, w));
                    nb[ib].push((Some(ia), xa - xb, w));
                }
                FragmentOrWorld::World => {
                    let d = b.centroid - xa;
                    nb[ia].push((None, d, b.area / d.length_squared().max(1e-300)));
                }
            }
        }
        (0..n)
            .map(|i| {
                let fid = FragmentId(r.start + i as u32);
                let (lambda, mu) = self.lame(asset, fid);
                let volume = asset.fragment(fid).mass.volume;
                let mut m = DMat3::ZERO;
                for (_, d, w) in &nb[i] {
                    m += DMat3::from_cols(*d * d.x, *d * d.y, *d * d.z) * *w;
                }
                let terms = if m.determinant().abs() > 1e-30 * m.col(0).length().max(1e-300).powi(3) {
                    let mi = m.inverse();
                    nb[i].iter().map(|(k, d, w)| (*k, mi * *d * *w)).collect()
                } else {
                    Vec::new()
                };
                VolOp { terms, lambda, mu, volume }
            })
            .collect()
    }

    /// Displacement-gradient estimate of fragment i from the displacements
    /// at the kinematic centers (`arms[i] = p_i − x_i`).
    fn grad(op: &VolOp, i: usize, disp: &[(DVec3, DVec3)], arms: &[DVec3]) -> DMat3 {
        let at = |k: usize| disp[k].0 + disp[k].1.cross(arms[k]);
        let ui = at(i);
        let mut g = DMat3::ZERO;
        for (k, c) in &op.terms {
            let uk = k.map(at).unwrap_or(DVec3::ZERO);
            let du = uk - ui;
            g += DMat3::from_cols(du * c.x, du * c.y, du * c.z);
        }
        g
    }

    pub fn springs(&self, asset: &Asset, b: &Bond) -> BondSprings {
        self.springs_with(asset, b, b.dist_a, b.dist_b)
    }

    /// Bond spring lengths `(d_a, d_b)` along the normal from the kinematic
    /// centers (Tensorial) or the centroids (other models).
    fn bond_lengths(&self, b: &Bond, centers: &[DVec3], r0: u32) -> (f64, f64) {
        if self.model != StiffnessModel::Tensorial {
            return (b.dist_a, b.dist_b);
        }
        let pa = centers[(b.a.0 - r0) as usize];
        match b.b {
            FragmentOrWorld::Fragment(f) => {
                let pb = centers[(f.0 - r0) as usize];
                let h = (pb - pa).dot(b.normal);
                let h0 = b.dist_a + b.dist_b;
                // guard against degenerate fits: keep within a factor of the centroid distance
                if !(h > 0.25 * h0) || !(h < 4.0 * h0) {
                    return (b.dist_a, b.dist_b);
                }
                let da = (b.centroid - pa).dot(b.normal).clamp(0.05 * h, 0.95 * h);
                (da, h - da)
            }
            FragmentOrWorld::World => {
                let da = (b.centroid - pa).dot(b.normal);
                if da > 0.25 * b.dist_a && da < 4.0 * b.dist_a { (da, 0.0) } else { (b.dist_a, b.dist_b) }
            }
        }
    }

    fn springs_with(&self, asset: &Asset, b: &Bond, dist_a: f64, dist_b: f64) -> BondSprings {
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
        let da = dist_a.max(1e-9);
        let db = if matches!(b.b, FragmentOrWorld::World) { 0.0 } else { dist_b.max(1e-9) };
        let ce = da / ea + if eb.is_finite() { db / eb } else { 0.0 } + if tm > 0.0 { tm / em } else { 0.0 };
        let cg = da / ga + if gb.is_finite() { db / gb } else { 0.0 } + if tm > 0.0 { tm / gm } else { 0.0 };
        if self.model == StiffnessModel::Tensorial {
            // facet bending/torsion integrate the true E and G over the facet
            let ea_t = self.lib.elastic(asset.fragment(b.a).material_mix.first().map(|x| x.0).unwrap_or(MaterialId(0)));
            let eb_t = match b.b {
                FragmentOrWorld::Fragment(f) => Some(self.lib.elastic(asset.fragment(f).material_mix.first().map(|x| x.0).unwrap_or(MaterialId(0)))),
                FragmentOrWorld::World => None,
            };
            let cer = da / ea_t.youngs + eb_t.map(|e| db / e.youngs).unwrap_or(0.0) + if tm > 0.0 { tm / em } else { 0.0 };
            let cgr = da / ea_t.shear + eb_t.map(|e| db / e.shear).unwrap_or(0.0) + if tm > 0.0 { tm / gm } else { 0.0 };
            return BondSprings { kn: b.area / ce, ks: b.area / cg, ktu: b.i_uu / cer, ktv: b.i_vv / cer, kt: b.j / cgr };
        }
        BondSprings { kn: b.area / ce, ks: b.area / cg, ktu: b.i_uu / ce, ktv: b.i_vv / ce, kt: b.j / cg }
    }

    fn arms(asset: &Asset, level: u8, centers: &[DVec3]) -> Vec<DVec3> {
        asset.level_fragments(level).iter().zip(centers).map(|(f, p)| *p - f.mass.com).collect()
    }

    /// Kinematic centers for the Tensorial model where they help: the fit is
    /// used only when it removes most of the connector misalignment
    /// (Voronoi-like levels); clustered levels keep the centroids.
    fn centers(&self, asset: &Asset, level: u8) -> Vec<DVec3> {
        let com: Vec<DVec3> = asset.level_fragments(level).iter().map(|f| f.mass.com).collect();
        if self.model != StiffnessModel::Tensorial {
            return com;
        }
        let fit = kinematic_centers(asset, level);
        let r0 = asset.hierarchy.level_ranges[level as usize].start;
        let misalignment = |p: &[DVec3]| {
            let (mut num, mut den) = (0.0, 0.0);
            for b in asset.level_bonds(level) {
                if let FragmentOrWorld::Fragment(f) = b.b {
                    let d = p[(f.0 - r0) as usize] - p[(b.a.0 - r0) as usize];
                    let t = d - b.normal * d.dot(b.normal);
                    num += b.area * t.length() / d.length().max(1e-300);
                    den += b.area;
                }
            }
            num / den.max(1e-300)
        };
        if misalignment(&fit) < 0.25 * misalignment(&com) { fit } else { com }
    }

    /// Coefficients of `tr ε̃_i` over the level's DOFs, with displacements
    /// sampled at the kinematic centers: `u(p_k) = t_k + θ_k × r_k`, so
    /// `g·u(p_k) = g·t_k + (r_k × g)·θ_k`.
    fn trace_row(op: &VolOp, i: usize, arms: &[DVec3]) -> Vec<(usize, f64)> {
        let mut t: Vec<(usize, f64)> = Vec::new();
        if op.terms.is_empty() {
            return t;
        }
        let mut gsum = DVec3::ZERO;
        for (kk, g) in &op.terms {
            gsum += *g;
            if let Some(kk) = kk {
                let rg = arms[*kk].cross(*g);
                for a in 0..3 {
                    t.push((6 * kk + a, g[a]));
                    t.push((6 * kk + 3 + a, rg[a]));
                }
            }
        }
        let rg = arms[i].cross(gsum);
        for a in 0..3 {
            t.push((6 * i + a, -gsum[a]));
            t.push((6 * i + 3 + a, -rg[a]));
        }
        t
    }

    /// Assemble the stiffness matrix (dense, 6N × 6N) for a level. `fv`
    /// selects the finite-volume volumetric coupling (statics) instead of the
    /// symmetric volumetric energy (modal).
    fn assemble(&self, asset: &Asset, level: u8, centers: &[DVec3], fv: bool) -> (Vec<f64>, usize, Vec<(BondId, BondSprings)>) {
        let r = asset.hierarchy.level_ranges[level as usize].clone();
        let n = (r.end - r.start) as usize;
        let dim = 6 * n;
        let mut k = vec![0.0f64; dim * dim];
        let mut springs = Vec::new();
        for b in asset.level_bonds(level) {
            let (da, db) = self.bond_lengths(b, centers, r.start);
            let s = self.springs_with(asset, b, da, db);
            springs.push((b.id, s));
            let ia = (b.a.0 - r.start) as usize;
            let ib = match b.b {
                FragmentOrWorld::Fragment(f) => Some((f.0 - r.start) as usize),
                FragmentOrWorld::World => None,
            };
            let c = b.centroid;
            let xa = asset.fragment(b.a).mass.com;
            // relative motion operator: δ = u_B(q) - u_A(q), Δθ = θ_B - θ_A,
            // measured at q and acting at c (both the centroid: there the
            // rotation-gradient term of the jump supplies the strain-gradient
            // correction by compatibility). Rows: [n,u,v] for δ, [u,v,n] for Δθ.
            let q = c;
            let frame = [b.normal, b.frame_u, b.frame_v];
            let kd = [s.kn, s.ks, s.ks];
            let kr = [s.kt, s.ktu, s.ktv];
            // For each row vector w, gradient wrt body DOFs at point y:
            // translation: w ; rotation: (y - x) × w
            let grad = |y: DVec3, w: DVec3| -> ([f64; 6], Option<[f64; 6]>) {
                let ra = (y - xa).cross(w);
                let ga = [-w.x, -w.y, -w.z, -ra.x, -ra.y, -ra.z];
                let gb = ib.map(|ib| {
                    let xb = asset.hierarchy.fragments[r.start as usize + ib].mass.com;
                    let rb = (y - xb).cross(w);
                    [w.x, w.y, w.z, rb.x, rb.y, rb.z]
                });
                (ga, gb)
            };
            // (stiffness, test vectors at c, trial vectors at q)
            #[allow(clippy::type_complexity)]
            let mut rows: Vec<(f64, ([f64; 6], Option<[f64; 6]>), ([f64; 6], Option<[f64; 6]>))> = Vec::new();
            for qq in 0..3 {
                let w = frame[qq];
                rows.push((kd[qq], grad(c, w), grad(q, w)));
                let g2 = ([0.0, 0.0, 0.0, -w.x, -w.y, -w.z], ib.map(|_| [0.0, 0.0, 0.0, w.x, w.y, w.z]));
                rows.push((kr[qq], g2, g2));
            }
            let expand = |g: ([f64; 6], Option<[f64; 6]>)| -> Vec<(usize, f64)> {
                let mut idx: Vec<(usize, f64)> = (0..6).map(|j| (6 * ia + j, g.0[j])).collect();
                if let (Some(ib), Some(gb)) = (ib, g.1) {
                    idx.extend((0..6).map(|j| (6 * ib + j, gb[j])));
                }
                idx
            };
            for (kk, test, trial) in rows {
                let (ti, tj) = (expand(test), expand(trial));
                for &(i, gi) in &ti {
                    for &(j, gj) in &tj {
                        k[i * dim + j] += kk * gi * gj;
                    }
                }
            }
        }
        if self.model == StiffnessModel::Tensorial {
            let arms = Self::arms(asset, level, centers);
            let ops = self.vol_ops(asset, level, centers);
            let rows: Vec<Vec<(usize, f64)>> = ops.iter().enumerate().map(|(i, op)| Self::trace_row(op, i, &arms)).collect();
            if fv {
                // finite-volume volumetric coupling: every bond transmits the
                // pressure p_b = ½(λ_a tr ε̃_a + λ_b tr ε̃_b) as the face force
                // p_b A n at its centroid (force on A along +n in tension).
                // Under any affine motion this is exactly ∮ λ tr(ε) n dA, so
                // uniform stress states are exact equilibria, free surfaces
                // included. The operator is not symmetric.
                for b in asset.level_bonds(level) {
                    let ia = (b.a.0 - r.start) as usize;
                    let ib = match b.b {
                        FragmentOrWorld::Fragment(f) => Some((f.0 - r.start) as usize),
                        FragmentOrWorld::World => None,
                    };
                    let mut pr: Vec<(usize, f64)> = Vec::new();
                    let wa = if ib.is_some() { 0.5 } else { 1.0 };
                    pr.extend(rows[ia].iter().map(|&(d, c)| (d, c * wa * ops[ia].lambda)));
                    if let Some(ib) = ib {
                        pr.extend(rows[ib].iter().map(|&(d, c)| (d, c * 0.5 * ops[ib].lambda)));
                    }
                    let fa = b.normal * b.area;
                    let ta = (b.centroid - asset.fragment(b.a).mass.com).cross(fa);
                    let mut targets: Vec<(usize, DVec3, DVec3, f64)> = vec![(ia, fa, ta, -1.0)];
                    if let (Some(ib), FragmentOrWorld::Fragment(fb)) = (ib, b.b) {
                        let tb = (b.centroid - asset.fragment(fb).mass.com).cross(fa);
                        targets.push((ib, fa, tb, 1.0));
                    }
                    // K u = −(internal force): rows of A get −p A n, rows of B +p A n
                    for (node, f, t, sg) in targets {
                        for &(d, c) in &pr {
                            for a in 0..3 {
                                k[(6 * node + a) * dim + d] += sg * f[a] * c;
                                k[(6 * node + 3 + a) * dim + d] += sg * t[a] * c;
                            }
                        }
                    }
                }
            } else {
                // volumetric energy ½ λ V (tr ε̃)² (symmetric; used for modal analysis)
                for (op, t) in ops.iter().zip(&rows) {
                    let c = op.lambda * op.volume;
                    for &(p, gp) in t {
                        for &(q, gq) in t {
                            k[p * dim + q] += c * gp * gq;
                        }
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
    let symmetric = (0..n).all(|i| (0..i).all(|j| a[i * n + j] == a[j * n + i]));
    let x = if symmetric {
        am.llt(faer::Side::Lower).ok()?.solve(&bm)
    } else {
        am.partial_piv_lu().solve(&bm)
    };
    if !(0..n).all(|i| x[(i, 0)].is_finite()) {
        return None;
    }
    Some((0..n).map(|i| x[(i, 0)]).collect())
}

impl<'a> ReferenceSolver<'a> {
    /// Uniform-strain patch test: fragments touching the free surface (or
    /// the world) are driven by the affine motion `u = ε p`, interior
    /// fragments are solved for. Returns, per interior–interior bond, the
    /// relative error `|t_net − σ·n| / |σ|` of the raw traction `F/A`
    /// (σ = λ tr(ε) I + 2μ ε of the bond's first fragment). A consistent
    /// network returns zeros.
    pub fn patch_test(&self, asset: &Asset, level: u8, eps: DMat3) -> Vec<f64> {
        self.patch_test_with(asset, level, eps, None).into_iter().map(|x| x.1).collect()
    }

    /// Patch test with an explicit set of driven fragments (`None` = every
    /// fragment touching the free surface or the world). Returns
    /// `(bond centroid, error)` for every bond not between two driven
    /// fragments.
    pub fn patch_test_with(&self, asset: &Asset, level: u8, eps: DMat3, driven: Option<&dyn Fn(&Fragment) -> bool>) -> Vec<(DVec3, f64)> {
        let e = (eps + eps.transpose()) * 0.5;
        let (lam, mu) = self.lame(asset, asset.level_fragments(level)[0].id);
        let sig = DMat3::from_diagonal(DVec3::splat(lam * (e.x_axis.x + e.y_axis.y + e.z_axis.z))) + e * (2.0 * mu);
        let w = (eps - eps.transpose()) * 0.5;
        let omega = DVec3::new(w.y_axis.z, w.z_axis.x, w.x_axis.y);
        self.patch_test_field(asset, level, &|x| eps * x, &|_| omega, &|_| sig, driven)
    }

    /// Patch test against an exact elasticity solution: driven fragments
    /// follow `u(x)`, `ω(x)` (rigid motion sampled at the kinematic center);
    /// every other fragment is solved for. Returns `(bond centroid,
    /// |t_net − σ(c)·n| / |σ(c)|_F)` per bond not between two driven fragments
    /// (`None` drives every fragment with open surface; then only bonds
    /// between two undriven fragments are reported).
    #[allow(clippy::type_complexity)]
    pub fn patch_test_field(
        &self,
        asset: &Asset,
        level: u8,
        u_of: &dyn Fn(DVec3) -> DVec3,
        w_of: &dyn Fn(DVec3) -> DVec3,
        sig_of: &dyn Fn(DVec3) -> DMat3,
        driven: Option<&dyn Fn(&Fragment) -> bool>,
    ) -> Vec<(DVec3, f64)> {
        let r = asset.hierarchy.level_ranges[level as usize].clone();
        let n = (r.end - r.start) as usize;
        let centers = self.centers(asset, level);
        let (mut k, dim, _) = self.assemble(asset, level, &centers, true);
        // boundary fragments: open bonded surface
        let mut avec = vec![DVec3::ZERO; n];
        let mut asum = vec![0.0f64; n];
        let mut world = vec![false; n];
        for b in asset.level_bonds(level) {
            let ia = (b.a.0 - r.start) as usize;
            avec[ia] += b.normal * b.area;
            asum[ia] += b.area;
            match b.b {
                FragmentOrWorld::Fragment(f) => {
                    let ib = (f.0 - r.start) as usize;
                    avec[ib] -= b.normal * b.area;
                    asum[ib] += b.area;
                }
                FragmentOrWorld::World => world[ia] = true,
            }
        }
        let boundary: Vec<bool> = match driven {
            Some(d) => (0..n).map(|i| d(&asset.hierarchy.fragments[r.start as usize + i])).collect(),
            None => (0..n).map(|i| world[i] || avec[i].length() > 1e-6 * asum[i].max(1e-300)).collect(),
        };
        let mut f = vec![0.0f64; dim];
        let big = k.iter().cloned().fold(0.0, f64::max).max(1.0) * 1e10;
        for i in 0..n {
            if !boundary[i] {
                continue;
            }
            let x = asset.hierarchy.fragments[r.start as usize + i].mass.com;
            let p = centers[i];
            let th = w_of(p);
            let t = u_of(p) + th.cross(x - p);
            for a in 0..3 {
                k[(6 * i + a) * dim + 6 * i + a] += big;
                f[6 * i + a] += big * t[a];
                k[(6 * i + 3 + a) * dim + 6 * i + 3 + a] += big;
                f[6 * i + 3 + a] += big * th[a];
            }
        }
        let Some(u) = dense_solve(&k, &f, dim) else { return Vec::new() };
        let disp: Vec<(DVec3, DVec3)> = (0..n).map(|i| (DVec3::new(u[6 * i], u[6 * i + 1], u[6 * i + 2]), DVec3::new(u[6 * i + 3], u[6 * i + 4], u[6 * i + 5]))).collect();
        let ops = self.vol_ops(asset, level, &centers);
        let arms = Self::arms(asset, level, &centers);
        let tr: Vec<f64> = (0..n)
            .map(|i| if ops[i].terms.is_empty() { 0.0 } else {
                let g = Self::grad(&ops[i], i, &disp, &arms);
                g.x_axis.x + g.y_axis.y + g.z_axis.z
            })
            .collect();
        let mut errs = Vec::new();
        for b in asset.level_bonds(level) {
            let FragmentOrWorld::Fragment(fb) = b.b else { continue };
            let (ia, ib) = ((b.a.0 - r.start) as usize, (fb.0 - r.start) as usize);
            if if driven.is_some() { boundary[ia] && boundary[ib] } else { boundary[ia] || boundary[ib] } {
                continue;
            }
            let (da, db) = self.bond_lengths(b, &centers, r.start);
            let s = self.springs_with(asset, b, da, db);
            let xa = asset.fragment(b.a).mass.com;
            let xb = asset.fragment(fb).mass.com;
            let q = b.centroid;
            let d = disp[ib].0 + disp[ib].1.cross(q - xb) - disp[ia].0 - disp[ia].1.cross(q - xa);
            let fvec = b.normal * (s.kn * d.dot(b.normal)) + (d - b.normal * d.dot(b.normal)) * s.ks;
            let (lam, _) = self.lame(asset, b.a);
            let t_net = fvec / b.area + b.normal * (lam * 0.5 * (tr[ia] + tr[ib]));
            let sig = sig_of(b.centroid);
            let t_ex = sig * b.normal;
            let snorm = (0..3).map(|c| sig.col(c).length_squared()).sum::<f64>().sqrt();
            errs.push((b.centroid, (t_net - t_ex).length() / snorm.max(1e-300)));
        }
        errs
    }
}

impl<'a> BondNetworkSolver for ReferenceSolver<'a> {
    fn static_solve(&self, asset: &Asset, level: u8, loads: &LoadCase) -> NetworkResult {
        let r = asset.hierarchy.level_ranges[level as usize].clone();
        let n = (r.end - r.start) as usize;
        let centers = self.centers(asset, level);
        let (mut k, dim, springs) = self.assemble(asset, level, &centers, true);
        let mut f = vec![0.0f64; dim];
        for i in 0..n {
            let fr = &asset.hierarchy.fragments[r.start as usize + i];
            let g = loads.gravity * fr.mass.mass;
            f[6 * i] += g.x;
            f[6 * i + 1] += g.y;
            f[6 * i + 2] += g.z;
        }
        for (q, &(fid, force)) in loads.forces.iter().enumerate() {
            let i = (fid.0 - r.start) as usize;
            let x = asset.hierarchy.fragments[r.start as usize + i].mass.com;
            let m = loads.force_points.get(q).map(|p| (*p - x).cross(force)).unwrap_or(DVec3::ZERO);
            for a in 0..3 {
                f[6 * i + a] += force[a];
                f[6 * i + 3 + a] += m[a];
            }
        }
        for &(fid, m) in &loads.moments {
            let i = (fid.0 - r.start) as usize;
            for a in 0..3 {
                f[6 * i + 3 + a] += m[a];
            }
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
        let mut stress = vec![DMat3::ZERO; n];
        let sym = |f: DVec3, r: DVec3| -> DMat3 {
            let o = DMat3::from_cols(f * r.x, f * r.y, f * r.z);
            (o + o.transpose()) * 0.5
        };
        for (bid, s) in springs {
            let b = &asset.bonds[bid.idx()];
            let ia = (b.a.0 - r.start) as usize;
            let xa = asset.fragment(b.a).mass.com;
            let c = b.centroid;
            let q = b.centroid;
            let ua = disp[ia].0 + disp[ia].1.cross(q - xa);
            let (ub, thb) = match b.b {
                FragmentOrWorld::Fragment(fb) => {
                    let ib = (fb.0 - r.start) as usize;
                    let xb = asset.fragment(fb).mass.com;
                    (disp[ib].0 + disp[ib].1.cross(q - xb), disp[ib].1)
                }
                FragmentOrWorld::World => (DVec3::ZERO, DVec3::ZERO),
            };
            let d = ub - ua;
            let dth = thb - disp[ia].1;
            let fnrm = s.kn * d.dot(b.normal);
            let shear = (b.frame_u * d.dot(b.frame_u) + b.frame_v * d.dot(b.frame_v)) * s.ks;
            let moment = b.frame_u * (s.ktu * dth.dot(b.frame_u)) + b.frame_v * (s.ktv * dth.dot(b.frame_v)) + b.normal * (s.kt * dth.dot(b.normal));
            // force exerted on A by the bond (tension pulls A towards B)
            let f_on_a = b.normal * fnrm + shear;
            stress[ia] += sym(f_on_a, c - xa);
            if let FragmentOrWorld::Fragment(fb) = b.b {
                let ib = (fb.0 - r.start) as usize;
                let xb = asset.fragment(fb).mass.com;
                stress[ib] += sym(-f_on_a, c - xb);
            }
            forces.push(BondForce {
                bond: bid,
                normal_force: fnrm,
                shear_force: shear,
                moment,
                traction: fnrm / b.area.max(1e-300),
                traction_vec: f_on_a / b.area.max(1e-300),
                recovered_traction: DVec3::ZERO,
            });
        }
        // external loads at their application points
        for (k, &(fid, force)) in loads.forces.iter().enumerate() {
            let i = (fid.0 - r.start) as usize;
            let x = asset.fragment(fid).mass.com;
            let p = loads.force_points.get(k).copied().unwrap_or(x);
            stress[i] += sym(force, p - x);
        }
        for i in 0..n {
            let v = asset.hierarchy.fragments[r.start as usize + i].mass.volume.max(1e-300);
            stress[i] = stress[i] * (1.0 / v);
        }
        let mut tr_eps = vec![0.0f64; n];
        if self.model == StiffnessModel::Tensorial {
            let ops = self.vol_ops(asset, level, &centers);
            let arms = Self::arms(asset, level, &centers);
            for i in 0..n {
                if ops[i].terms.is_empty() {
                    continue;
                }
                let g = Self::grad(&ops[i], i, &disp, &arms);
                let e = (g + g.transpose()) * 0.5;
                let tr = e.x_axis.x + e.y_axis.y + e.z_axis.z;
                tr_eps[i] = tr;
                stress[i] = DMat3::from_diagonal(DVec3::splat(ops[i].lambda * tr)) + e * (2.0 * ops[i].mu);
            }
            // bond traction includes the volumetric pressure term
            for bf in forces.iter_mut() {
                let b = &asset.bonds[bf.bond.idx()];
                let ia = (b.a.0 - r.start) as usize;
                let (lam_a, _) = self.lame(asset, b.a);
                let pb = match b.b {
                    FragmentOrWorld::Fragment(fb) => 0.5 * (lam_a * tr_eps[ia] + self.lame(asset, fb).0 * tr_eps[(fb.0 - r.start) as usize]),
                    FragmentOrWorld::World => lam_a * tr_eps[ia],
                };
                bf.traction_vec += b.normal * pb;
                bf.traction += pb;
            }
        }
        for bf in forces.iter_mut() {
            let b = &asset.bonds[bf.bond.idx()];
            let ia = (b.a.0 - r.start) as usize;
            let sa = stress[ia];
            let sb = match b.b {
                FragmentOrWorld::Fragment(fb) => stress[(fb.0 - r.start) as usize],
                FragmentOrWorld::World => sa,
            };
            bf.recovered_traction = (sa + sb) * 0.5 * b.normal;
        }
        NetworkResult { level, displacements: disp, stress, bond_forces: forces }
    }

    fn modal(&self, asset: &Asset, level: u8, nmodes: usize) -> Vec<f64> {
        use faer::Mat;
        let centers = self.centers(asset, level);
        let (k, dim, _) = self.assemble(asset, level, &centers, false);
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
