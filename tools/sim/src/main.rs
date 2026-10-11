//! Rigid-body destruction demo for a baked prefracture asset: Shardwright's
//! bonds drive a stress solver, and Rapier moves the pieces.
//!
//! * Chunks: the fragments of the starting level (default L2, the support
//!   graph). Every chunk keeps its baked convex hulls and mass.
//! * Islands: chunks joined by intact bonds form one rigid body (a compound
//!   of their hulls). Islands bonded to the ground are fixed; the others are
//!   dynamic and fall, tumble and collide.
//! * Stress solver: an exact sparse Cholesky solve (faer, AMD ordering) of
//!   the anchored bond network under gravity and impact loads, with each bond's
//!   normal / shear / bending / torsion springs (the `Spec` model of
//!   `frac_validate::network`, from the bond's exact area, second moments and
//!   materials). A bond breaks when its force or moment exceeds its capacity:
//!   tensile / compressive strength x area x the bond's baked strength scale
//!   (Weibull size effect and boundary weakening). Islands split on breaks.
//!   Moving islands hit hard are solved too, free-free with inertial relief.
//! * Design check: before the impact, bonds that cannot carry 2x the
//!   self-weight load are reinforced (bonds model no rebar).
//! * Refinement: chunks in the projectile's path, or hit hard, are replaced
//!   by their children at the next level (down to the leaf), with their bonds
//!   re-made through the bond hierarchy, so detail appears where it breaks.
//!
//! Writes every chunk's pose per output frame as JSON for
//! `prefracture preview --sim`. See tools/sim/README.md.
use frac_core::{Asset, FragmentOrWorld, MaterialId};
use frac_fem::sparse::{CsrMatrix, SparseCholesky};
use frac_material::MaterialLibrary;
use frac_validate::network::{BondSprings, ReferenceSolver, StiffnessModel};
use glam::DVec3;
use rapier3d::prelude::*;
use std::collections::{HashMap, HashSet};

type V3 = rapier3d::math::Vector;

/// Bonds within this fraction of an island's worst utilization break
/// together in one stress round; the rest wait for the re-solve.
const BREAK_FRACTION: f64 = 0.7;

fn outer(a: DVec3, b: DVec3) -> glam::DMat3 {
    glam::DMat3::from_cols(a * b.x, a * b.y, a * b.z)
}

fn v3(p: DVec3) -> V3 {
    V3::new(p.x as f32, p.y as f32, p.z as f32)
}
fn d3(p: V3) -> DVec3 {
    DVec3::new(p.x as f64, p.y as f64, p.z as f64)
}

struct Args {
    asset: String,
    out: String,
    level: u8,
    seconds: f32,
    fps: f32,
    hz: f32,
    projectile: V3,
    velocity: V3,
    radius: f32,
    density: f32,
    strength_scale: f64,
    refine_dv: f32,
    impact_time: f64,
    design_margin: f64,
    rounds: usize,
    impact_rounds: usize,
    impact_g: f64,
    analyze: Option<String>,
    remove: String,
}

fn parse() -> Args {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: frac-sim ASSET.json OUT.json [options] (see tools/sim/README.md)");
        std::process::exit(2);
    }
    let get = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let vec = |k: &str, d: [f32; 3]| {
        get(k).map_or(V3::new(d[0], d[1], d[2]), |s| {
            let v: Vec<f32> = s.split(',').map(|x| x.trim().parse().unwrap()).collect();
            V3::new(v[0], v[1], v[2])
        })
    };
    let num = |k: &str, d: f64| get(k).map_or(d, |s| s.parse().unwrap());
    Args {
        asset: a[1].clone(),
        out: a[2].clone(),
        level: num("--level", 2.0) as u8,
        seconds: num("--seconds", 3.5) as f32,
        fps: num("--fps", 60.0) as f32,
        hz: num("--substep-hz", 240.0) as f32,
        projectile: vec("--projectile", [13.0, 1.4, 8.15]),
        velocity: vec("--velocity", [-15.0, 0.0, 0.0]),
        radius: num("--radius", 0.4) as f32,
        density: num("--density", 7800.0) as f32,
        strength_scale: num("--strength-scale", 1.0),
        refine_dv: num("--refine-dv", 3.0) as f32,
        impact_time: num("--impact-time", 0.02),
        design_margin: num("--design-margin", 2.0),
        rounds: num("--rounds", 1.0) as usize,
        impact_rounds: num("--impact-rounds", 8.0) as usize,
        impact_g: num("--impact-g", 3.0),
        analyze: get("--analyze"),
        remove: get("--remove").unwrap_or_else(|| "f0_col_21".into()),
    }
}

/// Capacities of one bond: tension, compression, shear (N) and bending (N m).
#[derive(Clone, Copy)]
struct Cap {
    t: f64,
    c: f64,
    s: f64,
    m: f64,
}

impl Cap {
    fn scaled(self, k: f64) -> Cap {
        Cap {
            t: self.t * k,
            c: self.c * k,
            s: self.s * k,
            m: self.m * k,
        }
    }
}

struct Island {
    members: Vec<u32>,
    body: RigidBodyHandle,
    anchored: bool,
}

struct Sim<'a> {
    a: &'a Asset,
    caps: Vec<Cap>,
    springs: Vec<BondSprings>,
    shapes: HashMap<u32, Vec<(SharedShape, f32)>>,
    phys: PhysicsWorld,
    active: HashSet<u32>,
    refined: HashSet<u32>,
    bonds: HashSet<u32>,
    broken: HashSet<u32>,
    frag_bonds: Vec<Vec<u32>>,
    islands: Vec<Island>,
    island_of: HashMap<u32, usize>,
    chunk_of_collider: HashMap<ColliderHandle, u32>,
    /// factorizations of the islands' stiffness matrices, keyed by nodes and bonds
    fact: HashMap<Vec<u32>, SparseCholesky>,
    reinforce: HashMap<u32, f64>,
    /// when set, stress() records each solved bond's [utilization, |force|,
    /// normal force (+ tension)] here instead of breaking anything
    record: Option<HashMap<u32, [f64; 3]>>,
}

impl<'a> Sim<'a> {
    /// Capacity factor from the design check: the bond's own, or else its
    /// nearest reinforced ancestor's (refined bonds keep their parent's).
    fn reinforcement(&self, b: u32) -> f64 {
        let mut b = Some(b);
        while let Some(x) = b {
            if let Some(f) = self.reinforce.get(&x) {
                return *f;
            }
            b = self.a.bonds[x as usize].parent_bond.map(|p| p.0);
        }
        1.0
    }

    fn frag(&self, f: u32) -> &frac_core::Fragment {
        &self.a.hierarchy.fragments[f as usize]
    }

    /// The active chunk holding fragment `f` (itself or an ancestor).
    fn active_of(&self, f: u32) -> Option<u32> {
        let mut cur = Some(f);
        while let Some(c) = cur {
            if self.active.contains(&c) {
                return Some(c);
            }
            cur = self.frag(c).parent.map(|p| p.0);
        }
        None
    }

    /// Ends of an active bond: (chunk a, Some(chunk b) or None for ground).
    fn ends(&self, b: u32) -> Option<(u32, Option<u32>)> {
        let bond = &self.a.bonds[b as usize];
        let a = self.active_of(bond.a.0)?;
        let o = match bond.b {
            FragmentOrWorld::Fragment(f) => Some(self.active_of(f.0)?),
            FragmentOrWorld::World => None,
        };
        Some((a, o))
    }

    /// Activates bond `b`, or, past a refined end, its child bonds.
    fn connect(&mut self, b: u32) {
        if self.broken.contains(&b) {
            return;
        }
        let bond = &self.a.bonds[b as usize];
        let other = match bond.b {
            FragmentOrWorld::Fragment(f) => Some(f.0),
            FragmentOrWorld::World => None,
        };
        if self.refined.contains(&bond.a.0) || other.is_some_and(|o| self.refined.contains(&o)) {
            let kids: Vec<u32> = bond
                .child_bonds
                .clone()
                .map(|k| self.a.bond_children[k as usize].0)
                .collect();
            for cb in kids {
                self.connect(cb);
            }
            return;
        }
        if let Some((x, y)) = self.ends(b) {
            if Some(x) != y {
                self.bonds.insert(b);
            }
        }
    }

    /// Replaces chunk `f` by its children (same island until the next split).
    fn refine(&mut self, f: u32) -> bool {
        if !self.active.contains(&f) || self.frag(f).children.is_empty() {
            return false;
        }
        let mine: Vec<u32> = self
            .bonds
            .iter()
            .copied()
            .filter(|b| self.ends(*b).is_some_and(|(x, y)| x == f || y == Some(f)))
            .collect();
        for b in &mine {
            self.bonds.remove(b);
        }
        self.active.remove(&f);
        self.refined.insert(f);
        let kids: Vec<u32> = self.frag(f).children.clone().collect();
        let isl = self.island_of.get(&f).copied();
        for &c in &kids {
            self.active.insert(c);
            if let Some(i) = isl {
                self.island_of.insert(c, i);
            }
        }
        // the chunk's interior is as strong as its design-checked surface
        let k = mine
            .iter()
            .map(|b| self.reinforcement(*b))
            .fold(1.0, f64::max);
        for &c in &kids {
            for b in self.frag_bonds[c as usize].clone() {
                let bond = &self.a.bonds[b as usize];
                if bond.a.0 == c {
                    if let FragmentOrWorld::Fragment(o) = bond.b {
                        if self.frag(o.0).parent.map(|p| p.0) == Some(f) {
                            if k > 1.0 {
                                self.reinforce.insert(b, k);
                            }
                            self.connect(b);
                        }
                    }
                }
            }
        }
        // the removed bonds again: through their child bonds where an end is
        // now refined, as they are where both ends are already finer than f
        for b in mine {
            self.connect(b);
        }
        true
    }

    fn chunk_shapes(&mut self, f: u32) -> Vec<(SharedShape, f32)> {
        if let Some(s) = self.shapes.get(&f) {
            return s.clone();
        }
        let fr = self.frag(f).clone();
        let mut out = Vec::new();
        let mut vol = 0.0f32;
        for hi in fr.hulls.clone() {
            let pts: Vec<V3> = self.a.hulls[hi as usize]
                .vertices
                .iter()
                .map(|p| v3(*p))
                .collect();
            if let Some(s) = SharedShape::convex_hull(&pts) {
                vol += s.mass_properties(1.0).mass();
                out.push((s, 0.0));
            }
        }
        let density = (fr.mass.mass as f32 / vol.max(1e-9)).max(1.0);
        for o in &mut out {
            o.1 = density;
        }
        self.shapes.insert(f, out.clone());
        out
    }

    /// Recomputes the islands from the intact bonds and rebuilds the bodies
    /// of the islands that changed. All bodies use the rest frame: a chunk's
    /// rest point x is at `pose * x` in the world.
    fn rebuild(&mut self) {
        let chunks: Vec<u32> = {
            let mut v: Vec<u32> = self.active.iter().copied().collect();
            v.sort_unstable();
            v
        };
        let idx: HashMap<u32, usize> = chunks.iter().enumerate().map(|(i, c)| (*c, i)).collect();
        let mut parent: Vec<usize> = (0..chunks.len()).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        let mut anchored = vec![false; chunks.len()];
        for &b in &self.bonds {
            if let Some((x, y)) = self.ends(b) {
                let i = idx[&x];
                match y {
                    Some(y) => {
                        let (ri, rj) = (find(&mut parent, i), find(&mut parent, idx[&y]));
                        if ri != rj {
                            parent[ri] = rj;
                        }
                    }
                    None => anchored[i] = true,
                }
            }
        }
        let mut groups: HashMap<usize, Vec<u32>> = HashMap::new();
        for (i, c) in chunks.iter().enumerate() {
            let r = find(&mut parent, i);
            groups.entry(r).or_default().push(*c);
        }
        let mut anch_root: HashSet<usize> = HashSet::new();
        for (i, &an) in anchored.iter().enumerate() {
            if an {
                anch_root.insert(find(&mut parent, i));
            }
        }
        // reuse islands whose members did not change
        let mut old: HashMap<Vec<u32>, usize> = HashMap::new();
        for (k, isl) in self.islands.iter().enumerate() {
            let mut m = isl.members.clone();
            m.sort_unstable();
            old.insert(m, k);
        }
        let mut keep = vec![false; self.islands.len()];
        let mut new_islands: Vec<Island> = Vec::new();
        let mut pending: Vec<(Vec<u32>, bool)> = Vec::new();
        let mut roots: Vec<usize> = groups.keys().copied().collect();
        roots.sort_unstable();
        for root in roots {
            let mut members = groups.remove(&root).unwrap();
            members.sort_unstable();
            let anch = anch_root.contains(&root);
            if let Some(&k) = old.get(&members) {
                keep[k] = true;
                new_islands.push(Island {
                    members,
                    body: self.islands[k].body,
                    anchored: self.islands[k].anchored,
                });
            } else {
                pending.push((members, anch));
            }
        }
        // the motion of the old islands, handed to the pieces that leave them
        let mut motion: HashMap<u32, (Pose, V3, V3, V3)> = HashMap::new();
        for (k, isl) in self.islands.iter().enumerate() {
            if keep[k] {
                continue;
            }
            let rb = &self.phys.bodies[isl.body];
            let m = (
                *rb.position(),
                rb.linvel(),
                rb.angvel(),
                rb.center_of_mass(),
            );
            for &c in &isl.members {
                motion.insert(c, m);
            }
        }
        for (k, isl) in self.islands.iter().enumerate() {
            if !keep[k] {
                self.phys.remove_body_with_colliders(isl.body, true);
            }
        }
        for (members, anch) in pending {
            // the parent island of these chunks (through refined ancestors)
            let src = members.iter().find_map(|c| {
                let mut cur = Some(*c);
                while let Some(x) = cur {
                    if let Some(m) = motion.get(&x) {
                        return Some(*m);
                    }
                    cur = self.frag(x).parent.map(|p| p.0);
                }
                None
            });
            let (pose, lv, av, com) = src.unwrap_or((Pose::IDENTITY, V3::ZERO, V3::ZERO, V3::ZERO));
            let rb = if anch {
                RigidBodyBuilder::fixed().pose(pose)
            } else {
                RigidBodyBuilder::dynamic().pose(pose).ccd_enabled(true)
            };
            let h = self.phys.insert_body(rb.build());
            for &c in &members {
                for (s, dens) in self.chunk_shapes(c) {
                    let ch = self.phys.insert_collider(
                        ColliderBuilder::new(s)
                            .density(dens)
                            .friction(0.7)
                            .restitution(0.05),
                        Some(h),
                    );
                    self.chunk_of_collider.insert(ch, c);
                }
            }
            if !anch {
                let b = &mut self.phys.bodies[h];
                let off = b.center_of_mass() - com;
                b.set_linvel(lv + av.cross(off), true);
                b.set_angvel(av, true);
            }
            new_islands.push(Island {
                members,
                body: h,
                anchored: anch,
            });
        }
        self.islands = new_islands;
        self.island_of.clear();
        for (k, isl) in self.islands.iter().enumerate() {
            for &c in &isl.members {
                self.island_of.insert(c, k);
            }
        }
        let live: HashSet<ColliderHandle> = self.phys.colliders.iter().map(|(h, _)| h).collect();
        self.chunk_of_collider.retain(|h, _| live.contains(h));
    }

    /// Stress solve of the anchored islands under gravity `g` and `loads`
    /// (chunk, force, point, in the island's rest frame), and of the moving
    /// islands hit with more than `impact_g` x their weight. A moving island
    /// is solved free-free with inertial relief: its contact loads are
    /// balanced by the d'Alembert forces of the acceleration they cause (free
    /// fall itself is stress-free).
    /// Breaks overloaded bonds and returns how many.
    ///
    /// With `design = Some(margin)` nothing breaks: instead every bond whose
    /// utilization exceeds 1 / margin is reinforced to carry margin x its
    /// load (the design check of the intact structure), and the count of
    /// reinforced bonds is returned.
    fn stress(
        &mut self,
        loads: &[(u32, DVec3, DVec3)],
        impact_g: f64,
        g: DVec3,
        design: Option<f64>,
    ) -> usize {
        let diag = std::env::var_os("SIM_DIAG").is_some();
        // the bonds of each island: (bond, i, Some(j) or None = ground)
        let mut by_island: HashMap<usize, Vec<u32>> = HashMap::new();
        for &b in &self.bonds {
            if let Some((x, _)) = self.ends(b) {
                if let Some(&k) = self.island_of.get(&x) {
                    by_island.entry(k).or_default().push(b);
                }
            }
        }
        let mut gone = Vec::new();
        let mut used: HashSet<Vec<u32>> = HashSet::new();
        for k in 0..self.islands.len() {
            let free = !self.islands[k].anchored;
            let nodes = self.islands[k].members.clone();
            let Some(mut bonds) = by_island.remove(&k) else {
                continue;
            };
            bonds.sort_unstable();
            let idx: HashMap<u32, usize> = nodes.iter().enumerate().map(|(i, c)| (*c, i)).collect();
            let list: Vec<(u32, usize, Option<usize>)> = bonds
                .iter()
                .filter_map(|&b| {
                    let (x, y) = self.ends(b)?;
                    Some((
                        b,
                        *idx.get(&x)?,
                        match y {
                            Some(y) => Some(*idx.get(&y)?),
                            None => None,
                        },
                    ))
                })
                .collect();
            let n = nodes.len();
            let com: Vec<DVec3> = nodes.iter().map(|c| self.frag(*c).mass.com).collect();
            let mass: Vec<f64> = nodes.iter().map(|c| self.frag(*c).mass.mass).collect();
            let inertia: Vec<glam::DMat3> =
                nodes.iter().map(|c| self.frag(*c).mass.inertia).collect();
            let mut f = vec![0.0f64; 6 * n];
            let add = |f: &mut [f64], i: usize, force: DVec3, moment: DVec3| {
                for d in 0..3 {
                    f[6 * i + d] += force[d];
                    f[6 * i + 3 + d] += moment[d];
                }
            };
            let mut total = 0.0;
            for &(c, force, at) in loads {
                if let Some(&i) = idx.get(&c) {
                    add(&mut f, i, force, (at - com[i]).cross(force));
                    total += force.length();
                }
            }
            if free {
                if total <= impact_g * mass.iter().sum::<f64>() * g.length() {
                    continue;
                }
                // inertial relief about the island's centre of mass
                let mt: f64 = mass.iter().sum();
                let cm = com.iter().zip(&mass).map(|(c, m)| *c * *m).sum::<DVec3>() / mt;
                let (mut ft, mut tq) = (DVec3::ZERO, DVec3::ZERO);
                let mut it = glam::DMat3::ZERO;
                for i in 0..n {
                    let fi = DVec3::new(f[6 * i], f[6 * i + 1], f[6 * i + 2]);
                    let mi = DVec3::new(f[6 * i + 3], f[6 * i + 4], f[6 * i + 5]);
                    let r = com[i] - cm;
                    ft += fi;
                    tq += mi + r.cross(fi);
                    it += inertia[i]
                        + (glam::DMat3::IDENTITY * r.length_squared() - outer(r, r)) * mass[i];
                }
                let acc = ft / mt;
                let alpha = if it.determinant().abs() > 1e-30 {
                    it.inverse() * tq
                } else {
                    DVec3::ZERO
                };
                for i in 0..n {
                    let r = com[i] - cm;
                    add(
                        &mut f,
                        i,
                        -(acc + alpha.cross(r)) * mass[i],
                        -(inertia[i] * alpha),
                    );
                }
            } else {
                for i in 0..n {
                    add(&mut f, i, g * mass[i], DVec3::ZERO);
                }
            }
            let a = self.a;
            let springs = &self.springs;
            // a bond's spring force and moment on side B from the relative
            // motion of the two sides at the bond centroid
            let bond_fm = |x: &[f64], b: u32, i: usize, j: Option<usize>| -> (DVec3, DVec3) {
                let bd = &a.bonds[b as usize];
                let s = springs[b as usize];
                let ri = bd.centroid - com[i];
                let ui = DVec3::new(x[6 * i], x[6 * i + 1], x[6 * i + 2]);
                let ti = DVec3::new(x[6 * i + 3], x[6 * i + 4], x[6 * i + 5]);
                let (pj, tj) = match j {
                    Some(j) => {
                        let rj = bd.centroid - com[j];
                        let tj = DVec3::new(x[6 * j + 3], x[6 * j + 4], x[6 * j + 5]);
                        (
                            DVec3::new(x[6 * j], x[6 * j + 1], x[6 * j + 2]) + tj.cross(rj),
                            tj,
                        )
                    }
                    None => (DVec3::ZERO, DVec3::ZERO),
                };
                let d = pj - (ui + ti.cross(ri));
                let th = tj - ti;
                let (nn, uu, vv) = (bd.normal, bd.frame_u, bd.frame_v);
                // restoring force on B (opposes the separation)
                let fo =
                    -(nn * (s.kn * d.dot(nn)) + uu * (s.ks * d.dot(uu)) + vv * (s.ks * d.dot(vv)));
                let mo = -(uu * (s.ktu * th.dot(uu))
                    + vv * (s.ktv * th.dot(vv))
                    + nn * (s.kt * th.dot(nn)));
                (fo, mo)
            };
            // each bond's 12x12 stiffness block over (u_i, theta_i, u_j,
            // theta_j), assembled column by column from the linear force law
            let key: Vec<u32> = nodes
                .iter()
                .copied()
                .chain([u32::MAX])
                .chain(list.iter().map(|x| x.0))
                .collect();
            if !self.fact.contains_key(&key) {
                let mut t: Vec<(usize, usize, f64)> = Vec::with_capacity(list.len() * 144 + 9 * n);
                let mut x = vec![0.0f64; 6 * n];
                let mut dmax = 0.0f64;
                for &(b, i, j) in &list {
                    let c = a.bonds[b as usize].centroid;
                    let dofs: Vec<usize> = (0..6)
                        .map(|d| 6 * i + d)
                        .chain(j.into_iter().flat_map(|j| (0..6).map(move |d| 6 * j + d)))
                        .collect();
                    for &col in &dofs {
                        x[col] = 1.0;
                        let (fo, mo) = bond_fm(&x, b, i, j);
                        x[col] = 0.0;
                        // K x = -(spring forces on the nodes): A gets +fo, B gets -fo
                        let ma = mo + (c - com[i]).cross(fo);
                        for d in 0..3 {
                            t.push((6 * i + d, col, fo[d]));
                            t.push((6 * i + 3 + d, col, ma[d]));
                            if 6 * i + d == col {
                                dmax = dmax.max(fo[d].abs());
                            }
                        }
                        if let Some(j) = j {
                            let mb = mo + (c - com[j]).cross(fo);
                            for d in 0..3 {
                                t.push((6 * j + d, col, -fo[d]));
                                t.push((6 * j + 3 + d, col, -mb[d]));
                            }
                        }
                    }
                }
                // mass-proportional springs to the ground: they keep K regular,
                // and as they are proportional to the inertia the balanced
                // loads of a free island excite no rigid motion through them
                let mmean = mass.iter().sum::<f64>() / n as f64;
                let eps = if free { 1e-7 } else { 1e-9 } * dmax.max(1.0) / mmean.max(1e-9);
                for i in 0..n {
                    for d in 0..3 {
                        t.push((6 * i + d, 6 * i + d, eps * mass[i]));
                        for e in 0..3 {
                            t.push((6 * i + 3 + d, 6 * i + 3 + e, eps * inertia[i].col(e)[d]));
                        }
                    }
                }
                let kmat = CsrMatrix::from_triplets(6 * n, 6 * n, t);
                // symmetrize the rounding noise of the column-wise assembly
                let kmat = kmat.add(0.5, &kmat.transpose(), 0.5);
                match SparseCholesky::new(&kmat) {
                    Ok(ch) => {
                        self.fact.insert(key.clone(), ch);
                    }
                    Err(e) => {
                        eprintln!("stress: {e}");
                        continue;
                    }
                }
            }
            let mut x = f;
            self.fact[&key].solve_in_place(&mut x);
            if diag {
                eprintln!(
                    "stress: island {k} ({}), {n} chunks, {} bonds, factor nnz {}",
                    if free { "moving" } else { "anchored" },
                    list.len(),
                    self.fact[&key].factor_nnz()
                );
            }
            used.insert(key);
            // utilization; only the most overloaded bonds of the island break
            // in one round (within `BREAK_FRACTION` of the worst), so the load
            // re-routes before the next ones go, as a crack advances
            let mut over: Vec<(u32, f64)> = Vec::new();
            for &(b, i, j) in &list {
                let (fo, mo) = bond_fm(&x, b, i, j);
                let bd = &a.bonds[b as usize];
                let cap = self.caps[b as usize].scaled(self.reinforcement(b));
                // fo is the force the bond exerts on B: along +n it pulls B
                // towards A only when the sides are being pulled apart (tension)
                let fnr = -fo.dot(bd.normal);
                let sh = (fo + bd.normal * fnr).length();
                let mb = (mo - bd.normal * mo.dot(bd.normal)).length();
                let u = (fnr.max(0.0) / cap.t + mb / cap.m)
                    .max((-fnr).max(0.0) / cap.c)
                    .max(sh / cap.s);
                if let Some(rec) = self.record.as_mut() {
                    rec.insert(b, [u, fo.length(), fnr]);
                    continue;
                }
                if let Some(margin) = design {
                    if u * margin > 1.0 {
                        self.reinforce.insert(b, self.reinforcement(b) * u * margin);
                        gone.push(b);
                    }
                    continue;
                }
                if u > 1.0 {
                    over.push((b, u));
                    if diag && gone.len() + over.len() < 12 {
                        eprintln!(
                            "  break {b} L{} {:?} area {:.4} ss {:.2}: Fn {:.0}/{:.0}|{:.0} shear {:.0}/{:.0} M {:.0}/{:.0} u {:.2}",
                            bd.level,
                            bd.composition.first().map(|c| c.kind),
                            bd.area,
                            bd.strength_scale,
                            fnr,
                            cap.t,
                            cap.c,
                            sh,
                            cap.s,
                            mb,
                            cap.m,
                            u
                        );
                    }
                }
            }
            let umax = over.iter().map(|x| x.1).fold(0.0, f64::max);
            gone.extend(
                over.iter()
                    .filter(|x| x.1 >= BREAK_FRACTION * umax)
                    .map(|x| x.0),
            );
        }
        // keep the factorizations of the islands that still exist
        let live: HashSet<&[u32]> = self.islands.iter().map(|i| i.members.as_slice()).collect();
        self.fact.retain(|key, _| {
            used.contains(key)
                || key
                    .iter()
                    .position(|x| *x == u32::MAX)
                    .is_some_and(|p| live.contains(&key[..p]))
        });
        if design.is_some() || self.record.is_some() {
            return gone.len();
        }
        for b in &gone {
            self.bonds.remove(b);
            self.broken.insert(*b);
        }
        gone.len()
    }
}

fn main() {
    let args = parse();
    let asset: Asset = serde_json::from_reader(std::io::BufReader::new(
        std::fs::File::open(&args.asset).unwrap(),
    ))
    .unwrap();
    let lib = MaterialLibrary::builtin();
    let solver = ReferenceSolver {
        lib: &lib,
        model: StiffnessModel::Spec,
    };
    let mat_of = |f: u32| {
        asset.hierarchy.fragments[f as usize]
            .material_mix
            .first()
            .map(|x| x.0)
            .unwrap_or(MaterialId(0))
    };
    let strengths = |m: MaterialId| {
        let mm = lib.material(m);
        let ft = mm.tensile_strength.unwrap_or(1e6);
        (ft, mm.compressive_strength.unwrap_or(10.0 * ft))
    };
    let caps: Vec<Cap> = asset
        .bonds
        .iter()
        .map(|b| {
            let (fta, fca) = strengths(mat_of(b.a.0));
            let (ftb, fcb) = match b.b {
                FragmentOrWorld::Fragment(o) => strengths(mat_of(o.0)),
                FragmentOrWorld::World => (fta, fca),
            };
            let ft = b
                .composition
                .first()
                .and_then(|c| c.interface_material)
                .and_then(|m| lib.interface_material(m))
                .and_then(|im| im.tensile_strength)
                .unwrap_or(fta.min(ftb));
            let fc = fca.min(fcb);
            let k = b.strength_scale as f64 * args.strength_scale;
            let c = b.extent.half[0].max(b.extent.half[1]).max(1e-6);
            // bearings (no tensile strength) resist by friction only
            let ft_eff = if ft > 0.0 { ft } else { 0.05 * fc };
            Cap {
                t: (ft_eff * b.area * k).max(1e-6),
                c: (fc * b.area * k).max(1e-6),
                s: (1.5 * ft_eff * b.area * k).max(1e-6),
                m: (ft_eff * b.i_uu.min(b.i_vv) / c * k).max(1e-9),
            }
        })
        .collect();
    let springs: Vec<BondSprings> = asset
        .bonds
        .iter()
        .map(|b| solver.springs(&asset, b))
        .collect();
    let mut frag_bonds = vec![Vec::new(); asset.hierarchy.fragments.len()];
    for b in &asset.bonds {
        frag_bonds[b.a.0 as usize].push(b.id.0);
        if let FragmentOrWorld::Fragment(o) = b.b {
            frag_bonds[o.0 as usize].push(b.id.0);
        }
    }
    let mut phys = PhysicsWorld::new();
    phys.integration_parameters.dt = 1.0 / args.hz;
    phys.integration_parameters.num_solver_iterations = 4;
    let ground = phys.insert_body(RigidBodyBuilder::fixed().build());
    phys.insert_collider(
        ColliderBuilder::cuboid(60.0, 0.5, 60.0)
            .translation(V3::new(4.0, -0.5, 4.0))
            .friction(0.9),
        Some(ground),
    );
    let mut sim = Sim {
        a: &asset,
        caps,
        springs,
        shapes: HashMap::new(),
        phys,
        active: HashSet::new(),
        refined: HashSet::new(),
        bonds: HashSet::new(),
        broken: HashSet::new(),
        frag_bonds,
        islands: Vec::new(),
        island_of: HashMap::new(),
        chunk_of_collider: HashMap::new(),
        fact: HashMap::new(),
        reinforce: HashMap::new(),
        record: None,
    };
    for f in asset.hierarchy.level_ranges[args.level as usize].clone() {
        sim.active.insert(f);
    }
    for b in &asset.bonds {
        if b.level == args.level {
            sim.connect(b.id.0);
        }
    }
    let g = DVec3::new(0.0, -9.81, 0.0);
    sim.rebuild();
    // design check: the intact structure carries its own weight with a
    // margin (bonds below it are reinforced, as a real design would); with
    // --design-margin 0, what cannot carry its own weight breaks instead
    let reinforced = if args.design_margin > 0.0 {
        sim.stress(&[], 0.0, g, Some(args.design_margin))
    } else {
        0
    };
    for _ in 0..8 {
        if sim.stress(&[], 0.0, g, None) == 0 {
            break;
        }
        sim.rebuild();
    }
    if args.analyze.is_some() {
        analyze(&mut sim, &args, g);
        return;
    }
    if let Ok(ids) = std::env::var("SIM_TEST_REFINE") {
        // diagnostic: refine the given chunks of the intact structure, solve, stop
        for id in ids.split(',') {
            sim.refine(id.parse().unwrap());
        }
        sim.rebuild();
        eprintln!("test refine: {} breaks", sim.stress(&[], 0.0, g, None));
        return;
    }
    let initial_broken = sim.broken.len();
    let proj = sim.phys.insert_body(
        RigidBodyBuilder::dynamic()
            .translation(args.projectile)
            .linvel(args.velocity)
            .ccd_enabled(true)
            .build(),
    );
    sim.phys.insert_collider(
        ColliderBuilder::ball(args.radius)
            .density(args.density)
            .friction(0.5)
            .restitution(0.1),
        Some(proj),
    );
    let proj_mass = sim.phys.bodies[proj].mass();
    let dt = 1.0 / args.hz;
    let steps = (args.seconds * args.hz) as usize;
    let every = (args.hz / args.fps).round().max(1.0) as usize;
    let leaf = asset.hierarchy.levels - 1;
    let mut frames = Vec::new();
    let mut refinements = 0usize;
    eprintln!(
        "start: {} chunks, {} bonds, {} reinforced by the design check, {} broke under self-weight",
        sim.active.len(),
        sim.bonds.len(),
        reinforced,
        initial_broken
    );
    for step in 0..=steps {
        if step % every == 0 {
            let mut bodies = Vec::new();
            for isl in &sim.islands {
                let p = *sim.phys.bodies[isl.body].position();
                for &c in &isl.members {
                    let t = p.rotation * v3(sim.frag(c).mass.com) + p.translation;
                    let q = p.rotation;
                    let r4 = |x: f32| (x * 1e4).round() / 1e4;
                    bodies.push(serde_json::json!([
                        c,
                        r4(t.x),
                        r4(t.y),
                        r4(t.z),
                        r4(q.x),
                        r4(q.y),
                        r4(q.z),
                        r4(q.w)
                    ]));
                }
            }
            let pp = sim.phys.bodies[proj].translation();
            frames.push(serde_json::json!({
                "t": step as f32 * dt,
                "bodies": bodies,
                "projectile": [pp.x, pp.y, pp.z, args.radius],
                "broken": sim.broken.len() - initial_broken,
                "refined": refinements,
                "islands": sim.islands.len(),
            }));
        }
        if step == steps {
            break;
        }
        let mut changed = false;
        let mut loads: Vec<(u32, DVec3, DVec3)> = Vec::new();
        // the projectile's path this step: refine down to the leaf, and load
        // the chunks it is about to hit with its momentum over the impact time
        let pv = sim.phys.bodies[proj].linvel();
        let pp = sim.phys.bodies[proj].translation();
        if pv.length() > args.refine_dv {
            let reach = Ball::new(args.radius + pv.length() * dt * 2.0);
            let at = Pose::from_translation(pp + pv * dt);
            for _ in 0..leaf {
                let near: Vec<u32> = {
                    let mut v: Vec<u32> = sim
                        .phys
                        .intersect_shape(at, &reach, QueryFilter::default())
                        .filter_map(|(h, _)| sim.chunk_of_collider.get(&h).copied())
                        .collect();
                    v.sort_unstable();
                    v.dedup();
                    v
                };
                let mut any = false;
                for c in near {
                    if sim.frag(c).level < leaf && sim.refine(c) {
                        if std::env::var_os("SIM_DIAG").is_some() {
                            eprintln!("  refine {c} (L{}) at step {step}", sim.frag(c).level);
                        }
                        refinements += 1;
                        any = true;
                    }
                }
                if !any {
                    break;
                }
                sim.rebuild();
                changed = true;
            }
            let hit: Vec<u32> = {
                let mut v: Vec<u32> = sim
                    .phys
                    .intersect_shape(at, &Ball::new(args.radius * 1.05), QueryFilter::default())
                    .filter_map(|(h, _)| sim.chunk_of_collider.get(&h).copied())
                    .collect();
                v.sort_unstable();
                v.dedup();
                v
            };
            if !hit.is_empty() {
                let force = d3(pv) * (proj_mass as f64 / args.impact_time) / hit.len() as f64;
                for c in hit {
                    loads.push((c, force, sim.frag(c).mass.com));
                }
            }
        }
        // contact forces of the last step: on the fixed structure from moving
        // bodies, and on moving islands from anything
        let mut refine_now: Vec<u32> = Vec::new();
        for pair in sim.phys.contact_pairs() {
            let j = d3(pair.total_impulse());
            let imp = j.length();
            if imp <= 0.0 {
                continue;
            }
            // the impulse pushes collider 2 along +j and collider 1 along -j
            for (me, other, sign) in [
                (pair.collider1, pair.collider2, -1.0),
                (pair.collider2, pair.collider1, 1.0),
            ] {
                let (Some(&c), Some(oh)) = (
                    sim.chunk_of_collider.get(&me),
                    sim.phys.colliders.get(other).and_then(|c| c.parent()),
                ) else {
                    continue;
                };
                let Some(&k) = sim.island_of.get(&c) else {
                    continue;
                };
                let anchored_me = sim.islands[k].anchored;
                let ob = &sim.phys.bodies[oh];
                let rot = sim.phys.bodies[sim.islands[k].body].rotation();
                let force = d3(rot.inverse() * v3(j * (sign / dt as f64)));
                if !anchored_me || !ob.is_fixed() {
                    loads.push((c, force, sim.frag(c).mass.com));
                }
                if !anchored_me {
                    // a moving chunk hit hard (by anything) shatters a level finer
                    let fr = sim.frag(c);
                    if fr.level < leaf
                        && imp > fr.mass.mass * args.refine_dv as f64
                        && !refine_now.contains(&c)
                    {
                        refine_now.push(c);
                    }
                }
            }
        }
        for c in refine_now.into_iter().take(16) {
            if sim.refine(c) {
                refinements += 1;
                changed = true;
            }
        }
        if changed {
            sim.rebuild();
        }
        if step % 2 == 0 || !loads.is_empty() {
            // rounds of break + re-solve under this step's loads (more while
            // something is hitting the structure); a cascade under gravity
            // alone carries on over the following steps
            let rounds = if loads.is_empty() {
                args.rounds
            } else {
                args.impact_rounds
            };
            for round in 0..rounds {
                let k = sim.stress(&loads, args.impact_g, g, None);
                if std::env::var_os("SIM_ROUNDS").is_some() && k > 0 {
                    eprintln!(
                        "  step {step} round {round}: {k} breaks ({} loads)",
                        loads.len()
                    );
                }
                if k == 0 {
                    break;
                }
                sim.rebuild();
            }
        }
        sim.phys.step();
        if step % (args.hz as usize / 10) == 0 {
            eprintln!(
                "t={:.2}s chunks={} bonds={} broken={} islands={} refined={}",
                step as f32 * dt,
                sim.active.len(),
                sim.bonds.len(),
                sim.broken.len() - initial_broken,
                sim.islands.len(),
                refinements
            );
        }
    }
    let out = serde_json::json!({ "fps": args.fps, "level": args.level, "frames": frames });
    std::fs::write(&args.out, serde_json::to_vec(&out).unwrap()).unwrap();
    eprintln!("wrote {} ({} frames)", args.out, frames.len());
}

/// `--analyze`: no simulation. Records every bond's load under gravity on
/// the design-checked structure, then again with the component `--remove`
/// taken out (its chunks' bonds removed), and writes both to OUT.json (the
/// `--analyze` path) for tools/explainer/bond_graph.py.
fn analyze(sim: &mut Sim, args: &Args, g: DVec3) {
    let a = sim.a;
    sim.record = Some(HashMap::new());
    sim.stress(&[], 0.0, g, None);
    let intact = sim.record.replace(HashMap::new()).unwrap();
    let comp = a
        .components
        .iter()
        .position(|c| c.name == args.remove)
        .unwrap_or_else(|| panic!("--remove: no component {}", args.remove));
    let in_comp = |f: u32| sim.frag(f).component.0 as usize == comp;
    let removed_chunks: Vec<u32> = sim.active.iter().copied().filter(|&f| in_comp(f)).collect();
    let cut: Vec<u32> = sim
        .bonds
        .iter()
        .copied()
        .filter(|&b| {
            sim.ends(b)
                .is_some_and(|(x, y)| in_comp(x) || y.is_some_and(in_comp))
        })
        .collect();
    for b in &cut {
        sim.bonds.remove(b);
    }
    sim.rebuild();
    sim.stress(&[], 0.0, g, None);
    let after = sim.record.take().unwrap();
    let mut rows = Vec::new();
    let mut ids: Vec<u32> = intact.keys().copied().collect();
    ids.sort_unstable();
    for b in ids {
        let bd = &a.bonds[b as usize];
        let (x, y) = sim.ends(b).unwrap_or((bd.a.0, None));
        let other = match bd.b {
            FragmentOrWorld::Fragment(_) => y.map_or(-1, |y| y as i64),
            FragmentOrWorld::World => -1,
        };
        let i = intact[&b];
        let kind = bd
            .composition
            .first()
            .map_or("Monolithic".to_string(), |c| format!("{:?}", c.kind));
        let r4 = |v: f64| (v * 1e4).round() / 1e4;
        let mut row = vec![
            serde_json::json!(b),
            serde_json::json!(x),
            serde_json::json!(other),
            serde_json::json!(r4(bd.area)),
            serde_json::json!(kind),
            serde_json::json!([r4(bd.centroid.x), r4(bd.centroid.y), r4(bd.centroid.z)]),
            serde_json::json!([r4(i[0]), i[1].round(), i[2].round()]),
        ];
        row.push(match after.get(&b) {
            Some(v) => serde_json::json!([r4(v[0]), v[1].round(), v[2].round()]),
            None => serde_json::Value::Null,
        });
        rows.push(serde_json::Value::Array(row));
    }
    let over = after.values().filter(|v| v[0] > 1.0).count();
    let umax0 = intact.values().map(|v| v[0]).fold(0.0, f64::max);
    eprintln!(
        "analyze: {} bonds; intact max utilization {umax0:.2}; without {} ({} chunks, {} bonds cut): {over} over capacity",
        rows.len(),
        args.remove,
        removed_chunks.len(),
        cut.len()
    );
    let out = serde_json::json!({
        "level": args.level,
        "removed": args.remove,
        "removed_chunks": removed_chunks,
        "columns": ["bond", "a", "b (-1 ground)", "area", "kind", "centroid", "intact [u, |F|, Fn]", "removed [u, |F|, Fn]"],
        "bonds": rows,
    });
    std::fs::write(
        args.analyze.as_ref().unwrap(),
        serde_json::to_vec(&out).unwrap(),
    )
    .unwrap();
}
