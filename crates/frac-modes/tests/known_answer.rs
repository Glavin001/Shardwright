//! Known-answer tests (spec §13.2, without reference code): synthetic solids
//! whose weak regions are known from mechanics.

mod common;
use common::*;
use frac_geom::{DVec2, DVec3};
use frac_modes::*;

fn top_groups(out: &ModesOutput, n: usize) -> Vec<usize> {
    let mj = out.max_jump();
    let mut order: Vec<usize> = (0..out.groups.len()).collect();
    order.sort_by(|&a, &b| mj[b].total_cmp(&mj[a]).then(a.cmp(&b)));
    order.truncate(n);
    order
}

fn dump(out: &ModesOutput, cen: &[[f64; 3]], n: usize) {
    let mj = out.max_jump();
    for g in top_groups(out, n) {
        eprintln!(
            "  g {:?} c [{:.2}, {:.2}, {:.2}] area {:.3} mj {:.4} jumps {:?}",
            out.groups[g],
            cen[g][0],
            cen[g][1],
            cen[g][2],
            out.group_area[g],
            mj[g],
            out.jumps
                .iter()
                .map(|j| (j[g] * 1e3).round() / 1e3)
                .collect::<Vec<_>>()
        );
    }
    eprintln!("  energies {:?}", out.energies);
}

fn translational(disc: Discretization) -> bool {
    disc == Discretization::CellPolynomial(0)
}

/// Number of modes for the "weakest cut" assertions. P1 modes beyond the
/// first are mostly other relative rigid motions of the same two fragments
/// (opening, sliding, hinging); translational modes are distinct cut
/// patterns (each the cheapest cut orthogonal to the previous ones), so the
/// weakest cut is the first mode alone and later, smaller pieces would
/// dominate the max-over-modes jumps.
fn k_for(disc: Discretization, k_p1: usize) -> usize {
    if translational(disc) { 1 } else { k_p1 }
}

fn admm(k: usize) -> ModesParams {
    ModesParams {
        k,
        solver: Solver::Admm,
        ..Default::default()
    }
}

/// Grid x-index of a cell label of an 8 x 2 x 2 grid with no empty cells.
fn ix_of(cell: u32) -> u32 {
    cell % 8
}

/// Sorted x-indices of the two cells of a group (staircased interfaces can
/// pair cells that are also offset in y/z).
fn ix_pair(a: u32, b: u32) -> (u32, u32) {
    (ix_of(a).min(ix_of(b)), ix_of(a).max(ix_of(b)))
}

fn notched_bar_cuts_at_notch_impl(disc: Discretization) {
    let solid = notched_bar();
    assert!(solid.topology().is_closed_manifold());
    let c = case(&solid, 0.25, [0.0, 0.0, 0.0], [4.0, 1.0, 1.0], [8, 2, 2]);
    assert_eq!(c.n_cells, 32);
    let out = run_disc(&c, &|_, _| 1.0, &[], admm(k_for(disc, 4)), disc);
    let cen = group_centroids(&c.mesh, &c.cells, &out.groups);
    dump(&out, &cen, 8);
    assert!(out.converged.iter().all(|&b| b));
    // the highest-jump interfaces all lie in the notch cross-section x = 2
    for g in top_groups(&out, 4) {
        let (a, b) = out.groups[g];
        assert_eq!(
            ix_pair(a, b),
            (3, 4),
            "group {:?} at {:?} is not at the notch",
            out.groups[g],
            cen[g]
        );
    }
    // Level-1 with target 2 splits the bar at the notch
    let l1 = segment_level1(c.n_cells, &out.groups, &out.max_jump(), 2);
    assert!(l1.hit_target);
    assert_eq!(l1.n_fragments, 2);
    for cell in 0..32u32 {
        assert_eq!(
            l1.labels[cell as usize],
            u32::from(ix_of(cell) >= 4),
            "cell {cell}"
        );
    }
    // cut groups are exactly at the notch
    for (g, &(a, b)) in out.groups.iter().enumerate() {
        if l1.cut_groups[g] {
            assert_eq!(ix_pair(a, b), (3, 4));
        }
    }
}

fn notched_bar_anchored_end_cuts_at_notch_impl(disc: Discretization) {
    let solid = notched_bar();
    let c = case(&solid, 0.25, [0.0, 0.0, 0.0], [4.0, 1.0, 1.0], [8, 2, 2]);
    let anchors: Vec<u32> = (0..c.mesh.verts.len() as u32)
        .filter(|&v| c.mesh.verts[v as usize][0] < 1e-9)
        .collect();
    assert!(!anchors.is_empty());
    let out = run_disc(&c, &|_, _| 1.0, &anchors, admm(k_for(disc, 2)), disc);
    let cen = group_centroids(&c.mesh, &c.cells, &out.groups);
    dump(&out, &cen, 6);
    let g = top_groups(&out, 1)[0];
    let (a, b) = out.groups[g];
    assert_eq!(ix_pair(a, b), (3, 4));
    let l1 = segment_level1(c.n_cells, &out.groups, &out.max_jump(), 2);
    assert_eq!(l1.n_fragments, 2);
    for cell in 0..32u32 {
        assert_eq!(l1.labels[cell as usize], u32::from(ix_of(cell) >= 4));
    }
}

fn l_shape_weak_at_reentrant_corner_impl(disc: Discretization) {
    let poly = [
        DVec2::new(0.0, 0.0),
        DVec2::new(2.0, 0.0),
        DVec2::new(2.0, 1.0),
        DVec2::new(1.0, 1.0),
        DVec2::new(1.0, 2.0),
        DVec2::new(0.0, 2.0),
    ];
    let solid = extrude(&poly, 0.0, 0.5);
    assert!(solid.topology().is_closed_manifold());
    let c = case(&solid, 0.2, [0.0, 0.0, 0.0], [2.0, 2.0, 0.5], [4, 4, 1]);
    let out = run_disc(&c, &|_, _| 1.0, &[], admm(3), disc);
    let cen = group_centroids(&c.mesh, &c.cells, &out.groups);
    dump(&out, &cen, 8);
    // the weakest interfaces are the arm roots: the sections through the
    // re-entrant corner (x = 1, y < 1) and (y = 1, x < 1)
    let on_root = |p: [f64; 3]| {
        ((p[0] - 1.0).abs() < 0.1 && p[1] < 1.05) || ((p[1] - 1.0).abs() < 0.1 && p[0] < 1.05)
    };
    let mj = out.max_jump();
    let g = top_groups(&out, 1)[0];
    assert!(
        on_root(cen[g]),
        "top group {:?} at {:?} is not at the re-entrant corner",
        out.groups[g],
        cen[g]
    );
    // and they dominate the interfaces far from the corner
    let far: Vec<f64> = (0..out.groups.len())
        .filter(|&g| {
            let p = cen[g];
            !on_root(p) && ((p[0] - 1.0).powi(2) + (p[1] - 1.0).powi(2)).sqrt() > 0.6
        })
        .map(|g| mj[g])
        .collect();
    let far_max = far.iter().cloned().fold(0.0, f64::max);
    assert!(mj[g] > 1.2 * far_max, "corner {} vs far {}", mj[g], far_max);
    let l1 = segment_level1(c.n_cells, &out.groups, &mj, 2);
    assert_eq!(l1.n_fragments, 2);
    for (k, cut) in l1.cut_groups.iter().enumerate() {
        if *cut {
            assert!(
                on_root(cen[k]) || out.group_area[k] < 0.05,
                "cut group {:?} at {:?}",
                out.groups[k],
                cen[k]
            );
        }
    }
}

fn plate_with_hole_ring_is_weak_impl(disc: Discretization) {
    let n = 48;
    let mut outer = Vec::new();
    let mut inner = Vec::new();
    for i in 0..n {
        let a = 2.0 * std::f64::consts::PI * (i as f64 + 0.5) / n as f64;
        let (s, co) = (libm::sin(a), libm::cos(a));
        let t = 1.0 / co.abs().max(s.abs());
        outer.push(DVec2::new(co * t, s * t));
        inner.push(DVec2::new(0.3 * co, 0.3 * s));
    }
    let solid = extrude_ring(&outer, &inner, 0.0, 0.3);
    assert!(solid.topology().is_closed_manifold());
    assert!(solid.signed_volume() > 0.0);
    let c = case(&solid, 0.15, [-1.0, -1.0, 0.0], [1.0, 1.0, 0.3], [4, 4, 1]);
    assert_eq!(c.n_cells, 16);
    let out = run_disc(&c, &|_, _| 1.0, &[], admm(3), disc);
    let cen = group_centroids(&c.mesh, &c.cells, &out.groups);
    dump(&out, &cen, 8);
    // cells (ix, iy) in 1..=2 form the ring around the hole: labels 5, 6, 9, 10
    let ring = |c: u32| matches!(c, 5 | 6 | 9 | 10);
    let mj = out.max_jump();
    let (mut ring_s, mut ring_n, mut far_s, mut far_n) = (0.0, 0, 0.0, 0);
    for (g, &(a, b)) in out.groups.iter().enumerate() {
        if ring(a) && ring(b) {
            ring_s += mj[g];
            ring_n += 1;
        } else if !ring(a) && !ring(b) {
            far_s += mj[g];
            far_n += 1;
        }
    }
    let (ring_mean, far_mean) = (ring_s / ring_n as f64, far_s / far_n as f64);
    eprintln!("  ring mean {ring_mean} ({ring_n}) far mean {far_mean} ({far_n})");
    assert_eq!(ring_n, 4);
    assert!(
        ring_mean > 1.5 * far_mean,
        "ring {ring_mean} vs far {far_mean}"
    );
    // the single weakest interface crosses a hole ligament. Per-cell
    // translations open a straight cut across the plate through the hole with
    // the same jump on every interface of the cut, so there a ligament
    // interface must attain the maximum jump
    let g = top_groups(&out, 1)[0];
    if translational(disc) {
        let top = mj[g];
        assert!(
            (0..out.groups.len()).any(|h| ring(out.groups[h].0)
                && ring(out.groups[h].1)
                && mj[h] >= top * (1.0 - 1e-6)),
            "no ligament interface attains the max jump {top}"
        );
    } else {
        assert!(
            ring(out.groups[g].0) && ring(out.groups[g].1),
            "top group {:?}",
            out.groups[g]
        );
    }
}

fn material_weight_moves_cut_and_forbidden_never_cut_impl(disc: Discretization) {
    let solid = frac_geom::mesh::box_mesh(DVec3::ZERO, DVec3::new(4.0, 1.0, 1.0));
    let c = case(&solid, 0.25, [0.0, 0.0, 0.0], [4.0, 1.0, 1.0], [8, 2, 2]);
    assert_eq!(c.n_cells, 32);
    let section = |ga: u32, gb: u32| move |a: u32, b: u32| ix_pair(a, b) == (ga, gb);
    let at_center = section(3, 4);
    let at_one = section(1, 2);

    // geometric baseline: uniform bar breaks in the middle (checked on the
    // fast reduced discretization only, to keep the full test under budget)
    if disc != Discretization::Full {
        let base = run_disc(&c, &|_, _| 1.0, &[], admm(k_for(disc, 2)), disc);
        dump(&base, &group_centroids(&c.mesh, &c.cells, &base.groups), 4);
        let l1 = segment_level1(c.n_cells, &base.groups, &base.max_jump(), 2);
        assert_eq!(l1.n_fragments, 2);
        for (g, &(a, b)) in base.groups.iter().enumerate() {
            if l1.cut_groups[g] {
                assert!(at_center(a, b), "baseline cut {:?}", (a, b));
            }
        }
    }

    // material-aware: a weak interface (G_f ratio 0.01 -> w = 0.1) at x = 1
    let weak = move |a: u32, b: u32| if at_one(a, b) { 0.1 } else { 1.0 };
    let out = run_disc(&c, &weak, &[], admm(k_for(disc, 2)), disc);
    let cen = group_centroids(&c.mesh, &c.cells, &out.groups);
    dump(&out, &cen, 4);
    for g in top_groups(&out, 4) {
        let (a, b) = out.groups[g];
        assert!(
            at_one(a, b),
            "weak-section group expected, got {:?}",
            (a, b)
        );
    }
    let l1 = segment_level1(c.n_cells, &out.groups, &out.max_jump(), 2);
    assert_eq!(l1.n_fragments, 2);
    for cell in 0..32u32 {
        assert_eq!(
            l1.labels[cell as usize],
            u32::from(ix_of(cell) >= 2),
            "cell {cell}"
        );
    }

    // forbidden interfaces at the natural break (x = 2) never open
    let forb = move |a: u32, b: u32| if at_center(a, b) { f64::INFINITY } else { 1.0 };
    let out = run_disc(&c, &forb, &[], admm(k_for(disc, 2)), disc);
    dump(&out, &cen, 4);
    let mut n_forb = 0;
    for (g, &(a, b)) in out.groups.iter().enumerate() {
        if at_center(a, b) {
            n_forb += 1;
            for j in &out.jumps {
                assert!(
                    j[g].abs() < 1e-12,
                    "forbidden group {:?} has jump {}",
                    (a, b),
                    j[g]
                );
            }
        }
    }
    assert!(n_forb >= 4);
    let l1 = segment_level1(c.n_cells, &out.groups, &out.max_jump(), 2);
    assert_eq!(l1.n_fragments, 2);
    for (g, &(a, b)) in out.groups.iter().enumerate() {
        if at_center(a, b) {
            assert!(!l1.cut_groups[g]);
        }
    }
    // the two fragments are still separated cleanly somewhere else
    assert_ne!(l1.labels[0], l1.labels[7]);
}

#[test]
fn notched_bar_cuts_at_notch() {
    notched_bar_cuts_at_notch_impl(Discretization::Full);
}

#[test]
fn notched_bar_cuts_at_notch_reduced() {
    notched_bar_cuts_at_notch_impl(Discretization::CellPolynomial(1));
}

#[test]
fn notched_bar_anchored_end_cuts_at_notch() {
    notched_bar_anchored_end_cuts_at_notch_impl(Discretization::Full);
}

#[test]
fn notched_bar_anchored_end_cuts_at_notch_reduced() {
    notched_bar_anchored_end_cuts_at_notch_impl(Discretization::CellPolynomial(1));
}

#[test]
fn l_shape_weak_at_reentrant_corner() {
    l_shape_weak_at_reentrant_corner_impl(Discretization::Full);
}

#[test]
fn l_shape_weak_at_reentrant_corner_reduced() {
    l_shape_weak_at_reentrant_corner_impl(Discretization::CellPolynomial(1));
}

#[test]
fn plate_with_hole_ring_is_weak() {
    plate_with_hole_ring_is_weak_impl(Discretization::Full);
}

#[test]
fn plate_with_hole_ring_is_weak_reduced() {
    plate_with_hole_ring_is_weak_impl(Discretization::CellPolynomial(1));
}

#[test]
fn material_weight_moves_cut_and_forbidden_never_cut() {
    material_weight_moves_cut_and_forbidden_never_cut_impl(Discretization::Full);
}

#[test]
fn material_weight_moves_cut_and_forbidden_never_cut_reduced() {
    material_weight_moves_cut_and_forbidden_never_cut_impl(Discretization::CellPolynomial(1));
}

// The default model: the paper's per-cell translations (§3.6 of Sellán et
// al.), area-weighted interfaces, vector-Laplacian ICCM start.

#[test]
fn notched_bar_cuts_at_notch_translational() {
    notched_bar_cuts_at_notch_impl(Discretization::CellPolynomial(0));
}

#[test]
fn notched_bar_anchored_end_cuts_at_notch_translational() {
    notched_bar_anchored_end_cuts_at_notch_impl(Discretization::CellPolynomial(0));
}

#[test]
fn l_shape_weak_at_reentrant_corner_translational() {
    l_shape_weak_at_reentrant_corner_impl(Discretization::CellPolynomial(0));
}

#[test]
fn plate_with_hole_ring_is_weak_translational() {
    plate_with_hole_ring_is_weak_impl(Discretization::CellPolynomial(0));
}

#[test]
fn material_weight_moves_cut_and_forbidden_never_cut_translational() {
    material_weight_moves_cut_and_forbidden_never_cut_impl(Discretization::CellPolynomial(0));
}
