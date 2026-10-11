//! Cell-polynomial Galerkin reduction of the exploded problem.
//!
//! The fracture-mode subproblem is posed on the cell-exploded P1 space. Its
//! minimizers are (for small ω) nearly piecewise rigid per cell, with small,
//! smooth elastic deformation inside each cell. The reduction restricts the
//! displacement of every analysis cell (or super-cell: cells merged across
//! forbidden interfaces) to the nodal interpolant of a vector polynomial of
//! degree `p` (p = 1: 12 DOFs per cell incl. the 6 rigid modes; p = 2: 30),
//! i.e. `u = Φ α` with `Φ` a sparse prolongation onto the exploded P1 DOFs.
//! It is a genuine subspace of the full discretization, so:
//!
//! * `Q_r = Φᵀ Q̂ Φ` (exact Galerkin projection, block-diagonal per cell),
//! * `Φ` is chosen M̂-orthonormal per cell, so `M_r = I`,
//! * anchored copies stay exactly zero (each cell's polynomial space is
//!   restricted to the null space of its anchored nodal values),
//! * group norms are exact: `‖B̂_g Φ α‖ = ‖R_g α‖` where `R_gᵀR_g` is the
//!   (rank-revealed) Gram matrix of `B̂_g Φ` on the ≤ 2·3·s columns of the two
//!   cells involved (≤ 60 rows per group instead of 9 per fault face),
//! * rigid-mode rows, mode-orthogonality rows and initial vectors are the
//!   M̂-orthogonal projections of their full counterparts.
//!
//! Energies of the reduced problem are upper bounds of the full ones. The
//! number of unknowns drops from `3 × #exploded nodes` to `≈ 3 s × #cells`,
//! independent of the tet count.
//!
//! Degree 0 is the translational model of Sellán et al. (§3.6): one constant
//! displacement per (super-)cell, whose strain energy vanishes (`Q_r = 0`).
//! It is solved for a single displacement component (x): the objective
//! `Σ w_g ‖u_a − u_b‖` is isotropic and the constraints (rigid translations,
//! anchors, forbidden interfaces, mode orthogonality against single-axis
//! modes) separate per axis, so ICCM started from `φ e_x` stays in the
//! x component and the vector modes are exactly the scalar modes times
//! `e_x`, `e_y`, `e_z` (degenerate direction triples). Solving one component
//! gives `k` distinct cut patterns for `k` modes at a third of the unknowns.

use crate::problem::{NodeInfo, Problem, active_groups};
use frac_fem::dense;
use frac_fem::sparse::CsrMatrix;
use std::ops::Range;

fn monomials(x: [f64; 3], degree: u8, out: &mut Vec<f64>) {
    out.clear();
    out.push(1.0);
    if degree >= 1 {
        out.extend_from_slice(&[x[0], x[1], x[2]]);
    }
    if degree >= 2 {
        out.extend_from_slice(&[
            x[0] * x[0],
            x[1] * x[1],
            x[2] * x[2],
            x[0] * x[1],
            x[1] * x[2],
            x[2] * x[0],
        ]);
    }
}

/// Scalar basis of one super-cell: `phi[node][i]` for its free nodes.
struct CellBasis {
    nodes: Vec<usize>,
    s: usize,
    phi: Vec<f64>, // nodes.len() x s, row-major
    offset: usize, // first reduced column; component k uses offset + k*s .. + s
}

pub(crate) fn reduce(
    full: &Problem,
    info: &NodeInfo,
    degree: u8,
    omega: f64,
) -> Result<Problem, String> {
    let degree = degree.min(2);
    // displacement components carried by the reduced space: the translational
    // model (degree 0) is solved for one component (see below), P1/P2 for 3
    let ncomp = if degree == 0 { 1 } else { 3 };
    let nf = full.n / 3;
    assert_eq!(info.super_cell.len(), nf);
    let ns = info.n_super;
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); ns];
    for f in 0..nf {
        members[info.super_cell[f] as usize].push(f);
    }
    let mut anchored: Vec<Vec<[f64; 3]>> = vec![Vec::new(); ns];
    for (k, &sc) in info.anchored_super.iter().enumerate() {
        anchored[sc as usize].push(info.anchored_pos[k]);
    }
    let mut cells: Vec<CellBasis> = Vec::with_capacity(ns);
    let mut offset = 0usize;
    let mut mono = Vec::new();
    for sc in 0..ns {
        let nodes = std::mem::take(&mut members[sc]);
        if nodes.is_empty() {
            cells.push(CellBasis {
                nodes,
                s: 0,
                phi: Vec::new(),
                offset,
            });
            continue;
        }
        // local normalized coordinates
        let mut c = [0.0; 3];
        for &f in &nodes {
            for k in 0..3 {
                c[k] += info.pos[f][k];
            }
        }
        for v in &mut c {
            *v /= nodes.len() as f64;
        }
        let mut r2 = 0.0f64;
        for &f in &nodes {
            let d: f64 = (0..3).map(|k| (info.pos[f][k] - c[k]).powi(2)).sum();
            r2 = r2.max(d);
        }
        let scale = if r2 > 0.0 { 1.0 / r2.sqrt() } else { 1.0 };
        let loc = |p: [f64; 3]| {
            [
                (p[0] - c[0]) * scale,
                (p[1] - c[1]) * scale,
                (p[2] - c[2]) * scale,
            ]
        };
        let s0 = [1, 4, 10][degree as usize];
        // null space of anchored nodal values
        let mut nmat: Vec<f64> = (0..s0 * s0)
            .map(|i| if i % (s0 + 1) == 0 { 1.0 } else { 0.0 })
            .collect();
        let mut s1 = s0;
        if !anchored[sc].is_empty() {
            let mut g = vec![0.0; s0 * s0];
            for &p in &anchored[sc] {
                monomials(loc(p), degree, &mut mono);
                for i in 0..s0 {
                    for j in 0..s0 {
                        g[i * s0 + j] += mono[i] * mono[j];
                    }
                }
            }
            let (w, v) = dense::sym_eigen(&g, s0);
            let wmax = w.iter().cloned().fold(0.0, f64::max).max(1e-300);
            let keep: Vec<usize> = (0..s0).filter(|&i| w[i] <= 1e-10 * wmax).collect();
            s1 = keep.len();
            nmat = vec![0.0; s0 * s1];
            for (jj, &j) in keep.iter().enumerate() {
                for i in 0..s0 {
                    nmat[i * s1 + jj] = v[i * s0 + j];
                }
            }
        }
        // q values at nodes (nodes x s1)
        let mut qv = vec![0.0; nodes.len() * s1];
        for (a, &f) in nodes.iter().enumerate() {
            monomials(loc(info.pos[f]), degree, &mut mono);
            for jj in 0..s1 {
                let mut acc = 0.0;
                for i in 0..s0 {
                    acc += mono[i] * nmat[i * s1 + jj];
                }
                qv[a * s1 + jj] = acc;
            }
        }
        // M̂-orthonormalize (per component; masses are equal per component)
        let mut g = vec![0.0; s1 * s1];
        for (a, &f) in nodes.iter().enumerate() {
            let mf = full.m[3 * f];
            for i in 0..s1 {
                for j in 0..s1 {
                    g[i * s1 + j] += mf * qv[a * s1 + i] * qv[a * s1 + j];
                }
            }
        }
        let (w, v) = dense::sym_eigen(&g, s1);
        let wmax = w.iter().cloned().fold(0.0, f64::max).max(1e-300);
        let keep: Vec<usize> = (0..s1).filter(|&i| w[i] > 1e-12 * wmax).collect();
        let s2 = keep.len();
        let mut phi = vec![0.0; nodes.len() * s2];
        for a in 0..nodes.len() {
            for (ii, &i) in keep.iter().enumerate() {
                let mut acc = 0.0;
                for j in 0..s1 {
                    acc += qv[a * s1 + j] * v[j * s1 + i];
                }
                phi[a * s2 + ii] = acc / w[i].sqrt();
            }
        }
        cells.push(CellBasis {
            nodes,
            s: s2,
            phi,
            offset,
        });
        offset += ncomp * s2;
    }
    let nr = offset;
    if nr == 0 {
        return Err("reduced space is empty (everything anchored?)".into());
    }
    // node -> (cell, local index)
    let mut node_loc = vec![(0usize, 0usize); nf];
    for (sc, cb) in cells.iter().enumerate() {
        for (a, &f) in cb.nodes.iter().enumerate() {
            node_loc[f] = (sc, a);
        }
    }
    // prolongation Φ (full x reduced)
    let mut trip = Vec::new();
    for f in 0..nf {
        let (sc, a) = node_loc[f];
        let cb = &cells[sc];
        for k in 0..ncomp {
            for i in 0..cb.s {
                trip.push((3 * f + k, cb.offset + k * cb.s + i, cb.phi[a * cb.s + i]));
            }
        }
    }
    let phi = CsrMatrix::from_triplets(full.n, nr, trip);
    let phit = phi.transpose();
    let qr = if degree == 0 {
        // per-cell translations lie in the null space of the strain energy
        CsrMatrix::from_triplets(nr, nr, Vec::new())
    } else {
        let qr = phit.matmul(&full.q.matmul(&phi));
        qr.add(0.5, &qr.transpose(), 0.5)
    };

    // compressed group operators
    let mut rows_all: Vec<Range<usize>> = Vec::with_capacity(full.groups.len());
    let mut btrip: Vec<(usize, usize, f64)> = Vec::new();
    let mut row = 0usize;
    let mut cols: Vec<usize> = Vec::new(); // reduced comp-0 columns of this group
    let mut local = Vec::new();
    for g in 0..full.groups.len() {
        let start = row;
        let r = full.rows_all[g].clone();
        // collect columns
        cols.clear();
        for rr in r.clone() {
            if (rr - r.start) % 3 != 0 {
                continue;
            }
            let (c, _) = full.b_all.row(rr);
            for &d in c {
                let (sc, _) = node_loc[d / 3];
                let cb = &cells[sc];
                for i in 0..cb.s {
                    cols.push(cb.offset + i);
                }
            }
        }
        cols.sort_unstable();
        cols.dedup();
        let nc = cols.len();
        if nc > 0 {
            let mut gm = vec![0.0; nc * nc];
            local.resize(nc, 0.0);
            for rr in r.clone() {
                if (rr - r.start) % 3 != 0 {
                    continue;
                }
                local.iter_mut().for_each(|x| *x = 0.0);
                let (c, v) = full.b_all.row(rr);
                for e in 0..c.len() {
                    debug_assert_eq!(c[e] % 3, 0);
                    let (sc, a) = node_loc[c[e] / 3];
                    let cb = &cells[sc];
                    for i in 0..cb.s {
                        let col = cb.offset + i;
                        let li = cols.binary_search(&col).unwrap();
                        local[li] += v[e] * cb.phi[a * cb.s + i];
                    }
                }
                for i in 0..nc {
                    if local[i] != 0.0 {
                        for j in 0..nc {
                            gm[i * nc + j] += local[i] * local[j];
                        }
                    }
                }
            }
            let (w, vecs) = dense::sym_eigen(&gm, nc);
            let wmax = w.iter().cloned().fold(0.0, f64::max);
            if wmax > 0.0 {
                for i in 0..nc {
                    if w[i] > 1e-13 * wmax {
                        let sw = w[i].sqrt();
                        for k in 0..ncomp {
                            for (j, &col) in cols.iter().enumerate() {
                                // shift component-0 column to component k of the same cell
                                let (sc, _) = owner(&cells, col, ncomp);
                                let cb = &cells[sc];
                                let v = sw * vecs[j * nc + i];
                                if v != 0.0 {
                                    btrip.push((row, col + k * cb.s, v));
                                }
                            }
                            row += 1;
                        }
                    }
                }
            }
        }
        rows_all.push(start..row);
    }
    let b_all = CsrMatrix::from_triplets(row, nr, btrip);
    let (active, b_act, rows_act, lam_act) =
        active_groups(&b_all, &rows_all, &full.group_weight, omega);
    // Rigid-mode rows: only rigid modes contained in the subspace constrain
    // it (all six for degree >= 1). For degree 0 (per-cell translations of
    // one component, the paper's §3.6 space) the rotations are not
    // representable and are not zero-energy there, so only the
    // x translation is kept.
    let rigid_rows: Vec<Vec<f64>> = full
        .rigid_rows
        .iter()
        .map(|r| phit.mul(r))
        .filter(|a| a.iter().map(|x| x * x).sum::<f64>() > 1.0 - 1e-9)
        .collect();
    let mut init = Vec::with_capacity(full.init.len());
    for c in &full.init {
        let mc: Vec<f64> = (0..full.n).map(|i| full.m[i] * c[i]).collect();
        let mut a = phit.mul(&mc);
        let nn = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nn > 0.0 {
            a.iter_mut().for_each(|x| *x /= nn);
        }
        init.push(a);
    }
    Ok(Problem {
        n: nr,
        q: qr,
        m: vec![1.0; nr],
        groups: full.groups.clone(),
        group_area: full.group_area.clone(),
        group_weight: full.group_weight.clone(),
        b_all,
        rows_all,
        active,
        b_act,
        rows_act,
        lam_act,
        rigid_rows,
        eigenvalues: full.eigenvalues.clone(),
        init,
        init_cluster: full.init_cluster.clone(),
        length_scale: full.length_scale,
        prolong: Some(phi),
    })
}

/// Cell owning a reduced column.
fn owner(cells: &[CellBasis], col: usize, ncomp: usize) -> (usize, usize) {
    // cells are laid out contiguously by offset; binary search on offsets
    let mut lo = 0usize;
    let mut hi = cells.len();
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if cells[mid].offset <= col {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    // skip empty cells sharing the same offset
    let mut sc = lo;
    while cells[sc].s == 0 || col >= cells[sc].offset + ncomp * cells[sc].s {
        sc += 1;
    }
    (sc, col - cells[sc].offset)
}
