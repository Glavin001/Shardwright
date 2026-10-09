//! Global assembly and small analysis drivers (static solve, natural
//! frequencies) used for validation sanity checks.

use crate::eigen::{smallest_eigenpairs, EigenOptions};
use crate::element::tet_stiffness;
use crate::material::ElasticMaterial;
use crate::mesh::TetMesh;
use crate::sparse::{CsrMatrix, SparseCholesky};
use rayon::prelude::*;

/// Per-vertex sorted neighbor lists (including the vertex itself).
fn vertex_neighbors(nv: usize, tets: &[[u32; 4]]) -> Vec<Vec<u32>> {
    let mut nb: Vec<Vec<u32>> = vec![Vec::new(); nv];
    for t in tets {
        for &a in t {
            for &b in t {
                nb[a as usize].push(b);
            }
        }
    }
    for l in nb.iter_mut() {
        l.sort_unstable();
        l.dedup();
    }
    nb
}

/// Assembles the global stiffness (3 DOFs per vertex, DOF `3v + axis`) with
/// per-tet materials. Element matrices are computed in parallel (ordered
/// collection) and accumulated sequentially in tet order.
pub fn assemble_stiffness(mesh: &TetMesh, materials: &[ElasticMaterial]) -> CsrMatrix {
    assert_eq!(materials.len(), mesh.tets.len(), "one material per tet");
    let nv = mesh.verts.len();
    let nb = vertex_neighbors(nv, &mesh.tets);
    let n = 3 * nv;
    let mut row_ptr = vec![0usize; n + 1];
    for v in 0..nv {
        for a in 0..3 {
            row_ptr[3 * v + a + 1] = 3 * nb[v].len();
        }
    }
    for i in 0..n {
        row_ptr[i + 1] += row_ptr[i];
    }
    let mut col_idx = vec![0usize; row_ptr[n]];
    for v in 0..nv {
        for a in 0..3 {
            let base = row_ptr[3 * v + a];
            for (k, &w) in nb[v].iter().enumerate() {
                for b in 0..3 {
                    col_idx[base + 3 * k + b] = 3 * w as usize + b;
                }
            }
        }
    }
    let mut vals = vec![0.0; row_ptr[n]];
    // element matrices, with per-material stiffness cache for consecutive equal materials
    let kes: Vec<[[f64; 12]; 12]> = (0..mesh.tets.len())
        .into_par_iter()
        .map(|t| tet_stiffness(&mesh.tet_points(t), &materials[t].stiffness_voigt()))
        .collect();
    for (t, tt) in mesh.tets.iter().enumerate() {
        let ke = &kes[t];
        for (la, &a) in tt.iter().enumerate() {
            let list = &nb[a as usize];
            for (lb, &b) in tt.iter().enumerate() {
                let k = list.binary_search(&b).unwrap();
                for i in 0..3 {
                    let base = row_ptr[3 * a as usize + i] + 3 * k;
                    for j in 0..3 {
                        vals[base + j] += ke[3 * la + i][3 * lb + j];
                    }
                }
            }
        }
    }
    CsrMatrix { n_rows: n, n_cols: n, row_ptr, col_idx, vals }
}

/// Lumped (row-sum) mass: each tet contributes `ρ|V|/4` to each of its
/// vertices, for each of the 3 DOFs. Returns the diagonal (length `3 nv`).
pub fn assemble_lumped_mass(mesh: &TetMesh, materials: &[ElasticMaterial]) -> Vec<f64> {
    let mut m = vec![0.0; 3 * mesh.verts.len()];
    for (t, tt) in mesh.tets.iter().enumerate() {
        let w = materials[t].density * mesh.tet_volume(t).abs() * 0.25;
        for &v in tt {
            for a in 0..3 {
                m[3 * v as usize + a] += w;
            }
        }
    }
    m
}

/// The six rigid-body modes (3 translations, 3 rotations about the centroid
/// of `points`) as vectors of length `3 * points.len()`.
pub fn rigid_modes(points: &[[f64; 3]]) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut c = [0.0; 3];
    for p in points {
        for k in 0..3 {
            c[k] += p[k];
        }
    }
    for v in &mut c {
        *v /= n.max(1) as f64;
    }
    let mut out = vec![vec![0.0; 3 * n]; 6];
    for (i, p) in points.iter().enumerate() {
        let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        for a in 0..3 {
            out[a][3 * i + a] = 1.0;
        }
        // rotation about x: (0, -z, y); about y: (z, 0, -x); about z: (-y, x, 0)
        out[3][3 * i + 1] = -d[2];
        out[3][3 * i + 2] = d[1];
        out[4][3 * i] = d[2];
        out[4][3 * i + 2] = -d[0];
        out[5][3 * i] = -d[1];
        out[5][3 * i + 1] = d[0];
    }
    out
}

/// Free-DOF numbering: returns (map dof->free index or usize::MAX, n_free).
pub fn free_dof_map(nv: usize, fixed_vertices: &[u32]) -> (Vec<usize>, usize) {
    let mut fixed = vec![false; nv];
    for &v in fixed_vertices {
        fixed[v as usize] = true;
    }
    let mut map = vec![usize::MAX; 3 * nv];
    let mut nf = 0;
    for v in 0..nv {
        if !fixed[v] {
            for a in 0..3 {
                map[3 * v + a] = nf;
                nf += 1;
            }
        }
    }
    (map, nf)
}

/// Linear static solve `K u = f` with homogeneous Dirichlet conditions on
/// `fixed_vertices`. Returns per-vertex displacements. Fails if the
/// constrained stiffness is singular (e.g. no fixed vertices).
pub fn solve_static(
    mesh: &TetMesh,
    materials: &[ElasticMaterial],
    fixed_vertices: &[u32],
    nodal_forces: &[[f64; 3]],
) -> Result<Vec<[f64; 3]>, String> {
    let nv = mesh.verts.len();
    if nodal_forces.len() != nv {
        return Err("nodal_forces must have one entry per vertex".into());
    }
    if fixed_vertices.is_empty() {
        return Err("solve_static needs at least one fixed vertex (rigid modes)".into());
    }
    let k = assemble_stiffness(mesh, materials);
    let (map, nf) = free_dof_map(nv, fixed_vertices);
    let kf = k.restrict(&map, nf, &map, nf);
    let chol = SparseCholesky::new(&kf)?;
    let mut b = vec![0.0; nf];
    for v in 0..nv {
        for a in 0..3 {
            if map[3 * v + a] != usize::MAX {
                b[map[3 * v + a]] = nodal_forces[v][a];
            }
        }
    }
    chol.solve_in_place(&mut b);
    let mut u = vec![[0.0; 3]; nv];
    for v in 0..nv {
        for a in 0..3 {
            if map[3 * v + a] != usize::MAX {
                u[v][a] = b[map[3 * v + a]];
            }
        }
    }
    Ok(u)
}

/// Smallest generalized eigenpairs of the (constrained) stiffness/lumped mass
/// pair. When `fixed_vertices` is empty the six rigid modes are deflated and
/// any further near-zero eigenvalues (disconnected parts) are skipped.
/// Returns `(eigenvalues λ = ω², vectors over all 3·nv DOFs)`.
pub fn eigenmodes(
    mesh: &TetMesh,
    materials: &[ElasticMaterial],
    fixed_vertices: &[u32],
    n: usize,
    seed: u64,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), String> {
    let nv = mesh.verts.len();
    let k = assemble_stiffness(mesh, materials);
    let m = assemble_lumped_mass(mesh, materials);
    let (map, nf) = free_dof_map(nv, fixed_vertices);
    let kf = k.restrict(&map, nf, &map, nf);
    let mut mf = vec![0.0; nf];
    for i in 0..3 * nv {
        if map[i] != usize::MAX {
            mf[map[i]] = m[i];
        }
    }
    let unanchored = fixed_vertices.is_empty();
    let defl = if unanchored { rigid_modes(&mesh.verts) } else { Vec::new() };
    let extra = if unanchored { 6 } else { 0 };
    let res = smallest_eigenpairs(
        &kf,
        &mf,
        &defl,
        &EigenOptions { n: n + extra, seed, ..Default::default() },
    )?;
    let trk: f64 = kf.diagonal().iter().sum();
    let trm: f64 = mf.iter().sum();
    let zero_tol = 1e-11 * trk / trm;
    let mut vals = Vec::new();
    let mut vecs = Vec::new();
    for (j, &l) in res.values.iter().enumerate() {
        if l <= zero_tol {
            continue;
        }
        if vals.len() == n {
            break;
        }
        vals.push(l);
        let mut full = vec![0.0; 3 * nv];
        for i in 0..3 * nv {
            if map[i] != usize::MAX {
                full[i] = res.vectors[j][map[i]];
            }
        }
        vecs.push(full);
    }
    Ok((vals, vecs))
}

/// Lowest `n` natural frequencies in Hz (`sqrt(λ)/2π`). For a free body
/// (no fixed vertices) the zero-frequency rigid modes are skipped.
pub fn natural_frequencies(
    mesh: &TetMesh,
    materials: &[ElasticMaterial],
    fixed_vertices: &[u32],
    n: usize,
) -> Result<Vec<f64>, String> {
    let (vals, _) = eigenmodes(mesh, materials, fixed_vertices, n, 0x00f2_ee9e)?;
    Ok(vals.iter().map(|&l| l.max(0.0).sqrt() / (2.0 * std::f64::consts::PI)).collect())
}
