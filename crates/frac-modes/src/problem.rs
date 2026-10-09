//! Cell-exploded discretization (spec §4.1) and problem normalization.

use crate::ModesInput;
use frac_fem::analysis::free_dof_map;
use frac_fem::sparse::CsrMatrix;
use frac_fem::{assemble_lumped_mass, assemble_stiffness, rigid_modes, TetMesh};
use std::ops::Range;

/// Regularization added to the quadratic term in both solvers:
/// `½ uᵀ(Q̂ + δM̂)u`. It makes the ADMM system matrix SPD when the exploded
/// problem has zero-energy motions (global rigid modes of a free body,
/// rigid motions behind zero-weight interfaces); with `uᵀM̂u ≈ 1` it changes
/// energies by `≈ δ/2`.
pub const DELTA: f64 = 1e-8;

pub(crate) struct Problem {
    /// Number of free DOFs (unknowns).
    pub n: usize,
    /// Normalized stiffness `Q̂ = Q / (λ₁ m_tot)` on free DOFs (exploded).
    pub q: CsrMatrix,
    /// Normalized lumped mass `M̂ = M / m_tot` on free DOFs.
    pub m: Vec<f64>,
    /// All groups (sorted (a<b)), physical areas, weights.
    pub groups: Vec<(u32, u32)>,
    pub group_area: Vec<f64>,
    pub group_weight: Vec<f64>,
    /// Jump operator for all groups (rows of group g in `rows_all[g]`),
    /// normalized by the length scale: `‖B̂_g u‖² = ∫_g ‖D‖² dA / L²`.
    pub b_all: CsrMatrix,
    pub rows_all: Vec<Range<usize>>,
    /// Active groups (0 < w < ∞): indices into `groups`, the stacked
    /// operator and per-active-group row ranges / penalty `ω w_g`.
    pub active: Vec<usize>,
    pub b_act: CsrMatrix,
    pub rows_act: Vec<Range<usize>>,
    pub lam_act: Vec<f64>,
    /// `M̂ R` rows for rigid-mode orthogonality (empty when anchored).
    pub rigid_rows: Vec<Vec<f64>>,
    /// Continuous-mesh eigenvalues used for initialization (physical units).
    pub eigenvalues: Vec<f64>,
    /// Initial vectors (continuous eigenvectors mapped to free exploded DOFs).
    pub init: Vec<Vec<f64>>,
    pub length_scale: f64,
}

struct Dsu(Vec<u32>);
impl Dsu {
    fn find(&mut self, mut x: u32) -> u32 {
        while self.0[x as usize] != x {
            let p = self.0[self.0[x as usize] as usize];
            self.0[x as usize] = p;
            x = p;
        }
        x
    }
    fn union(&mut self, a: u32, b: u32) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            self.0[hi as usize] = lo;
        }
    }
}

pub(crate) fn mdot(a: &[f64], b: &[f64], m: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * m[i] * b[i];
    }
    s
}

pub(crate) fn build(input: &ModesInput, timings: &mut Vec<(String, f64)>) -> Result<Problem, String> {
    let mesh = input.mesh;
    let nt = mesh.tets.len();
    let nv = mesh.verts.len();
    if nt == 0 {
        return Err("empty tet mesh".into());
    }
    if input.tet_material.len() != nt || input.tet_cell.len() != nt {
        return Err("tet_material and tet_cell must have one entry per tet".into());
    }
    for m in input.tet_material {
        m.validate()?;
    }
    for &a in input.anchored_vertices {
        if a as usize >= nv {
            return Err(format!("anchored vertex {a} out of range"));
        }
    }
    let p = &input.params;
    if p.k == 0 {
        return Err("k must be positive".into());
    }
    if !(p.omega >= 0.0 && p.omega.is_finite()) {
        return Err("omega must be finite and non-negative".into());
    }
    let t0 = std::time::Instant::now();

    // ---- continuous eigenproblem (initialization), physical units ----
    let unanchored = input.anchored_vertices.is_empty();
    let (evals, evecs) =
        frac_fem::analysis::eigenmodes(mesh, input.tet_material, input.anchored_vertices, p.k, p.seed)?;
    if evals.is_empty() {
        return Err("no non-rigid eigenmodes found on the continuous mesh".into());
    }
    timings.push(("eigen_init".into(), t0.elapsed().as_secs_f64() * 1e3));
    let t1 = std::time::Instant::now();

    // ---- exploded nodes: one per (vertex, cell) pair ----
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(4 * nt);
    for (t, tt) in mesh.tets.iter().enumerate() {
        for &v in tt {
            pairs.push((v, input.tet_cell[t]));
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    let node_of = |v: u32, c: u32| -> u32 { pairs.binary_search(&(v, c)).unwrap() as u32 };

    // ---- fault faces ----
    let faces = mesh.sorted_faces();
    struct Fault {
        verts: [u32; 3],
        ca: u32,
        cb: u32,
    }
    let mut faults: Vec<Fault> = Vec::new();
    let mut i = 0;
    while i < faces.len() {
        let mut j = i + 1;
        while j < faces.len() && faces[j].0 == faces[i].0 {
            j += 1;
        }
        if j - i == 2 {
            let c1 = input.tet_cell[faces[i].1 as usize];
            let c2 = input.tet_cell[faces[i + 1].1 as usize];
            if c1 != c2 {
                let (ca, cb) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
                faults.push(Fault { verts: faces[i].0, ca, cb });
            }
        }
        i = j;
    }
    faults.sort_by(|a, b| (a.ca, a.cb, a.verts).cmp(&(b.ca, b.cb, b.verts)));
    let mut groups: Vec<(u32, u32)> = faults.iter().map(|f| (f.ca, f.cb)).collect();
    groups.dedup();
    let mut group_weight = Vec::with_capacity(groups.len());
    for &(a, b) in &groups {
        let w = (input.group_weight)(a, b);
        if w.is_nan() || w < 0.0 {
            return Err(format!("invalid group weight {w} for cells ({a},{b})"));
        }
        group_weight.push(w);
    }
    let group_index = |a: u32, b: u32| groups.binary_search(&(a, b)).unwrap();

    // ---- forbidden groups: merge DOF copies (B_g u = 0 <=> equal vertex copies) ----
    let mut dsu = Dsu((0..pairs.len() as u32).collect());
    for f in &faults {
        if group_weight[group_index(f.ca, f.cb)].is_infinite() {
            for &v in &f.verts {
                dsu.union(node_of(v, f.ca), node_of(v, f.cb));
            }
        }
    }
    let mut node_id = vec![u32::MAX; pairs.len()];
    let mut n_nodes = 0u32;
    let mut node_vertex: Vec<u32> = Vec::new();
    for k in 0..pairs.len() as u32 {
        let r = dsu.find(k);
        if node_id[r as usize] == u32::MAX {
            node_id[r as usize] = n_nodes;
            node_vertex.push(pairs[r as usize].0);
            n_nodes += 1;
        }
        node_id[k as usize] = node_id[r as usize];
    }
    let node = |v: u32, c: u32| node_id[node_of(v, c) as usize];
    let n_nodes = n_nodes as usize;

    // exploded mesh
    let ex = TetMesh {
        verts: node_vertex.iter().map(|&v| mesh.verts[v as usize]).collect(),
        tets: mesh
            .tets
            .iter()
            .enumerate()
            .map(|(t, tt)| tt.map(|v| node(v, input.tet_cell[t])))
            .collect(),
    };
    let k_full = assemble_stiffness(&ex, input.tet_material);
    let m_full = assemble_lumped_mass(&ex, input.tet_material);
    let m_tot: f64 = m_full.iter().sum::<f64>() / 3.0;
    let volume = mesh.volume().abs();
    let length_scale = libm::cbrt(volume);
    let lambda1 = evals[0];

    // anchored: all copies of anchored vertices
    let mut anchored = vec![false; nv];
    for &a in input.anchored_vertices {
        anchored[a as usize] = true;
    }
    let fixed_nodes: Vec<u32> =
        (0..n_nodes as u32).filter(|&nd| anchored[node_vertex[nd as usize] as usize]).collect();
    let (map, n) = free_dof_map(n_nodes, &fixed_nodes);
    if n == 0 {
        return Err("all DOFs are anchored".into());
    }
    let qs = 1.0 / (lambda1 * m_tot);
    let mut q = k_full.restrict(&map, n, &map, n);
    q.vals.iter_mut().for_each(|v| *v *= qs);
    let mut m = vec![0.0; n];
    for d in 0..3 * n_nodes {
        if map[d] != usize::MAX {
            m[map[d]] = m_full[d] / m_tot;
        }
    }

    // ---- jump operator ----
    let mut trip: Vec<(usize, usize, f64)> = Vec::with_capacity(faults.len() * 36);
    let mut rows_all: Vec<Range<usize>> = vec![0..0; groups.len()];
    let mut group_area = vec![0.0; groups.len()];
    let mut row = 0usize;
    let mut fi = 0;
    for (g, &(a, b)) in groups.iter().enumerate() {
        let start = row;
        while fi < faults.len() && (faults[fi].ca, faults[fi].cb) == (a, b) {
            let f = &faults[fi];
            let p: Vec<[f64; 3]> = f.verts.iter().map(|&v| mesh.verts[v as usize]).collect();
            let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
            let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
            let cr = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            let area = 0.5 * (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
            group_area[g] += area;
            // edge-midpoint rule (exact for quadratics), weights area/3
            let s = 0.5 * (area / 3.0).sqrt() / length_scale;
            for e in 0..3 {
                let (vi, vj) = (f.verts[e], f.verts[(e + 1) % 3]);
                for k in 0..3 {
                    for (v, c, sign) in [(vi, a, 1.0), (vj, a, 1.0), (vi, b, -1.0), (vj, b, -1.0)] {
                        let d = map[3 * node(v, c) as usize + k];
                        if d != usize::MAX {
                            trip.push((row, d, sign * s));
                        }
                    }
                    row += 1;
                }
            }
            fi += 1;
        }
        rows_all[g] = start..row;
    }
    let b_all = CsrMatrix::from_triplets(row, n, trip);
    // drop exact zeros (merged copies) to keep the operator lean
    let b_all = prune(&b_all);

    let mut active = Vec::new();
    let mut rows_act = Vec::new();
    let mut lam_act = Vec::new();
    let mut keep_row = vec![usize::MAX; b_all.n_rows];
    let mut nr = 0;
    for g in 0..groups.len() {
        let w = group_weight[g];
        if w > 0.0 && w.is_finite() {
            active.push(g);
            lam_act.push(p.omega * w);
            let s = nr;
            for r in rows_all[g].clone() {
                keep_row[r] = nr;
                nr += 1;
            }
            rows_act.push(s..nr);
        }
    }
    let ident: Vec<usize> = (0..n).collect();
    let b_act = b_all.restrict(&keep_row, nr, &ident, n);

    // ---- rigid-mode rows (unanchored) ----
    let mut rigid_rows = Vec::new();
    if unanchored {
        let full = rigid_modes(&ex.verts);
        let mut basis: Vec<Vec<f64>> = Vec::new();
        for r in full {
            let mut v: Vec<f64> = vec![0.0; n];
            for d in 0..3 * n_nodes {
                if map[d] != usize::MAX {
                    v[map[d]] = r[d];
                }
            }
            for _ in 0..2 {
                for q in &basis {
                    let c = mdot(q, &v, &m);
                    for i in 0..n {
                        v[i] -= c * q[i];
                    }
                }
            }
            let nn = mdot(&v, &v, &m).sqrt();
            if nn > 1e-10 {
                v.iter_mut().for_each(|x| *x /= nn);
                basis.push(v);
            }
        }
        rigid_rows = basis.iter().map(|q| (0..n).map(|i| m[i] * q[i]).collect()).collect();
    }

    // ---- initial vectors ----
    let mut init = Vec::with_capacity(evecs.len());
    for ev in &evecs {
        let mut v = vec![0.0; n];
        for nd in 0..n_nodes {
            let vert = node_vertex[nd] as usize;
            for k in 0..3 {
                let d = map[3 * nd + k];
                if d != usize::MAX {
                    v[d] = ev[3 * vert + k];
                }
            }
        }
        let nn = mdot(&v, &v, &m).sqrt();
        if nn > 0.0 {
            v.iter_mut().for_each(|x| *x /= nn);
        }
        init.push(v);
    }
    timings.push(("assemble".into(), t1.elapsed().as_secs_f64() * 1e3));
    Ok(Problem {
        n,
        q,
        m,
        groups,
        group_area,
        group_weight,
        b_all,
        rows_all,
        active,
        b_act,
        rows_act,
        lam_act,
        rigid_rows,
        eigenvalues: evals,
        init,
        length_scale,
    })
}

fn prune(a: &CsrMatrix) -> CsrMatrix {
    let mut row_ptr = vec![0usize; a.n_rows + 1];
    let mut col_idx = Vec::with_capacity(a.nnz());
    let mut vals = Vec::with_capacity(a.nnz());
    for i in 0..a.n_rows {
        let (c, v) = a.row(i);
        for k in 0..c.len() {
            if v[k] != 0.0 {
                col_idx.push(c[k]);
                vals.push(v[k]);
            }
        }
        row_ptr[i + 1] = col_idx.len();
    }
    CsrMatrix { n_rows: a.n_rows, n_cols: a.n_cols, row_ptr, col_idx, vals }
}

impl Problem {
    /// `½ uᵀ(Q̂+δM̂)u + Σ_active ω w_g ‖B̂_g u‖`.
    pub fn objective(&self, u: &[f64]) -> f64 {
        let qu = self.q.mul(u);
        let mut e = 0.0;
        for i in 0..self.n {
            e += 0.5 * u[i] * qu[i] + 0.5 * DELTA * self.m[i] * u[i] * u[i];
        }
        let bu = self.b_act.mul(u);
        for (k, r) in self.rows_act.iter().enumerate() {
            let s: f64 = bu[r.clone()].iter().map(|x| x * x).sum();
            e += self.lam_act[k] * s.sqrt();
        }
        e
    }

    /// `‖B̂_g u‖` for all groups.
    pub fn group_norms(&self, u: &[f64]) -> Vec<f64> {
        let bu = self.b_all.mul(u);
        self.rows_all.iter().map(|r| bu[r.clone()].iter().map(|x| x * x).sum::<f64>().sqrt()).collect()
    }
}
