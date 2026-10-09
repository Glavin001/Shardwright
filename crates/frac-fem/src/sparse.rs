//! Deterministic CSR sparse matrices and a sequential sparse Cholesky
//! wrapper over faer.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, LltRef, SymbolicCholesky,
    SymmetricOrdering,
};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, Mat, MatMut, Par, Side};
use rayon::prelude::*;

/// Compressed sparse row matrix with sorted, unique column indices per row.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CsrMatrix {
    pub n_rows: usize,
    pub n_cols: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub vals: Vec<f64>,
}

impl CsrMatrix {
    /// Builds from triplets; duplicates are summed in input order
    /// (stable sort), so the result is deterministic.
    pub fn from_triplets(n_rows: usize, n_cols: usize, mut t: Vec<(usize, usize, f64)>) -> Self {
        t.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        let mut row_ptr = vec![0usize; n_rows + 1];
        let mut col_idx = Vec::with_capacity(t.len());
        let mut vals: Vec<f64> = Vec::with_capacity(t.len());
        let mut last: Option<(usize, usize)> = None;
        for &(r, c, v) in &t {
            assert!(r < n_rows && c < n_cols, "triplet out of bounds");
            if last == Some((r, c)) {
                *vals.last_mut().unwrap() += v;
            } else {
                col_idx.push(c);
                vals.push(v);
                row_ptr[r + 1] += 1;
                last = Some((r, c));
            }
        }
        for i in 0..n_rows {
            row_ptr[i + 1] += row_ptr[i];
        }
        CsrMatrix { n_rows, n_cols, row_ptr, col_idx, vals }
    }

    pub fn nnz(&self) -> usize {
        self.vals.len()
    }

    pub fn row(&self, i: usize) -> (&[usize], &[f64]) {
        let (a, b) = (self.row_ptr[i], self.row_ptr[i + 1]);
        (&self.col_idx[a..b], &self.vals[a..b])
    }

    /// `y = A x` (row-parallel, deterministic: each row is an independent
    /// sequential dot product).
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n_cols);
        assert_eq!(y.len(), self.n_rows);
        let f = |(i, yi): (usize, &mut f64)| {
            let (c, v) = self.row(i);
            let mut s = 0.0;
            for k in 0..c.len() {
                s += v[k] * x[c[k]];
            }
            *yi = s;
        };
        if self.nnz() > 200_000 {
            y.par_iter_mut().enumerate().for_each(f);
        } else {
            y.iter_mut().enumerate().for_each(f);
        }
    }

    pub fn mul(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.n_rows];
        self.matvec(x, &mut y);
        y
    }

    /// `y = A^T x` (sequential scatter in row order; deterministic).
    pub fn matvec_t(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n_rows);
        assert_eq!(y.len(), self.n_cols);
        y.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..self.n_rows {
            let xi = x[i];
            if xi == 0.0 {
                continue;
            }
            let (c, v) = self.row(i);
            for k in 0..c.len() {
                y[c[k]] += v[k] * xi;
            }
        }
    }

    pub fn transpose(&self) -> CsrMatrix {
        let mut cnt = vec![0usize; self.n_cols + 1];
        for &c in &self.col_idx {
            cnt[c + 1] += 1;
        }
        for i in 0..self.n_cols {
            cnt[i + 1] += cnt[i];
        }
        let row_ptr = cnt.clone();
        let mut pos = cnt;
        let mut col_idx = vec![0usize; self.nnz()];
        let mut vals = vec![0.0; self.nnz()];
        for i in 0..self.n_rows {
            let (c, v) = self.row(i);
            for k in 0..c.len() {
                let p = pos[c[k]];
                col_idx[p] = i;
                vals[p] = v[k];
                pos[c[k]] += 1;
            }
        }
        CsrMatrix { n_rows: self.n_cols, n_cols: self.n_rows, row_ptr, col_idx, vals }
    }

    /// Diagonal entries (zero where absent).
    pub fn diagonal(&self) -> Vec<f64> {
        let n = self.n_rows.min(self.n_cols);
        let mut d = vec![0.0; n];
        for (i, di) in d.iter_mut().enumerate() {
            let (c, v) = self.row(i);
            if let Ok(k) = c.binary_search(&i) {
                *di = v[k];
            }
        }
        d
    }

    /// Restriction to the given rows and columns: `map_r[i]`/`map_c[j]` give
    /// the new index or `usize::MAX` to drop.
    pub fn restrict(&self, map_r: &[usize], n_r: usize, map_c: &[usize], n_c: usize) -> CsrMatrix {
        let mut row_ptr = vec![0usize; n_r + 1];
        let mut rows: Vec<(usize, usize)> = Vec::new(); // (new row, old row)
        for i in 0..self.n_rows {
            if map_r[i] != usize::MAX {
                rows.push((map_r[i], i));
            }
        }
        rows.sort_unstable();
        let mut col_idx = Vec::new();
        let mut vals = Vec::new();
        let mut buf: Vec<(usize, f64)> = Vec::new();
        for &(nr, or) in &rows {
            let (c, v) = self.row(or);
            buf.clear();
            for k in 0..c.len() {
                let nc = map_c[c[k]];
                if nc != usize::MAX {
                    buf.push((nc, v[k]));
                }
            }
            buf.sort_by(|a, b| a.0.cmp(&b.0));
            for &(nc, val) in &buf {
                col_idx.push(nc);
                vals.push(val);
            }
            row_ptr[nr + 1] = buf.len();
        }
        for i in 0..n_r {
            row_ptr[i + 1] += row_ptr[i];
        }
        CsrMatrix { n_rows: n_r, n_cols: n_c, row_ptr, col_idx, vals }
    }

    /// `alpha*A + beta*B` (same shape; union pattern).
    pub fn add(&self, alpha: f64, b: &CsrMatrix, beta: f64) -> CsrMatrix {
        assert_eq!(self.n_rows, b.n_rows);
        assert_eq!(self.n_cols, b.n_cols);
        let mut row_ptr = vec![0usize; self.n_rows + 1];
        let mut col_idx = Vec::with_capacity(self.nnz() + b.nnz());
        let mut vals = Vec::with_capacity(self.nnz() + b.nnz());
        for i in 0..self.n_rows {
            let (ca, va) = self.row(i);
            let (cb, vb) = b.row(i);
            let (mut p, mut q) = (0, 0);
            while p < ca.len() || q < cb.len() {
                let take_a = q >= cb.len() || (p < ca.len() && ca[p] <= cb[q]);
                let take_b = p >= ca.len() || (q < cb.len() && cb[q] <= ca[p]);
                if take_a && take_b {
                    col_idx.push(ca[p]);
                    vals.push(alpha * va[p] + beta * vb[q]);
                    p += 1;
                    q += 1;
                } else if take_a {
                    col_idx.push(ca[p]);
                    vals.push(alpha * va[p]);
                    p += 1;
                } else {
                    col_idx.push(cb[q]);
                    vals.push(beta * vb[q]);
                    q += 1;
                }
            }
            row_ptr[i + 1] = col_idx.len();
        }
        CsrMatrix { n_rows: self.n_rows, n_cols: self.n_cols, row_ptr, col_idx, vals }
    }

    /// Adds `d[i]` to the diagonal (the diagonal must be in the pattern or
    /// will be inserted).
    pub fn add_diagonal(&self, d: &[f64]) -> CsrMatrix {
        let n = self.n_rows;
        let dm = CsrMatrix {
            n_rows: n,
            n_cols: self.n_cols,
            row_ptr: (0..=n).collect(),
            col_idx: (0..n).collect(),
            vals: d.to_vec(),
        };
        self.add(1.0, &dm, 1.0)
    }

    /// `A^T diag(w) A` for a rectangular `A` (result is symmetric, CSR,
    /// deterministic).
    pub fn ata(&self, w: Option<&[f64]>) -> CsrMatrix {
        let at = self.transpose();
        // row j of result = sum over rows i of A with A_ij != 0 of A_ij * w_i * A_i:
        let n = self.n_cols;
        let mut row_ptr = vec![0usize; n + 1];
        let mut col_idx = Vec::new();
        let mut vals = Vec::new();
        let mut acc = vec![0.0f64; n];
        let mut mark = vec![usize::MAX; n];
        let mut cols: Vec<usize> = Vec::new();
        for j in 0..n {
            cols.clear();
            let (ri, rv) = at.row(j);
            for k in 0..ri.len() {
                let i = ri[k];
                let f = rv[k] * w.map_or(1.0, |w| w[i]);
                let (c, v) = self.row(i);
                for m in 0..c.len() {
                    if mark[c[m]] != j {
                        mark[c[m]] = j;
                        acc[c[m]] = 0.0;
                        cols.push(c[m]);
                    }
                    acc[c[m]] += f * v[m];
                }
            }
            cols.sort_unstable();
            for &c in &cols {
                col_idx.push(c);
                vals.push(acc[c]);
            }
            row_ptr[j + 1] = col_idx.len();
        }
        CsrMatrix { n_rows: n, n_cols: n, row_ptr, col_idx, vals }
    }

    /// Dense copy (row-major), for tests and tiny problems.
    pub fn to_dense(&self) -> Vec<f64> {
        let mut d = vec![0.0; self.n_rows * self.n_cols];
        for i in 0..self.n_rows {
            let (c, v) = self.row(i);
            for k in 0..c.len() {
                d[i * self.n_cols + c[k]] += v[k];
            }
        }
        d
    }
}

/// Sparse Cholesky `A = L L^T` of a symmetric positive definite matrix stored
/// with full (both triangles) pattern in CSR form. Always sequential
/// (`Par::Seq`) for bitwise reproducibility. The symbolic analysis (AMD
/// ordering, elimination tree) can be reused across numeric refactorizations
/// with the same pattern.
pub struct SparseCholesky {
    symbolic: SymbolicCholesky<usize>,
    values: Vec<f64>,
    n: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
}

impl SparseCholesky {
    pub fn new(a: &CsrMatrix) -> Result<Self, String> {
        assert_eq!(a.n_rows, a.n_cols);
        let n = a.n_rows;
        // a symmetric CSR matrix is its own CSC
        let sym = SymbolicSparseColMatRef::new_checked(n, n, &a.row_ptr, None, &a.col_idx);
        let symbolic = factorize_symbolic_cholesky(
            sym,
            Side::Lower,
            SymmetricOrdering::Amd,
            CholeskySymbolicParams::default(),
        )
        .map_err(|e| format!("symbolic cholesky failed: {e:?}"))?;
        let mut s = SparseCholesky {
            values: vec![0.0; symbolic.len_val()],
            symbolic,
            n,
            row_ptr: a.row_ptr.clone(),
            col_idx: a.col_idx.clone(),
        };
        s.refactor(a)?;
        Ok(s)
    }

    /// Numeric refactorization for a matrix with the same pattern.
    pub fn refactor(&mut self, a: &CsrMatrix) -> Result<(), String> {
        if a.row_ptr != self.row_ptr || a.col_idx != self.col_idx {
            return Err("refactor: sparsity pattern changed".into());
        }
        let n = self.n;
        let sym = SymbolicSparseColMatRef::new_checked(n, n, &a.row_ptr, None, &a.col_idx);
        let mat = SparseColMatRef::new(sym, &a.vals);
        let mut mem = MemBuffer::new(
            self.symbolic.factorize_numeric_llt_scratch::<f64>(Par::Seq, Default::default()),
        );
        let stack = MemStack::new(&mut mem);
        self.symbolic
            .factorize_numeric_llt::<f64>(
                &mut self.values,
                mat,
                Side::Lower,
                LltRegularization::default(),
                Par::Seq,
                stack,
                Default::default(),
            )
            .map_err(|e| format!("sparse cholesky failed (matrix not SPD?): {e:?}"))?;
        Ok(())
    }

    pub fn dim(&self) -> usize {
        self.n
    }

    /// Solves `A X = B` in place for a column-major block of right-hand sides.
    pub fn solve_mat(&self, rhs: MatMut<'_, f64>) {
        let mut mem = MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(rhs.ncols(), Par::Seq));
        let stack = MemStack::new(&mut mem);
        LltRef::new(&self.symbolic, &self.values).solve_in_place_with_conj(Conj::No, rhs, Par::Seq, stack);
    }

    /// Solves `A x = b` in place.
    pub fn solve_in_place(&self, b: &mut [f64]) {
        assert_eq!(b.len(), self.n);
        let mut m = Mat::<f64>::zeros(self.n, 1);
        for i in 0..self.n {
            m[(i, 0)] = b[i];
        }
        self.solve_mat(m.as_mut());
        for i in 0..self.n {
            b[i] = m[(i, 0)];
        }
    }

    /// Solves for several right-hand sides at once.
    pub fn solve_many(&self, bs: &mut [Vec<f64>]) {
        if bs.is_empty() {
            return;
        }
        let k = bs.len();
        let mut m = Mat::<f64>::zeros(self.n, k);
        for (j, b) in bs.iter().enumerate() {
            for i in 0..self.n {
                m[(i, j)] = b[i];
            }
        }
        self.solve_mat(m.as_mut());
        for (j, b) in bs.iter_mut().enumerate() {
            for i in 0..self.n {
                b[i] = m[(i, j)];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_solves_laplacian() {
        let n = 50;
        let mut t = Vec::new();
        for i in 0..n {
            t.push((i, i, 2.5));
            if i + 1 < n {
                t.push((i, i + 1, -1.0));
                t.push((i + 1, i, -1.0));
            }
        }
        let a = CsrMatrix::from_triplets(n, n, t);
        let ch = SparseCholesky::new(&a).unwrap();
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let mut b = a.mul(&x);
        ch.solve_in_place(&mut b);
        for i in 0..n {
            assert!((b[i] - x[i]).abs() < 1e-12);
        }
        let ata = a.ata(None);
        let d = a.to_dense();
        let dd = ata.to_dense();
        for i in 0..n {
            for j in 0..n {
                let s: f64 = (0..n).map(|k| d[k * n + i] * d[k * n + j]).sum();
                assert!((s - dd[i * n + j]).abs() < 1e-12);
            }
        }
    }
}
