//! Deterministic CSR sparse matrices and a sequential sparse Cholesky
//! wrapper over faer.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, LltRef, SymbolicCholesky,
    SymbolicCholeskyRaw, SymmetricOrdering,
};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, Mat, MatMut, MatRef, Par, Side};
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

    /// Sparse product `A B` (Gustavson; deterministic accumulation order).
    pub fn matmul(&self, b: &CsrMatrix) -> CsrMatrix {
        assert_eq!(self.n_cols, b.n_rows);
        let n = b.n_cols;
        let mut row_ptr = vec![0usize; self.n_rows + 1];
        let mut col_idx = Vec::new();
        let mut vals = Vec::new();
        let mut acc = vec![0.0f64; n];
        let mut mark = vec![usize::MAX; n];
        let mut cols: Vec<usize> = Vec::new();
        for i in 0..self.n_rows {
            cols.clear();
            let (ca, va) = self.row(i);
            for k in 0..ca.len() {
                let (cb, vb) = b.row(ca[k]);
                for m in 0..cb.len() {
                    let c = cb[m];
                    if mark[c] != i {
                        mark[c] = i;
                        acc[c] = 0.0;
                        cols.push(c);
                    }
                    acc[c] += va[k] * vb[m];
                }
            }
            cols.sort_unstable();
            for &c in &cols {
                col_idx.push(c);
                vals.push(acc[c]);
            }
            row_ptr[i + 1] = col_idx.len();
        }
        CsrMatrix { n_rows: self.n_rows, n_cols: n, row_ptr, col_idx, vals }
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

/// Number of subtree tasks of [`TreeSolve`]. Fixed (independent of the
/// thread count) so the floating-point operation order never depends on how
/// many threads run the tasks.
const TREE_TASKS: usize = 16;

/// Supernodal triangular solves parallelized over disjoint subtrees of the
/// supernodal elimination tree, on top of faer's factor layout (supernode
/// `s` stores its columns `begin[s]..begin[s+1]` as a dense column-major
/// block: diagonal block first, then the rows listed in its pattern).
///
/// Forward solve: every task (a set of disjoint subtrees) eliminates its
/// supernodes in increasing order on a private copy of its rows; updates of
/// rows outside its subtrees (always "top" rows: ancestors of several tasks)
/// are accumulated in a private buffer. The buffers are then subtracted in
/// task order and the top supernodes are eliminated sequentially. Backward
/// solve: top supernodes first (sequentially), then the tasks in parallel
/// (they only read top rows). The partition is fixed at construction, so
/// results are bitwise identical for any number of threads.
struct TreeSolve {
    begin: Vec<usize>,
    rptr: Vec<usize>,
    vptr: Vec<usize>,
    rows: Vec<usize>,
    /// Tasks: supernodes in increasing order, and the (permuted) rows they own.
    tasks: Vec<(Vec<usize>, Vec<usize>)>,
    top: Vec<usize>,
    top_rows: Vec<usize>,
    /// Per permuted row: owning task (`usize::MAX` = top) and local index.
    owner: Vec<(usize, usize)>,
    max_pattern: usize,
}

impl TreeSolve {
    fn new(sn: &faer::sparse::linalg::cholesky::supernodal::SymbolicSupernodalCholesky<usize>) -> TreeSolve {
        let ns = sn.n_supernodes();
        let n = sn.nrows();
        let mut begin: Vec<usize> = sn.supernode_begin().to_vec();
        begin.push(sn.supernode_end()[ns - 1]);
        let rptr = sn.col_ptr_for_row_idx().to_vec();
        let vptr = sn.col_ptr_for_val().to_vec();
        let rows = sn.row_idx().to_vec();
        let mut sn_of_row = vec![0usize; n];
        for s in 0..ns {
            for r in begin[s]..begin[s + 1] {
                sn_of_row[r] = s;
            }
        }
        // supernodal elimination tree and subtree work (stored values)
        let mut parent = vec![usize::MAX; ns];
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); ns];
        for s in 0..ns {
            if rptr[s + 1] > rptr[s] {
                let p = sn_of_row[rows[rptr[s]..rptr[s + 1]].iter().copied().min().unwrap()];
                parent[s] = p;
                children[p].push(s);
            }
        }
        let mut work: Vec<usize> = (0..ns).map(|s| vptr[s + 1] - vptr[s]).collect();
        for s in 0..ns {
            // children have smaller indices than their parent
            if parent[s] != usize::MAX {
                work[parent[s]] += work[s];
            }
        }
        let total: usize = (0..ns).filter(|&s| parent[s] == usize::MAX).map(|s| work[s]).sum();
        // split the heaviest subtree until there are enough of them
        let mut subtrees: Vec<usize> = (0..ns).filter(|&s| parent[s] == usize::MAX).collect();
        let mut top: Vec<usize> = Vec::new();
        loop {
            let (k, &r) = match subtrees.iter().enumerate().max_by_key(|&(_, &s)| (work[s], usize::MAX - s)) {
                Some(x) => x,
                None => break,
            };
            if subtrees.len() >= 4 * TREE_TASKS || work[r] * 2 * TREE_TASKS <= total || children[r].is_empty() {
                break;
            }
            subtrees.swap_remove(k);
            top.push(r);
            subtrees.extend(children[r].iter().copied());
        }
        // longest-processing-time assignment of subtrees to tasks
        subtrees.sort_by_key(|&s| (usize::MAX - work[s], s));
        let mut bins: Vec<(usize, Vec<usize>)> = vec![(0, Vec::new()); TREE_TASKS.min(subtrees.len()).max(1)];
        for &r in &subtrees {
            let b = (0..bins.len()).min_by_key(|&b| (bins[b].0, b)).unwrap();
            bins[b].0 += work[r];
            bins[b].1.push(r);
        }
        let mut owner = vec![(usize::MAX, 0usize); n];
        let mut tasks = Vec::new();
        for (_, roots) in bins {
            let mut sns = Vec::new();
            let mut stack = roots;
            while let Some(s) = stack.pop() {
                sns.push(s);
                stack.extend(children[s].iter().copied());
            }
            sns.sort_unstable();
            let t = tasks.len();
            let mut own = Vec::new();
            for &s in &sns {
                own.extend(begin[s]..begin[s + 1]);
            }
            own.sort_unstable();
            for (k, &r) in own.iter().enumerate() {
                owner[r] = (t, k);
            }
            tasks.push((sns, own));
        }
        top.sort_unstable();
        let mut top_rows = Vec::new();
        for &s in &top {
            top_rows.extend(begin[s]..begin[s + 1]);
        }
        top_rows.sort_unstable();
        for (k, &r) in top_rows.iter().enumerate() {
            owner[r] = (usize::MAX, k);
        }
        let max_pattern = (0..ns).map(|s| rptr[s + 1] - rptr[s]).max().unwrap_or(0);
        TreeSolve { begin, rptr, vptr, rows, tasks, top, top_rows, owner, max_pattern }
    }

    /// Diagonal block and below-diagonal block of supernode `s`.
    fn blocks<'a>(&self, s: usize, l: &'a [f64]) -> (MatRef<'a, f64>, MatRef<'a, f64>) {
        let ncols = self.begin[s + 1] - self.begin[s];
        let nrows = ncols + self.rptr[s + 1] - self.rptr[s];
        let m = MatRef::from_column_major_slice(&l[self.vptr[s]..self.vptr[s + 1]], nrows, ncols);
        m.split_at_row(ncols)
    }

    /// Forward elimination of supernode `s` on `x` (the supernode's own rows
    /// at `x[off..off + size]`); returns the update `L_bot x_s` in `tmp`.
    fn fwd_node(&self, s: usize, l: &[f64], xs: &mut [f64], tmp: &mut [f64]) {
        let (top, bot) = self.blocks(s, l);
        let size = xs.len();
        let mut xm = MatMut::from_column_major_slice_mut(xs, size, 1);
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(top, xm.as_mut(), Par::Seq);
        let np = bot.nrows();
        let tm = MatMut::from_column_major_slice_mut(&mut tmp[..np], np, 1);
        faer::linalg::matmul::matmul(tm, faer::Accum::Replace, bot, xm.as_ref(), 1.0, Par::Seq);
    }

    /// Backward step of supernode `s`: `x_s ← L_topᵀ⁻¹ (x_s − L_botᵀ tmp)`.
    fn bwd_node(&self, s: usize, l: &[f64], xs: &mut [f64], tmp: &[f64]) {
        let (top, bot) = self.blocks(s, l);
        let size = xs.len();
        let np = bot.nrows();
        let mut xm = MatMut::from_column_major_slice_mut(xs, size, 1);
        let tm = MatRef::from_column_major_slice(&tmp[..np], np, 1);
        faer::linalg::matmul::matmul(xm.as_mut(), faer::Accum::Add, bot.transpose(), tm, -1.0, Par::Seq);
        faer::linalg::triangular_solve::solve_upper_triangular_in_place(top.transpose(), xm, Par::Seq);
    }

    /// Solves `L Lᵀ x = b` in the permuted space, in place.
    fn solve(&self, l: &[f64], x: &mut [f64]) {
        let nt = self.top_rows.len();
        // ---- forward: tasks ----
        let parts: Vec<(Vec<f64>, Vec<f64>)> = self
            .tasks
            .par_iter()
            .enumerate()
            .map(|(t, (sns, own))| {
                let mut xl: Vec<f64> = own.iter().map(|&r| x[r]).collect();
                let mut acc = vec![0.0; nt];
                let mut tmp = vec![0.0; self.max_pattern];
                for &s in sns {
                    let off = self.owner[self.begin[s]].1;
                    let size = self.begin[s + 1] - self.begin[s];
                    self.fwd_node(s, l, &mut xl[off..off + size], &mut tmp);
                    for (idx, &r) in self.rows[self.rptr[s]..self.rptr[s + 1]].iter().enumerate() {
                        let (o, k) = self.owner[r];
                        if o == t {
                            xl[k] -= tmp[idx];
                        } else {
                            acc[k] += tmp[idx];
                        }
                    }
                }
                (xl, acc)
            })
            .collect();
        for (t, (xl, acc)) in parts.iter().enumerate() {
            for (k, &r) in self.tasks[t].1.iter().enumerate() {
                x[r] = xl[k];
            }
            for k in 0..nt {
                x[self.top_rows[k]] -= acc[k];
            }
        }
        // ---- forward/backward: top supernodes (sequential) ----
        let mut tmp = vec![0.0; self.max_pattern];
        for &s in &self.top {
            let (b, e) = (self.begin[s], self.begin[s + 1]);
            self.fwd_node(s, l, &mut x[b..e], &mut tmp);
            for (idx, &r) in self.rows[self.rptr[s]..self.rptr[s + 1]].iter().enumerate() {
                x[r] -= tmp[idx];
            }
        }
        for &s in self.top.iter().rev() {
            let (b, e) = (self.begin[s], self.begin[s + 1]);
            for (idx, &r) in self.rows[self.rptr[s]..self.rptr[s + 1]].iter().enumerate() {
                tmp[idx] = x[r];
            }
            self.bwd_node(s, l, &mut x[b..e], &tmp);
        }
        // ---- backward: tasks (read top rows only) ----
        let xr: &[f64] = x;
        let parts: Vec<Vec<f64>> = self
            .tasks
            .par_iter()
            .enumerate()
            .map(|(t, (sns, own))| {
                let mut xl: Vec<f64> = own.iter().map(|&r| xr[r]).collect();
                let mut tmp = vec![0.0; self.max_pattern];
                for &s in sns.iter().rev() {
                    for (idx, &r) in self.rows[self.rptr[s]..self.rptr[s + 1]].iter().enumerate() {
                        let (o, k) = self.owner[r];
                        tmp[idx] = if o == t { xl[k] } else { xr[r] };
                    }
                    let off = self.owner[self.begin[s]].1;
                    let size = self.begin[s + 1] - self.begin[s];
                    self.bwd_node(s, l, &mut xl[off..off + size], &tmp);
                }
                xl
            })
            .collect();
        for (t, xl) in parts.iter().enumerate() {
            for (k, &r) in self.tasks[t].1.iter().enumerate() {
                x[r] = xl[k];
            }
        }
    }
}

/// Scratch buffers for [`SparseCholesky::solve_in_place_work`] (tied to one
/// factorization's dimensions).
pub struct SolveWork {
    mem: Option<MemBuffer>,
    mat: Mat<f64>,
    x: Vec<f64>,
}

impl Default for SolveWork {
    fn default() -> Self {
        SolveWork { mem: None, mat: Mat::zeros(0, 0), x: Vec::new() }
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
    /// Simplicial fast path: (L col_ptr, L row_idx, perm fwd, perm inv).
    simplicial: Option<(Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>)>,
    /// Supernodal path: subtree-parallel solves and (perm fwd, perm inv).
    tree: Option<(TreeSolve, Vec<usize>, Vec<usize>)>,
}

impl SparseCholesky {
    pub fn new(a: &CsrMatrix) -> Result<Self, String> {
        Self::with_threshold(a, faer::sparse::linalg::SupernodalThreshold::AUTO)
    }

    /// Forces the simplicial (column-by-column) factorization, whose solves
    /// run through a lean allocation-free loop; best for small/medium
    /// matrices solved many times.
    pub fn new_simplicial(a: &CsrMatrix) -> Result<Self, String> {
        Self::with_threshold(a, faer::sparse::linalg::SupernodalThreshold::FORCE_SIMPLICIAL)
    }

    /// Forces the supernodal factorization (dense blocks; solves parallelized
    /// over subtrees of the elimination tree, deterministically); best for
    /// large matrices.
    pub fn new_supernodal(a: &CsrMatrix) -> Result<Self, String> {
        Self::with_threshold(a, faer::sparse::linalg::SupernodalThreshold::FORCE_SUPERNODAL)
    }

    fn with_threshold(a: &CsrMatrix, th: faer::sparse::linalg::SupernodalThreshold) -> Result<Self, String> {
        assert_eq!(a.n_rows, a.n_cols);
        let n = a.n_rows;
        // a symmetric CSR matrix is its own CSC
        let sym = SymbolicSparseColMatRef::new_checked(n, n, &a.row_ptr, None, &a.col_idx);
        let params = CholeskySymbolicParams { supernodal_flop_ratio_threshold: th, ..Default::default() };
        let symbolic = factorize_symbolic_cholesky(sym, Side::Lower, SymmetricOrdering::Amd, params)
            .map_err(|e| format!("symbolic cholesky failed: {e:?}"))?;
        let simplicial = match symbolic.raw() {
            SymbolicCholeskyRaw::Simplicial(sm) => {
                let cp = sm.col_ptr().to_vec();
                let ri = sm.row_idx().to_vec();
                // faer stores the diagonal first in each column
                let ok = (0..n).all(|j| cp[j] < cp[j + 1] && ri[cp[j]] == j);
                let (fwd, inv) = match symbolic.perm() {
                    Some(p) => (p.arrays().0.to_vec(), p.arrays().1.to_vec()),
                    None => ((0..n).collect(), (0..n).collect()),
                };
                if ok { Some((cp, ri, fwd, inv)) } else { None }
            }
            _ => None,
        };
        let tree = match symbolic.raw() {
            SymbolicCholeskyRaw::Supernodal(sn) if sn.n_supernodes() > 0 => {
                let (fwd, inv) = match symbolic.perm() {
                    Some(p) => (p.arrays().0.to_vec(), p.arrays().1.to_vec()),
                    None => ((0..n).collect(), (0..n).collect()),
                };
                Some((TreeSolve::new(sn), fwd, inv))
            }
            _ => None,
        };
        let mut s = SparseCholesky {
            values: vec![0.0; symbolic.len_val()],
            symbolic,
            n,
            row_ptr: a.row_ptr.clone(),
            col_idx: a.col_idx.clone(),
            simplicial,
            tree,
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

    /// Number of stored values in the factor `L`.
    pub fn factor_nnz(&self) -> usize {
        self.values.len()
    }

    /// Solves `A X = B` in place for a column-major block of right-hand sides.
    pub fn solve_mat(&self, rhs: MatMut<'_, f64>) {
        let mut mem = MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(rhs.ncols(), Par::Seq));
        let stack = MemStack::new(&mut mem);
        LltRef::new(&self.symbolic, &self.values).solve_in_place_with_conj(Conj::No, rhs, Par::Seq, stack);
    }

    /// Solves `A x = b` in place.
    pub fn solve_in_place(&self, b: &mut [f64]) {
        let mut w = SolveWork::default();
        self.solve_in_place_work(b, &mut w);
    }

    /// [`Self::solve_in_place`] reusing the scratch buffers in `w` (no
    /// allocation after the first call; for solves repeated thousands of times).
    pub fn solve_in_place_work(&self, b: &mut [f64], w: &mut SolveWork) {
        assert_eq!(b.len(), self.n);
        let n = self.n;
        if let Some((cp, ri, fwd, inv)) = &self.simplicial {
            let l = &self.values;
            w.x.resize(n, 0.0);
            let x = &mut w.x;
            for (xi, &f) in x.iter_mut().zip(fwd) {
                *xi = b[f];
            }
            // L x = b (column oriented)
            for j in 0..n {
                let (s, e) = (cp[j], cp[j + 1]);
                let xj = x[j] / l[s];
                x[j] = xj;
                if xj != 0.0 {
                    for k in s + 1..e {
                        x[ri[k]] -= l[k] * xj;
                    }
                }
            }
            // L^T x = y
            for j in (0..n).rev() {
                let (s, e) = (cp[j], cp[j + 1]);
                let mut acc = x[j];
                for k in s + 1..e {
                    acc -= l[k] * x[ri[k]];
                }
                x[j] = acc / l[s];
            }
            for i in 0..n {
                b[i] = x[inv[i]];
            }
            return;
        }
        if let Some((tree, fwd, inv)) = &self.tree {
            w.x.resize(n, 0.0);
            for (xi, &f) in w.x.iter_mut().zip(fwd) {
                *xi = b[f];
            }
            tree.solve(&self.values, &mut w.x);
            for i in 0..n {
                b[i] = w.x[inv[i]];
            }
            return;
        }
        if w.mat.nrows() != n || w.mat.ncols() != 1 {
            w.mat = Mat::<f64>::zeros(n, 1);
        }
        if w.mem.is_none() {
            w.mem = Some(MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(1, Par::Seq)));
        }
        for i in 0..n {
            w.mat[(i, 0)] = b[i];
        }
        let stack = MemStack::new(w.mem.as_mut().unwrap());
        LltRef::new(&self.symbolic, &self.values).solve_in_place_with_conj(Conj::No, w.mat.as_mut(), Par::Seq, stack);
        for i in 0..n {
            b[i] = w.mat[(i, 0)];
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
        assert!(ch.simplicial.is_some());
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let mut b = a.mul(&x);
        ch.solve_in_place(&mut b);
        for i in 0..n {
            assert!((b[i] - x[i]).abs() < 1e-12);
        }
        // 2D grid Laplacian + shift, forced simplicial vs supernodal-capable path
        let g = 30;
        let mut t = Vec::new();
        for i in 0..g {
            for j in 0..g {
                let p = i * g + j;
                t.push((p, p, 4.1));
                if i + 1 < g {
                    t.push((p, p + g, -1.0));
                    t.push((p + g, p, -1.0));
                }
                if j + 1 < g {
                    t.push((p, p + 1, -1.0));
                    t.push((p + 1, p, -1.0));
                }
            }
        }
        let a2 = CsrMatrix::from_triplets(g * g, g * g, t);
        let x2: Vec<f64> = (0..g * g).map(|i| (i as f64 * 0.11).cos()).collect();
        for ch in [SparseCholesky::new_simplicial(&a2).unwrap(), SparseCholesky::new(&a2).unwrap()] {
            let mut b = a2.mul(&x2);
            ch.solve_in_place(&mut b);
            for i in 0..g * g {
                assert!((b[i] - x2[i]).abs() < 1e-11);
            }
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

    #[test]
    fn supernodal_tree_solve_matches_and_is_thread_independent() {
        // 3D grid, 3 DOFs per node with coupled blocks: many supernodes and subtrees
        let g = 11;
        let id = |i: usize, j: usize, k: usize| (i * g + j) * g + k;
        let mut t = Vec::new();
        for i in 0..g {
            for j in 0..g {
                for k in 0..g {
                    let p = id(i, j, k);
                    for a in 0..3 {
                        for b in 0..3 {
                            t.push((3 * p + a, 3 * p + b, if a == b { 6.5 } else { 0.3 }));
                        }
                    }
                    let mut nb = Vec::new();
                    if i + 1 < g {
                        nb.push(id(i + 1, j, k));
                    }
                    if j + 1 < g {
                        nb.push(id(i, j + 1, k));
                    }
                    if k + 1 < g {
                        nb.push(id(i, j, k + 1));
                    }
                    for q in nb {
                        for a in 0..3 {
                            t.push((3 * p + a, 3 * q + a, -1.0));
                            t.push((3 * q + a, 3 * p + a, -1.0));
                        }
                    }
                }
            }
        }
        let n = 3 * g * g * g;
        let a = CsrMatrix::from_triplets(n, n, t);
        let sn = SparseCholesky::new_supernodal(&a).unwrap();
        let tree = sn.tree.as_ref().expect("supernodal factor");
        assert!(tree.0.tasks.len() > 1 && !tree.0.top.is_empty());
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.013).sin() + 0.5).collect();
        let b = a.mul(&x);
        let solve = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let mut v = b.clone();
                sn.solve_in_place(&mut v);
                v
            })
        };
        let y1 = solve(1);
        let y4 = solve(4);
        assert_eq!(y1, y4, "tree solve depends on the thread count");
        let mut z = b.clone();
        sn.solve_mat(faer::MatMut::from_column_major_slice_mut(&mut z, n, 1));
        for i in 0..n {
            assert!((y1[i] - x[i]).abs() < 1e-10, "{i}: {} vs {}", y1[i], x[i]);
            assert!((y1[i] - z[i]).abs() < 1e-11);
        }
    }
}
