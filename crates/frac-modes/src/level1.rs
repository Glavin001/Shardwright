//! From modes to Level-1 fragments (spec §4.4).

/// Level-1 segmentation of the analysis-cell graph.
#[derive(Clone, Debug, PartialEq)]
pub struct Level1 {
    /// Canonical fragment label per cell (fragment containing the lowest cell
    /// id is 0, the next unseen one 1, ...).
    pub labels: Vec<u32>,
    pub n_fragments: u32,
    /// Threshold used: group g is cut iff `max_jump[g] > sigma`.
    pub sigma: f64,
    pub cut_groups: Vec<bool>,
    /// Whether `n_fragments` is within ±10% of the target.
    pub hit_target: bool,
}

fn components(n_cells: u32, groups: &[(u32, u32)], cut: &[bool]) -> (u32, Vec<u32>) {
    let n = n_cells as usize;
    let mut parent: Vec<u32> = (0..n_cells).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    for (g, &(a, b)) in groups.iter().enumerate() {
        if cut[g] || a >= n_cells || b >= n_cells {
            continue;
        }
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            parent[hi as usize] = lo;
        }
    }
    let mut label = vec![u32::MAX; n];
    let mut out = vec![0u32; n];
    let mut count = 0;
    for c in 0..n {
        let r = find(&mut parent, c as u32) as usize;
        if label[r] == u32::MAX {
            label[r] = count;
            count += 1;
        }
        out[c] = label[r];
    }
    (count, out)
}

/// Cuts interfaces whose max jump exceeds σ and labels connected components
/// of the remaining cell adjacency graph. σ is chosen by a deterministic
/// bisection over the sorted distinct jump values (the fragment count is
/// monotone in σ and only changes at those values), aiming at `target`
/// fragments with the largest such σ (fewest cut interfaces); σ is placed midway between consecutive distinct values (or at
/// half the smallest positive value when cutting everything), so groups with
/// zero jump (e.g. forbidden interfaces) are never cut. If the target is not
/// reachable within ±10%, the closest achievable count is returned with
/// `hit_target = false`.
pub fn segment_level1(n_cells: u32, groups: &[(u32, u32)], max_jump: &[f64], target: u32) -> Level1 {
    assert_eq!(groups.len(), max_jump.len());
    // candidate thresholds: L = [0] ∪ {positive jumps}, sorted, unique
    let mut vals: Vec<f64> = max_jump.iter().copied().filter(|v| *v > 0.0 && v.is_finite()).collect();
    vals.push(0.0);
    vals.sort_by(|a, b| a.total_cmp(b));
    vals.dedup();
    // sigma_j for j = 0..len: midpoint between vals[j] and vals[j+1]; last = max (no cuts)
    let sigma_at = |j: usize| -> f64 {
        if j + 1 < vals.len() {
            0.5 * (vals[j] + vals[j + 1])
        } else {
            vals[j]
        }
    };
    let eval = |sigma: f64| -> (u32, Vec<u32>, Vec<bool>) {
        let cut: Vec<bool> = max_jump.iter().map(|&v| v > sigma).collect();
        let (n, l) = components(n_cells, groups, &cut);
        (n, l, cut)
    };
    // f(j) = fragments at sigma_at(j), non-increasing in j. Find the largest j
    // (largest σ, fewest cuts) with f(j) >= target, then compare with j + 1.
    let last = vals.len() - 1;
    let (mut lo, mut hi) = (0usize, last);
    let f0 = eval(sigma_at(0)).0;
    let jstar = if f0 < target {
        0
    } else if eval(sigma_at(hi)).0 >= target {
        hi
    } else {
        // invariant: f(lo) >= target > f(hi)
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if eval(sigma_at(mid)).0 >= target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };
    let (fa, la, ca) = eval(sigma_at(jstar));
    let mut best = (sigma_at(jstar), fa, la, ca);
    if jstar < last && fa > target {
        let (fb, lb, cb) = eval(sigma_at(jstar + 1));
        let da = (fa as i64 - target as i64).abs();
        let db = (fb as i64 - target as i64).abs();
        // ties prefer the larger σ (fewer cuts)
        if db <= da {
            best = (sigma_at(jstar + 1), fb, lb, cb);
        }
    }
    let (sigma, nf, labels, cut_groups) = best;
    let hit_target = ((nf as f64) - (target as f64)).abs() <= 0.1 * target as f64;
    Level1 { labels, n_fragments: nf, sigma, cut_groups, hit_target }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_segmentation() {
        // chain 0-1-2-3-4 with jumps; target 3 -> cut the two largest
        let groups = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
        let jumps = vec![0.1, 0.9, 0.2, 0.8];
        let l = segment_level1(5, &groups, &jumps, 3);
        assert_eq!(l.n_fragments, 3);
        assert_eq!(l.labels, vec![0, 0, 1, 1, 2]);
        assert_eq!(l.cut_groups, vec![false, true, false, true]);
        assert!(l.hit_target);
        assert!(l.sigma > 0.2 && l.sigma < 0.8);
        // target 1 -> no cuts
        let l = segment_level1(5, &groups, &jumps, 1);
        assert_eq!(l.n_fragments, 1);
        // impossible target: zero-jump groups never cut, isolated cell 5 stays alone
        let l = segment_level1(6, &groups, &[0.0, 0.0, 0.0, 0.5], 10);
        assert_eq!(l.n_fragments, 3);
        assert!(!l.hit_target);
        assert_eq!(l.labels, vec![0, 0, 0, 0, 1, 2]);
    }
}
