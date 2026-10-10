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

fn find(p: &mut [u32], mut x: u32) -> u32 {
    while p[x as usize] != x {
        p[x as usize] = p[p[x as usize] as usize];
        x = p[x as usize];
    }
    x
}

fn components(n_cells: u32, groups: &[(u32, u32)], cut: &[bool]) -> (u32, Vec<u32>) {
    let n = n_cells as usize;
    let mut parent: Vec<u32> = (0..n_cells).collect();
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
pub fn segment_level1(
    n_cells: u32,
    groups: &[(u32, u32)],
    max_jump: &[f64],
    target: u32,
) -> Level1 {
    assert_eq!(groups.len(), max_jump.len());
    // candidate thresholds: L = [0] ∪ {positive jumps}, sorted, unique
    let mut vals: Vec<f64> = max_jump
        .iter()
        .copied()
        .filter(|v| *v > 0.0 && v.is_finite())
        .collect();
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
    Level1 {
        labels,
        n_fragments: nf,
        sigma,
        cut_groups,
        hit_target,
    }
}

/// Merge components below `min_volume` into the adjacent component they
/// share the most interface area with (smallest first, deterministic
/// tie-breaks). Returns the relabelled components (dense labels).
fn merge_small(
    n_comp: u32,
    labels: &[u32],
    groups: &[(u32, u32)],
    area: &[f64],
    volume: &[f64],
    min_volume: f64,
) -> (u32, Vec<u32>) {
    let mut parent: Vec<u32> = (0..n_comp).collect();
    let mut vol = vec![0.0f64; n_comp as usize];
    for (c, &l) in labels.iter().enumerate() {
        vol[l as usize] += volume[c];
    }
    loop {
        // current roots and their shared areas
        let root = |p: &mut Vec<u32>, x: u32| find(p, x);
        let mut shared: std::collections::BTreeMap<(u32, u32), f64> =
            std::collections::BTreeMap::new();
        for (g, &(a, b)) in groups.iter().enumerate() {
            let (ra, rb) = (
                root(&mut parent, labels[a as usize]),
                root(&mut parent, labels[b as usize]),
            );
            if ra != rb {
                *shared.entry((ra.min(rb), ra.max(rb))).or_insert(0.0) += area[g];
            }
        }
        // smallest undersized root that has a neighbour
        let mut small: Option<(f64, u32)> = None;
        for r in 0..n_comp {
            if parent[r as usize] != r || vol[r as usize] >= min_volume {
                continue;
            }
            if shared.keys().any(|&(a, b)| a == r || b == r)
                && small.is_none_or(|(v, _)| vol[r as usize] < v)
            {
                small = Some((vol[r as usize], r));
            }
        }
        let Some((_, r)) = small else { break };
        let mut best: Option<(f64, u32)> = None;
        for (&(a, b), &ar) in &shared {
            let o = if a == r {
                b
            } else if b == r {
                a
            } else {
                continue;
            };
            if best.is_none_or(|(ba, bo)| ar > ba || (ar == ba && o < bo)) {
                best = Some((ar, o));
            }
        }
        let (_, o) = best.unwrap();
        let (lo, hi) = if r < o { (r, o) } else { (o, r) };
        parent[hi as usize] = lo;
        vol[lo as usize] += vol[hi as usize];
    }
    let mut dense = vec![u32::MAX; n_comp as usize];
    let mut count = 0;
    let out = labels
        .iter()
        .map(|&l| {
            let r = find(&mut parent, l) as usize;
            if dense[r] == u32::MAX {
                dense[r] = count;
                count += 1;
            }
            dense[r]
        })
        .collect();
    (count, out)
}

/// Size-balanced Level-1 segmentation. Thresholding the mode jumps alone
/// favours tiny surface chips (isolating a corner cell is the cheapest
/// discontinuity a mode can have). Here, at every candidate threshold σ the
/// components smaller than `min_volume` are merged into the neighbour they
/// share the most area with, and σ is chosen (largest first, i.e. fewest
/// cuts) so that the merged fragment count is closest to `target`.
#[allow(clippy::too_many_arguments)]
pub fn segment_level1_balanced(
    n_cells: u32,
    groups: &[(u32, u32)],
    max_jump: &[f64],
    area: &[f64],
    volume: &[f64],
    target: u32,
    min_volume: f64,
) -> Level1 {
    assert_eq!(groups.len(), max_jump.len());
    assert_eq!(groups.len(), area.len());
    let mut vals: Vec<f64> = max_jump
        .iter()
        .copied()
        .filter(|v| *v > 0.0 && v.is_finite())
        .collect();
    vals.push(0.0);
    vals.sort_by(|a, b| a.total_cmp(b));
    vals.dedup();
    let sigma_at = |j: usize| -> f64 {
        if j + 1 < vals.len() {
            0.5 * (vals[j] + vals[j + 1])
        } else {
            vals[j]
        }
    };
    let eval = |sigma: f64| -> (u32, Vec<u32>) {
        let cut: Vec<bool> = max_jump.iter().map(|&v| v > sigma).collect();
        let (n, l) = components(n_cells, groups, &cut);
        merge_small(n, &l, groups, area, volume, min_volume)
    };
    // scan from the largest σ (no cuts) down; keep the closest count,
    // preferring larger σ on ties; stop once the count overshoots by 50%
    let mut best: Option<(i64, f64, u32, Vec<u32>)> = None;
    for j in (0..vals.len()).rev() {
        let sg = sigma_at(j);
        let (nf, l) = eval(sg);
        let d = (nf as i64 - target as i64).abs();
        if best.as_ref().is_none_or(|b| d < b.0) {
            best = Some((d, sg, nf, l));
        }
        if d == 0 || nf as f64 > 1.5 * target as f64 {
            break;
        }
    }
    let (_, sigma, nf, labels) = best.unwrap();
    let cut_groups = groups
        .iter()
        .map(|&(a, b)| labels[a as usize] != labels[b as usize])
        .collect();
    let hit_target = ((nf as f64) - (target as f64)).abs() <= 0.1 * target as f64;
    Level1 {
        labels,
        n_fragments: nf,
        sigma,
        cut_groups,
        hit_target,
    }
}

/// Level-1 segmentation from mode jumps over an exact analysis-cell
/// adjacency `(a < b, shared area, w_g)`: groups present in the adjacency but
/// not in the tet staircase get zero jump (never cut first), shared areas are
/// taken from the adjacency, then [`segment_level1_balanced`]. Returns the
/// segmentation and the `(groups, max_jump)` it was computed from.
pub fn segment_from_jumps(
    n_cells: u32,
    mode_groups: &[(u32, u32)],
    max_jump: &[f64],
    adjacency: &[(u32, u32, f64, f64)],
    volume: &[f64],
    target: u32,
    min_volume: f64,
) -> (Level1, Vec<(u32, u32)>, Vec<f64>) {
    let mut groups = mode_groups.to_vec();
    let mut mj = max_jump.to_vec();
    let present: std::collections::BTreeSet<(u32, u32)> = groups.iter().copied().collect();
    for &(a, b, _, _) in adjacency {
        if !present.contains(&(a, b)) {
            groups.push((a, b));
            mj.push(0.0);
        }
    }
    let area_of: std::collections::BTreeMap<(u32, u32), f64> = adjacency
        .iter()
        .map(|&(a, b, ar, _)| ((a, b), ar))
        .collect();
    let areas: Vec<f64> = groups
        .iter()
        .map(|&(a, b)| *area_of.get(&(a.min(b), a.max(b))).unwrap_or(&0.0))
        .collect();
    let l1 = segment_level1_balanced(n_cells, &groups, &mj, &areas, volume, target, min_volume);
    (l1, groups, mj)
}

/// Adjusted Rand index of two labelings (1 = identical partitions).
pub fn adjusted_rand_index(a: &[u32], b: &[u32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let c2 = |x: f64| 0.5 * x * (x - 1.0);
    let mut cont: std::collections::BTreeMap<(u32, u32), f64> = std::collections::BTreeMap::new();
    let mut sa: std::collections::BTreeMap<u32, f64> = std::collections::BTreeMap::new();
    let mut sb: std::collections::BTreeMap<u32, f64> = std::collections::BTreeMap::new();
    for (&x, &y) in a.iter().zip(b) {
        *cont.entry((x, y)).or_insert(0.0) += 1.0;
        *sa.entry(x).or_insert(0.0) += 1.0;
        *sb.entry(y).or_insert(0.0) += 1.0;
    }
    let idx: f64 = cont.values().map(|&v| c2(v)).sum();
    let ea: f64 = sa.values().map(|&v| c2(v)).sum();
    let eb: f64 = sb.values().map(|&v| c2(v)).sum();
    let exp = ea * eb / c2(n).max(1.0);
    let mx = 0.5 * (ea + eb);
    if (mx - exp).abs() < 1e-12 {
        1.0
    } else {
        (idx - exp) / (mx - exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ari_known_values() {
        assert_eq!(adjusted_rand_index(&[0, 0, 1, 1], &[5, 5, 2, 2]), 1.0);
        // sklearn: adjusted_rand_score([0,0,1,1],[0,0,1,2]) = 0.5714285714
        assert!(
            (adjusted_rand_index(&[0, 0, 1, 1], &[0, 0, 1, 2]) - 0.5714285714285714).abs() < 1e-12
        );
    }

    #[test]
    fn balanced_segmentation_merges_chips() {
        // chain of 6 equal cells; the largest jumps isolate the end cell
        // (a "chip"); balanced segmentation must split in the middle instead
        let groups = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
        let jumps = vec![0.9, 0.1, 0.5, 0.1, 0.8];
        let area = vec![1.0; 5];
        let vol = vec![1.0; 6];
        let l = segment_level1_balanced(6, &groups, &jumps, &area, &vol, 2, 2.0);
        assert_eq!(l.n_fragments, 2);
        assert_eq!(l.labels, vec![0, 0, 0, 1, 1, 1]);
    }

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
