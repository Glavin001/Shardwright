//! Deterministic blue-noise (Poisson-disk) seeding with variable radius,
//! via greedy sample elimination over a candidate pool and bisection on the
//! radius to hit a target count.

use frac_geom::DVec3;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeMap;

/// Sample about `target` points from `candidates` with a Poisson-disk
/// constraint `|x - y| >= r * min(s(x), s(y))`, where `s` is a relative
/// spacing field (1 = nominal; smaller = denser). Candidates are visited in
/// a random (seeded) order. Returns accepted points in candidate order.
pub fn eliminate(candidates: &[DVec3], spacing: &dyn Fn(DVec3) -> f64, target: usize, rng: &mut ChaCha8Rng) -> Vec<DVec3> {
    if candidates.is_empty() || target == 0 {
        return Vec::new();
    }
    if target >= candidates.len() {
        return candidates.to_vec();
    }
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    // Fisher–Yates with the seeded RNG
    for i in (1..order.len()).rev() {
        let j = rng.gen_range(0..=i);
        order.swap(i, j);
    }
    let s: Vec<f64> = candidates.iter().map(|&p| spacing(p).max(1e-6)).collect();
    let bb = frac_geom::Aabb::from_points(candidates.iter());
    let ext = bb.extent();
    let dim = ext.to_array().iter().filter(|&&e| e > 1e-12 * bb.diagonal()).count().max(1) as f64;
    let measure: f64 = ext.to_array().iter().filter(|&&e| e > 1e-12 * bb.diagonal()).product();
    let r0 = (measure / target as f64).powf(1.0 / dim);
    let run = |r: f64| -> Vec<usize> {
        let smax = s.iter().cloned().fold(0.0, f64::max);
        let cell = (r * smax).max(1e-12);
        let key = |p: DVec3| {
            let q = (p - bb.min) / cell;
            (q.x.floor() as i64, q.y.floor() as i64, q.z.floor() as i64)
        };
        let mut grid: BTreeMap<(i64, i64, i64), Vec<usize>> = BTreeMap::new();
        let mut acc = Vec::new();
        for &i in &order {
            let p = candidates[i];
            let k = key(p);
            let mut ok = true;
            'o: for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(v) = grid.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                            for &j in v {
                                let rr = r * s[i].min(s[j]);
                                if (candidates[j] - p).length_squared() < rr * rr {
                                    ok = false;
                                    break 'o;
                                }
                            }
                        }
                    }
                }
            }
            if ok {
                grid.entry(k).or_default().push(i);
                acc.push(i);
            }
        }
        acc
    };
    // bisection on r: count is decreasing in r
    let (mut lo, mut hi) = (r0 * 0.05, r0 * 3.0);
    let mut best = run(r0 * 0.7);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let a = run(mid);
        let diff = (a.len() as i64 - target as i64).abs();
        if diff < (best.len() as i64 - target as i64).abs() {
            best = a.clone();
        }
        if a.len() > target {
            lo = mid;
        } else {
            hi = mid;
        }
        if diff as f64 <= 0.01 * target as f64 {
            break;
        }
    }
    let mut idx = best;
    idx.sort_unstable();
    idx.into_iter().map(|i| candidates[i]).collect()
}

/// Uniform candidates inside a region defined by an inside test, over a box.
pub fn candidates_in(
    lo: DVec3,
    hi: DVec3,
    inside: &(dyn Fn(DVec3) -> bool + Sync),
    want: usize,
    flat_axis: Option<(usize, f64)>,
    rng: &mut ChaCha8Rng,
) -> Vec<DVec3> {
    let mut out = Vec::with_capacity(want);
    let mut tries = 0usize;
    let max_tries = want * 400 + 10_000;
    while out.len() < want && tries < max_tries {
        let batch = (want - out.len()).max(64) * 2;
        let pts: Vec<DVec3> = (0..batch)
            .map(|_| {
                let mut p = DVec3::new(rng.gen_range(lo.x..=hi.x), rng.gen_range(lo.y..=hi.y), rng.gen_range(lo.z..=hi.z));
                if let Some((ax, v)) = flat_axis {
                    p[ax] = v;
                }
                p
            })
            .collect();
        tries += batch;
        for p in pts {
            if inside(p) {
                out.push(p);
                if out.len() >= want {
                    break;
                }
            }
        }
    }
    out
}
