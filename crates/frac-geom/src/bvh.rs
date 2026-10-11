//! A simple deterministic bounding volume hierarchy over boxes.

use crate::aabb::Aabb;
use glam::DVec3;

#[derive(Clone, Debug)]
struct Node {
    bbox: Aabb,
    /// For leaves: `start..start+count` in `order`. For internal nodes,
    /// `count == 0` and `start` is the index of the left child (right = left+1).
    start: u32,
    count: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Bvh {
    nodes: Vec<Node>,
    order: Vec<u32>,
}

const LEAF: usize = 4;

impl Bvh {
    pub fn build(boxes: &[Aabb]) -> Bvh {
        let mut order: Vec<u32> = (0..boxes.len() as u32).collect();
        let centers: Vec<DVec3> = boxes.iter().map(|b| b.center()).collect();
        let mut nodes = Vec::with_capacity(2 * boxes.len() / LEAF + 1);
        if boxes.is_empty() {
            return Bvh { nodes, order };
        }
        nodes.push(Node {
            bbox: Aabb::EMPTY,
            start: 0,
            count: 0,
        });
        // iterative build: stack of (node index, start, end)
        let mut stack = vec![(0usize, 0usize, boxes.len())];
        while let Some((ni, s, e)) = stack.pop() {
            let mut bb = Aabb::EMPTY;
            let mut cb = Aabb::EMPTY;
            for &i in &order[s..e] {
                bb = bb.union(&boxes[i as usize]);
                cb.grow(centers[i as usize]);
            }
            nodes[ni].bbox = bb;
            if e - s <= LEAF {
                nodes[ni].start = s as u32;
                nodes[ni].count = (e - s) as u32;
                continue;
            }
            let axis = cb.longest_axis();
            let mid = (s + e) / 2;
            order[s..e].select_nth_unstable_by(mid - s, |&a, &b| {
                let ca = centers[a as usize][axis];
                let cbv = centers[b as usize][axis];
                ca.partial_cmp(&cbv)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            let left = nodes.len();
            nodes.push(Node {
                bbox: Aabb::EMPTY,
                start: 0,
                count: 0,
            });
            nodes.push(Node {
                bbox: Aabb::EMPTY,
                start: 0,
                count: 0,
            });
            nodes[ni].start = left as u32;
            nodes[ni].count = 0;
            stack.push((left, s, mid));
            stack.push((left + 1, mid, e));
        }
        Bvh { nodes, order }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Visit all items whose box overlaps `q` (in deterministic order).
    pub fn query(&self, q: &Aabb, mut f: impl FnMut(u32)) {
        if self.nodes.is_empty() {
            return;
        }
        let mut stack = vec![0u32];
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni as usize];
            if !n.bbox.overlaps(q) {
                continue;
            }
            if n.count > 0 {
                for &i in &self.order[n.start as usize..(n.start + n.count) as usize] {
                    f(i);
                }
            } else {
                stack.push(n.start + 1);
                stack.push(n.start);
            }
        }
    }

    /// Collect sorted, deduplicated overlapping items.
    pub fn query_vec(&self, q: &Aabb) -> Vec<u32> {
        let mut v = Vec::new();
        self.query(q, |i| v.push(i));
        v.sort_unstable();
        v
    }

    /// Generic traversal with a node-pruning predicate on node boxes.
    pub fn traverse(&self, mut visit_node: impl FnMut(&Aabb) -> bool, mut leaf: impl FnMut(u32)) {
        if self.nodes.is_empty() {
            return;
        }
        let mut stack = vec![0u32];
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni as usize];
            if !visit_node(&n.bbox) {
                continue;
            }
            if n.count > 0 {
                for &i in &self.order[n.start as usize..(n.start + n.count) as usize] {
                    leaf(i);
                }
            } else {
                stack.push(n.start + 1);
                stack.push(n.start);
            }
        }
    }

    /// Nearest item only if every item is farther than `r2` (squared):
    /// returns `None` as soon as an item with `item_d2 <= r2` is found
    /// (early exit for Hausdorff-style maxima), else the nearest item.
    pub fn nearest_beyond(
        &self,
        p: DVec3,
        r2: f64,
        mut item_d2: impl FnMut(u32) -> f64,
    ) -> Option<(u32, f64)> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best: Option<(u32, f64)> = None;
        let mut stack: Vec<(f64, u32)> = vec![(self.nodes[0].bbox.dist2(p), 0)];
        while let Some((d, ni)) = stack.pop() {
            if let Some((_, bd)) = best {
                if d > bd {
                    continue;
                }
            }
            let n = &self.nodes[ni as usize];
            if n.count > 0 {
                for &i in &self.order[n.start as usize..(n.start + n.count) as usize] {
                    let di = item_d2(i);
                    if di <= r2 {
                        return None;
                    }
                    match best {
                        Some((bi, bd)) if di > bd || (di == bd && i > bi) => {}
                        _ => best = Some((i, di)),
                    }
                }
            } else {
                let l = n.start;
                let r = n.start + 1;
                let dl = self.nodes[l as usize].bbox.dist2(p);
                let dr = self.nodes[r as usize].bbox.dist2(p);
                if dl <= dr {
                    stack.push((dr, r));
                    stack.push((dl, l));
                } else {
                    stack.push((dl, l));
                    stack.push((dr, r));
                }
            }
        }
        best
    }

    /// Best-first nearest search with a user distance (squared) function.
    /// Returns (item, d2) minimizing `item_d2`.
    pub fn nearest(&self, p: DVec3, mut item_d2: impl FnMut(u32) -> f64) -> Option<(u32, f64)> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best: Option<(u32, f64)> = None;
        let mut stack: Vec<(f64, u32)> = vec![(self.nodes[0].bbox.dist2(p), 0)];
        while let Some((d, ni)) = stack.pop() {
            if let Some((_, bd)) = best {
                if d > bd {
                    continue;
                }
            }
            let n = &self.nodes[ni as usize];
            if n.count > 0 {
                for &i in &self.order[n.start as usize..(n.start + n.count) as usize] {
                    let di = item_d2(i);
                    match best {
                        Some((bi, bd)) if di > bd || (di == bd && i > bi) => {}
                        _ => best = Some((i, di)),
                    }
                }
            } else {
                let l = n.start;
                let r = n.start + 1;
                let dl = self.nodes[l as usize].bbox.dist2(p);
                let dr = self.nodes[r as usize].bbox.dist2(p);
                // push farther first so nearer is popped first
                if dl <= dr {
                    stack.push((dr, r));
                    stack.push((dl, l));
                } else {
                    stack.push((dl, l));
                    stack.push((dr, r));
                }
            }
        }
        best
    }

    /// Visit all pairs (i, j), i < j, of overlapping boxes.
    pub fn self_pairs(&self, boxes: &[Aabb], mut f: impl FnMut(u32, u32)) {
        if self.nodes.is_empty() {
            return;
        }
        let mut stack = vec![(0u32, 0u32)];
        while let Some((a, b)) = stack.pop() {
            let na = &self.nodes[a as usize];
            let nb = &self.nodes[b as usize];
            if !na.bbox.overlaps(&nb.bbox) {
                continue;
            }
            match (na.count > 0, nb.count > 0) {
                (true, true) => {
                    let ia = &self.order[na.start as usize..(na.start + na.count) as usize];
                    let ib = &self.order[nb.start as usize..(nb.start + nb.count) as usize];
                    for &x in ia {
                        for &y in ib {
                            if a == b && x >= y {
                                continue;
                            }
                            if a != b && x == y {
                                continue;
                            }
                            if boxes[x as usize].overlaps(&boxes[y as usize]) {
                                if x < y { f(x, y) } else { f(y, x) }
                            }
                        }
                    }
                }
                (false, true) => {
                    stack.push((na.start, b));
                    stack.push((na.start + 1, b));
                }
                (true, false) => {
                    stack.push((a, nb.start));
                    stack.push((a, nb.start + 1));
                }
                (false, false) => {
                    if a == b {
                        stack.push((na.start, na.start));
                        stack.push((na.start + 1, na.start + 1));
                        stack.push((na.start, na.start + 1));
                    } else {
                        stack.push((na.start, nb.start));
                        stack.push((na.start, nb.start + 1));
                        stack.push((na.start + 1, nb.start));
                        stack.push((na.start + 1, nb.start + 1));
                    }
                }
            }
        }
    }
}
