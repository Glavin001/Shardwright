use frac_cells::clip::{ClipOutput, Clipper};
use frac_cells::complex::Complex;
use frac_geom::mesh::{box_mesh, icosphere};
use frac_geom::{DVec3, TriMesh, VolumeIntegrals};
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;

/// Oriented boundary polygons of every complex cell.
fn cell_polys(out: &ClipOutput, ncells: usize) -> Vec<Vec<Vec<u32>>> {
    let mut cells: Vec<Vec<Vec<u32>>> = vec![Vec::new(); ncells];
    for e in &out.ext {
        cells[e.cell as usize].push(e.verts.clone());
    }
    for p in &out.patches {
        for t in &p.tris {
            cells[p.cells[0] as usize].push(t.to_vec());
            cells[p.cells[1] as usize].push(vec![t[0], t[2], t[1]]);
        }
    }
    cells
}

fn check(out: &ClipOutput, ncells: usize, expect_volume: f64) {
    let cells = cell_polys(out, ncells);
    let mut total = 0.0;
    for (c, polys) in cells.iter().enumerate() {
        if polys.is_empty() {
            continue;
        }
        let mut de: BTreeMap<(u32, u32), i32> = BTreeMap::new();
        for p in polys {
            for k in 0..p.len() {
                let (a, b) = (p[k], p[(k + 1) % p.len()]);
                *de.entry((a, b)).or_default() += 1;
            }
        }
        for (&(a, b), &n) in &de {
            let m = de.get(&(b, a)).copied().unwrap_or(0);
            if n != m {
                eprintln!(
                    "cell {c}: edge ({a},{b}) {:?} -> {:?} used {n} rev {m}",
                    out.keys[a as usize], out.keys[b as usize]
                );
                eprintln!(
                    "  pa {:?} pb {:?}",
                    out.verts[a as usize], out.verts[b as usize]
                );
                for p in polys {
                    if p.contains(&a) || p.contains(&b) {
                        eprintln!(
                            "  poly {:?}",
                            p.iter().map(|&i| out.keys[i as usize]).collect::<Vec<_>>()
                        );
                    }
                }
                for pt in &out.patches {
                    if pt.loops.iter().any(|l| l.contains(&a) || l.contains(&b)) {
                        eprintln!(
                            "  patch cells {:?} face {} loops {:?}",
                            pt.cells,
                            pt.face,
                            pt.loops
                                .iter()
                                .map(|l| l
                                    .iter()
                                    .map(|&i| out.keys[i as usize])
                                    .collect::<Vec<_>>())
                                .collect::<Vec<_>>()
                        );
                    }
                }
                panic!("not closed");
            }
        }
        let mut vi = VolumeIntegrals::default();
        let r = out.verts[polys[0][0] as usize];
        for p in polys {
            let pts: Vec<DVec3> = p.iter().map(|&i| out.verts[i as usize]).collect();
            vi.add_polygon(r, &pts);
        }
        assert!(vi.volume > -1e-12, "cell {c} negative volume {}", vi.volume);
        total += vi.volume;
    }
    assert!(
        (total / expect_volume - 1.0).abs() < 1e-9,
        "volume {total} vs {expect_volume}"
    );
}

fn seeds_in(rng: &mut rand_chacha::ChaCha8Rng, lo: DVec3, hi: DVec3, n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|_| {
            [
                rng.gen_range(lo.x..hi.x),
                rng.gen_range(lo.y..hi.y),
                rng.gen_range(lo.z..hi.z),
            ]
        })
        .collect()
}

#[test]
fn cube_voronoi() {
    let m = box_mesh(DVec3::ZERO, DVec3::ONE);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let seeds = seeds_in(&mut rng, DVec3::ZERO, DVec3::ONE, 60);
    let cx = Complex::voronoi(&seeds, [0.0; 3], [1.0; 3]).unwrap();
    let out = Clipper::new(&m, &cx).run().unwrap();
    check(&out, cx.cells.len(), 1.0);
}

#[test]
fn sphere_voronoi() {
    let m = icosphere(DVec3::new(0.1, 0.2, 0.3), 1.0, 3);
    let vol = m.signed_volume();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(2);
    let seeds = seeds_in(&mut rng, DVec3::splat(-0.8), DVec3::splat(1.1), 150);
    let cx = Complex::voronoi(&seeds, [-1.0; 3], [1.5; 3]).unwrap();
    let out = Clipper::new(&m, &cx).run().unwrap();
    check(&out, cx.cells.len(), vol);
}

fn l_shape() -> TriMesh {
    // union of boxes, built as a single closed mesh: extrude an L polygon
    let poly = [
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [0.0, 2.0],
    ];
    extrude(&poly, 0.0, 0.7)
}

fn extrude(poly: &[[f64; 2]], z0: f64, z1: f64) -> TriMesh {
    let n = poly.len();
    let mut verts = Vec::new();
    for p in poly {
        verts.push(DVec3::new(p[0], p[1], z0));
    }
    for p in poly {
        verts.push(DVec3::new(p[0], p[1], z1));
    }
    let mut tris = Vec::new();
    // caps via ear clipping of the simple polygon
    let pts: Vec<[f64; 2]> = poly.to_vec();
    let t2 = frac_cells::tri2d::triangulate(&pts, &[(0..n).collect()]);
    for t in &t2 {
        tris.push([t[0] as u32, t[2] as u32, t[1] as u32]);
        tris.push([(t[0] + n) as u32, (t[1] + n) as u32, (t[2] + n) as u32]);
    }
    for i in 0..n {
        let j = (i + 1) % n;
        let (a, b, c, d) = (i as u32, j as u32, (j + n) as u32, (i + n) as u32);
        tris.push([a, b, c]);
        tris.push([a, c, d]);
    }
    TriMesh { verts, tris }
}

#[test]
fn l_shape_voronoi() {
    let m = l_shape();
    assert!(m.topology().is_closed_manifold());
    let vol = m.signed_volume();
    assert!((vol - 3.0 * 0.7).abs() < 1e-12);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
    let seeds = seeds_in(&mut rng, DVec3::ZERO, DVec3::new(2.0, 2.0, 0.7), 80);
    let cx = Complex::voronoi(&seeds, [0.0; 3], [2.0, 2.0, 0.7]).unwrap();
    let out = Clipper::new(&m, &cx).run().unwrap();
    check(&out, cx.cells.len(), vol);
}

#[test]
fn grid_seeds_on_box_degenerate() {
    // seeds on a regular grid: bisector planes pass exactly through mesh vertices
    // and edges; exercises symbolic perturbation everywhere.
    let m = box_mesh(DVec3::ZERO, DVec3::new(2.0, 2.0, 2.0));
    let mut seeds = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                seeds.push([
                    0.25 + 0.5 * i as f64,
                    0.25 + 0.5 * j as f64,
                    0.25 + 0.5 * k as f64,
                ]);
            }
        }
    }
    let cx = Complex::voronoi(&seeds, [0.0; 3], [2.0; 3]).unwrap();
    let out = Clipper::new(&m, &cx).run().unwrap();
    check(&out, cx.cells.len(), 8.0);
}

#[test]
fn bricks_box_complex() {
    // running-bond bricks covering a wall with a window opening
    let wall = {
        // wall 0..3 x 0..2, thickness 0.2, window hole [1,2]x[0.6,1.4]
        let mut m = TriMesh::default();
        // build as union of 4 boxes (non-overlapping, sharing faces) then weld:
        // easier: extrude a polygon with a hole is not supported by extrude();
        // use a U-shaped outline instead (window touching the top).
        let poly = [
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 2.0],
            [2.0, 2.0],
            [2.0, 0.6],
            [1.0, 0.6],
            [1.0, 2.0],
            [0.0, 2.0],
        ];
        m.append(&extrude(&poly, 0.0, 0.2));
        m
    };
    assert!(wall.topology().is_closed_manifold());
    let vol = wall.signed_volume();
    let (bw, bh) = (0.5, 0.25);
    let mut boxes = Vec::new();
    let big = 10.0;
    let rows = 8;
    for r in 0..rows {
        let y0 = if r == 0 { -big } else { r as f64 * bh };
        let y1 = if r == rows - 1 {
            big
        } else {
            (r + 1) as f64 * bh
        };
        let off = if r % 2 == 0 { 0.0 } else { bw * 0.5 };
        let mut xs = vec![-big];
        let mut x = off + bw;
        while x < 3.0 {
            xs.push(x);
            x += bw;
        }
        xs.push(big);
        for w in xs.windows(2) {
            // two wythes split at z = 0.1
            boxes.push(([w[0], y0, -big], [w[1], y1, 0.1]));
            boxes.push(([w[0], y0, 0.1], [w[1], y1, big]));
        }
    }
    let cx = Complex::boxes(&boxes).unwrap();
    let out = Clipper::new(&wall, &cx).run().unwrap();
    check(&out, cx.cells.len(), vol);
}

mod props {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig { cases: std::env::var("FRAC_CASES").ok().and_then(|v| v.parse().ok()).unwrap_or(24), .. ProptestConfig::default() })]
        #[test]
        fn random_seeds_sphere(seed in 0u64..10_000, n in 5usize..120, sub in 1u32..3) {
            let m = icosphere(DVec3::new(0.3, -0.2, 0.1), 0.7, sub);
            let vol = m.signed_volume();
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
            let seeds = seeds_in(&mut rng, DVec3::splat(-0.6), DVec3::splat(0.9), n);
            let cx = Complex::voronoi(&seeds, [-0.5; 3], [1.1; 3]).unwrap();
            let out = Clipper::new(&m, &cx).run().unwrap();
            check(&out, cx.cells.len(), vol);
        }

        #[test]
        fn quantized_seeds_box(seed in 0u64..10_000, n in 5usize..80) {
            // seeds on a coarse lattice => many exact degeneracies with the box faces
            let m = box_mesh(DVec3::ZERO, DVec3::ONE);
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
            let mut seeds: Vec<[f64; 3]> = (0..n).map(|_| [rng.gen_range(0..8) as f64 / 8.0 + 0.0625, rng.gen_range(0..8) as f64 / 8.0 + 0.0625, rng.gen_range(0..8) as f64 / 8.0]).collect();
            seeds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            seeds.dedup();
            let cx = Complex::voronoi(&seeds, [0.0; 3], [1.0; 3]).unwrap();
            let out = Clipper::new(&m, &cx).run().unwrap();
            check(&out, cx.cells.len(), 1.0);
        }
    }
}

#[test]
#[ignore]
fn perf_10k_cells() {
    let m = icosphere(DVec3::ZERO, 1.0, 5);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(9);
    let seeds = seeds_in(&mut rng, DVec3::splat(-1.0), DVec3::splat(1.0), 10_000);
    let t0 = std::time::Instant::now();
    let cx = Complex::voronoi(&seeds, [-1.0; 3], [1.0; 3]).unwrap();
    let t1 = t0.elapsed();
    let out = Clipper::new(&m, &cx).run().unwrap();
    let t2 = t0.elapsed();
    eprintln!(
        "complex {:?} clip {:?} verts {} ext {} patches {}",
        t1,
        t2 - t1,
        out.verts.len(),
        out.ext.len(),
        out.patches.len()
    );
    check(&out, cx.cells.len(), m.signed_volume());
}

#[test]
fn regression_quantized_3028() {
    let m = box_mesh(DVec3::ZERO, DVec3::ONE);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3028);
    let mut seeds: Vec<[f64; 3]> = (0..59)
        .map(|_| {
            [
                rng.gen_range(0..8) as f64 / 8.0 + 0.0625,
                rng.gen_range(0..8) as f64 / 8.0 + 0.0625,
                rng.gen_range(0..8) as f64 / 8.0,
            ]
        })
        .collect();
    seeds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    seeds.dedup();
    let cx = Complex::voronoi(&seeds, [0.0; 3], [1.0; 3]).unwrap();
    let out = Clipper::new(&m, &cx).run().unwrap();
    check(&out, cx.cells.len(), 1.0);
}
