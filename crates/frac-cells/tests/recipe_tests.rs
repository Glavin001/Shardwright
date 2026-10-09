use frac_cells::cellset::CellSet;
use frac_cells::recipes::{build_cells, BondPattern, CellParams, MasonryLayout, Recipe};
use frac_geom::mesh::{box_mesh, icosphere};
use frac_geom::{DVec3, TriMesh};
use std::collections::BTreeMap;

fn check_cells(cs: &CellSet, vol: f64) {
    let polys = cs.cell_polygons();
    let mut total = 0.0;
    for (c, ps) in polys.iter().enumerate() {
        let mut de: BTreeMap<(u32, u32), i32> = BTreeMap::new();
        for p in ps {
            for k in 0..p.len() {
                *de.entry((p[k], p[(k + 1) % p.len()])).or_default() += 1;
            }
        }
        for (&(a, b), &n) in &de {
            assert_eq!(n, de.get(&(b, a)).copied().unwrap_or(0), "cell {c} edge {a}-{b}");
        }
        total += cs.cells[c].vi.volume;
        assert!(cs.cells[c].vi.volume > 0.0);
    }
    assert!((total / vol - 1.0).abs() < 1e-9, "{total} vs {vol}");
}

fn params(recipe: Recipe, na: usize) -> CellParams<'static> {
    CellParams {
        recipe,
        analysis_target: na,
        fine_per_analysis: 6,
        max_fine: 5000,
        asset_seed: 42,
        component: 0,
        variant: 0,
        grain: None,
        min_cell_volume: 1e-9,
        min_thickness_ratio: 0.05,
        spacing: None,
    }
}

fn wall_with_window() -> TriMesh {
    // 2.0 x 1.5 x 0.2 wall with a 0.6 x 0.6 window, built as union of 4 boxes
    // welded into one closed mesh via an outline extrusion with a slot
    let outline = [[0.0, 0.0], [2.0, 0.0], [2.0, 1.5], [1.3, 1.5], [1.3, 0.5], [0.7, 0.5], [0.7, 1.5], [0.0, 1.5]];
    let n = outline.len();
    let mut verts = Vec::new();
    for z in [0.0, 0.2] {
        for p in outline {
            verts.push(DVec3::new(p[0], p[1], z));
        }
    }
    let t2 = frac_cells::tri2d::triangulate(&outline, &[(0..n).collect()]);
    let mut tris = Vec::new();
    for t in &t2 {
        tris.push([t[0] as u32, t[2] as u32, t[1] as u32]);
        tris.push([(t[0] + n) as u32, (t[1] + n) as u32, (t[2] + n) as u32]);
    }
    for i in 0..n {
        let j = (i + 1) % n;
        tris.push([i as u32, j as u32, (j + n) as u32]);
        tris.push([i as u32, (j + n) as u32, (i + n) as u32]);
    }
    TriMesh { verts, tris }
}

#[test]
fn concrete_box() {
    let m = box_mesh(DVec3::ZERO, DVec3::new(0.4, 2.0, 0.4));
    let b = build_cells(&m, &params(Recipe::ClusteredVoronoi, 20)).unwrap();
    check_cells(&b.cells, m.signed_volume());
    assert!(b.n_clusters >= 10, "clusters {}", b.n_clusters);
    assert!(b.cells.cells.len() > 60);
}

#[test]
fn ceramic_sphere_shell_like() {
    let m = icosphere(DVec3::ZERO, 0.2, 3);
    let b = build_cells(&m, &params(Recipe::ClusteredVoronoi, 8)).unwrap();
    check_cells(&b.cells, m.signed_volume());
}

#[test]
fn masonry_bonds() {
    let m = wall_with_window();
    let vol = m.signed_volume();
    for bond in [BondPattern::Stretcher, BondPattern::Stack, BondPattern::English, BondPattern::Flemish] {
        let layout = MasonryLayout { brick: [0.2, 0.065, 0.09], bond, mortar: 0.01, ..Default::default() };
        let b = build_cells(&m, &params(Recipe::Masonry(layout), 1)).unwrap();
        check_cells(&b.cells, vol);
        assert!(b.n_clusters > 50, "{bond:?}: {}", b.n_clusters);
    }
}

#[test]
fn wood_beam_grain_alignment() {
    let m = box_mesh(DVec3::ZERO, DVec3::new(3.0, 0.2, 0.1));
    let mut p = params(Recipe::Wood { stretch: 6.0, fine_stretch: 9.0 }, 12);
    p.grain = Some(DVec3::X);
    let b = build_cells(&m, &p).unwrap();
    check_cells(&b.cells, m.signed_volume());
    // mean |cos| between cell principal axis and grain
    let mut s = 0.0;
    let mut w = 0.0;
    for c in &b.cells.cells {
        let vi = c.vi;
        let com = vi.com();
        let cov = vi.second - frac_geom::polygon::outer(com, com) * vi.volume;
        let (_, vecs) = frac_geom::integrals::sym_eigen3(&cov);
        s += vecs.col(2).x.abs() * vi.volume;
        w += vi.volume;
    }
    assert!(s / w > 0.85, "grain alignment {}", s / w);
}

#[test]
fn glass_annealed_and_tempered() {
    let m = box_mesh(DVec3::ZERO, DVec3::new(1.0, 0.8, 0.006));
    let b = build_cells(&m, &params(Recipe::GlassAnnealed { impact: None, rings: 6, spokes: 14 }, 2)).unwrap();
    check_cells(&b.cells, m.signed_volume());
    assert!(b.cells.cells.len() > 40);
    let t = build_cells(&m, &params(Recipe::GlassTempered { cell_size: Some(0.04) }, 2)).unwrap();
    check_cells(&t.cells, m.signed_volume());
    assert!(t.cells.cells.len() > 200, "{}", t.cells.cells.len());
}

#[test]
fn panel_steel_stone() {
    let panel = box_mesh(DVec3::ZERO, DVec3::new(1.2, 2.4, 0.0125));
    let b = build_cells(&panel, &params(Recipe::Panel, 6)).unwrap();
    check_cells(&b.cells, panel.signed_volume());
    let beam = box_mesh(DVec3::ZERO, DVec3::new(4.0, 0.3, 0.15));
    let s = build_cells(&beam, &params(Recipe::Steel, 1)).unwrap();
    check_cells(&s.cells, beam.signed_volume());
    assert_eq!(s.cells.cells.len(), 1);
    let block = box_mesh(DVec3::ZERO, DVec3::new(1.0, 1.0, 1.0));
    let st = build_cells(&block, &params(Recipe::Stone { bedding: [0.0, 1.0, 0.0], layer: 0.2, flatten: 2.5 }, 10)).unwrap();
    check_cells(&st.cells, 1.0);
}

#[test]
fn determinism() {
    let m = icosphere(DVec3::ZERO, 0.5, 2);
    let a = build_cells(&m, &params(Recipe::ClusteredVoronoi, 10)).unwrap();
    let b = build_cells(&m, &params(Recipe::ClusteredVoronoi, 10)).unwrap();
    assert_eq!(a.cells.verts.len(), b.cells.verts.len());
    for (x, y) in a.cells.verts.iter().zip(b.cells.verts.iter()) {
        assert_eq!(x.to_array().map(f64::to_bits), y.to_array().map(f64::to_bits));
    }
}

#[test]
fn masonry_chips_clip_level_closed() {
    use frac_cells::clip::Clipper;
    use frac_cells::complex::Complex;
    let m = wall_with_window();
    let layout = MasonryLayout { brick: [0.2, 0.065, 0.09], bond: BondPattern::Stretcher, mortar: 0.01, ..Default::default() };
    let bb = m.aabb();
    let boxes = frac_cells::recipes::masonry_boxes(&layout, bb.min, bb.max, 42 ^ frac_core::determinism::stable_hash(&[42, 0, 0, 77]));
    let cx = Complex::boxes(&boxes.iter().map(|b| (b.0, b.1)).collect::<Vec<_>>()).unwrap();
    let out = Clipper::new(&m, &cx).run().unwrap();
    let mut cells: Vec<Vec<Vec<u32>>> = vec![Vec::new(); cx.cells.len()];
    for e in &out.ext { cells[e.cell as usize].push(e.verts.clone()); }
    for p in &out.patches { for t in &p.tris { cells[p.cells[0] as usize].push(t.to_vec()); cells[p.cells[1] as usize].push(vec![t[0], t[2], t[1]]); } }
    for (c, ps) in cells.iter().enumerate() {
        let mut de: BTreeMap<(u32, u32), i32> = BTreeMap::new();
        for p in ps { for k in 0..p.len() { *de.entry((p[k], p[(k + 1) % p.len()])).or_default() += 1; } }
        for (&(a, b), &n) in &de {
            let r = de.get(&(b, a)).copied().unwrap_or(0);
            if n != r {
                eprintln!("cell {c} box {:?} edge {:?}->{:?} {:?} {:?} n {n} r {r}", boxes[c], out.keys[a as usize], out.keys[b as usize], out.verts[a as usize], out.verts[b as usize]);
                for p in ps { if p.contains(&a) && p.contains(&b) { eprintln!("   poly {:?}", p.iter().map(|&i| out.keys[i as usize]).collect::<Vec<_>>()); } }
                for pt in &out.patches { if pt.loops.iter().any(|l| l.contains(&a) && l.contains(&b)) { eprintln!("   patch {:?} {:?}", pt.cells, pt.loops.iter().map(|l| l.iter().map(|&i| out.keys[i as usize]).collect::<Vec<_>>()).collect::<Vec<_>>()); } }
                panic!();
            }
        }
    }
}
