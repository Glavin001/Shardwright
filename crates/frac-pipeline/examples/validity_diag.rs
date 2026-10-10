//! Clean-geometry validity diagnostics on a baked asset: for every fragment
//! whose clean boundary mesh fails `mesh_validity`, report the level, cell
//! count and the first offending triangle pair (with owning cells).
//! Usage: cargo run --release -p frac-pipeline --example validity_diag -- <X.asset.json> [max_reports]
use frac_core::Asset;
use frac_geom::TriMesh;
use frac_validate::gates::mesh_validity;
use rayon::prelude::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let asset: Asset = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let max: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let bad: Vec<(usize, String)> = asset
        .hierarchy
        .fragments
        .par_iter()
        .enumerate()
        .filter_map(|(i, f)| {
            let m = frac_collision::cells_boundary_mesh(&asset, asset.fragment_cells(f));
            let (ok, d) = mesh_validity(&m);
            (!ok).then_some((i, d))
        })
        .collect();
    let mut per_level = vec![0usize; asset.hierarchy.levels as usize];
    for (i, _) in &bad {
        per_level[asset.hierarchy.fragments[*i].level as usize] += 1;
    }
    println!("{} invalid clean fragments; per level {:?}", bad.len(), per_level);
    let ones: Vec<_> = bad.iter().filter(|(i, _)| asset.fragment_cells(&asset.hierarchy.fragments[*i]).len() == 1).collect();
    let pick: Vec<_> = if ones.is_empty() { bad.iter().collect() } else { ones };
    for (i, d) in pick.into_iter().take(max) {
        let f = &asset.hierarchy.fragments[*i];
        let cells = asset.fragment_cells(f);
        let comp = &asset.components[asset.cells[cells[0].idx()].component.idx()];
        println!("fragment {i} level {} component '{}' cells {}: {d}", f.level, comp.name, cells.len());
        if cells.len() == 1 {
            let c = cells[0];
            let g = &comp.geometry;
            let pr = |l: &[u32]| l.iter().map(|&v| { let p = g.verts[v as usize]; format!("{v}:({:.4},{:.4},{:.4})", p.x, p.y, p.z) }).collect::<Vec<_>>().join(" ");
            for e in g.ext_polys.iter().filter(|e| e.cell == c) {
                let degen = e.tris.iter().filter(|t| TriMesh { verts: g.verts.clone(), tris: vec![**t] }.is_degenerate(0)).count();
                if degen > 0 {
                    println!("  ext src_tri {} loop [{}] tris {:?} ({degen} degenerate)", e.src_tri, pr(&e.verts), e.tris);
                }
            }
            for p in g.patches.iter().filter(|p| p.cells.0 == c || p.cells.1 == c) {
                let degen = p.tris.iter().filter(|t| TriMesh { verts: g.verts.clone(), tris: vec![**t] }.is_degenerate(0)).count();
                if degen > 0 {
                    println!("  patch cells {:?} loops {:?} tris {:?} ({degen} degenerate)", p.cells, p.loops.iter().map(|l| pr(l)).collect::<Vec<_>>(), p.tris);
                }
            }
        }
        let m = frac_collision::cells_boundary_mesh(&asset, cells);
        let em = m.edge_map();
        for (&(a, b), ts) in &em {
            let r = em.get(&(b, a)).map(|v| v.len()).unwrap_or(0);
            if ts.len() != 1 || r != 1 {
                println!("   edge {a}->{b} {:?}->{:?}: {} fwd {} rev; tris {:?}", m.verts[a as usize], m.verts[b as usize], ts.len(), r, ts.iter().map(|&t| m.tris[t as usize]).collect::<Vec<_>>());
            }
        }
        for (a, b) in m.self_intersections(2) {
            for t in [a, b] {
                println!("   tri {t} {:?} degenerate {}", m.tri_points(t as usize), m.is_degenerate(t as usize));
            }
        }
    }
}
