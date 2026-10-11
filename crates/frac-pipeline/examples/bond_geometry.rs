//! Why exact bond geometry matters: the uniform-strain patch test of a baked
//! asset's finest level with its exact bonds, and with the approximations a
//! connectivity estimate without the contact geometry would use:
//! * `centre-line normals`: exact areas, normals and centroids taken from
//!   the line between the two fragments' centres;
//! * `adjacency only`: every bond gets the mean area, centre-line normal and
//!   midpoint centroid (only the neighbour graph is known).
//!
//! Prints the relative traction error `|t − σ·n| / |σ|` (p50 / p95 / mean)
//! per variant, for an axial and a shear strain.
//! Usage: cargo run --release -p frac-pipeline --example bond_geometry -- out/rc_column.asset.json
use frac_core::*;
use frac_validate::network::{ReferenceSolver, StiffnessModel};
use glam::{DMat3, DVec3};

fn stats(mut e: Vec<f64>) -> (f64, f64, f64) {
    e.retain(|x| x.is_finite());
    e.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = e.len().max(1);
    (
        e[n / 2],
        e[(n * 95 / 100).min(n - 1)],
        e.iter().sum::<f64>() / n as f64,
    )
}

/// Replace the contact geometry of every bond at `level` by what the
/// fragment centres alone give; `keep_area` keeps the exact areas.
fn approximate(asset: &Asset, level: u8, keep_area: bool) -> Asset {
    let mut a = asset.clone();
    let lb: Vec<usize> = (0..a.bonds.len())
        .filter(|&i| a.bonds[i].level == level)
        .collect();
    let mean_area = lb.iter().map(|&i| a.bonds[i].area).sum::<f64>() / lb.len().max(1) as f64;
    for &i in &lb {
        let b = a.bonds[i].clone();
        let pa = a.fragment(b.a).mass.com;
        let (pb, c) = match b.b {
            FragmentOrWorld::Fragment(f) => {
                let pb = a.fragment(f).mass.com;
                (pb, (pa + pb) * 0.5)
            }
            // anchors keep their contact point (the ground plane is known)
            FragmentOrWorld::World => (b.centroid, b.centroid),
        };
        let d = pb - pa;
        if d.length() < 1e-12 {
            continue;
        }
        let n = d.normalize();
        let area = if keep_area { b.area } else { mean_area };
        let u = n.any_orthonormal_vector();
        let v = n.cross(u);
        let bb = &mut a.bonds[i];
        bb.normal = n;
        bb.centroid = c;
        bb.frame_u = u;
        bb.frame_v = v;
        // second moments of a square of the same area
        let s = area * area / 12.0;
        bb.i_uu = s;
        bb.i_vv = s;
        bb.i_uv = 0.0;
        bb.j = 2.0 * s;
        bb.area = area;
        bb.dist_a = (c - pa).length();
        bb.dist_b = match b.b {
            FragmentOrWorld::Fragment(_) => (pb - c).length(),
            FragmentOrWorld::World => 0.0,
        };
    }
    a
}

fn main() {
    let p = std::env::args()
        .nth(1)
        .expect("usage: bond_geometry <X.asset.json>");
    let asset: Asset =
        serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(p).unwrap())).unwrap();
    let lib = frac_material::MaterialLibrary::builtin();
    let solver = ReferenceSolver {
        lib: &lib,
        model: StiffnessModel::Tensorial,
    };
    let lv = asset.hierarchy.levels - 1;
    // driven fragments: those with open surface or a ground anchor in the
    // exact asset (the same set for every variant)
    let r0 = asset.hierarchy.level_ranges[lv as usize].start;
    let n = asset.level_fragments(lv).len();
    let mut open = vec![DVec3::ZERO; n];
    let mut total = vec![0.0f64; n];
    let mut driven = vec![false; n];
    for b in asset.level_bonds(lv) {
        let ia = (b.a.0 - r0) as usize;
        open[ia] += b.normal * b.area;
        total[ia] += b.area;
        match b.b {
            FragmentOrWorld::Fragment(f) => {
                let ib = (f.0 - r0) as usize;
                open[ib] -= b.normal * b.area;
                total[ib] += b.area;
            }
            FragmentOrWorld::World => driven[ia] = true,
        }
    }
    for i in 0..n {
        driven[i] |= open[i].length() > 1e-6 * total[i].max(1e-300);
    }
    let is_driven = |f: &Fragment| driven[(f.id.0 - r0) as usize];
    let variants = [
        ("exact", asset.clone()),
        ("centre-line normals", approximate(&asset, lv, true)),
        ("adjacency only", approximate(&asset, lv, false)),
    ];
    let strains = [
        ("axial", DMat3::from_diagonal(DVec3::new(0.0, 1e-4, 0.0))),
        (
            "shear",
            DMat3::from_cols(
                DVec3::new(0.0, 1e-4, 0.0),
                DVec3::new(1e-4, 0.0, 0.0),
                DVec3::ZERO,
            ),
        ),
    ];
    println!("variant\tstrain\tbonds\tp50\tp95\tmean");
    for (name, a) in &variants {
        for (sn, e) in &strains {
            let errs: Vec<f64> = solver
                .patch_test_with(a, lv, *e, Some(&is_driven))
                .into_iter()
                .map(|x| x.1)
                .collect();
            let (p50, p95, mean) = stats(errs.clone());
            println!(
                "{name}\t{sn}\t{}\t{p50:.4}\t{p95:.4}\t{mean:.4}",
                errs.len()
            );
        }
    }
}
