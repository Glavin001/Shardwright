//! Bond-network diagnostics on a baked asset JSON: kinematic-center fit
//! quality and patch tests (uniform, uniaxial with free lateral surface,
//! pure bending). The end/lateral geometry assumes the `rc_column` benchmark
//! (0.4 × 3 × 0.4 m, axis y).
//! Usage: cargo run --release -p frac-pipeline --example network_diag -- out/rc_column.asset.json
use frac_core::*;
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let asset: Asset = serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(p).unwrap())).unwrap();
    let lib = frac_material::MaterialLibrary::builtin();
    for model in [frac_validate::network::StiffnessModel::Tensorial, frac_validate::network::StiffnessModel::Calibrated] {
        let solver = frac_validate::network::ReferenceSolver { lib: &lib, model };
        for (name, e) in [("axial", glam::DMat3::from_diagonal(glam::DVec3::new(0.0, 1e-4, 0.0))), ("shear", glam::DMat3::from_cols(glam::DVec3::new(0.0, 1e-4, 0.0), glam::DVec3::new(1e-4, 0.0, 0.0), glam::DVec3::ZERO))] {
            let lv = asset.hierarchy.levels - 1;
            let mut errs = solver.patch_test(&asset, lv, e);
            errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if errs.is_empty() { continue; }
            println!("patch test {model:?} {name} L{lv}: {} interior bonds, err p50 {:.4} p95 {:.4} max {:.4}", errs.len(), errs[errs.len() / 2], errs[errs.len() * 95 / 100], errs[errs.len() - 1]);
        }
    }
    {
        let solver = frac_validate::network::ReferenceSolver { lib: &lib, model: frac_validate::network::StiffnessModel::Tensorial };
        let nu = 0.2;
        let e = glam::DMat3::from_diagonal(glam::DVec3::new(-nu, 1.0, -nu) * 1e-4);
        let lv = asset.hierarchy.levels - 1;
        let ends = |f: &Fragment| f.mass.com.y < 0.12 || f.mass.com.y > 2.88;
        let mut r = solver.patch_test_with(&asset, lv, e, Some(&ends));
        r.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let errs: Vec<f64> = r.iter().map(|x| x.1).collect();
        println!("uniaxial (ends driven, lateral free) L{lv}: {} bonds, err p50 {:.4} p95 {:.4} max {:.4}", errs.len(), errs[errs.len() / 2], errs[errs.len() * 95 / 100], errs[errs.len() - 1]);
        // by distance to the lateral surface
        for (lo, hi) in [(0.0, 0.03), (0.03, 0.08), (0.08, 1.0)] {
            let mut s: Vec<f64> = r.iter().filter(|x| { let d = (0.2 - x.0.x.abs()).min(0.2 - x.0.z.abs()); d >= lo && d < hi }).map(|x| x.1).collect();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if !s.is_empty() { println!("   lateral dist [{lo},{hi}): n={} p50 {:.4} p95 {:.4}", s.len(), s[s.len() / 2], s[s.len() * 95 / 100]); }
        }
    }
    {
        // pure bending (axis y, curvature κ about z): σ_yy = E κ x
        let solver = frac_validate::network::ReferenceSolver { lib: &lib, model: frac_validate::network::StiffnessModel::Tensorial };
        let el = lib.elastic(asset.components[0].material);
        let (e_mod, nu, kap) = (el.youngs, el.poisson, 1e-4);
        let lv = asset.hierarchy.levels - 1;
        let ends = |f: &Fragment| f.mass.com.y < 0.12 || f.mass.com.y > 2.88;
        let u = |p: glam::DVec3| glam::DVec3::new(-0.5 * kap * (p.y * p.y + nu * (p.x * p.x - p.z * p.z)), kap * p.x * p.y, -nu * kap * p.x * p.z);
        let w = |p: glam::DVec3| glam::DVec3::new(0.0, nu * kap * p.z, kap * p.y);
        let sg = |p: glam::DVec3| glam::DMat3::from_diagonal(glam::DVec3::new(0.0, e_mod * kap * p.x, 0.0));
        let r = solver.patch_test_field(&asset, lv, &u, &w, &sg, Some(&ends));
        // normalise by the peak stress instead of the local one
        let smax = e_mod * kap * 0.2;
        let mut abs: Vec<f64> = r.iter().map(|x| x.1 * (e_mod * kap * x.0.x.abs()) / smax).collect();
        abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (lo, hi) in [(0.0, 0.03), (0.03, 0.08), (0.08, 1.0)] {
            let mut v: Vec<f64> = r.iter().filter(|x| { let d = (0.2 - x.0.x.abs()).min(0.2 - x.0.z.abs()); d >= lo && d < hi && x.0.y > 0.4 && x.0.y < 2.6 }).map(|x| x.1 * (e_mod * kap * x.0.x.abs()) / smax).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if !v.is_empty() { println!("   mid, lateral dist [{lo},{hi}): n={} p50 {:.4} p95 {:.4}", v.len(), v[v.len() / 2], v[v.len() * 95 / 100]); }
        }
        let mut v: Vec<f64> = r.iter().filter(|x| x.0.y <= 0.4 || x.0.y >= 2.6).map(|x| x.1 * (e_mod * kap * x.0.x.abs()) / smax).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("   near driven ends: n={} p50 {:.4} p95 {:.4}", v.len(), v[v.len() / 2], v[v.len() * 95 / 100]);
        println!("pure bending (ends driven) L{lv}: {} bonds, |t_net - σn|/σ_max p50 {:.4} p95 {:.4} max {:.4}", abs.len(), abs[abs.len() / 2], abs[abs.len() * 95 / 100], abs[abs.len() - 1]);
    }
    for level in 1..asset.hierarchy.levels {
        let c = frac_validate::network::kinematic_centers(&asset, level);
        let fr = asset.level_fragments(level);
        let mut d: Vec<f64> = fr.iter().zip(&c).map(|(f, p)| (*p - f.mass.com).length() / f.mass.volume.cbrt()).collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let r0 = asset.hierarchy.level_ranges[level as usize].start;
        let mut hr = Vec::new();
        let mut tang = Vec::new();
        for b in asset.level_bonds(level) {
            if let FragmentOrWorld::Fragment(f) = b.b {
                let pa = c[(b.a.0 - r0) as usize];
                let pb = c[(f.0 - r0) as usize];
                let h = (pb - pa).dot(b.normal);
                hr.push(h / (b.dist_a + b.dist_b));
                let xa = asset.fragment(b.a).mass.com;
                let xb = asset.fragment(f).mass.com;
                let t0 = (xb - xa - b.normal * (xb - xa).dot(b.normal)).length() / (xb - xa).length();
                let t1 = (pb - pa - b.normal * (pb - pa).dot(b.normal)).length() / (pb - pa).length();
                tang.push((t0, t1));
            }
        }
        hr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |v: &Vec<f64>, x: f64| v[((v.len() - 1) as f64 * x) as usize];
        let mut t0: Vec<f64> = tang.iter().map(|t| t.0).collect();
        let mut t1: Vec<f64> = tang.iter().map(|t| t.1).collect();
        t0.sort_by(|a, b| a.partial_cmp(b).unwrap());
        t1.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("L{level}: n={} |p-com|/size p50 {:.3} p95 {:.3} max {:.3}; h/h0 p5 {:.3} p50 {:.3} p95 {:.3}; tangential com p50 {:.3} p95 {:.3} -> centers p50 {:.3} p95 {:.3}",
            fr.len(), q(&d, 0.5), q(&d, 0.95), q(&d, 1.0), q(&hr, 0.05), q(&hr, 0.5), q(&hr, 0.95), q(&t0, 0.5), q(&t0, 0.95), q(&t1, 0.5), q(&t1, 0.95));
    }
}

