//! Bakes a small non-convex asset (a U-shaped block) and checks the gates
//! and the collision hulls: every level passes `no_overlap_hulls`, the
//! whole-asset fragment needs several hulls, and the hulls stay within the
//! budget and close to the fragment volume.

use frac_core::input::{AuthoringMeta, InputPart, InputScene, PartMeta};
use frac_core::Settings;
use frac_geom::mesh::box_mesh;
use frac_geom::{DVec3, TriMesh};
use frac_material::MaterialLibrary;
use frac_pipeline::{run, InputSpec};

/// Closed mesh of a union of axis-aligned boxes (shared grid).
fn union_boxes(boxes: &[(DVec3, DVec3)]) -> TriMesh {
    let mut axes: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (lo, hi) in boxes {
        for k in 0..3 {
            axes[k].extend([lo[k], hi[k]]);
        }
    }
    for a in axes.iter_mut() {
        a.sort_by(f64::total_cmp);
        a.dedup();
    }
    let n = [axes[0].len() - 1, axes[1].len() - 1, axes[2].len() - 1];
    let filled = |i: isize, j: isize, k: isize| -> bool {
        if i < 0 || j < 0 || k < 0 || i >= n[0] as isize || j >= n[1] as isize || k >= n[2] as isize {
            return false;
        }
        let c = DVec3::new(
            (axes[0][i as usize] + axes[0][i as usize + 1]) * 0.5,
            (axes[1][j as usize] + axes[1][j as usize + 1]) * 0.5,
            (axes[2][k as usize] + axes[2][k as usize + 1]) * 0.5,
        );
        boxes.iter().any(|(lo, hi)| c.cmpgt(*lo).all() && c.cmplt(*hi).all())
    };
    let mut m = TriMesh::default();
    for i in 0..n[0] as isize {
        for j in 0..n[1] as isize {
            for k in 0..n[2] as isize {
                if !filled(i, j, k) {
                    continue;
                }
                let lo = DVec3::new(axes[0][i as usize], axes[1][j as usize], axes[2][k as usize]);
                let hi = DVec3::new(axes[0][i as usize + 1], axes[1][j as usize + 1], axes[2][k as usize + 1]);
                let b = box_mesh(lo, hi);
                for t in &b.tris {
                    let [p, q, r] = [b.verts[t[0] as usize], b.verts[t[1] as usize], b.verts[t[2] as usize]];
                    let nn = (q - p).cross(r - p).normalize();
                    if !filled(i + nn.x.round() as isize, j + nn.y.round() as isize, k + nn.z.round() as isize) {
                        let base = m.verts.len() as u32;
                        m.verts.extend([p, q, r]);
                        m.tris.push([base, base + 1, base + 2]);
                    }
                }
            }
        }
    }
    m.weld_exact()
}

#[test]
fn nonconvex_u_block_hulls() {
    let mesh = union_boxes(&[
        (DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.2, 0.4, 0.4)),
        (DVec3::new(0.0, 0.0, 0.0), DVec3::new(0.4, 1.2, 0.4)),
        (DVec3::new(0.8, 0.0, 0.0), DVec3::new(1.2, 1.2, 0.4)),
    ]);
    assert!(mesh.topology().is_closed_manifold());
    let n = mesh.tris.len();
    let part = InputPart { name: "u".into(), mesh, material_slot: vec![0; n], material_names: vec!["concrete_c30".into()], extras: serde_json::Value::Null, ..Default::default() };
    let pm = PartMeta { material: Some("concrete_c30".into()), ..Default::default() };
    let mut meta = AuthoringMeta::default();
    meta.parts.insert("u".into(), pm);
    let input = InputSpec { name: "u".into(), scene: InputScene { source: "test".into(), parts: vec![part] }, meta, variant: 0 };
    let mut settings = Settings::default();
    settings.modes.enabled = false;
    settings.cells.target_analysis_cells_per_m3 = 60.0;
    settings.cells.fine_per_analysis = 4;
    let lib = MaterialLibrary::builtin();
    let out = run(&input, &settings, &lib).expect("pipeline");
    let sc = &out.report.scorecard;
    assert!(sc.all_pass(), "gate failures: {:?}", sc.failures());
    let g = sc.gates.iter().find(|g| g.name == "no_overlap_hulls").unwrap();
    assert_eq!(g.status, frac_validate::GateStatus::Pass);
    let a = &out.asset;
    let root = &a.hierarchy.fragments[a.hierarchy.level_ranges[0].start as usize];
    let nh = root.hulls.len();
    assert!(nh >= 2 && nh <= settings.collision.max_hulls_per_fragment as usize, "root fragment hulls: {nh}");
    // hulls of the root cover most of the U and do not fill its notch
    let hv: f64 = a.hulls[root.hulls.start as usize..root.hulls.end as usize].iter().map(|h| frac_pipeline::hull_polytope(h).volume()).sum();
    let v = root.mass.volume;
    assert!(hv > 0.85 * v && hv < 1.25 * v, "hull volume {hv} vs fragment {v}");
    for f in &a.hierarchy.fragments {
        assert!(f.hulls.len() <= settings.collision.max_hulls_per_fragment as usize);
    }
}
