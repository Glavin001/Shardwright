use frac_core::Settings;
use frac_material::MaterialLibrary;
use frac_pipeline::{InputSpec, run};

#[test]
#[ignore]
fn dbg_wall_clean_validity() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/assets/brick_wall_window.glb");
    let scene = frac_io::load_scene(
        &path,
        &frac_io::ImportOptions {
            unit_scale: 1.0,
            z_up: false,
        },
    )
    .unwrap();
    let meta = frac_core::input::AuthoringMeta::from_json(
        &std::fs::read_to_string(path.with_extension("glb.meta.json")).unwrap(),
    )
    .unwrap();
    let mut s = Settings::default();
    s.modes.enabled = false;
    s.render.noise = false;
    s.render.chipping = false;
    let input = InputSpec {
        name: "wall".into(),
        scene,
        meta,
        variant: 0,
    };
    let out = run(&input, &s, &MaterialLibrary::builtin()).unwrap();
    let a = &out.asset;
    let mut shown = 0;
    for f in a.level_fragments(3) {
        let m = frac_collision::cells_boundary_mesh(a, a.fragment_cells(f));
        let si = m.self_intersections(4);
        let si: Vec<_> = si
            .into_iter()
            .filter(|&(x, y)| !m.is_degenerate(x as usize) && !m.is_degenerate(y as usize))
            .collect();
        if si.is_empty() {
            continue;
        }
        shown += 1;
        if shown > 3 {
            continue;
        }
        let (x, y) = si[0];
        eprintln!("cell frag {} vol {:e}", f.id.0, f.mass.volume);
        for t in [x, y] {
            let p = m.tri_points(t as usize);
            eprintln!("  tri {t} {:?} {:?}", m.tris[t as usize], p);
        }
    }
    eprintln!("bad leaf cells: {shown} of {}", a.level_fragments(3).len());
}
