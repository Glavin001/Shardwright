//! Differential test against Voro++ (spec §13.2): cell volumes and vertex
//! sets must match within 1e-9 relative.
//!
//! The Voro++ output for each case is frozen in `tests/golden/` (a golden
//! oracle dataset), so the test always runs. When the oracle binary is
//! available (`VORO_ORACLE`, else `$ORACLES/voro_oracle`, else
//! /opt/oracles/voro_oracle; `tools/setup.sh` builds it) it is run as well
//! and must reproduce the frozen output; `VORO_WRITE_GOLDEN=1` refreshes it.

use frac_cells::clip::Clipper;
use frac_cells::complex::Complex;
use frac_geom::mesh::box_mesh;
use frac_geom::{DVec3, VolumeIntegrals};
use rand::{Rng, SeedableRng};
use std::io::Write;
use std::process::{Command, Stdio};

fn oracle() -> Option<String> {
    let p = std::env::var("VORO_ORACLE").unwrap_or_else(|_| {
        format!(
            "{}/voro_oracle",
            std::env::var("ORACLES").unwrap_or_else(|_| "/opt/oracles".into())
        )
    });
    if std::path::Path::new(&p).exists() {
        Some(p)
    } else {
        None
    }
}

fn run_oracle(bin: &str, seeds: &[[f64; 3]], lo: DVec3, hi: DVec3) -> String {
    let mut child = Command::new(bin)
        .args([lo.x, hi.x, lo.y, hi.y, lo.z, hi.z].map(|v| format!("{v:.17e}")))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    {
        let mut si = child.stdin.take().unwrap();
        for (i, s) in seeds.iter().enumerate() {
            writeln!(si, "{} {:.17e} {:.17e} {:.17e}", i, s[0], s[1], s[2]).unwrap();
        }
    }
    String::from_utf8(child.wait_with_output().unwrap().stdout).unwrap()
}

/// Voro++ output for a case: frozen golden file, cross-checked against (or
/// refreshed from) the live oracle when it is installed.
fn oracle_output(name: &str, seeds: &[[f64; 3]], lo: DVec3, hi: DVec3) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join(format!("voro_{name}.txt"));
    let live = oracle().map(|bin| run_oracle(&bin, seeds, lo, hi));
    if std::env::var_os("VORO_WRITE_GOLDEN").is_some() {
        let text = live.expect("VORO_WRITE_GOLDEN needs the voro_oracle binary");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &text).unwrap();
        return text;
    }
    let frozen = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing golden {}: {e} (run with VORO_WRITE_GOLDEN=1 and the oracle installed)",
            path.display()
        )
    });
    if let Some(text) = live {
        assert_eq!(
            text,
            frozen,
            "live Voro++ output differs from the frozen golden file {}",
            path.display()
        );
    }
    frozen
}

fn run_case(name: &str, seed: u64, n: usize, lo: DVec3, hi: DVec3) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let seeds: Vec<[f64; 3]> = (0..n)
        .map(|_| {
            [
                rng.gen_range(lo.x..hi.x),
                rng.gen_range(lo.y..hi.y),
                rng.gen_range(lo.z..hi.z),
            ]
        })
        .collect();
    // ours
    let m = box_mesh(lo, hi);
    let cx = Complex::voronoi(&seeds, lo.to_array(), hi.to_array()).unwrap();
    let out = Clipper::new(&m, &cx).run().unwrap();
    let mut polys: Vec<Vec<Vec<u32>>> = vec![Vec::new(); n];
    for e in &out.ext {
        polys[e.cell as usize].push(e.verts.clone());
    }
    for p in &out.patches {
        for t in &p.tris {
            polys[p.cells[0] as usize].push(t.to_vec());
            polys[p.cells[1] as usize].push(vec![t[0], t[2], t[1]]);
        }
    }
    // oracle
    let text = oracle_output(name, &seeds, lo, hi);
    let scale = (hi - lo).length();
    let mut max_vol_err: f64 = 0.0;
    let mut max_vert_err: f64 = 0.0;
    let mut ncells = 0;
    for line in text.lines() {
        let f: Vec<f64> = line
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        let id = f[0] as usize;
        let vol = f[1];
        let nv = f[2] as usize;
        let ref_verts: Vec<DVec3> = (0..nv)
            .map(|k| DVec3::new(f[3 + 3 * k], f[4 + 3 * k], f[5 + 3 * k]))
            .collect();
        // our volume
        let ps = &polys[id];
        let mut vi = VolumeIntegrals::default();
        let r = out.verts[ps[0][0] as usize];
        for p in ps {
            let pts: Vec<DVec3> = p.iter().map(|&i| out.verts[i as usize]).collect();
            vi.add_polygon(r, &pts);
        }
        let rel = (vi.volume - vol).abs() / vol;
        max_vol_err = max_vol_err.max(rel);
        // vertex sets: every reference vertex matched by one of ours
        let mut ours: Vec<DVec3> = ps
            .iter()
            .flatten()
            .map(|&i| out.verts[i as usize])
            .collect();
        ours.sort_by(|a, b| a.to_array().partial_cmp(&b.to_array()).unwrap());
        ours.dedup();
        for rv in &ref_verts {
            let d = ours
                .iter()
                .map(|o| (*o - *rv).length())
                .fold(f64::INFINITY, f64::min);
            max_vert_err = max_vert_err.max(d / scale);
        }
        // and every one of ours lies on a reference vertex or on a box face
        // diagonal (our box faces are triangulated, adding collinear points)
        for ov in &ours {
            let d = ref_verts
                .iter()
                .map(|r| (*r - *ov).length())
                .fold(f64::INFINITY, f64::min);
            if d / scale > 1e-9 {
                let on_face = (0..3).any(|k| {
                    (ov[k] - lo[k]).abs() < 1e-12 * scale || (ov[k] - hi[k]).abs() < 1e-12 * scale
                });
                assert!(on_face, "cell {id}: extra interior vertex {ov:?}");
            }
        }
        ncells += 1;
    }
    assert_eq!(ncells, n);
    eprintln!(
        "voro++ diff: n={n} max volume rel err {max_vol_err:e}, max vertex err {max_vert_err:e}"
    );
    assert!(max_vol_err < 1e-9, "volume error {max_vol_err}");
    assert!(max_vert_err < 1e-9, "vertex error {max_vert_err}");
}

#[test]
fn voro_pp_unit_box() {
    run_case("unit_box", 5, 400, DVec3::ZERO, DVec3::ONE);
}

#[test]
fn voro_pp_offset_box() {
    run_case(
        "offset_box",
        6,
        1500,
        DVec3::new(10.0, -3.0, 100.0),
        DVec3::new(12.5, -1.0, 101.0),
    );
}
