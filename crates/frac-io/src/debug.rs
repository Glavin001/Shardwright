//! JSON debug dump (spec §6.3) and PLY/OBJ geometry dumps.

use crate::IoError;
use frac_core::Asset;
use frac_geom::TriMesh;
use glam::DVec3;
use std::fmt::Write as _;
use std::path::Path;

/// Pretty JSON dump of the full asset. Field order is the struct declaration
/// order; floats use serde_json's shortest round-trip formatting (exact).
pub fn asset_to_json(asset: &Asset) -> String {
    serde_json::to_string_pretty(asset).expect("Asset serialization cannot fail")
}

fn write_file(path: &Path, s: &str) -> Result<(), IoError> {
    if let Some(dir) = path.parent()
        && !dir.as_os_str().is_empty()
    {
        std::fs::create_dir_all(dir).map_err(|e| IoError::io(dir, e))?;
    }
    std::fs::write(path, s).map_err(|e| IoError::io(path, e))
}

/// Write an ASCII PLY (double-precision vertices) of a triangle mesh.
pub fn write_ply(path: impl AsRef<Path>, mesh: &TriMesh) -> Result<(), IoError> {
    let polys: Vec<Vec<u32>> = mesh.tris.iter().map(|t| t.to_vec()).collect();
    write_ply_polys(path, &mesh.verts, &polys, None)
}

/// Write an ASCII PLY of polygons, optionally with a per-polygon RGB color.
pub fn write_ply_polys(
    path: impl AsRef<Path>,
    verts: &[DVec3],
    polys: &[Vec<u32>],
    per_poly_color: Option<&[[u8; 3]]>,
) -> Result<(), IoError> {
    if let Some(c) = per_poly_color
        && c.len() != polys.len()
    {
        return Err(IoError::InvalidScene(format!(
            "{} colors for {} polygons",
            c.len(),
            polys.len()
        )));
    }
    let mut s = String::new();
    s.push_str("ply\nformat ascii 1.0\ncomment Shardwright frac-io debug dump\n");
    let _ = writeln!(s, "element vertex {}", verts.len());
    s.push_str("property double x\nproperty double y\nproperty double z\n");
    let _ = writeln!(s, "element face {}", polys.len());
    s.push_str("property list uchar uint vertex_indices\n");
    if per_poly_color.is_some() {
        s.push_str("property uchar red\nproperty uchar green\nproperty uchar blue\n");
    }
    s.push_str("end_header\n");
    for v in verts {
        let _ = writeln!(s, "{} {} {}", v.x, v.y, v.z);
    }
    for (i, p) in polys.iter().enumerate() {
        if p.len() > 255 {
            return Err(IoError::InvalidScene(format!(
                "polygon {i} has {} vertices (max 255)",
                p.len()
            )));
        }
        let _ = write!(s, "{}", p.len());
        for idx in p {
            let _ = write!(s, " {idx}");
        }
        if let Some(c) = per_poly_color {
            let _ = write!(s, " {} {} {}", c[i][0], c[i][1], c[i][2]);
        }
        s.push('\n');
    }
    write_file(path.as_ref(), &s)
}

/// Write a Wavefront OBJ of a triangle mesh (1-based indices).
pub fn write_obj(path: impl AsRef<Path>, mesh: &TriMesh) -> Result<(), IoError> {
    let mut s = String::from("# Shardwright frac-io debug dump\n");
    for v in &mesh.verts {
        let _ = writeln!(s, "v {} {} {}", v.x, v.y, v.z);
    }
    for t in &mesh.tris {
        let _ = writeln!(s, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1);
    }
    write_file(path.as_ref(), &s)
}
