//! Analysis tetrahedralization.
//!
//! [`tetrahedralize`] is a pure-Rust mesher for closed manifold solids:
//! a body-centred-cubic (BCC) lattice — whose tets have edge lengths in
//! `[0.866h, h]` and good dihedral angles — is laid over the solid's
//! bounding box; lattice vertices close to the surface (unsigned distance
//! `< 0.3h`) are snapped to their closest surface point (an
//! "isosurface stuffing"-lite in the spirit of Labelle & Shewchuk 2007, but
//! without cutting stencils); tets whose (snapped) centroid lies inside the
//! solid are kept; remaining outside vertices of kept tets are snapped to the
//! surface when that keeps all incident tets valid; inverted and sliver tets
//! (volume `< 0.01 h³`, a regular BCC tet has `h³/12`) are dropped; tiny
//! disconnected pieces (volume `< 1e-3` of the total) are removed; vertices
//! are compacted in lattice order. The result is deterministic.
//!
//! [`tetrahedralize_external`] runs an fTetWild binary as a subprocess and
//! parses its Gmsh `.msh` output (2.x ASCII/binary, 4.x ASCII).

use crate::mesh::{TetMesh, tet_signed_volume};
use frac_geom::inside::MeshQuery;
use frac_geom::{DVec3, TriMesh};
use rayon::prelude::*;

/// Fraction of `h` within which lattice vertices snap to the surface.
const SNAP_ALPHA: f64 = 0.3;
/// Tets whose centroid is outside but within this fraction of `h` from the
/// surface are kept (their outside vertices get snapped).
const KEEP_BETA: f64 = 0.15;
/// Minimum kept tet volume relative to `h³`.
const MIN_VOL_REL: f64 = 0.01;

/// Tetrahedralizes a closed manifold triangle mesh with target edge length
/// `target_edge`, coarsening (growing the lattice spacing) until the tet
/// count is at most `max_tets` (`max_tets == 0` means unlimited).
pub fn tetrahedralize(solid: &TriMesh, target_edge: f64, max_tets: usize) -> TetMesh {
    if solid.tris.is_empty() || !(target_edge > 0.0) {
        return TetMesh::default();
    }
    let vol = solid.signed_volume().abs();
    let mut h = target_edge;
    if max_tets > 0 && vol > 0.0 {
        // a BCC lattice has 12 tets per h^3
        let h_min = libm::cbrt(12.0 * vol / max_tets as f64);
        if h_min > h {
            h = h_min * 1.02;
        }
    }
    let query = MeshQuery::new(solid);
    let mut best = TetMesh::default();
    for _ in 0..12 {
        let m = bcc_mesh(solid, &query, h);
        let count = m.tets.len();
        best = m;
        if max_tets == 0 || count <= max_tets {
            break;
        }
        h *= libm::cbrt(count as f64 / max_tets as f64) * 1.02;
    }
    best
}

fn bcc_mesh(solid: &TriMesh, query: &MeshQuery, h: f64) -> TetMesh {
    let bb = solid.aabb();
    let ext = bb.extent();
    let ctr = bb.center();
    let nx = (ext.x / h).ceil() as usize + 2;
    let ny = (ext.y / h).ceil() as usize + 2;
    let nz = (ext.z / h).ceil() as usize + 2;
    let origin = ctr - DVec3::new(nx as f64, ny as f64, nz as f64) * (0.5 * h);
    let corner_id = |i: usize, j: usize, k: usize| i + (nx + 1) * (j + (ny + 1) * k);
    let n_corner = (nx + 1) * (ny + 1) * (nz + 1);
    let center_id = |i: usize, j: usize, k: usize| n_corner + i + nx * (j + ny * k);
    let n_nodes = n_corner + nx * ny * nz;
    let mut pos = vec![DVec3::ZERO; n_nodes];
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                pos[corner_id(i, j, k)] = origin + DVec3::new(i as f64, j as f64, k as f64) * h;
            }
        }
    }
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                pos[center_id(i, j, k)] =
                    origin + DVec3::new(i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5) * h;
            }
        }
    }
    // tets: for each pair of face-adjacent cubes
    let mut tets: Vec<[u32; 4]> = Vec::new();
    let dims = [nx, ny, nz];
    for axis in 0..3 {
        let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = [i, j, k];
                    if c[axis] + 1 >= dims[axis] {
                        continue;
                    }
                    let mut c2 = c;
                    c2[axis] += 1;
                    let a = center_id(c[0], c[1], c[2]) as u32;
                    let b = center_id(c2[0], c2[1], c2[2]) as u32;
                    // shared face corners in cyclic order
                    let mut q = [[0usize; 3]; 4];
                    for (m, (du, dw)) in [(0, 0), (1, 0), (1, 1), (0, 1)].iter().enumerate() {
                        let mut g = c;
                        g[axis] += 1;
                        g[u] += du;
                        g[w] += dw;
                        q[m] = g;
                    }
                    for m in 0..4 {
                        let p = q[m];
                        let r = q[(m + 1) % 4];
                        let mut t = [
                            a,
                            b,
                            corner_id(p[0], p[1], p[2]) as u32,
                            corner_id(r[0], r[1], r[2]) as u32,
                        ];
                        let pts = [
                            pos[t[0] as usize],
                            pos[t[1] as usize],
                            pos[t[2] as usize],
                            pos[t[3] as usize],
                        ];
                        if tet_signed_volume(&pts.map(|v| v.to_array())) < 0.0 {
                            t.swap(2, 3);
                        }
                        tets.push(t);
                    }
                }
            }
        }
    }
    // per-node closest point and inside flag
    let info: Vec<(DVec3, f64, bool)> = pos
        .par_iter()
        .map(|&p| {
            let (cp, d2) = query
                .closest_point(p)
                .map(|c| (c.0, c.1))
                .unwrap_or((p, f64::INFINITY));
            let inside = query.contains(p);
            (cp, d2.sqrt(), inside)
        })
        .collect();
    let mut snapped = vec![false; n_nodes];
    let mut cur = pos.clone();
    for i in 0..n_nodes {
        if info[i].1 < SNAP_ALPHA * h {
            cur[i] = info[i].0;
            snapped[i] = true;
        }
    }
    // keep tets with centroid inside
    let keep: Vec<bool> = tets
        .par_iter()
        .map(|t| {
            let far_in = t
                .iter()
                .all(|&v| info[v as usize].2 && info[v as usize].1 > h);
            if far_in {
                return true;
            }
            let far_out = t
                .iter()
                .all(|&v| !info[v as usize].2 && info[v as usize].1 > h);
            if far_out {
                return false;
            }
            let c =
                (cur[t[0] as usize] + cur[t[1] as usize] + cur[t[2] as usize] + cur[t[3] as usize])
                    * 0.25;
            if query.contains(c) {
                return true;
            }
            // tets whose centroid lies on (or just outside) the surface are kept too;
            // their outside vertices are snapped onto the surface below
            query
                .closest_point(c)
                .map(|q| q.1.sqrt() <= KEEP_BETA * h)
                .unwrap_or(false)
        })
        .collect();
    let mut kept: Vec<[u32; 4]> = tets
        .iter()
        .zip(&keep)
        .filter(|(_, k)| **k)
        .map(|(t, _)| *t)
        .collect();
    let vmin = MIN_VOL_REL * h * h * h;
    let vol_of = |cur: &[DVec3], t: &[u32; 4]| {
        tet_signed_volume(&[
            cur[t[0] as usize].to_array(),
            cur[t[1] as usize].to_array(),
            cur[t[2] as usize].to_array(),
            cur[t[3] as usize].to_array(),
        ])
    };
    // snap remaining outside vertices of kept tets when valid
    {
        let mut inc: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];
        for (ti, t) in kept.iter().enumerate() {
            for &v in t {
                inc[v as usize].push(ti as u32);
            }
        }
        for v in 0..n_nodes {
            if inc[v].is_empty() || snapped[v] || info[v].2 {
                continue;
            }
            let old = cur[v];
            cur[v] = info[v].0;
            let ok = inc[v]
                .iter()
                .all(|&ti| vol_of(&cur, &kept[ti as usize]) >= vmin);
            if ok {
                snapped[v] = true;
            } else {
                cur[v] = old;
            }
        }
    }
    kept.retain(|t| vol_of(&cur, t) >= vmin);
    let mut mesh = TetMesh {
        verts: cur.iter().map(|p| p.to_array()).collect(),
        tets: kept,
    };
    mesh.remove_unreferenced();
    remove_small_components(&mut mesh, 1e-3);
    mesh
}

/// Removes face-connected components with volume below `rel` of the total.
pub fn remove_small_components(mesh: &mut TetMesh, rel: f64) {
    if mesh.tets.is_empty() {
        return;
    }
    let (nc, ids) = mesh.tet_components();
    if nc <= 1 {
        return;
    }
    let mut cv = vec![0.0; nc];
    for t in 0..mesh.tets.len() {
        cv[ids[t] as usize] += mesh.tet_volume(t);
    }
    let total: f64 = cv.iter().sum();
    let tets: Vec<[u32; 4]> = mesh
        .tets
        .iter()
        .enumerate()
        .filter(|(t, _)| cv[ids[*t] as usize] >= rel * total)
        .map(|(_, t)| *t)
        .collect();
    mesh.tets = tets;
    mesh.remove_unreferenced();
}

// ---------------------------------------------------------------------------
// External fTetWild
// ---------------------------------------------------------------------------

/// Runs an fTetWild binary (`bin -i in.obj -o out.msh -l <rel_edge>`) on
/// `solid` with absolute target edge `edge` (converted to fTetWild's relative
/// edge length w.r.t. the bounding-box diagonal), and parses the resulting
/// `.msh` (Gmsh 2.x ASCII or binary, 4.x ASCII). Tets are re-oriented to be
/// positive and unreferenced vertices removed.
pub fn tetrahedralize_external(bin: &str, solid: &TriMesh, edge: f64) -> Result<TetMesh, String> {
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    if solid.tris.is_empty() {
        return Err("empty input mesh".into());
    }
    let diag = solid.aabb().diagonal();
    if !(edge > 0.0) || !(diag > 0.0) {
        return Err("edge length and bounding box must be positive".into());
    }
    let rel = edge / diag;
    let dir = std::env::temp_dir().join(format!(
        "frac-fem-ftetwild-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create temp dir: {e}"))?;
    let input = dir.join("input.obj");
    let output = dir.join("output.msh");
    let result = (|| {
        {
            let f =
                std::fs::File::create(&input).map_err(|e| format!("cannot write input: {e}"))?;
            let mut w = std::io::BufWriter::new(f);
            for v in &solid.verts {
                writeln!(w, "v {:.17e} {:.17e} {:.17e}", v.x, v.y, v.z)
                    .map_err(|e| e.to_string())?;
            }
            for t in &solid.tris {
                writeln!(w, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1)
                    .map_err(|e| e.to_string())?;
            }
            w.flush().map_err(|e| e.to_string())?;
        }
        let out = std::process::Command::new(bin)
            .arg("-i")
            .arg(&input)
            .arg("-o")
            .arg(&output)
            .arg("-l")
            .arg(format!("{rel}"))
            .output()
            .map_err(|e| format!("failed to run {bin}: {e}"))?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(format!(
                "{bin} exited with {}: {}",
                out.status,
                stderr.trim()
            ));
        }
        let bytes =
            std::fs::read(&output).map_err(|e| format!("cannot read {}: {e}", output.display()))?;
        let mut mesh = parse_msh(&bytes)?;
        mesh.fix_orientation();
        mesh.tets.retain(|t| {
            let p = [
                mesh.verts[t[0] as usize],
                mesh.verts[t[1] as usize],
                mesh.verts[t[2] as usize],
                mesh.verts[t[3] as usize],
            ];
            tet_signed_volume(&p) > 0.0
        });
        mesh.remove_unreferenced();
        Ok(mesh)
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cursor<'a> {
    fn line(&mut self) -> Option<&'a str> {
        if self.p >= self.b.len() {
            return None;
        }
        let s = self.p;
        while self.p < self.b.len() && self.b[self.p] != b'\n' {
            self.p += 1;
        }
        let e = self.p;
        if self.p < self.b.len() {
            self.p += 1;
        }
        std::str::from_utf8(&self.b[s..e]).ok().map(|x| x.trim())
    }
    fn nonempty_line(&mut self) -> Option<&'a str> {
        loop {
            let l = self.line()?;
            if !l.is_empty() {
                return Some(l);
            }
        }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.p + n > self.b.len() {
            return Err("unexpected end of binary msh data".into());
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
}

fn nums<T: std::str::FromStr>(l: &str) -> Result<Vec<T>, String> {
    l.split_whitespace()
        .map(|t| {
            t.parse::<T>()
                .map_err(|_| format!("bad number '{t}' in msh"))
        })
        .collect()
}

/// Parses a Gmsh `.msh` file (2.x ASCII/binary, 4.0/4.1 ASCII), returning
/// its vertices and 4-node tetrahedra (type 4). Orientation is not fixed.
pub fn parse_msh(bytes: &[u8]) -> Result<TetMesh, String> {
    let mut c = Cursor { b: bytes, p: 0 };
    let mut version = 2.2f64;
    let mut binary = false;
    let mut swap = false;
    let mut node_tags: Vec<u64> = Vec::new();
    let mut node_pos: Vec<[f64; 3]> = Vec::new();
    let mut tets_tagged: Vec<[u64; 4]> = Vec::new();
    while let Some(l) = c.nonempty_line() {
        match l {
            "$MeshFormat" => {
                let f = c.nonempty_line().ok_or("truncated $MeshFormat")?;
                let parts: Vec<&str> = f.split_whitespace().collect();
                if parts.len() < 3 {
                    return Err("bad $MeshFormat".into());
                }
                version = parts[0].parse().map_err(|_| "bad msh version")?;
                binary = parts[1] == "1";
                let dsize: usize = parts[2].parse().map_err(|_| "bad data size")?;
                if binary {
                    if version >= 4.0 {
                        return Err("binary MSH 4 is not supported; write ASCII".into());
                    }
                    if dsize != 8 {
                        return Err("only 8-byte doubles are supported".into());
                    }
                    let one = c.take(4)?;
                    let v = i32::from_le_bytes([one[0], one[1], one[2], one[3]]);
                    swap = v != 1;
                    // rest of the line
                    c.line();
                }
                skip_to(&mut c, "$EndMeshFormat")?;
            }
            "$Nodes" => {
                if version < 4.0 {
                    let n: usize = c
                        .nonempty_line()
                        .ok_or("truncated $Nodes")?
                        .parse()
                        .map_err(|_| "bad node count")?;
                    node_tags.reserve(n);
                    node_pos.reserve(n);
                    if binary {
                        for _ in 0..n {
                            let id = read_i32(&mut c, swap)? as u64;
                            let mut p = [0.0; 3];
                            for v in &mut p {
                                *v = read_f64(&mut c, swap)?;
                            }
                            node_tags.push(id);
                            node_pos.push(p);
                        }
                    } else {
                        for _ in 0..n {
                            let v: Vec<f64> = nums(c.nonempty_line().ok_or("truncated nodes")?)?;
                            if v.len() < 4 {
                                return Err("bad node line".into());
                            }
                            node_tags.push(v[0] as u64);
                            node_pos.push([v[1], v[2], v[3]]);
                        }
                    }
                } else {
                    let h: Vec<u64> = nums(c.nonempty_line().ok_or("truncated $Nodes")?)?;
                    let nblocks = *h.first().ok_or("bad $Nodes header")? as usize;
                    for _ in 0..nblocks {
                        let bh: Vec<i64> = nums(c.nonempty_line().ok_or("truncated node block")?)?;
                        if bh.len() < 4 {
                            return Err("bad node block header".into());
                        }
                        let parametric = bh[2] != 0;
                        let nn = bh[3] as usize;
                        if version >= 4.1 {
                            let mut tags = Vec::with_capacity(nn);
                            for _ in 0..nn {
                                let t: Vec<u64> =
                                    nums(c.nonempty_line().ok_or("truncated node tags")?)?;
                                tags.push(*t.first().ok_or("bad node tag")?);
                            }
                            for t in tags {
                                let v: Vec<f64> =
                                    nums(c.nonempty_line().ok_or("truncated node coords")?)?;
                                if v.len() < 3 {
                                    return Err("bad node coords".into());
                                }
                                node_tags.push(t);
                                node_pos.push([v[0], v[1], v[2]]);
                            }
                        } else {
                            for _ in 0..nn {
                                let v: Vec<f64> =
                                    nums(c.nonempty_line().ok_or("truncated nodes")?)?;
                                if v.len() < 4 {
                                    return Err("bad node line".into());
                                }
                                node_tags.push(v[0] as u64);
                                node_pos.push([v[1], v[2], v[3]]);
                            }
                        }
                        let _ = parametric;
                    }
                }
                skip_to(&mut c, "$EndNodes")?;
            }
            "$Elements" => {
                if version < 4.0 {
                    let n: usize = c
                        .nonempty_line()
                        .ok_or("truncated $Elements")?
                        .parse()
                        .map_err(|_| "bad element count")?;
                    if binary {
                        let mut read = 0;
                        while read < n {
                            let ty = read_i32(&mut c, swap)?;
                            let cnt = read_i32(&mut c, swap)? as usize;
                            let ntags = read_i32(&mut c, swap)? as usize;
                            let nn = nodes_per_element(ty)
                                .ok_or(format!("unknown msh element type {ty}"))?;
                            for _ in 0..cnt {
                                let _id = read_i32(&mut c, swap)?;
                                for _ in 0..ntags {
                                    read_i32(&mut c, swap)?;
                                }
                                let mut vs = [0u64; 4];
                                for k in 0..nn {
                                    let v = read_i32(&mut c, swap)? as u64;
                                    if k < 4 {
                                        vs[k] = v;
                                    }
                                }
                                if ty == 4 {
                                    tets_tagged.push(vs);
                                }
                            }
                            read += cnt;
                        }
                    } else {
                        for _ in 0..n {
                            let v: Vec<u64> = nums(c.nonempty_line().ok_or("truncated elements")?)?;
                            if v.len() < 3 {
                                return Err("bad element line".into());
                            }
                            let ty = v[1];
                            let ntags = v[2] as usize;
                            if ty == 4 {
                                let s = 3 + ntags;
                                if v.len() < s + 4 {
                                    return Err("bad tet element line".into());
                                }
                                tets_tagged.push([v[s], v[s + 1], v[s + 2], v[s + 3]]);
                            }
                        }
                    }
                } else {
                    let h: Vec<u64> = nums(c.nonempty_line().ok_or("truncated $Elements")?)?;
                    let nblocks = *h.first().ok_or("bad $Elements header")? as usize;
                    for _ in 0..nblocks {
                        let bh: Vec<i64> =
                            nums(c.nonempty_line().ok_or("truncated element block")?)?;
                        if bh.len() < 4 {
                            return Err("bad element block header".into());
                        }
                        let ty = bh[2];
                        let ne = bh[3] as usize;
                        for _ in 0..ne {
                            let v: Vec<u64> = nums(c.nonempty_line().ok_or("truncated elements")?)?;
                            if ty == 4 {
                                if v.len() < 5 {
                                    return Err("bad tet element line".into());
                                }
                                tets_tagged.push([v[1], v[2], v[3], v[4]]);
                            }
                        }
                    }
                }
                skip_to(&mut c, "$EndElements")?;
            }
            s if s.starts_with('$') && !s.starts_with("$End") => {
                let end = format!("$End{}", &s[1..]);
                if binary {
                    skip_to_binary(&mut c, &end)?;
                } else {
                    skip_to(&mut c, &end)?;
                }
            }
            _ => {}
        }
    }
    if node_pos.is_empty() || tets_tagged.is_empty() {
        return Err("msh file contains no nodes or no tetrahedra".into());
    }
    // map tags -> indices (sorted lookup; deterministic)
    let mut order: Vec<(u64, usize)> = node_tags.iter().enumerate().map(|(i, &t)| (t, i)).collect();
    order.sort_unstable();
    let lookup = |t: u64| -> Result<u32, String> {
        order
            .binary_search_by(|e| e.0.cmp(&t))
            .map(|k| order[k].1 as u32)
            .map_err(|_| format!("element references unknown node {t}"))
    };
    let mut tets = Vec::with_capacity(tets_tagged.len());
    for t in &tets_tagged {
        tets.push([lookup(t[0])?, lookup(t[1])?, lookup(t[2])?, lookup(t[3])?]);
    }
    Ok(TetMesh {
        verts: node_pos,
        tets,
    })
}

fn nodes_per_element(ty: i32) -> Option<usize> {
    Some(match ty {
        1 => 2,
        2 => 3,
        3 => 4,
        4 => 4,
        5 => 8,
        6 => 6,
        7 => 5,
        8 => 3,
        9 => 6,
        10 => 9,
        11 => 10,
        15 => 1,
        _ => return None,
    })
}

fn skip_to(c: &mut Cursor, end: &str) -> Result<(), String> {
    while let Some(l) = c.line() {
        if l == end {
            return Ok(());
        }
    }
    Err(format!("missing {end}"))
}

fn skip_to_binary(c: &mut Cursor, end: &str) -> Result<(), String> {
    let pat = end.as_bytes();
    while c.p + pat.len() <= c.b.len() {
        if &c.b[c.p..c.p + pat.len()] == pat {
            c.p += pat.len();
            c.line();
            return Ok(());
        }
        c.p += 1;
    }
    Err(format!("missing {end}"))
}

fn read_i32(c: &mut Cursor, swap: bool) -> Result<i32, String> {
    let b = c.take(4)?;
    let a = [b[0], b[1], b[2], b[3]];
    Ok(if swap {
        i32::from_be_bytes(a)
    } else {
        i32::from_le_bytes(a)
    })
}

fn read_f64(c: &mut Cursor, swap: bool) -> Result<f64, String> {
    let b = c.take(8)?;
    let mut a = [0u8; 8];
    a.copy_from_slice(b);
    Ok(if swap {
        f64::from_be_bytes(a)
    } else {
        f64::from_le_bytes(a)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_msh2_ascii_and_binary_and_msh4() {
        let a = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n$EndNodes\n$Elements\n2\n1 2 2 0 1 1 2 3\n2 4 2 0 1 1 2 3 4\n$EndElements\n";
        let m = parse_msh(a.as_bytes()).unwrap();
        assert_eq!(m.tets, vec![[0, 1, 2, 3]]);
        // binary
        let mut b: Vec<u8> = b"$MeshFormat\n2.2 1 8\n".to_vec();
        b.extend_from_slice(&1i32.to_le_bytes());
        b.extend_from_slice(b"\n$EndMeshFormat\n$Nodes\n4\n");
        let p = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        for (i, q) in p.iter().enumerate() {
            b.extend_from_slice(&((i + 1) as i32).to_le_bytes());
            for v in q {
                b.extend_from_slice(&(*v as f64).to_le_bytes());
            }
        }
        b.extend_from_slice(b"\n$EndNodes\n$Elements\n1\n");
        for v in [4i32, 1, 2, 7, 0, 0, 4, 3, 2, 1] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(b"\n$EndElements\n$ElementData\n1\n\"x\"\n$EndElementData\n");
        let m = parse_msh(&b).unwrap();
        assert_eq!(m.tets, vec![[3, 2, 1, 0]]);
        let c4 = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 4 1 4\n3 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n$EndNodes\n$Elements\n1 1 1 1\n3 1 4 1\n1 1 2 3 4\n$EndElements\n";
        let m = parse_msh(c4.as_bytes()).unwrap();
        assert_eq!(m.tets, vec![[0, 1, 2, 3]]);
        assert_eq!(m.verts[3], [0.0, 0.0, 1.0]);
    }
}
