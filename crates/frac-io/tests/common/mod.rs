#![allow(dead_code)]

use frac_core::*;
use frac_geom::MassProps;
use frac_io::{RenderMaterial, RenderMesh, RenderNode, RenderPrimitive, RenderScene};
use glam::{DMat3, DVec3};
use serde_json::json;
use smallvec::smallvec;
use std::path::PathBuf;

/// Fresh, unique scratch directory for a test.
pub fn temp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("frac-io-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Axis-aligned box centered at the origin; faces with `interior_faces`
/// indices (0..6: +X,-X,+Y,-Y,+Z,-Z) go to primitive 1.
pub fn box_mesh(name: &str, half: [f32; 3], interior_faces: &[usize]) -> RenderMesh {
    let mut m = RenderMesh {
        name: name.into(),
        uvs: Some(vec![]),
        tangents: Some(vec![]),
        ..Default::default()
    };
    let mut prims = vec![
        RenderPrimitive {
            material: 0,
            indices: vec![],
        },
        RenderPrimitive {
            material: 1,
            indices: vec![],
        },
    ];
    let e = |k: usize| {
        let mut v = [0f32; 3];
        v[k] = 1.0;
        v
    };
    let mut face = 0;
    for k in 0..3 {
        for s in [1.0f32, -1.0] {
            let n = e(k).map(|x| x * s);
            let u = e((k + 1) % 3);
            let v = [
                n[1] * u[2] - n[2] * u[1],
                n[2] * u[0] - n[0] * u[2],
                n[0] * u[1] - n[1] * u[0],
            ];
            let base = m.positions.len() as u32;
            for (a, b) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let p: [f32; 3] = std::array::from_fn(|i| (n[i] + a * u[i] + b * v[i]) * half[i]);
                m.positions.push(p);
                m.normals.push(n);
                m.uvs
                    .as_mut()
                    .unwrap()
                    .push([(a + 1.0) * 0.5, (1.0 - b) * 0.5]);
                m.tangents.as_mut().unwrap().push([u[0], u[1], u[2], 1.0]);
            }
            let prim = if interior_faces.contains(&face) { 1 } else { 0 };
            prims[prim].indices.extend_from_slice(&[
                base,
                base + 1,
                base + 2,
                base,
                base + 2,
                base + 3,
            ]);
            face += 1;
        }
    }
    m.primitives = prims;
    m
}

/// Tetrahedron (a "LOD" mesh), normals only.
pub fn tet_mesh(name: &str, s: f32) -> RenderMesh {
    let p = [[s, s, s], [s, -s, -s], [-s, s, -s], [-s, -s, s]];
    let tris = [[0u32, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];
    let mut m = RenderMesh {
        name: name.into(),
        ..Default::default()
    };
    for t in tris {
        let base = m.positions.len() as u32;
        let a = glam::Vec3::from(p[t[0] as usize]);
        let b = glam::Vec3::from(p[t[1] as usize]);
        let c = glam::Vec3::from(p[t[2] as usize]);
        let n = (b - a).cross(c - a).normalize();
        for v in [a, b, c] {
            m.positions.push(v.into());
            m.normals.push(n.into());
        }
        let _ = base;
    }
    m.primitives = vec![RenderPrimitive {
        material: 0,
        indices: (0..12).collect(),
    }];
    m
}

/// A grid mesh with more than 65535 vertices (forces u32 indices).
pub fn big_grid_mesh(n: u32) -> RenderMesh {
    let mut m = RenderMesh {
        name: "grid".into(),
        ..Default::default()
    };
    for j in 0..n {
        for i in 0..n {
            m.positions.push([i as f32 * 0.01, 0.0, j as f32 * 0.01]);
            m.normals.push([0.0, 1.0, 0.0]);
        }
    }
    let mut idx = Vec::new();
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let a = j * n + i;
            let (b, c, d) = (a + 1, a + n, a + n + 1);
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    m.primitives = vec![RenderPrimitive {
        material: 0,
        indices: idx,
    }];
    m
}

/// A small fractured-asset render scene: root node, one L0 fragment with an
/// MSFT_lod chain, two L1 child fragments.
pub fn sample_render_scene() -> RenderScene {
    RenderScene {
        name: "sample".into(),
        materials: vec![
            RenderMaterial {
                name: "concrete".into(),
                base_color: [0.6, 0.6, 0.6, 1.0],
                metallic: 0.0,
                roughness: 0.9,
                double_sided: false,
            },
            RenderMaterial {
                name: "concrete_interior".into(),
                base_color: [0.8, 0.3, 0.2, 1.0],
                metallic: 0.0,
                roughness: 1.0,
                double_sided: true,
            },
        ],
        meshes: vec![
            box_mesh("frag0", [1.0, 0.5, 0.5], &[]),
            box_mesh("frag1", [0.5, 0.5, 0.5], &[0]),
            box_mesh("frag2", [0.5, 0.5, 0.5], &[1]),
            tet_mesh("frag0_lod1", 0.5),
        ],
        nodes: vec![
            RenderNode {
                name: "asset".into(),
                parent: None,
                translation: [0.0, 0.0, 0.0],
                mesh_lods: vec![],
                extras: json!({"asset": "sample"}),
            },
            RenderNode {
                name: "frag_0".into(),
                parent: Some(0),
                translation: [0.0, 1.5, 0.0],
                mesh_lods: vec![0, 3],
                extras: json!({"fragment_id": 0, "level": 0, "parent": null, "children": [1, 2]}),
            },
            RenderNode {
                name: "frag_1".into(),
                parent: Some(1),
                translation: [0.5, 1.5, 0.0],
                mesh_lods: vec![1],
                extras: json!({"fragment_id": 1, "level": 1, "parent": 0, "children": []}),
            },
            RenderNode {
                name: "frag_2".into(),
                parent: Some(1),
                translation: [-0.5, 1.5, 0.0],
                mesh_lods: vec![2],
                extras: json!({"fragment_id": 2, "level": 1, "parent": 0, "children": []}),
            },
        ],
        extras: json!({"prefracture": {"levels": 2}}),
    }
}

fn mass(m: f64, c: DVec3) -> MassProps {
    MassProps {
        volume: m / 2400.0,
        mass: m,
        com: c,
        inertia: DMat3::from_diagonal(DVec3::new(1.0, 2.0, 3.0)) * m,
    }
}

fn fragment(
    id: u32,
    level: u8,
    parent: Option<u32>,
    children: std::ops::Range<u32>,
    cells: std::ops::Range<u32>,
    m: f64,
    c: DVec3,
) -> Fragment {
    Fragment {
        id: FragmentId(id),
        level,
        component: ComponentId(0),
        parent: parent.map(FragmentId),
        children,
        cells,
        material_mix: smallvec![(MaterialId(3), 0.75), (MaterialId(7), 0.25)],
        mass: mass(m, c),
        hulls: id..id + 1,
        render: RenderRefs {
            gltf_node: id as i32 + 1,
            lod_meshes: vec![id],
        },
        particle_candidate: level == 1,
        role: ComponentRole::Wall,
        interior_area: 0.5 * id as f64,
    }
}

fn bond(id: u32, level: u8, a: u32, b: FragmentOrWorld, area: f64) -> Bond {
    Bond {
        id: BondId(id),
        level,
        a: FragmentId(a),
        b,
        area,
        centroid: DVec3::new(0.0, 1.0, 0.0),
        normal: DVec3::X,
        planarity: 1.0,
        frame_u: DVec3::Y,
        frame_v: DVec3::Z,
        i_uu: 0.1,
        i_vv: 0.2,
        i_uv: 0.0,
        j: 0.3,
        extent: Obb2 {
            center: DVec3::new(0.0, 1.0, 0.0),
            axis_u: DVec3::Y,
            axis_v: DVec3::Z,
            half: [0.5, 0.25],
        },
        dist_a: 0.5,
        dist_b: 0.5,
        composition: smallvec![
            BondComposition {
                kind: InterfaceKind::Monolithic,
                interface_material: None,
                fraction: 0.8
            },
            BondComposition {
                kind: InterfaceKind::MortarJoint,
                interface_material: Some(MaterialId(9)),
                fraction: 0.2
            }
        ],
        reinforcement: RebarCrossing {
            count: 2,
            steel_area: 2e-4,
            dir: DVec3::new(3.0, 0.0, 0.0),
        },
        strength_scale: 1.0,
        parent_bond: None,
        child_bonds: 0..0,
        boundary_loops: id..id + 1,
        spawn: 2 * id..2 * id + 2,
        anchor: matches!(b, FragmentOrWorld::World),
        interfaces: vec![InterfaceId(id), InterfaceId(id + 10)],
    }
}

fn cube_hull(f: u32, c: DVec3, h: f64) -> Hull {
    let mut vertices = Vec::new();
    for i in 0..8 {
        vertices.push(
            c + DVec3::new(
                if i & 1 != 0 { h } else { -h },
                if i & 2 != 0 { h } else { -h },
                if i & 4 != 0 { h } else { -h },
            ),
        );
    }
    let faces = vec![
        vec![0, 2, 3, 1],
        vec![4, 5, 7, 6],
        vec![0, 1, 5, 4],
        vec![2, 6, 7, 3],
        vec![0, 4, 6, 2],
        vec![1, 3, 7, 5],
    ];
    Hull {
        fragment: FragmentId(f),
        vertices,
        faces,
    }
}

/// Hand-built two-level asset: L0 = {0, 1}, L1 = {2, 3} (children of 0),
/// {4} (child of 1). Bonds sorted by (level, a, b), with world anchors.
pub fn sample_asset() -> Asset {
    let fragments = vec![
        fragment(0, 0, None, 2..4, 0..2, 200.0, DVec3::new(0.0, 1.0, 0.0)),
        fragment(1, 0, None, 4..5, 2..3, 100.0, DVec3::new(2.0, 1.0, 0.0)),
        fragment(2, 1, Some(0), 0..0, 0..1, 120.0, DVec3::new(-0.5, 1.0, 0.0)),
        fragment(3, 1, Some(0), 0..0, 1..2, 80.0, DVec3::new(0.5, 1.0, 0.0)),
        fragment(4, 1, Some(1), 0..0, 2..3, 100.0, DVec3::new(2.0, 1.0, 0.0)),
    ];
    let mut bonds = vec![
        bond(0, 0, 0, FragmentOrWorld::Fragment(FragmentId(1)), 1.0),
        bond(1, 0, 0, FragmentOrWorld::World, 2.0),
        bond(2, 1, 2, FragmentOrWorld::Fragment(FragmentId(3)), 0.5),
        bond(3, 1, 2, FragmentOrWorld::World, 1.0),
        bond(4, 1, 3, FragmentOrWorld::Fragment(FragmentId(4)), 1.0),
        bond(5, 1, 3, FragmentOrWorld::World, 1.0),
    ];
    bonds[0].child_bonds = 0..1;
    bonds[1].child_bonds = 1..3;
    bonds[4].parent_bond = Some(BondId(0));
    bonds[3].parent_bond = Some(BondId(1));
    bonds[5].parent_bond = Some(BondId(1));
    let hulls = (0..5)
        .map(|i| cube_hull(i, fragments[i as usize].mass.com, 0.5))
        .collect();
    let loops = (0..6)
        .map(|i| {
            vec![
                DVec3::new(0.0, 0.0, i as f64),
                DVec3::new(0.0, 1.0, i as f64),
                DVec3::new(0.0, 1.0, i as f64 + 1.0),
            ]
        })
        .collect();
    let spawn = (0..12)
        .map(|i| SpawnPoint {
            p: DVec3::new(i as f64 * 0.1, 1.0, 0.0),
            n: DVec3::X,
        })
        .collect();
    Asset {
        meta: AssetMeta {
            name: "sample".into(),
            seed: 42,
            variant: 1,
            tool_version: "0.1.0".into(),
            settings_hash: "abc123".into(),
            material_library_id: "builtin".into(),
            material_library_version: "1".into(),
        },
        hierarchy: Hierarchy {
            levels: 2,
            fragments,
            level_ranges: vec![0..2, 2..5],
            cell_order: vec![CellId(0), CellId(1), CellId(2)],
            cell_fragment: vec![
                vec![FragmentId(0), FragmentId(0), FragmentId(1)],
                vec![FragmentId(2), FragmentId(3), FragmentId(4)],
            ],
        },
        bonds,
        bond_children: vec![BondId(4), BondId(3), BondId(5)],
        hulls,
        loops,
        spawn,
        ..Default::default()
    }
}
