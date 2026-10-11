//! Geometry kernels for the pre-fracture library.
//!
//! Everything here is deterministic: no hash-map iteration, no platform
//! transcendental functions in decision paths (only `libm`), and exact
//! predicates (with filtered evaluation) for every combinatorial decision.

pub mod aabb;
pub mod bvh;
pub mod exact;
pub mod hull;
pub mod inside;
pub mod integrals;
pub mod mesh;
pub mod polygon;
pub mod predicates;

pub use aabb::Aabb;
pub use glam::{DMat3, DVec2, DVec3};
pub use integrals::{MassProps, VolumeIntegrals};
pub use mesh::TriMesh;

/// 30-bit-per-axis Morton code of a point normalized to a box.
pub fn morton3(p: DVec3, bbox: &Aabb) -> u64 {
    let e = bbox.extent().max(DVec3::splat(1e-300));
    let q = ((p - bbox.min) / e).clamp(DVec3::ZERO, DVec3::ONE);
    let s = (1u64 << 21) as f64 - 1.0;
    let spread = |x: u64| -> u64 {
        let mut x = x & 0x1fffff;
        x = (x | (x << 32)) & 0x1f00000000ffff;
        x = (x | (x << 16)) & 0x1f0000ff0000ff;
        x = (x | (x << 8)) & 0x100f00f00f00f00f;
        x = (x | (x << 4)) & 0x10c30c30c30c30c3;
        x = (x | (x << 2)) & 0x1249249249249249;
        x
    };
    let xi = (q.x * s) as u64;
    let yi = (q.y * s) as u64;
    let zi = (q.z * s) as u64;
    spread(xi) | (spread(yi) << 1) | (spread(zi) << 2)
}
