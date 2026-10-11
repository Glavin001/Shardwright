//! Shardwright I/O: scene import (glTF/GLB, OBJ, PLY, STL), glTF 2.0 render
//! export (with optional `EXT_meshopt_compression` and `MSFT_lod`), the
//! FlatBuffers physics payload (`.fracphys`, schema `schemas/frac.fbs`),
//! JSON debug dumps and PLY/OBJ geometry dumps, plus hooks for the external
//! validators (Khronos glTF Validator, `flatc`).
//!
//! Every output path is deterministic: identical input produces
//! byte-identical output (no hash-map iteration, canonical ordering, floats
//! formatted with shortest round-trip representations).

mod debug;
mod error;
mod glb;
mod gltf_out;
mod import;
mod physics;
mod validate;

#[allow(
    unused_imports,
    dead_code,
    clippy::all,
    non_camel_case_types,
    non_snake_case,
    unsafe_op_in_unsafe_fn,
    mismatched_lifetime_syntaxes,
    unknown_lints
)]
#[rustfmt::skip]
#[path = "generated/frac_generated.rs"]
mod frac_generated;

/// Generated FlatBuffers bindings for `schemas/frac.fbs` (namespace `frac`).
pub mod fb {
    pub use crate::frac_generated::frac::*;
}

pub use debug::{asset_to_json, write_asset_json, write_obj, write_ply, write_ply_polys};
pub use error::IoError;
pub use gltf_out::{
    GlbSummary, GltfOptions, RenderMaterial, RenderMesh, RenderNode, RenderPrimitive, RenderScene,
    decompress_meshopt_glb, read_glb_summary, release_free_memory, write_glb, write_glb_owned,
};
pub use import::{ImportOptions, load_scene, load_scene_bytes};
pub use physics::{
    PhysicsBond, PhysicsFragment, PhysicsView, physics_to_json, read_physics, write_physics,
};
pub use validate::{FRAC_FBS, find_flatc, flatc_scratch_check, flatc_validate, khronos_validate};
