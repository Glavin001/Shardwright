//! Analysis tetrahedral meshing and linear-elastic P1 finite elements.
//!
//! This crate provides the "Stage 3" analysis mesh of the pre-fracture
//! pipeline and the FEM building blocks used by the weak-region analysis
//! (`frac-modes`) and by validation:
//!
//! * [`tetrahedralize`]: a pure-Rust BCC-lattice mesher with surface snapping
//!   ("isosurface stuffing"-lite) for closed manifold solids, plus
//!   [`tetrahedralize_external`] which drives an fTetWild binary.
//! * [`ElasticMaterial`]: isotropic and transversely isotropic (wood-like)
//!   linear elasticity, [`element::tet_stiffness`] (12x12 P1 element matrix),
//!   lumped mass, deterministic sparse assembly ([`sparse::CsrMatrix`]).
//! * [`sparse::SparseCholesky`]: a thin deterministic (sequential) wrapper
//!   over faer's supernodal/simplicial sparse Cholesky.
//! * [`eigen::smallest_eigenpairs`]: shift-invert block subspace iteration with
//!   Rayleigh–Ritz for the generalized problem `K x = λ M x` with lumped `M`.
//! * [`solve_static`] and [`natural_frequencies`] for sanity checks.
//!
//! Determinism: no hash-map iteration; parallel loops only with ordered
//! collection; all reductions are sequential; faer is always invoked with
//! `Par::Seq`; transcendental functions go through `libm`.

pub mod analysis;
pub mod dense;
pub mod eigen;
pub mod element;
pub mod material;
pub mod mesh;
pub mod sparse;
pub mod tetgen;

pub use analysis::{
    assemble_lumped_mass, assemble_stiffness, natural_frequencies, rigid_modes, solve_static,
};
pub use material::{ElasticMaterial, TransverseIsotropic};
pub use mesh::TetMesh;
pub use tetgen::{tetrahedralize, tetrahedralize_external};
