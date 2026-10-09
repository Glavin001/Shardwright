//! Cell complexes for fracture: seeding, Delaunay/Voronoi and box
//! partitions, exact clipping of a closed solid into cells with shared
//! interfaces, and material recipes.

pub mod delaunay;
pub mod planes;
pub mod complex;
pub mod clip;
pub mod tri2d;
pub mod cellset;
pub mod seeding;
pub mod recipes;
