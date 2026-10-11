//! Cell complexes for fracture: seeding, Delaunay/Voronoi and box
//! partitions, exact clipping of a closed solid into cells with shared
//! interfaces, and material recipes.

pub mod cellset;
pub mod clip;
pub mod complex;
pub mod delaunay;
pub mod planes;
pub mod recipes;
pub mod seeding;
pub mod tri2d;
