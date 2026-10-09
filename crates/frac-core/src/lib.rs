//! Core data model, identifiers, errors, settings and determinism utilities.

pub mod determinism;
pub mod error;
pub mod ids;
pub mod model;
pub mod settings;

pub use determinism::*;
pub use error::{FracError, Stage};
pub use ids::*;
pub use model::*;
pub use settings::Settings;

/// Tool version embedded in outputs.
pub const TOOL_VERSION: &str = env!("CARGO_PKG_VERSION");
/// Physics payload schema version.
pub const SCHEMA_VERSION: u16 = 1;
