use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Stage {
    Config,
    Ingest,
    Components,
    Cells,
    AnalysisMesh,
    Modes,
    Hierarchy,
    Bonds,
    Mass,
    Collision,
    Render,
    Crack,
    Patterns,
    Validate,
    Export,
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Debug, thiserror::Error, Clone, Serialize, Deserialize)]
#[error("[{stage}] {context}: {message}")]
pub struct FracError {
    pub stage: Stage,
    /// Component / cell / interface context, e.g. "component 3, cell 17".
    pub context: String,
    pub message: String,
}

impl FracError {
    pub fn new(stage: Stage, context: impl Into<String>, message: impl Into<String>) -> Self {
        FracError { stage, context: context.into(), message: message.into() }
    }
}

pub type FracResult<T> = Result<T, FracError>;
