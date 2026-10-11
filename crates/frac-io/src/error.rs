use std::path::PathBuf;

/// Errors produced by `frac-io`.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error("I/O error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("unsupported file format '{0}' (expected gltf, glb, obj, ply or stl)")]
    UnsupportedFormat(String),
    #[error("glTF error: {0}")]
    Gltf(String),
    #[error("OBJ error: {0}")]
    Obj(String),
    #[error("PLY parse error: {0}")]
    Ply(String),
    #[error("STL parse error: {0}")]
    Stl(String),
    #[error("invalid render scene: {0}")]
    InvalidScene(String),
    #[error("meshopt error: {0}")]
    Meshopt(String),
    #[error("invalid physics payload: {0}")]
    Physics(String),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl IoError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        IoError::Io {
            path: path.into(),
            source,
        }
    }
}
