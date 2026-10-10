//! Hooks for external validators: the Khronos glTF Validator (via Node and
//! `tools/gltf_validate`) and `flatc` (schema decode of `.fracphys`).

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The FlatBuffers schema of the physics payload (`schemas/frac.fbs`).
pub const FRAC_FBS: &str = include_str!("../../../schemas/frac.fbs");

/// Directory holding `tools/`: the working directory, an ancestor of the
/// executable (`target/<profile>/prefracture`, or an installed copy next to
/// the repository tree), else the source tree this crate was built from.
fn repo_root() -> PathBuf {
    let has_tools = |p: &Path| p.join("tools").join("gltf_validate").is_dir();
    let mut roots: Vec<PathBuf> = vec![PathBuf::from(".")];
    if let Ok(exe) = std::env::current_exe() {
        roots.extend(exe.ancestors().skip(1).take(4).map(|p| p.to_path_buf()));
    }
    roots.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(".."));
    roots.into_iter().find(|r| has_tools(r)).unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(".."))
}

fn runs(cmd: &Path, arg: &str) -> bool {
    Command::new(cmd)
        .arg(arg)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Remove `frac-io-{tag}-{pid}-*` scratch directories left behind by killed
/// processes (Linux: the pid has no `/proc` entry); the flatc dumps of
/// building-scale assets are gigabytes each.
fn remove_stale_scratch(tag: &str) {
    if !Path::new("/proc/self").exists() {
        return;
    }
    let prefix = format!("frac-io-{tag}-");
    let Ok(rd) = std::fs::read_dir(std::env::temp_dir()) else { return };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let Some(rest) = name.strip_prefix(&prefix) else { continue };
        let Some(pid) = rest.split('-').next().and_then(|p| p.parse::<u32>().ok()) else { continue };
        if pid != std::process::id() && !Path::new(&format!("/proc/{pid}")).exists() {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
}

fn scratch_dir(tag: &str) -> std::io::Result<PathBuf> {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "frac-io-{tag}-{}-{}-{nanos}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Locate `flatc`: `$FLATC`, then `$ORACLES/flatbuffers/build/flatc`
/// (tools/setup.sh), `/opt/oracles/flatbuffers/build/flatc`, then `flatc` on
/// `PATH`.
pub fn find_flatc() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(p) = std::env::var_os("FLATC") {
        candidates.push(PathBuf::from(p));
    }
    if let Some(o) = std::env::var_os("ORACLES") {
        candidates.push(PathBuf::from(o).join("flatbuffers/build/flatc"));
    }
    candidates.push(PathBuf::from("/opt/oracles/flatbuffers/build/flatc"));
    candidates.push(PathBuf::from("flatc"));
    candidates.into_iter().find(|c| runs(c, "--version"))
}

/// Decode a `.fracphys` file with `flatc --json --strict-json --raw-binary`
/// against [`FRAC_FBS`] (the spec's schema gate). `None` when `flatc` is
/// unavailable; `Some(Err(msg))` when decoding fails.
pub fn flatc_validate(path: &Path) -> Option<Result<(), String>> {
    let flatc = find_flatc()?;
    remove_stale_scratch("flatc");
    Some((|| {
        let dir = scratch_dir("flatc").map_err(|e| e.to_string())?;
        let schema = dir.join("frac.fbs");
        std::fs::write(&schema, FRAC_FBS).map_err(|e| e.to_string())?;
        let out = Command::new(&flatc)
            .arg("--json")
            .arg("--strict-json")
            .arg("--raw-binary")
            .arg("-o")
            .arg(&dir)
            .arg(&schema)
            .arg("--")
            .arg(path)
            .output()
            .map_err(|e| format!("failed to run flatc: {e}"))?;
        let stem = path
            .file_stem()
            .map(|s| s.to_os_string())
            .unwrap_or_default();
        let json_out = dir.join(stem).with_extension("json");
        let result = if !out.status.success() {
            Err(format!(
                "flatc failed ({}): {}{}",
                out.status,
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            ))
        } else {
            // streamed syntax check (the dump reaches gigabytes on
            // building-scale assets; no in-memory document)
            match std::fs::File::open(&json_out) {
                Ok(f) => serde_json::from_reader::<_, serde::de::IgnoredAny>(std::io::BufReader::with_capacity(1 << 20, f))
                    .map(|_| ())
                    .map_err(|e| format!("flatc JSON output invalid: {e}")),
                Err(e) => Err(format!("flatc produced no JSON output: {e}")),
            }
        };
        let _ = std::fs::remove_dir_all(&dir);
        result
    })())
}

/// Validate a `.glb`/`.gltf` file with the Khronos glTF Validator, using
/// `tools/gltf_validate/validate.mjs` (override the directory with
/// `$SHARDWRIGHT_GLTF_VALIDATE_DIR`, the Node binary with `$NODE`).
///
/// Returns `None` when Node or the validator package (`npm install` in that
/// directory) is unavailable; `Some(Ok(report))` when the report has zero
/// errors (warnings/infos/hints allowed; inspect `report.issues`); and
/// `Some(Err(msg))` when the report has errors (message lists the counts and
/// the first error messages) or the validator could not run.
pub fn khronos_validate(path: &Path) -> Option<Result<Value, String>> {
    let dir = std::env::var_os("SHARDWRIGHT_GLTF_VALIDATE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("tools").join("gltf_validate"));
    let script = dir.join("validate.mjs");
    if !script.is_file() || !dir.join("node_modules").join("gltf-validator").is_dir() {
        return None;
    }
    let node = std::env::var_os("NODE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("node"));
    if !runs(&node, "--version") {
        return None;
    }
    let out = match Command::new(&node).arg(&script).arg(path).output() {
        Ok(o) => o,
        Err(e) => return Some(Err(format!("failed to run node: {e}"))),
    };
    let report: Value = match serde_json::from_slice(&out.stdout) {
        Ok(v) => v,
        Err(_) => {
            return Some(Err(format!(
                "validator produced no report ({}): {}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            )));
        }
    };
    let num_errors = report
        .pointer("/issues/numErrors")
        .and_then(Value::as_u64)
        .unwrap_or(u64::MAX);
    if num_errors == 0 {
        return Some(Ok(report));
    }
    let msgs: Vec<String> = report
        .pointer("/issues/messages")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter(|m| m.get("severity").and_then(Value::as_u64) == Some(0))
                .take(10)
                .map(|m| {
                    format!(
                        "{} at {}: {}",
                        m.get("code").and_then(Value::as_str).unwrap_or("?"),
                        m.get("pointer").and_then(Value::as_str).unwrap_or(""),
                        m.get("message").and_then(Value::as_str).unwrap_or("")
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    Some(Err(format!(
        "glTF validator: {num_errors} error(s), {} warning(s): {}",
        report
            .pointer("/issues/numWarnings")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        msgs.join("; ")
    )))
}
