//! Determinism utilities: stable hashing, seeded RNG streams, canonical
//! ordering helpers.

use rand::SeedableRng;
pub use rand_chacha::ChaCha8Rng;

/// Pipeline stage identifiers for RNG stream derivation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum RngStage {
    Seeding = 1,
    Recipe = 2,
    Hierarchy = 3,
    Bonds = 4,
    Render = 5,
    Noise = 6,
    Collision = 7,
    Validate = 8,
    Patterns = 9,
}

/// Stable 64-bit hash of a list of integers (BLAKE3, platform independent).
pub fn stable_hash(parts: &[u64]) -> u64 {
    let mut h = blake3::Hasher::new();
    for p in parts {
        h.update(&p.to_le_bytes());
    }
    let out = h.finalize();
    u64::from_le_bytes(out.as_bytes()[..8].try_into().unwrap())
}

/// Stable hash of bytes, hex encoded (for settings hashes and cache keys).
pub fn stable_hash_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// RNG for `(asset_seed, component, stage, level, variant)` (spec §10).
pub fn rng_for(
    asset_seed: u64,
    component: u32,
    stage: RngStage,
    level: u32,
    variant: u32,
) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(stable_hash(&[
        asset_seed,
        component as u64,
        stage as u64,
        level as u64,
        variant as u64,
    ]))
}

/// Derive a sub-stream seed.
pub fn sub_seed(seed: u64, k: u64) -> u64 {
    stable_hash(&[seed, k])
}

/// Uniform f64 in [0,1) from a u64 hash (53-bit mantissa).
#[inline]
pub fn unit_f64(h: u64) -> f64 {
    (h >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Total-order key for f64 (for deterministic sorting).
#[inline]
pub fn f64_key(x: f64) -> i64 {
    let b = x.to_bits() as i64;
    if b < 0 { b ^ i64::MAX } else { b }
}
