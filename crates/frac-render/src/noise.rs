//! Deterministic fractal noise (value-gradient lattice noise, fBm) using
//! integer hashing and `libm` only, so results are bit-identical across
//! platforms.

use glam::DVec3;

#[inline]
fn hash3(x: i64, y: i64, z: i64, seed: u64) -> u64 {
    let mut h = seed ^ 0x9E37_79B9_7F4A_7C15;
    for v in [x as u64, y as u64, z as u64] {
        h ^= v.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h = h.rotate_left(27).wrapping_mul(0x94D0_49BB_1331_11EB);
        h ^= h >> 31;
    }
    h
}

#[inline]
fn grad(h: u64, d: DVec3) -> f64 {
    // 12 edge gradients of a cube
    match h % 12 {
        0 => d.x + d.y,
        1 => -d.x + d.y,
        2 => d.x - d.y,
        3 => -d.x - d.y,
        4 => d.x + d.z,
        5 => -d.x + d.z,
        6 => d.x - d.z,
        7 => -d.x - d.z,
        8 => d.y + d.z,
        9 => -d.y + d.z,
        10 => d.y - d.z,
        _ => -d.y - d.z,
    }
}

#[inline]
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Gradient noise in roughly [-1, 1].
pub fn gradient_noise(p: DVec3, seed: u64) -> f64 {
    let i = p.floor();
    let f = p - i;
    let (ix, iy, iz) = (i.x as i64, i.y as i64, i.z as i64);
    let u = DVec3::new(fade(f.x), fade(f.y), fade(f.z));
    let mut acc = [0.0f64; 8];
    for c in 0..8 {
        let (dx, dy, dz) = ((c & 1) as i64, ((c >> 1) & 1) as i64, ((c >> 2) & 1) as i64);
        let h = hash3(ix + dx, iy + dy, iz + dz, seed);
        acc[c] = grad(h, f - DVec3::new(dx as f64, dy as f64, dz as f64));
    }
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let x00 = lerp(acc[0], acc[1], u.x);
    let x10 = lerp(acc[2], acc[3], u.x);
    let x01 = lerp(acc[4], acc[5], u.x);
    let x11 = lerp(acc[6], acc[7], u.x);
    let y0 = lerp(x00, x10, u.y);
    let y1 = lerp(x01, x11, u.y);
    lerp(y0, y1, u.z)
}

/// Fractional Brownian motion with octaves from `max_wl` down to `min_wl`,
/// octave amplitude ∝ wavelength^H (Hurst exponent), normalized so the
/// largest octave has unit amplitude.
pub fn fbm(p: DVec3, min_wl: f64, max_wl: f64, hurst: f64, seed: u64) -> f64 {
    let mut wl = max_wl.max(min_wl);
    let mut sum = 0.0;
    let mut norm = 0.0;
    let mut k = 0u64;
    while wl >= min_wl * 0.999 && k < 16 {
        let amp = libm::pow(wl / max_wl, hurst);
        sum += amp * gradient_noise(p / wl, seed.wrapping_add(k * 7919));
        norm += amp;
        wl *= 0.5;
        k += 1;
    }
    if norm > 0.0 { sum / norm * 1.6 } else { 0.0 }
}
