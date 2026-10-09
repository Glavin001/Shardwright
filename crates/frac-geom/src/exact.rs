//! Exact floating-point expansion arithmetic (Shewchuk 1997) and a filtered
//! evaluation scheme.
//!
//! Predicates are written once, generically over [`Field`], and evaluated
//! first with [`F64E`] (an `f64` that carries a rigorous-in-practice forward
//! error bound). Only when the sign is not certified do we re-evaluate with
//! [`Expansion`], which is exact for any polynomial in `f64` inputs.

use smallvec::SmallVec;

#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let bv = x - a;
    let av = x - bv;
    let br = b - bv;
    let ar = a - av;
    (x, ar + br)
}

#[inline(always)]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let bv = x - a;
    (x, b - bv)
}

#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let x = a * b;
    (x, a.mul_add(b, -x))
}

/// A nonoverlapping expansion: components sorted by increasing magnitude,
/// zero-eliminated. The represented value is the exact sum of components.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Expansion(pub SmallVec<[f64; 8]>);

impl Expansion {
    pub fn zero() -> Self {
        Expansion(SmallVec::new())
    }
    pub fn from_f64(x: f64) -> Self {
        let mut v = SmallVec::new();
        if x != 0.0 {
            v.push(x);
        }
        Expansion(v)
    }
    /// Sign of the exact value: -1, 0 or 1.
    pub fn sign(&self) -> i8 {
        match self.0.last() {
            None => 0,
            Some(&x) if x > 0.0 => 1,
            Some(&x) if x < 0.0 => -1,
            _ => 0,
        }
    }
    /// Approximation of the value (sum of components, largest last).
    pub fn approx(&self) -> f64 {
        // Summing from smallest to largest gives a faithful rounding for
        // nonoverlapping expansions in practice.
        let mut s = 0.0;
        for &c in self.0.iter() {
            s += c;
        }
        s
    }
    fn grow(&self, b: f64) -> Expansion {
        let mut h = SmallVec::with_capacity(self.0.len() + 1);
        let mut q = b;
        for &e in self.0.iter() {
            let (qn, hh) = two_sum(q, e);
            q = qn;
            if hh != 0.0 {
                h.push(hh);
            }
        }
        if q != 0.0 {
            h.push(q);
        }
        Expansion(h)
    }
    pub fn add(&self, o: &Expansion) -> Expansion {
        // fast_expansion_sum_zeroelim
        let e = &self.0;
        let f = &o.0;
        if e.is_empty() {
            return o.clone();
        }
        if f.is_empty() {
            return self.clone();
        }
        if f.len() == 1 {
            return self.grow(f[0]);
        }
        if e.len() == 1 {
            return o.grow(e[0]);
        }
        let mut h: SmallVec<[f64; 8]> = SmallVec::with_capacity(e.len() + f.len());
        let (mut ei, mut fi) = (0usize, 0usize);
        let mut enow = e[0];
        let mut fnow = f[0];
        let mut q;
        if (fnow > enow) == (fnow > -enow) {
            q = enow;
            ei += 1;
        } else {
            q = fnow;
            fi += 1;
        }
        if ei < e.len() && fi < f.len() {
            enow = e[ei];
            fnow = f[fi];
            let (qn, hh) = if (fnow > enow) == (fnow > -enow) {
                ei += 1;
                fast_two_sum(enow, q)
            } else {
                fi += 1;
                fast_two_sum(fnow, q)
            };
            q = qn;
            if hh != 0.0 {
                h.push(hh);
            }
            while ei < e.len() && fi < f.len() {
                enow = e[ei];
                fnow = f[fi];
                let (qn, hh) = if (fnow > enow) == (fnow > -enow) {
                    ei += 1;
                    two_sum(q, enow)
                } else {
                    fi += 1;
                    two_sum(q, fnow)
                };
                q = qn;
                if hh != 0.0 {
                    h.push(hh);
                }
            }
        }
        while ei < e.len() {
            let (qn, hh) = two_sum(q, e[ei]);
            ei += 1;
            q = qn;
            if hh != 0.0 {
                h.push(hh);
            }
        }
        while fi < f.len() {
            let (qn, hh) = two_sum(q, f[fi]);
            fi += 1;
            q = qn;
            if hh != 0.0 {
                h.push(hh);
            }
        }
        if q != 0.0 {
            h.push(q);
        }
        Expansion(h)
    }
    pub fn neg(&self) -> Expansion {
        Expansion(self.0.iter().map(|x| -x).collect())
    }
    pub fn sub(&self, o: &Expansion) -> Expansion {
        self.add(&o.neg())
    }
    pub fn scale(&self, b: f64) -> Expansion {
        let e = &self.0;
        if e.is_empty() || b == 0.0 {
            return Expansion::zero();
        }
        let mut h: SmallVec<[f64; 8]> = SmallVec::with_capacity(2 * e.len());
        let (mut q, hh) = two_prod(e[0], b);
        if hh != 0.0 {
            h.push(hh);
        }
        for &ei in e.iter().skip(1) {
            let (p1, p0) = two_prod(ei, b);
            let (s, hh) = two_sum(q, p0);
            if hh != 0.0 {
                h.push(hh);
            }
            let (qn, hh) = fast_two_sum(p1, s);
            q = qn;
            if hh != 0.0 {
                h.push(hh);
            }
        }
        if q != 0.0 {
            h.push(q);
        }
        Expansion(h)
    }
    pub fn mul(&self, o: &Expansion) -> Expansion {
        let (a, b) = if self.0.len() <= o.0.len() { (self, o) } else { (o, self) };
        let mut acc = Expansion::zero();
        for &c in a.0.iter() {
            acc = acc.add(&b.scale(c));
        }
        acc
    }
}

/// Error-tracked double: `v` approximates the exact value with
/// `|exact - v| <= e`.
#[derive(Clone, Copy, Debug)]
pub struct F64E {
    pub v: f64,
    pub e: f64,
}

const U: f64 = f64::EPSILON * 0.5; // unit roundoff 2^-53

impl F64E {
    /// Certified sign, or `None` when the filter cannot decide.
    #[inline]
    pub fn certain_sign(&self) -> Option<i8> {
        // The bound itself was computed in floating point; inflate it
        // slightly to cover the roundoff of the bound computation.
        let bound = self.e * (1.0 + 16.0 * U) + f64::MIN_POSITIVE;
        if self.v > bound {
            Some(1)
        } else if self.v < -bound {
            Some(-1)
        } else if self.v == 0.0 && self.e == 0.0 {
            Some(0)
        } else {
            None
        }
    }
}

/// Numeric field abstraction used to write predicates once.
pub trait Field: Clone {
    fn from_f64(x: f64) -> Self;
    fn add(&self, o: &Self) -> Self;
    fn sub(&self, o: &Self) -> Self;
    fn mul(&self, o: &Self) -> Self;
    fn neg(&self) -> Self;
    fn approx(&self) -> f64;
    #[inline]
    fn zero() -> Self {
        Self::from_f64(0.0)
    }
    #[inline]
    fn one() -> Self {
        Self::from_f64(1.0)
    }
}

impl Field for F64E {
    #[inline(always)]
    fn from_f64(x: f64) -> Self {
        F64E { v: x, e: 0.0 }
    }
    #[inline(always)]
    fn add(&self, o: &Self) -> Self {
        let v = self.v + o.v;
        F64E { v, e: self.e + o.e + v.abs() * U }
    }
    #[inline(always)]
    fn sub(&self, o: &Self) -> Self {
        let v = self.v - o.v;
        F64E { v, e: self.e + o.e + v.abs() * U }
    }
    #[inline(always)]
    fn mul(&self, o: &Self) -> Self {
        let v = self.v * o.v;
        F64E {
            v,
            e: self.v.abs() * o.e + o.v.abs() * self.e + self.e * o.e + v.abs() * U,
        }
    }
    #[inline(always)]
    fn neg(&self) -> Self {
        F64E { v: -self.v, e: self.e }
    }
    #[inline(always)]
    fn approx(&self) -> f64 {
        self.v
    }
}

impl Field for Expansion {
    fn from_f64(x: f64) -> Self {
        Expansion::from_f64(x)
    }
    fn add(&self, o: &Self) -> Self {
        Expansion::add(self, o)
    }
    fn sub(&self, o: &Self) -> Self {
        Expansion::sub(self, o)
    }
    fn mul(&self, o: &Self) -> Self {
        Expansion::mul(self, o)
    }
    fn neg(&self) -> Self {
        Expansion::neg(self)
    }
    fn approx(&self) -> f64 {
        Expansion::approx(self)
    }
}

/// Plain f64 (no error tracking) for fast approximate evaluation.
impl Field for f64 {
    #[inline(always)]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline(always)]
    fn add(&self, o: &Self) -> Self {
        self + o
    }
    #[inline(always)]
    fn sub(&self, o: &Self) -> Self {
        self - o
    }
    #[inline(always)]
    fn mul(&self, o: &Self) -> Self {
        self * o
    }
    #[inline(always)]
    fn neg(&self) -> Self {
        -self
    }
    #[inline(always)]
    fn approx(&self) -> f64 {
        *self
    }
}

/// Evaluate a generic polynomial expression and return its exact sign.
/// `$f` must be a generic function/closure-like macro body using the type
/// parameter `$F`.
#[macro_export]
macro_rules! exact_sign {
    (|$F:ident| $body:expr) => {{
        let filtered = {
            type $F = $crate::exact::F64E;
            $body
        };
        match filtered.certain_sign() {
            Some(s) => s,
            None => {
                type $F = $crate::exact::Expansion;
                let ex: $crate::exact::Expansion = $body;
                ex.sign()
            }
        }
    }};
}

/// Evaluate a generic polynomial and return the best `f64` approximation of
/// its value (exact evaluation, then rounded).
#[macro_export]
macro_rules! exact_value {
    (|$F:ident| $body:expr) => {{
        type $F = $crate::exact::Expansion;
        let ex: $crate::exact::Expansion = $body;
        ex.approx()
    }};
}

#[inline]
pub fn det2<F: Field>(a: &F, b: &F, c: &F, d: &F) -> F {
    // | a b |
    // | c d |
    a.mul(d).sub(&b.mul(c))
}

#[inline]
pub fn det3<F: Field>(m: &[[F; 3]; 3]) -> F {
    let c0 = det2(&m[1][1], &m[1][2], &m[2][1], &m[2][2]);
    let c1 = det2(&m[1][0], &m[1][2], &m[2][0], &m[2][2]);
    let c2 = det2(&m[1][0], &m[1][1], &m[2][0], &m[2][1]);
    m[0][0].mul(&c0).sub(&m[0][1].mul(&c1)).add(&m[0][2].mul(&c2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expansion_exact_cancellation() {
        let a = Expansion::from_f64(1e20);
        let b = Expansion::from_f64(1.0);
        let c = a.add(&b).sub(&a);
        assert_eq!(c.approx(), 1.0);
        let p = Expansion::from_f64(0.1).mul(&Expansion::from_f64(0.1));
        let q = Expansion::from_f64(0.1 * 0.1);
        // 0.1*0.1 in exact arithmetic differs from its rounding
        assert_ne!(p.sub(&q).sign(), 0);
    }

    #[test]
    fn filter_agrees_with_exact() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        for _ in 0..10000 {
            let v: Vec<f64> = (0..6).map(|_| rng.gen_range(-1.0..1.0)).collect();
            // near-degenerate: c ~ a*b/d
            let s = exact_sign!(|F| det2(
                &F::from_f64(v[0]),
                &F::from_f64(v[1]),
                &F::from_f64(v[2]),
                &F::from_f64(v[0] * v[3] / v[1] * (1.0 + v[4] * 1e-15))
            ));
            let e = det2(
                &Expansion::from_f64(v[0]),
                &Expansion::from_f64(v[1]),
                &Expansion::from_f64(v[2]),
                &Expansion::from_f64(v[0] * v[3] / v[1] * (1.0 + v[4] * 1e-15)),
            )
            .sign();
            assert_eq!(s, e);
        }
    }
}
