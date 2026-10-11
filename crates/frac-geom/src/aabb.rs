use glam::DVec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Aabb {
    pub min: DVec3,
    pub max: DVec3,
}

impl Default for Aabb {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb {
        min: DVec3::splat(f64::INFINITY),
        max: DVec3::splat(f64::NEG_INFINITY),
    };
    pub fn from_points<'a>(pts: impl IntoIterator<Item = &'a DVec3>) -> Aabb {
        let mut b = Aabb::EMPTY;
        for p in pts {
            b.grow(*p);
        }
        b
    }
    #[inline]
    pub fn grow(&mut self, p: DVec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }
    #[inline]
    pub fn union(&self, o: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(o.min),
            max: self.max.max(o.max),
        }
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }
    #[inline]
    pub fn overlaps(&self, o: &Aabb) -> bool {
        self.min.x <= o.max.x
            && self.max.x >= o.min.x
            && self.min.y <= o.max.y
            && self.max.y >= o.min.y
            && self.min.z <= o.max.z
            && self.max.z >= o.min.z
    }
    #[inline]
    pub fn contains(&self, p: DVec3) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }
    pub fn center(&self) -> DVec3 {
        (self.min + self.max) * 0.5
    }
    pub fn extent(&self) -> DVec3 {
        self.max - self.min
    }
    pub fn diagonal(&self) -> f64 {
        self.extent().length()
    }
    pub fn expanded(&self, r: f64) -> Aabb {
        Aabb {
            min: self.min - DVec3::splat(r),
            max: self.max + DVec3::splat(r),
        }
    }
    /// Squared distance from a point to the box (0 if inside).
    pub fn dist2(&self, p: DVec3) -> f64 {
        let d = (self.min - p).max(DVec3::ZERO).max(p - self.max);
        d.length_squared()
    }
    pub fn longest_axis(&self) -> usize {
        let e = self.extent();
        if e.x >= e.y && e.x >= e.z {
            0
        } else if e.y >= e.z {
            1
        } else {
            2
        }
    }
}
