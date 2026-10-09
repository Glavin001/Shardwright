use serde::{Deserialize, Serialize};

macro_rules! id_type {
    ($name:ident, $t:ty) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
        #[serde(transparent)]
        pub struct $name(pub $t);
        impl $name {
            #[inline]
            pub fn idx(self) -> usize {
                self.0 as usize
            }
        }
        impl From<usize> for $name {
            fn from(v: usize) -> Self {
                $name(v as $t)
            }
        }
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

id_type!(ComponentId, u32);
id_type!(CellId, u32);
id_type!(InterfaceId, u32);
id_type!(FragmentId, u32);
id_type!(BondId, u32);
id_type!(HullId, u32);
id_type!(MaterialId, u16);

/// Bond endpoint B: another fragment or the world (anchor).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FragmentOrWorld {
    Fragment(FragmentId),
    World,
}

impl FragmentOrWorld {
    pub fn as_i64(self) -> i64 {
        match self {
            FragmentOrWorld::Fragment(f) => f.0 as i64,
            FragmentOrWorld::World => -1,
        }
    }
    pub fn fragment(self) -> Option<FragmentId> {
        match self {
            FragmentOrWorld::Fragment(f) => Some(f),
            FragmentOrWorld::World => None,
        }
    }
}

/// Cell endpoint of an interface: a cell or the world (anchor).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CellOrWorld {
    Cell(CellId),
    World,
}

impl CellOrWorld {
    pub fn cell(self) -> Option<CellId> {
        match self {
            CellOrWorld::Cell(c) => Some(c),
            CellOrWorld::World => None,
        }
    }
}
