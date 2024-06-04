//! Three-valued [`BitVec`]

use std::fmt;

use bitvec::vec::BitVec;

/// Three-valued [`BitVec`]
#[derive(Clone, PartialEq, Eq, Default)]
pub struct TVBitVec(BitVec);

impl TVBitVec {
    #[allow(unused)] // used in tests
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_capacity(len: usize) -> Self {
        Self(BitVec::with_capacity(2 * len))
    }

    pub fn push(&mut self, value: Option<bool>) {
        let (b0, b1) = match value {
            Some(v) => (true, v),
            None => (false, false),
        };
        self.0.push(b0);
        self.0.push(b1);
    }

    #[allow(unused)]
    pub fn set(&mut self, index: usize, value: Option<bool>) {
        let (b0, b1) = match value {
            Some(v) => (true, v),
            None => (false, false),
        };
        self.0.set(2 * index, b0);
        self.0.set(2 * index + 1, b1);
    }
}

impl std::ops::Index<usize> for TVBitVec {
    type Output = Option<bool>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        if self.0[2 * index] {
            if self.0[2 * index + 1] {
                &Some(true)
            } else {
                &Some(false)
            }
        } else {
            &None
        }
    }
}

/// Helper for nice Debug output
#[derive(Clone, Copy)]
enum OptBoolDebug {
    None,
    False,
    True,
}

impl fmt::Debug for OptBoolDebug {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "-"),
            Self::False => write!(f, "0"),
            Self::True => write!(f, "1"),
        }
    }
}

impl fmt::Debug for TVBitVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.0.chunks(2).map(|bits| {
                if !bits[0] {
                    OptBoolDebug::None
                } else if bits[1] {
                    OptBoolDebug::True
                } else {
                    OptBoolDebug::False
                }
            }))
            .finish()
    }
}
