//! Three-valued bit vector

use std::fmt::{self, Write};

/// Three-valued bit vector
#[derive(Clone, PartialEq, Eq, Default)]
pub struct TVBitVec {
    data: Vec<u32>,
    len: usize,
}

impl TVBitVec {
    const BITS_PER_ELEMENT: usize = 2;
    const ELEMENTS_PER_BLOCK: usize = u32::BITS as usize / Self::BITS_PER_ELEMENT;

    #[allow(unused)] // used in tests
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_capacity(len: usize) -> Self {
        Self {
            data: Vec::with_capacity(len.div_ceil(Self::ELEMENTS_PER_BLOCK)),
            len,
        }
    }

    pub fn push(&mut self, value: Option<bool>) {
        let bits = match value {
            Some(v) => 0b10 | v as u32,
            None => 0b00,
        };
        let offset = self.data.len() % Self::ELEMENTS_PER_BLOCK;
        if offset == 0 {
            self.data.push(bits);
        } else {
            let block = self.data.last_mut().unwrap();
            *block |= bits << (Self::BITS_PER_ELEMENT * offset);
        }
    }

    pub fn at(&self, index: usize) -> Option<bool> {
        if index >= self.len {
            panic!(
                "index out of bounds: the len is {} but the index is {index}",
                self.len
            );
        }
        let block = self.data[index / Self::ELEMENTS_PER_BLOCK];
        let i = index % Self::ELEMENTS_PER_BLOCK;
        if block & (1 << (i + 1)) != 0 {
            Some(block & (1 << i) != 0)
        } else {
            None
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
        f.write_char(match self {
            Self::None => '-',
            Self::False => '0',
            Self::True => '1',
        })
    }
}

impl fmt::Debug for TVBitVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries((0..self.len).map(|i| {
                let block = self.data[i / Self::ELEMENTS_PER_BLOCK];
                let i = i % Self::ELEMENTS_PER_BLOCK;
                if block & (1 << (i + 1)) != 0 {
                    if block & (1 << i) != 0 {
                        OptBoolDebug::True
                    } else {
                        OptBoolDebug::False
                    }
                } else {
                    OptBoolDebug::None
                }
            }))
            .finish()
    }
}
