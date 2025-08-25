//! Error types

use std::{fmt, ops::Range};

use crate::VarNo;

/// Out of memory error
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct OutOfMemory;

impl fmt::Display for OutOfMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("decision diagram operation ran out of memory")
    }
}
impl std::error::Error for OutOfMemory {}

impl From<OutOfMemory> for std::io::Error {
    fn from(_: OutOfMemory) -> Self {
        std::io::ErrorKind::OutOfMemory.into()
    }
}

/// Error details for labelling a variable with a name that is already in use
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct DuplicateVarName {
    /// The variable name
    pub name: String,
    /// Variable number already using the name
    pub present_var: VarNo,
    /// Range of variables that have been successfully added before the error
    /// occurred
    pub added_vars: Range<VarNo>,
}

impl fmt::Display for DuplicateVarName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the variable name '{}' is already in use for variable number {}",
            self.name, self.present_var
        )
    }
}
