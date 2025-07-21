//! Im- and export to the [DDDMP] format used by CUDD
//!
//! Currently, versions 2.0 and 3.0 are supported, where version 2.0 is the
//! version bundled with the CUDD 3.0 release, while version 3.0 was probably
//! defined by the authors of tools like [BDDSampler] and [Logic2BDD] and solely
//! adds the `.varnames` field.
//!
//! [DDDMP]: https://github.com/ssoelvsten/cudd/tree/main/dddmp
//! [BDDSampler]: https://github.com/davidfa71/BDDSampler
//! [Logic2BDD]: https://github.com/davidfa71/Extending-Logic

// spell-checker:ignore varnames

mod import;
pub use import::{import, DumpHeader};
mod export;
pub use export::{DDDMPVersion, ExportSettings};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum VarInfo {
    VariableID,
    PermutationID,
    AuxiliaryID,
    VariableName,
    None,
}

/// Encoding of the variable ID and then/else edges in binary format
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
enum Code {
    Terminal,
    AbsoluteID,
    RelativeID,
    Relative1,
}

impl From<u8> for Code {
    fn from(value: u8) -> Self {
        match value {
            0 => Code::Terminal,
            1 => Code::AbsoluteID,
            2 => Code::RelativeID,
            3 => Code::Relative1,
            _ => panic!(),
        }
    }
}
