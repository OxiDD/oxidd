use std::fs;
use std::io::{self, Write as _};

use anyhow::Result;

mod stub_gen;

fn main() -> Result<()> {
    let out_path = "../../bindings/python/oxidd/_oxidd.pyi";

    let mut output = io::BufWriter::new(fs::File::create(out_path)?);
    output.write_all(
        r#"from __future__ import annotations

__all__ = [
    "BDDFunction",
    "BDDManager",
    "BDDSubstitution",
    "BCDDFunction",
    "BCDDManager",
    "BCDDSubstitution",
    "ZBDDFunction",
    "ZBDDManager",
    "DDMemoryError",
    "DuplicateVarName",
    "BooleanOperator",
    "DDDMPFile",
    "DDDMPVersion",
]

import enum
from collections.abc import Iterable
from os import PathLike
from types import TracebackType
from typing import final

from typing_extensions import Never, Self, deprecated

class BooleanOperator(enum.Enum):
    """Binary operators on Boolean functions."""

    AND = ...
    """Conjunction ``lhs ∧ rhs``"""
    OR = ...
    """Disjunction ``lhs ∨ rhs``"""
    XOR = ...
    """Exclusive disjunction ``lhs ⊕ rhs``"""
    EQUIV = ...
    """Equivalence ``lhs ↔ rhs``"""
    NAND = ...
    """Negated conjunction ``lhs ⊼ rhs``"""
    NOR = ...
    """Negated disjunction ``lhs ⊽ rhs``"""
    IMP = ...
    """Implication ``lhs → rhs`` (or `lhs ≤ rhs)`"""
    IMP_STRICT = ...
    """Strict implication ``lhs < rhs``"""

class DDDMPVersion(enum.Enum):
    """DDDMP format version version."""

    V2_0 = ...
    """Version 2.0, bundled with `CUDD <https://github.com/cuddorg/cudd>` 3.0"""
    V3_0 = ...
    """Version 3.0, used by `BDDSampler`_ and `Logic2BDD`_

    .. _BDDSampler: https://github.com/davidfa71/BDDSampler
    .. _Logic2BDD: https://github.com/davidfa71/Extending-Logic
    """
"#
        .as_bytes(),
    )?;

    let mut env = stub_gen::TypeEnv::new();
    {
        let types = &[
            "type",
            "None",
            "bool",
            "int",
            "float",
            "str",
            "tuple",
            "list",
            "range",
            "Never",
            "Self",
            "PathLike",
            "Iterable",
            "BaseException",
            "TracebackType",
            "BooleanOperator",
            "DDDMPVersion",
        ];
        for ty in types {
            env.register_python_type(ty.to_string())?;
        }

        let handle = env.register_python_type("MemoryError".into())?;
        env.register_rust_type("PyMemoryError", handle)?;
    }

    let mut writer = stub_gen::StubGen::new(env);
    writer.process_files([
        "src/bdd.rs",
        "src/bcdd.rs",
        "src/zbdd.rs",
        "src/util/mod.rs",
        "src/util/dddmp.rs",
    ])?;
    writer.write(&mut output)?;
    drop(output);

    // Format the resulting .pyi file using Ruff (if Ruff is installed)
    if let Ok(mut process) = std::process::Command::new("ruff")
        .arg("format")
        .arg(out_path)
        .spawn()
    {
        process.wait().ok();
    }

    Ok(())
}
