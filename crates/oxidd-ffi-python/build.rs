use std::fs;
use std::io::{self, Write as _};

use anyhow::Result;

mod stub_gen;

fn main() -> Result<()> {
    let out_path = "../../bindings/python/oxidd/_oxidd.pyi";

    let mut output = io::BufWriter::new(fs::File::create(out_path)?);
    output.write_all(
        "\
from __future__ import annotations

__all__ = [
    \"BDDFunction\",
    \"BDDManager\",
    \"BDDSubstitution\",
    \"BCDDFunction\",
    \"BCDDManager\",
    \"BCDDSubstitution\",
    \"ZBDDFunction\",
    \"ZBDDManager\",
    \"DDMemoryError\",
    \"BooleanOperator\",
]

import enum
from collections.abc import Iterable
from os import PathLike
from typing import final

from typing_extensions import Never, Self, deprecated
"
        .as_bytes(),
    )?;

    let mut env = stub_gen::TypeEnv::new();
    {
        let types = [
            "None",
            "bool",
            "int",
            "float",
            "str",
            "tuple",
            "list",
            "Never",
            "Self",
            "PathLike",
            "Iterable",
            "BooleanOperator",
        ];
        for ty in types {
            env.register_python_type(ty.to_string())?;
        }

        let handle = env.register_python_type("MemoryError".into())?;
        env.register_rust_type("PyMemoryError", handle)?;
    }

    let mut writer = stub_gen::StubGen::new(env);
    writer.process_files(["src/bdd.rs", "src/bcdd.rs", "src/zbdd.rs", "src/util.rs"])?;
    writer.write(&mut output)?;
    output.write_all(
        "
class BooleanOperator(enum.Enum):
    \"\"\"Binary operators on Boolean functions.\"\"\"

    AND = ...
    \"\"\"Conjunction ``lhs ∧ rhs``\"\"\"
    OR = ...
    \"\"\"Disjunction ``lhs ∨ rhs``\"\"\"
    XOR = ...
    \"\"\"Exclusive disjunction ``lhs ⊕ rhs``\"\"\"
    EQUIV = ...
    \"\"\"Equivalence ``lhs ↔ rhs``\"\"\"
    NAND = ...
    \"\"\"Negated conjunction ``lhs ⊼ rhs``\"\"\"
    NOR = ...
    \"\"\"Negated disjunction ``lhs ⊽ rhs``\"\"\"
    IMP = ...
    \"\"\"Implication ``lhs → rhs`` (or `lhs ≤ rhs)`\"\"\"
    IMP_STRICT = ...
    \"\"\"Strict implication ``lhs < rhs``\"\"\"
"
        .as_bytes(),
    )?;
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
