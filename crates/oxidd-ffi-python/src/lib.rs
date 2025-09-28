//! All items of this crate are not intended to be used from other Rust crates
//! but only from Python. Consult the Python API documentation for details on
//! the provided classes.

use pyo3::prelude::*;

// Docstrings in this crate follow the Google Python Style Guide (see
// https://google.github.io/styleguide/pyguide.html and
// https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)

mod bcdd;
mod bdd;
mod zbdd;

mod util;

#[pymodule(gil_used = false)]
mod _oxidd {
    use oxidd::BooleanOperator;
    use oxidd_dump::dddmp;
    use pyo3::prelude::*;

    // bdd
    #[pymodule_export]
    use crate::bdd::BDDFunction;
    #[pymodule_export]
    use crate::bdd::BDDManager;
    #[pymodule_export]
    use crate::bdd::BDDSubstitution;

    // bcdd
    #[pymodule_export]
    use crate::bcdd::BCDDFunction;
    #[pymodule_export]
    use crate::bcdd::BCDDManager;
    #[pymodule_export]
    use crate::bcdd::BCDDSubstitution;

    // zbdd
    #[pymodule_export]
    use crate::zbdd::ZBDDFunction;
    #[pymodule_export]
    use crate::zbdd::ZBDDManager;

    // util
    #[pymodule_export]
    use crate::util::DDDMPFile;
    #[pymodule_export]
    use crate::util::DDMemoryError;
    #[pymodule_export]
    use crate::util::DuplicateVarName;

    fn add_enum(
        m: &Bound<PyModule>,
        enum_cls: &Bound<PyAny>,
        name: &str,
        variants: &[(&str, u8)],
    ) -> PyResult<()> {
        m.add(name, enum_cls.call((name, variants), None)?)
    }

    #[pymodule_init]
    fn init(m: &Bound<PyModule>) -> PyResult<()> {
        let enum_cls = PyModule::import(m.py(), "enum")?.getattr("Enum")?;

        let variants = &[
            ("AND", BooleanOperator::And as u8),
            ("OR", BooleanOperator::Or as u8),
            ("XOR", BooleanOperator::Xor as u8),
            ("EQUIV", BooleanOperator::Equiv as u8),
            ("NAND", BooleanOperator::Nand as u8),
            ("NOR", BooleanOperator::Nor as u8),
            ("IMP", BooleanOperator::Imp as u8),
            ("IMP_STRICT", BooleanOperator::ImpStrict as u8),
        ];
        add_enum(m, &enum_cls, "BooleanOperator", variants)?;

        let variants = &[
            ("V2_0", dddmp::DDDMPVersion::V2_0 as u8),
            ("V3_0", dddmp::DDDMPVersion::V3_0 as u8),
        ];
        add_enum(m, &enum_cls, "DDDMPVersion", variants)?;

        Ok(())
    }
}
