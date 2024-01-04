#[cfg(not(miri))]
#[test]
fn tests() {
    let t = trybuild::TestCases::new();

    // Countable
    #[cfg(feature = "nightly-ui-tests")]
    {
        t.compile_fail("tests/countable/union.rs");
        t.compile_fail("tests/countable/enum_repr_default.rs");
        t.compile_fail("tests/countable/enum_repr_u64.rs");
        t.compile_fail("tests/countable/enum_fields.rs");
        t.compile_fail("tests/countable/enum_discriminants.rs");
        t.compile_fail("tests/countable/enum_zero_variants.rs");
        t.compile_fail("tests/countable/struct_fields.rs");
    }
    t.pass("tests/countable/valid.rs");

    // Function
    #[cfg(feature = "nightly-ui-tests")]
    {
        t.compile_fail("tests/function/enum_union.rs");
        t.compile_fail("tests/function/field_count.rs");
    }
    t.pass("tests/function/valid.rs");
}
