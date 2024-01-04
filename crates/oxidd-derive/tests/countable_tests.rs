use oxidd_core::Countable;
use oxidd_derive::Countable;

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug, Countable)]
#[repr(u8)]
pub enum EdgeTag {
    #[default]
    None,
    Complemented,
}

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug, Countable)]
pub struct NoTag;

#[test]
fn test() {
    assert_eq!(EdgeTag::MAX_VALUE, 1);
    for tag in [EdgeTag::None, EdgeTag::Complemented] {
        assert_eq!(tag as usize, tag.as_usize());
        assert_eq!(EdgeTag::from_usize(tag.as_usize()), tag);
    }

    assert_eq!(NoTag::MAX_VALUE, 0);
    assert_eq!(NoTag.as_usize(), 0);
    assert_eq!(NoTag::from_usize(0), NoTag);
}
