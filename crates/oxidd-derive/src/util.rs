use proc_macro_error::abort;
use syn::spanned::Spanned;

/// Extract the value of the `repr(...)` attribute from `attrs`, if present
///
/// Aborts if multiple `repr()` attributes are given. In this case, the error
/// message is:
/// "only one `repr()` attribute allowed for deriving `<trait_name>`"
pub fn get_repr<'a>(
    attrs: &'a [syn::Attribute],
    trait_name: &str,
) -> Option<&'a proc_macro2::TokenStream> {
    let mut repr = None;
    for attr in attrs {
        let syn::AttrStyle::Outer = attr.style else {
            continue;
        };
        if let syn::Meta::List(attr) = &attr.meta {
            if !attr.path.is_ident("repr") {
                continue;
            }
            if repr.is_some() {
                abort!(
                    attr.span(),
                    "only one `repr()` attribute allowed for deriving `{}`",
                    trait_name
                );
            }
            repr = Some(&attr.tokens);
        }
    }
    repr
}

/// Returns `true` if `tokens` is exactly `u8`
pub fn is_repr_u8(tokens: proc_macro2::TokenStream) -> bool {
    let mut iter = tokens.into_iter();
    let Some(proc_macro2::TokenTree::Ident(i)) = iter.next() else {
        return false;
    };
    iter.next().is_none() && i == "u8"
}
