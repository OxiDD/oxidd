use crate::util::{get_repr, is_repr_u8};
use proc_macro2::TokenStream;
use proc_macro_error::{abort, emit_error};
use quote::quote;
use syn::spanned::Spanned;

pub fn derive_countable(input: syn::DeriveInput) -> TokenStream {
    let name = &input.ident;

    // We can only derive for `enum`s and structs with zero fields
    match &input.data {
        syn::Data::Enum(enumeration) => {
            // We need `repr(u8)`
            let repr_err = "can only derive `Countable` for `enum`s with `repr(u8)`";
            if let Some(repr_tokens) = get_repr(&input.attrs, "Operation") {
                if !is_repr_u8(repr_tokens.clone()) {
                    emit_error!(repr_tokens.span(), repr_err);
                }
            } else {
                emit_error!(input.span(), repr_err);
            }

            for variant in &enumeration.variants {
                if !variant.fields.is_empty() {
                    emit_error!(
                        variant.fields.span(),
                        "can only derive `Countable` for fieldless `enum`s"
                    );
                }
                if let Some((_, disc)) = &variant.discriminant {
                    emit_error!(
                        disc.span(),
                        "cannot derive `Countable` for `enum`s with explicit discriminants"
                    )
                }
            }
            let num_variants = enumeration.variants.len();
            if num_variants == 0 {
                emit_error!(
                    input.span(),
                    "cannot derive `Countable` for `enum`s with zero variants"
                );
            }

            proc_macro_error::abort_if_dirty();

            let max_value = num_variants - 1;

            // SAFETY of the generated code:
            //
            // We forbid explicit discriminants, and since we have `repr(u8)`
            // and no fields, values of `Self` are simply `u8`s in range
            // `0..num_variants`, or equivalently `0..=max_value`. Due to the
            // assertion, the `transmute` operation is safe.
            quote! {
                unsafe impl ::oxidd_core::Countable for #name {
                    const MAX_VALUE: usize = #max_value;

                    #[inline]
                    fn as_usize(self) -> usize {
                        self as usize
                    }

                    #[inline]
                    fn from_usize(value: usize) -> Self {
                        assert!(value <= Self::MAX_VALUE);
                        unsafe { std::mem::transmute(value as u8) }
                    }
                }
            }
        }
        syn::Data::Struct(structure) => {
            let from_usize_body = match &structure.fields {
                syn::Fields::Named(fields) => {
                    if !fields.named.is_empty() {
                        abort!(
                            fields.span(),
                            "`Countable` can only be derived for `struct`s with zero fields"
                        )
                    }
                    quote!(Self {})
                }
                syn::Fields::Unnamed(fields) => {
                    if !fields.unnamed.is_empty() {
                        abort!(
                            fields.span(),
                            "`Countable` can only be derived for `struct`s with zero fields"
                        )
                    }
                    quote!(Self())
                }
                syn::Fields::Unit => quote!(Self),
            };

            // SAFETY of the generated code: The struct has zero fields, hence
            // there is only one value of that type. There clearly is a
            // bijection to the range `0..=0`.
            quote! {
                unsafe impl ::oxidd_core::Countable for #name {
                    const MAX_VALUE: usize = 0;

                    #[inline]
                    fn as_usize(self) -> usize {
                        0
                    }

                    #[inline]
                    fn from_usize(value: usize) -> Self {
                        #from_usize_body
                    }
                }
            }
        }
        syn::Data::Union(u) => {
            abort!(
                u.union_token.span,
                "`Countable` cannot be derived for `union`s"
            );
        }
    }
}
