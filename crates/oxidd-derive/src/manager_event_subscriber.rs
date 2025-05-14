use proc_macro2::{Literal, Span, TokenStream, TokenTree};
use proc_macro_error::{abort, emit_warning};
use quote::{quote, ToTokens};

pub fn derive_manager_event_subscriber(input: syn::DeriveInput) -> TokenStream {
    let mut manager_ty: Option<syn::Type> = None;
    let mut trait_bounds = true;
    for attr in &input.attrs {
        if attr.path().is_ident("subscribe") {
            let res = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("manager") {
                    if manager_ty.is_none() {
                        manager_ty = Some(meta.value()?.parse()?);
                    } else {
                        proc_macro_error::emit_error!(
                            meta.input.span(),
                            "`manager` attribute given multiple times"
                        );
                    }
                } else if meta.path.is_ident("no_trait_bounds") {
                    trait_bounds = false;
                } else {
                    emit_warning!(
                        meta.input.span(),
                        "ignoring unknown attribute `{}`",
                        meta.input
                    );
                }
                Ok(())
            });
            if let Err(err) = res {
                return err.into_compile_error();
            }
        }
    }

    let only_structs_msg = "`ManagerEventSubscriber` can only be derived for `struct`s";
    let s = match &input.data {
        syn::Data::Struct(s) => s,
        syn::Data::Enum(e) => abort!(e.enum_token.span, only_structs_msg),
        syn::Data::Union(u) => abort!(u.union_token.span, only_structs_msg),
    };

    let Some(manager_ty) = manager_ty else {
        proc_macro_error::abort_call_site!(
            "Missing manager attribute";
            hint = "To implement `ManagerEventSubscriber<MyManagerType>`, add an attribute `#[subscribe(manager = MyManagerType)]` to the struct"
        );
    };
    let subscriber_trait = quote!(::oxidd_core::ManagerEventSubscriber<#manager_ty>);

    let default_span = Span::call_site();
    let body_item = |s, is_unsafe| {
        (
            proc_macro2::Ident::new(s, default_span),
            TokenStream::new(),
            is_unsafe,
        )
    };
    let mut bodies = [
        body_item("init", false),
        body_item("pre_gc", false),
        body_item("post_gc", true),
        body_item("pre_reorder", false),
        body_item("post_reorder", false),
    ];
    let mut bodies_mut = [
        body_item("init_mut", false),
        body_item("pre_reorder_mut", false),
        body_item("post_reorder_mut", false),
    ];
    let mut where_predicates = TokenStream::new();

    'outer: for (i, field) in s.fields.iter().enumerate() {
        for attr in &field.attrs {
            if attr.path().is_ident("subscribe") {
                let mut skip = false;
                let res = attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("skip") || meta.path.is_ident("ignore") {
                        skip = true;
                    } else {
                        emit_warning!(
                            meta.input.span(),
                            "ignoring unknown attribute `{}`",
                            meta.input
                        );
                    }
                    Ok(())
                });
                if let Err(err) = res {
                    return err.into_compile_error();
                }
                if skip {
                    continue 'outer;
                }
            }
        }

        let field_id = match &field.ident {
            Some(id) => TokenTree::Ident(id.clone()),
            None => TokenTree::Literal(Literal::usize_unsuffixed(i)),
        };
        let ty = &field.ty;
        if trait_bounds {
            where_predicates.extend(quote!(#ty: #subscriber_trait,));
        }

        for (f, ts, _) in &mut bodies {
            ts.extend(quote! {
                <#ty as #subscriber_trait>::#f(&self.#field_id, manager);
            });
        }
        for (f, ts, _) in &mut bodies_mut {
            ts.extend(quote! {
                <#ty as #subscriber_trait>::#f(manager);
            });
        }
    }

    proc_macro_error::abort_if_dirty();

    let methods = bodies.into_iter().map(|(id, body, is_unsafe)| {
        if is_unsafe {
            // SAFETY of the method calls in `#body` is ensured by the caller
            quote! {
                unsafe fn #id(&self, manager: &#manager_ty) {
                    unsafe { #body }
                }
            }
        } else {
            quote! {
                fn #id(&self, manager: &#manager_ty) {
                    #body
                }
            }
        }
    });
    let methods_mut = bodies_mut.into_iter().map(|(id, body, is_unsafe)| {
        debug_assert!(!is_unsafe); // currently, there is no such unsafe method
        quote! {
            fn #id(manager: &mut #manager_ty) {
                #body
            }
        }
    });

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let where_clause = if trait_bounds {
        let generics_where_clauses = where_clause.map(|wc| &wc.predicates);
        quote! {
            where
                #manager_ty: ::oxidd_core::Manager,
                #where_predicates
                #generics_where_clauses
        }
    } else {
        where_clause.to_token_stream()
    };

    let name = &input.ident;
    quote! {
        impl #impl_generics ::oxidd_core::ManagerEventSubscriber<#manager_ty> for #name #ty_generics #where_clause
        {
            #(#methods)*
            #(#methods_mut)*
        }
    }
}
