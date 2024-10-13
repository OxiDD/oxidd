use proc_macro2::{Span, TokenStream};
use proc_macro_error::{abort, emit_error};
use quote::{quote, ToTokens};
use syn::{spanned::Spanned, Token};

struct StructField {
    ident: TokenStream,
    ty: syn::Type,
    named: bool,
}

impl StructField {
    fn gen_from_inner(&self, inner: TokenStream) -> TokenStream {
        let ident = &self.ident;
        if self.named {
            quote!(Self { #ident: #inner })
        } else {
            quote!(Self(#inner))
        }
    }

    fn single_from_item(item: syn::Data, trait_name: &str) -> Self {
        let structure = match item {
            syn::Data::Struct(s) => s,
            syn::Data::Union(syn::DataUnion {
                union_token: syn::token::Union { span },
                ..
            })
            | syn::Data::Enum(syn::DataEnum {
                enum_token: syn::token::Enum { span },
                ..
            }) => {
                abort!(span, "`{}` can only be derived for `struct`s", trait_name);
            }
        };

        match structure.fields {
            syn::Fields::Named(syn::FieldsNamed {
                named: mut fields, ..
            })
            | syn::Fields::Unnamed(syn::FieldsUnnamed {
                unnamed: mut fields,
                ..
            }) => {
                if fields.len() != 1 {
                    abort!(
                        fields.span(),
                        "can only derive `{}` for `struct`s with precisely one field",
                        trait_name
                    );
                }
                let field = fields.pop().unwrap().into_value();
                let ty = field.ty;
                match &field.ident {
                    Some(ident) => Self {
                        ident: ident.to_token_stream(),
                        ty,
                        named: true,
                    },
                    None => Self {
                        ident: quote!(0),
                        ty,
                        named: false,
                    },
                }
            }
            syn::Fields::Unit => abort!(
                Span::call_site(),
                "can only derive `{}` for `struct`s with precisely one field",
                trait_name
            ),
        }
    }
}

struct UseManagerRefAttr {
    ty: syn::Type,
    #[allow(unused)]
    delim: Token![,],
    conversion_expr: syn::Expr,
}

impl syn::parse::Parse for UseManagerRefAttr {
    fn parse(input: syn::parse::ParseStream) -> syn::parse::Result<Self> {
        let span = input.span();
        let res = Self {
            ty: input.parse()?,
            delim: input.parse()?,
            conversion_expr: input.parse()?,
        };

        if input.is_empty() || (input.parse::<Token![,]>().is_ok() && input.is_empty()) {
            Ok(res)
        } else {
            Err(syn::parse::Error::new(
                span,
                "expected `ManagerRefType, expr_to_convert_from(inner)`",
            ))
        }
    }
}

pub fn derive_function(input: syn::DeriveInput) -> TokenStream {
    let ident = input.ident;

    let mut manager_ref_attr = None;
    for attr in input.attrs {
        let syn::AttrStyle::Outer = attr.style else {
            continue;
        };
        let meta = attr.meta;
        if meta.path().is_ident("use_manager_ref") {
            if manager_ref_attr.is_some() {
                emit_error!(
                    meta.span(),
                    "the `use_manager_ref` attribute may only be given once per item"
                );
            }
            let mut span = meta.span();
            if let syn::Meta::List(ml) = &meta {
                span = ml.tokens.span();
                if let Ok(attr) = ml.parse_args::<UseManagerRefAttr>() {
                    manager_ref_attr = Some(attr);
                    continue;
                }
            }
            emit_error!(
                span,
                "expected `#[use_manager_ref(ManagerRefType, expr_to_convert_from(inner))]`"
            );
        }
    }

    let struct_field = StructField::single_from_item(input.data, "Function");
    proc_macro_error::abort_if_dirty();
    let ty = &struct_field.ty;
    let from_edge_body = struct_field.gen_from_inner(
        quote!(<#ty as ::oxidd_core::function::Function>::from_edge(manager, edge)),
    );
    let field = struct_field.ident;

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    // Note: We don't add any trait bounds like `#ty: Function`. Adding those
    // does not seem to work well for the way we instantiate MTBDDs (i.e.
    // `Function`s that are generic over the terminal type).

    let (manager_ref, manager_ref_expr) = match manager_ref_attr {
        Some(UseManagerRefAttr {
            ty,
            conversion_expr,
            ..
        }) => (ty.into_token_stream(), conversion_expr.into_token_stream()),
        None => (
            quote!(<#ty as ::oxidd_core::function::Function>::ManagerRef),
            quote!(inner),
        ),
    };

    // SAFETY of the generated implementation is inherited from the inner
    // `Function` implementation
    quote! {
        unsafe impl #impl_generics ::oxidd_core::function::Function for #ident #ty_generics #where_clause {
            type Manager<'__id> = <#ty as ::oxidd_core::function::Function>::Manager<'__id>;

            type ManagerRef = #manager_ref;

            #[inline]
            fn from_edge<'__id>(
                manager: &Self::Manager<'__id>,
                edge: <Self::Manager<'__id> as ::oxidd_core::Manager>::Edge,
            ) -> Self {
                #from_edge_body
            }

            #[inline]
            fn as_edge<'__id>(
                &self,
                manager: &Self::Manager<'__id>,
            ) -> &<Self::Manager<'__id> as ::oxidd_core::Manager>::Edge {
                self.#field.as_edge(manager)
            }

            #[inline]
            fn into_edge<'__id>(self, manager: &Self::Manager<'__id>) -> <Self::Manager<'__id> as ::oxidd_core::Manager>::Edge {
                self.#field.into_edge(manager)
            }

            #[inline]
            fn manager_ref(&self) -> #manager_ref {
                let inner = self.#field.manager_ref();
                #manager_ref_expr
            }

            #[inline]
            fn with_manager_shared<__F, __T>(&self, f: __F) -> __T
            where
                __F: for<'__id> ::std::ops::FnOnce(&Self::Manager<'__id>, &<Self::Manager<'__id> as ::oxidd_core::Manager>::Edge) -> __T,
            {
                self.#field.with_manager_shared(f)
            }

            #[inline]
            fn with_manager_exclusive<__F, __T>(&self, f: __F) -> __T
            where
                __F: for<'__id> ::std::ops::FnOnce(&mut Self::Manager<'__id>, &<Self::Manager<'__id> as ::oxidd_core::Manager>::Edge) -> __T,
            {
                self.#field.with_manager_exclusive(f)
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Method {
    Terminal(&'static str),
    NewVar(&'static str),
    Unary(&'static str),
    UnaryOwned(&'static str),
    Binary(&'static str),
    Ternary(&'static str),
}

impl Method {
    fn expand(
        self,
        trait_path: &TokenStream,
        manager_ty: &TokenStream,
        edge_ty: &TokenStream,
        struct_field: &StructField,
    ) -> TokenStream {
        let field = &struct_field.ident;
        let inner = &struct_field.ty;

        match self {
            Method::Terminal(n) => {
                let method = syn::Ident::new(n, Span::call_site());
                let method_edge = syn::Ident::new(&format!("{n}_edge"), Span::call_site());
                let func =
                    struct_field.gen_from_inner(quote!(<#inner as #trait_path>::#method(manager)));

                quote! {
                    #[inline]
                    fn #method<'__id>(manager: &#manager_ty) -> Self {
                        #func
                    }
                    #[inline]
                    fn #method_edge<'__id>(manager: &#manager_ty) -> #edge_ty {
                        <#inner as #trait_path>::#method_edge(manager)
                    }
                }
            }

            Method::NewVar(n) => {
                let method = syn::Ident::new(n, Span::call_site());
                let func =
                    struct_field.gen_from_inner(quote!(<#inner as #trait_path>::#method(manager)?));

                quote! {
                    #[inline]
                    fn #method<'__id>(manager: &mut #manager_ty) -> ::oxidd_core::util::AllocResult<Self> {
                        ::std::result::Result::Ok(#func)
                    }
                }
            }

            Method::Unary(n) => {
                let method = syn::Ident::new(n, Span::call_site());
                let method_edge = syn::Ident::new(&format!("{n}_edge"), Span::call_site());
                let func = struct_field.gen_from_inner(quote!(#trait_path::#method(&self.#field)?));

                quote! {
                    #[inline]
                    fn #method(&self) -> ::oxidd_core::util::AllocResult<Self> {
                        ::std::result::Result::Ok(#func)
                    }
                    #[inline]
                    fn #method_edge<'__id>(manager: &#manager_ty, edge: &#edge_ty) -> ::oxidd_core::util::AllocResult<#edge_ty> {
                        <#inner as #trait_path>::#method_edge(manager, edge)
                    }
                }
            }

            Method::UnaryOwned(n) => {
                let method = syn::Ident::new(&format!("{n}_owned"), Span::call_site());
                let method_edge = syn::Ident::new(&format!("{n}_edge_owned"), Span::call_site());
                let func = struct_field.gen_from_inner(quote!(#trait_path::#method(self.#field)?));

                quote! {
                    #[inline]
                    fn #method(self) -> ::oxidd_core::util::AllocResult<Self> {
                        ::std::result::Result::Ok(#func)
                    }
                    #[inline]
                    fn #method_edge<'__id>(manager: &#manager_ty, edge: #edge_ty) -> ::oxidd_core::util::AllocResult<#edge_ty> {
                        <#inner as #trait_path>::#method_edge(manager, edge)
                    }
                }
            }

            Method::Binary(n) => {
                let method = syn::Ident::new(n, Span::call_site());
                let method_edge = syn::Ident::new(&format!("{n}_edge"), Span::call_site());
                let func = struct_field
                    .gen_from_inner(quote!(#trait_path::#method(&self.#field, &rhs.#field)?));

                quote! {
                    #[inline]
                    fn #method(&self, rhs: &Self) -> ::oxidd_core::util::AllocResult<Self> {
                        ::std::result::Result::Ok(#func)
                    }
                    #[inline]
                    fn #method_edge<'__id>(manager: &#manager_ty, lhs: &#edge_ty, rhs: &#edge_ty) -> ::oxidd_core::util::AllocResult<#edge_ty> {
                        <#inner as #trait_path>::#method_edge(manager, lhs, rhs)
                    }
                }
            }

            Method::Ternary(n) => {
                let method = syn::Ident::new(n, Span::call_site());
                let method_edge = syn::Ident::new(&format!("{n}_edge"), Span::call_site());
                let func = struct_field.gen_from_inner(
                    quote!(#trait_path::#method(&self.#field, &f1.#field, &f2.#field)?),
                );

                quote! {
                    #[inline]
                    fn #method(&self, f1: &Self, f2: &Self) -> ::oxidd_core::util::AllocResult<Self> {
                        ::std::result::Result::Ok(#func)
                    }
                    #[inline]
                    fn #method_edge<'__id>(manager: &#manager_ty, e0: &#edge_ty, e1: &#edge_ty, e2: &#edge_ty) -> ::oxidd_core::util::AllocResult<#edge_ty> {
                        <#inner as #trait_path>::#method_edge(manager, e0, e1, e2)
                    }
                }
            }
        }
    }
}

struct CustomMethodsCtx<'a> {
    trait_path: &'a TokenStream,
    manager_ty: &'a TokenStream,
    edge_ty: &'a TokenStream,
    struct_field: StructField,
}

fn derive_function_trait(
    input: syn::DeriveInput,
    trait_name: &str,
    methods: &[Method],
    custom_methods: impl for<'a> FnOnce(CustomMethodsCtx<'a>) -> TokenStream,
) -> TokenStream {
    let ident = input.ident;
    let struct_field = StructField::single_from_item(input.data, trait_name);
    proc_macro_error::abort_if_dirty();
    let trait_ident = syn::Ident::new(trait_name, Span::call_site());
    let trait_path = quote!(::oxidd_core::function::#trait_ident);

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    // Note: We don't add any trait bounds like `Self: Function`. Adding those
    // does not seem to work well for the way we instantiate MTBDDs (i.e.,
    // `Function`s that are generic over the terminal type).

    let manager_ty = quote!(<Self as ::oxidd_core::function::Function>::Manager<'__id>);
    let edge_ty = quote!(<#manager_ty as ::oxidd_core::Manager>::Edge);

    let methods: TokenStream = methods
        .iter()
        .map(|method| method.expand(&trait_path, &manager_ty, &edge_ty, &struct_field))
        .collect();
    let custom_methods = custom_methods(CustomMethodsCtx {
        trait_path: &trait_path,
        manager_ty: &manager_ty,
        edge_ty: &edge_ty,
        struct_field,
    });

    quote! {
        impl #impl_generics #trait_path for #ident #ty_generics #where_clause {
            #methods
            #custom_methods
        }
    }
}

pub fn derive_function_subst(input: syn::DeriveInput) -> TokenStream {
    derive_function_trait(input, "FunctionSubst", &[], |ctx| {
        let CustomMethodsCtx {
            trait_path,
            manager_ty,
            edge_ty,
            struct_field,
        } = ctx;
        let field = &struct_field.ident;
        let inner = &struct_field.ty;

        let from_res = struct_field.gen_from_inner(quote!(res));

        quote! {
            #[inline]
            fn substitute<'__a>(
                &'__a self,
                substitution: impl ::oxidd_core::util::Substitution<Var = &'__a Self, Replacement = &'__a Self>,
            ) -> ::oxidd_core::util::AllocResult<Self> {
                let res = <#inner as #trait_path>::substitute(
                    &self.#field,
                    ::oxidd_core::util::Substitution::map(
                        &substitution,
                        |(v, r)| (&v.#field, &r.#field),
                    ),
                )?;
                ::std::result::Result::Ok(#from_res)
            }

            #[inline]
            fn substitute_edge<'__id, '__a>(
                manager: &'__a #manager_ty,
                edge: &'__a #edge_ty,
                substitution: impl ::oxidd_core::util::Substitution<
                    Var = ::oxidd_core::util::Borrowed<'__a, #edge_ty>,
                    Replacement = ::oxidd_core::util::Borrowed<'__a, #edge_ty>,
                >,
            ) -> ::oxidd_core::util::AllocResult<#edge_ty> {
                <#inner as #trait_path>::substitute_edge(manager, edge, substitution)
            }
        }
    })
}

pub fn derive_boolean_function(input: syn::DeriveInput) -> TokenStream {
    use Method::*;
    derive_function_trait(
        input,
        "BooleanFunction",
        &[
            Terminal("f"),
            Terminal("t"),
            NewVar("new_var"),
            Unary("not"),
            UnaryOwned("not"),
            Binary("and"),
            Binary("or"),
            Binary("nand"),
            Binary("nor"),
            Binary("xor"),
            Binary("equiv"),
            Binary("imp"),
            Binary("imp_strict"),
            Ternary("ite"),
            Binary("pick_cube_dd_set"),
        ],
        |ctx| {
            let CustomMethodsCtx {
                trait_path,
                manager_ty,
                edge_ty,
                struct_field,
            } = ctx;
            let field = &struct_field.ident;
            let inner = &struct_field.ty;

            let from_ft = struct_field.gen_from_inner(quote!(ft));
            let from_ff = struct_field.gen_from_inner(quote!(ff));
            let from_res = struct_field.gen_from_inner(quote!(res));

            quote! {
                #[inline]
                fn cofactors(&self) -> ::std::option::Option<(Self, Self)> {
                    let (ft, ff) = <#inner as #trait_path>::cofactors(&self.#field)?;
                    ::std::option::Option::Some((#from_ft, #from_ff))
                }
                #[inline]
                fn cofactor_true(&self) -> ::std::option::Option<Self> {
                    let ft = <#inner as #trait_path>::cofactor_true(&self.#field)?;
                    ::std::option::Option::Some(#from_ft)
                }
                #[inline]
                fn cofactor_false(&self) -> ::std::option::Option<Self> {
                    let ff = <#inner as #trait_path>::cofactor_false(&self.#field)?;
                    ::std::option::Option::Some(#from_ff)
                }
                #[inline]
                fn cofactors_edge<'__a, '__id>(
                    manager: &'__a #manager_ty,
                    f: &'__a #edge_ty,
                ) -> ::std::option::Option<(
                    ::oxidd_core::util::Borrowed<'__a, #edge_ty>,
                    ::oxidd_core::util::Borrowed<'__a, #edge_ty>,
                )> {
                    <#inner as #trait_path>::cofactors_edge(manager, f)
                }

                #[inline]
                fn cofactors_node<'__a, '__id>(
                    tag: ::oxidd_core::function::ETagOfFunc<'__id, Self>,
                    node: &'__a ::oxidd_core::function::INodeOfFunc<'__id, Self>,
                ) -> (
                    ::oxidd_core::util::Borrowed<'__a, #edge_ty>,
                    ::oxidd_core::util::Borrowed<'__a, #edge_ty>,
                ) {
                    <#inner as #trait_path>::cofactors_node(tag, node)
                }

                #[inline]
                fn satisfiable(&self) -> bool {
                    <#inner as #trait_path>::satisfiable(&self.#field)
                }
                #[inline]
                fn valid(&self) -> bool {
                    <#inner as #trait_path>::valid(&self.#field)
                }

                #[inline]
                fn sat_count<__N, __S>(
                    &self,
                    vars: ::oxidd_core::LevelNo,
                    cache: &mut ::oxidd_core::util::SatCountCache<__N, __S>,
                ) -> __N
                where
                    __N: ::oxidd_core::util::SatCountNumber,
                    __S: ::std::hash::BuildHasher,
                {
                    <#inner as #trait_path>::sat_count(&self.#field, vars, cache)
                }

                #[inline]
                fn sat_count_edge<'__id, __N, __S>(
                    manager: &#manager_ty,
                    edge: &#edge_ty,
                    vars: ::oxidd_core::LevelNo,
                    cache: &mut ::oxidd_core::util::SatCountCache<__N, __S>,
                ) -> __N
                where
                    __N: ::oxidd_core::util::SatCountNumber,
                    __S: ::std::hash::BuildHasher,
                {
                    <#inner as #trait_path>::sat_count_edge(manager, edge, vars, cache)
                }

                #[inline]
                fn pick_cube<'__a, __I: ::std::iter::ExactSizeIterator<Item = &'__a Self>>(
                    &'__a self,
                    order: impl ::std::iter::IntoIterator<IntoIter = __I>,
                    choice: impl for<'__id> ::std::ops::FnMut(&#manager_ty, &#edge_ty, ::oxidd_core::LevelNo) -> bool,
                ) -> ::std::option::Option<::std::vec::Vec<::oxidd_core::util::OptBool>> {
                    <#inner as #trait_path>::pick_cube(&self.#field, order.into_iter().map(|f| &f.#field), choice)
                }

                #[inline]
                fn pick_cube_edge<'__id, '__a, __I>(
                    manager: &'__a #manager_ty,
                    edge: &'__a #edge_ty,
                    order: impl ::std::iter::IntoIterator<IntoIter = __I>,
                    choice: impl ::std::ops::FnMut(&#manager_ty, &#edge_ty, ::oxidd_core::LevelNo) -> bool,
                ) -> ::std::option::Option<::std::vec::Vec<::oxidd_core::util::OptBool>>
                where
                    __I: ::std::iter::ExactSizeIterator<Item = &'__a #edge_ty>,
                {
                    <#inner as #trait_path>::pick_cube_edge(manager, edge, order, choice)
                }

                #[inline]
                fn pick_cube_dd(
                    &self,
                    choice: impl for<'__id> ::std::ops::FnMut(&#manager_ty, &#edge_ty, ::oxidd_core::LevelNo) -> bool,
                ) -> ::oxidd_core::util::AllocResult<Self> {
                    let res = <#inner as #trait_path>::pick_cube_dd(&self.#field, choice)?;
                    Ok(#from_res)
                }

                #[inline]
                fn pick_cube_dd_edge<'__id>(
                    manager: &#manager_ty,
                    edge: &#edge_ty,
                    choice: impl ::std::ops::FnMut(&#manager_ty, &#edge_ty, ::oxidd_core::LevelNo) -> bool,
                ) -> ::oxidd_core::util::AllocResult<#edge_ty> {
                    <#inner as #trait_path>::pick_cube_dd_edge(manager, edge, choice)
                }

                #[inline]
                fn pick_cube_uniform<'__a, I: ::std::iter::ExactSizeIterator<Item = &'__a Self>, S: ::std::hash::BuildHasher>(
                    &'__a self,
                    order: impl ::std::iter::IntoIterator<IntoIter = I>,
                    cache: &mut ::oxidd_core::util::SatCountCache<::oxidd_core::util::num::F64, S>,
                    rng: &mut ::oxidd_core::util::Rng,
                ) -> ::std::option::Option<::std::vec::Vec<::oxidd_core::util::OptBool>> {
                    <#inner as #trait_path>::pick_cube_uniform(&self.#field, order.into_iter().map(|f| &f.#field), cache, rng)
                }

                #[inline]
                fn pick_cube_uniform_edge<'__id, '__a, I, S>(
                    manager: &'__a #manager_ty,
                    edge: &'__a #edge_ty,
                    order: impl ::std::iter::IntoIterator<IntoIter = I>,
                    cache: &mut ::oxidd_core::util::SatCountCache<::oxidd_core::util::num::F64, S>,
                    rng: &mut ::oxidd_core::util::Rng,
                ) -> ::std::option::Option<::std::vec::Vec<::oxidd_core::util::OptBool>>
                where
                    I: ::std::iter::ExactSizeIterator<Item = &'__a #edge_ty>,
                    S: ::std::hash::BuildHasher
                {
                    <#inner as #trait_path>::pick_cube_uniform_edge(manager, edge, order, cache, rng)
                }

                #[inline]
                fn eval<'__a>(
                    &'__a self,
                    args: impl ::std::iter::IntoIterator<Item = (&'__a Self, bool)>,
                ) -> bool {
                    <#inner as #trait_path>::eval(&self.#field, args.into_iter().map(|(f, b)| (&f.#field, b)))
                }

                #[inline]
                fn eval_edge<'__id, '__a>(
                    manager: &'__a #manager_ty,
                    edge: &'__a #edge_ty,
                    args: impl ::std::iter::IntoIterator<Item = (::oxidd_core::util::Borrowed<'__a, #edge_ty>, bool)>,
                ) -> bool {
                    <#inner as #trait_path>::eval_edge(manager, edge, args)
                }
            }
        },
    )
}

pub fn derive_boolean_function_quant(input: syn::DeriveInput) -> TokenStream {
    use Method::*;
    derive_function_trait(
        input,
        "BooleanFunctionQuant",
        &[
            Binary("restrict"),
            Binary("forall"),
            Binary("exist"),
            Binary("unique"),
        ],
        |ctx| {
            let CustomMethodsCtx {
                trait_path,
                manager_ty,
                edge_ty,
                struct_field,
            } = ctx;

            let field = &struct_field.ident;
            let inner = &struct_field.ty;

            let mut methods = TokenStream::new();

            for name in ["apply_forall", "apply_exist", "apply_unique"] {
                let method = syn::Ident::new(name, Span::call_site());
                let func = struct_field.gen_from_inner(
                    quote!(#trait_path::#method(&self.#field, op, &rhs.#field, &vars.#field)?),
                );
                let method_edge = syn::Ident::new(&format!("{name}_edge"), Span::call_site());

                let method = quote! {
                    #[inline]
                    fn #method(
                        &self,
                        op: ::oxidd_core::function::BooleanOperator,
                        rhs: &Self,
                        vars: &Self,
                    ) -> ::oxidd_core::util::AllocResult<Self> {
                        ::std::result::Result::Ok(#func)
                    }
                    #[inline]
                    fn #method_edge<'__id>(
                        manager: &#manager_ty,
                        op: ::oxidd_core::function::BooleanOperator,
                        lhs: &#edge_ty,
                        rhs: &#edge_ty,
                        vars: &#edge_ty,
                    ) -> ::oxidd_core::util::AllocResult<#edge_ty> {
                        <#inner as #trait_path>::#method_edge(manager, op, lhs, rhs, vars)
                    }
                };

                methods.extend(method);
            }

            methods
        },
    )
}

pub fn derive_boolean_vec_set(input: syn::DeriveInput) -> TokenStream {
    use Method::*;
    derive_function_trait(
        input,
        "BooleanVecSet",
        &[
            NewVar("new_singleton"),
            Terminal("empty"),
            Terminal("base"),
            Binary("subset0"),
            Binary("subset1"),
            Binary("change"),
            Binary("union"),
            Binary("intsec"),
            Binary("diff"),
        ],
        |_| TokenStream::new(),
    )
}

pub fn derive_pseudo_boolean_function(input: syn::DeriveInput) -> TokenStream {
    use Method::*;
    derive_function_trait(
        input,
        "PseudoBooleanFunction",
        &[
            NewVar("new_var"),
            Binary("add"),
            Binary("sub"),
            Binary("mul"),
            Binary("div"),
            Binary("min"),
            Binary("max"),
        ],
        |ctx| {
            let CustomMethodsCtx {
                trait_path,
                manager_ty,
                edge_ty,
                struct_field,
            } = ctx;
            let inner = &struct_field.ty;

            let constant_func = struct_field
                .gen_from_inner(quote!(<#inner as #trait_path>::constant(manager, value)?));

            quote! {
                type Number = <#inner as #trait_path>::Number;

                #[inline]
                fn constant<'__id>(manager: &#manager_ty, value: <Self as #trait_path>::Number) -> ::oxidd_core::util::AllocResult<Self> {
                    ::std::result::Result::Ok(#constant_func)
                }
                #[inline]
                fn constant_edge<'__id>(manager: &#manager_ty, value: <Self as #trait_path>::Number) -> ::oxidd_core::util::AllocResult<#edge_ty> {
                    <#inner as #trait_path>::constant_edge(manager, value)
                }
            }
        },
    )
}

pub fn derive_tvl_function(input: syn::DeriveInput) -> TokenStream {
    use Method::*;
    derive_function_trait(
        input,
        "TVLFunction",
        &[
            Terminal("f"),
            Terminal("t"),
            Terminal("u"),
            NewVar("new_var"),
            Unary("not"),
            UnaryOwned("not"),
            Binary("and"),
            Binary("or"),
            Binary("nand"),
            Binary("nor"),
            Binary("xor"),
            Binary("equiv"),
            Binary("imp"),
            Binary("imp_strict"),
            Ternary("ite"),
        ],
        |ctx| {
            let CustomMethodsCtx {
                trait_path,
                manager_ty,
                edge_ty,
                struct_field,
            } = ctx;
            let field = &struct_field.ident;
            let inner = &struct_field.ty;

            let from_ft = struct_field.gen_from_inner(quote!(ft));
            let from_fu = struct_field.gen_from_inner(quote!(fu));
            let from_ff = struct_field.gen_from_inner(quote!(ff));

            quote! {
                #[inline]
                fn cofactors(&self) -> ::std::option::Option<(Self, Self, Self)> {
                    let (ft, fu, ff) = <#inner as #trait_path>::cofactors(&self.#field)?;
                    ::std::option::Option::Some((#from_ft, #from_fu, #from_ff))
                }
                #[inline]
                fn cofactor_true(&self) -> ::std::option::Option<Self> {
                    let ft = <#inner as #trait_path>::cofactor_true(&self.#field)?;
                    ::std::option::Option::Some(#from_ft)
                }
                #[inline]
                fn cofactor_unknown(&self) -> ::std::option::Option<Self> {
                    let fu = <#inner as #trait_path>::cofactor_unknown(&self.#field)?;
                    ::std::option::Option::Some(#from_fu)
                }
                #[inline]
                fn cofactor_false(&self) -> ::std::option::Option<Self> {
                    let ff = <#inner as #trait_path>::cofactor_false(&self.#field)?;
                    ::std::option::Option::Some(#from_ff)
                }
                #[inline]
                fn cofactors_edge<'__a, '__id>(
                    manager: &'__a #manager_ty,
                    f: &'__a #edge_ty,
                ) -> ::std::option::Option<(
                    ::oxidd_core::util::Borrowed<'__a, #edge_ty>,
                    ::oxidd_core::util::Borrowed<'__a, #edge_ty>,
                    ::oxidd_core::util::Borrowed<'__a, #edge_ty>,
                )> {
                    <#inner as #trait_path>::cofactors_edge(manager, f)
                }
            }
        },
    )
}
