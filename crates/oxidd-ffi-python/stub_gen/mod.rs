//! Generate stub definitions with types from PyO3/Rust source code with
//! Google-style docstrings

use std::collections::HashMap;
use std::fmt::{self, Write as _};
use std::fs;
use std::io;
use std::path::Path;

use anyhow::{bail, Result};
use proc_macro2::TokenStream;
use quote::ToTokens;

mod util;
use util::{identifier_to_string, Indent, OrdRc};

// spell-checker:ignore punct

#[derive(Debug)]
struct Item {
    name: String,
    doc: String,
    kind: ItemKind,
}

#[derive(Debug)]
enum ItemKind {
    Attribute {
        ty: Type,
    },
    Function {
        kind: FunctionKind,
        params: Vec<Parameter>,
        return_type: Type,
    },
    Class {
        items: Vec<Item>,
        bases: Vec<OrdRc<String>>,
        constructible: bool,
        subclassable: bool,
        impl_eq: bool,
        impl_eq_int: bool,
        impl_ord: bool,
        impl_hash: bool,
    },
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum FunctionKind {
    Function,
    FunctionPassModule,
    Method,
    Constructor,
    Getter,
    Setter(String),
    Classmethod,
    Staticmethod,
}

#[derive(Debug)]
struct Parameter {
    name: String,
    ty: Option<Type>,
    default_value: Option<String>,
}

impl Parameter {
    fn name_only(name: impl ToString) -> Self {
        Self {
            name: name.to_string(),
            ty: None,
            default_value: None,
        }
    }

    fn params_for(
        name: &str,
        kind: &FunctionKind,
        rust: &syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
        pyo3: Option<&TokenStream>,
        py: Option<&str>,
    ) -> Result<Vec<Parameter>> {
        let mut params = Vec::with_capacity(rust.len() + 1);

        // If there is a signature attribute, then PyO3 and the Rust compiler
        // make sure that the PyO3 signature and the Rust function signature
        // agree. In this case, we only look at the PyO3 signature. Otherwise,
        // we infer the parameters from the Rust signature.
        if let Some(tokens) = pyo3 {
            let mut iter = tokens.clone().into_iter();
            match kind {
                FunctionKind::Method | FunctionKind::Getter | FunctionKind::Setter(_) => {
                    params.push(Self::name_only("self"))
                }
                FunctionKind::Constructor | FunctionKind::Classmethod => {
                    params.push(Self::name_only("cls"))
                }
                _ => {}
            }
            loop {
                let mut kv = iter
                    .by_ref()
                    .take_while(|tok| !util::parse::is_punct(',', tok));
                let mut name = String::new();
                for tok in kv
                    .by_ref()
                    .take_while(|tok| !util::parse::is_punct('=', tok))
                {
                    write!(name, "{tok}")?;
                }
                if name.is_empty() {
                    break;
                }
                let mut v = String::new();
                for tok in kv {
                    write!(v, "{tok}")?;
                }
                params.push(Parameter {
                    name,
                    ty: None,
                    default_value: if v.is_empty() { None } else { Some(v) },
                });
            }
        } else {
            let mut add_positional_delimiter = true;
            if *kind == FunctionKind::Constructor {
                params.push(Self::name_only("cls"));
                params.push(Self::name_only("/"));
                add_positional_delimiter = false;
            }
            let mut iter = rust.iter();
            if *kind == FunctionKind::FunctionPassModule {
                iter.next(); // skip over module parameter
            }
            for param in iter {
                match param {
                    syn::FnArg::Receiver(_) => {
                        params.push(Self::name_only("self"));
                        if !name.starts_with("__") {
                            params.push(Self::name_only("/"));
                            add_positional_delimiter = false;
                        }
                    }
                    syn::FnArg::Typed(pat_type) => {
                        let syn::Pat::Ident(pat_ident) = &*pat_type.pat else {
                            bail!("expected Rust parameters to be given an identifier")
                        };
                        if let syn::Type::Path(path) = &*pat_type.ty {
                            if let Some(last) = path.path.segments.last() {
                                if last.ident == "Python" {
                                    continue;
                                }
                            }
                        }
                        let mut name = pat_ident.ident.to_string();
                        if let Some(n) = name.strip_prefix('_') {
                            name = n.to_string();
                        }
                        params.push(Self::name_only(name));
                    }
                }
            }
            if !params.is_empty() && add_positional_delimiter {
                params.push(Self::name_only("/"));
            }
        }

        if let Some(sig) = py {
            // Check that the text_signature matches the signature from above
            // and add default arguments.

            let Some(sig) = sig.strip_prefix('(').and_then(|s| s.strip_suffix(')')) else {
                bail!("text signatures must start with '(' and end with ')'")
            };

            let mut iter = sig.bytes().enumerate();
            let mut par_depth = 0u32;
            let mut in_str = 0u8;
            let mut in_esc = false;
            let mut param_iter = params.iter_mut();
            loop {
                let mut inner = iter
                    .by_ref()
                    .filter(|(_, c)| !c.is_ascii_whitespace())
                    .take_while(|&(_, c)| {
                        match c {
                            b',' if par_depth == 0 && in_str == 0 => return false,
                            b'(' | b'[' | b'{' => par_depth += 1,
                            b')' | b']' | b'}' => par_depth -= 1,
                            b'"' if !in_esc => in_str ^= b'"',
                            b'\'' if !in_esc => in_str ^= b'\'',
                            b'\\' => {
                                in_esc = true;
                                return true;
                            }
                            _ => {}
                        }
                        true
                    });

                let Some((name_start, _)) = inner.next() else {
                    if let Some(param) = param_iter.next() {
                        bail!(
                            "parameter '{}' is missing in text_signature '{sig}'",
                            param.name
                        )
                    }
                    break;
                };
                let name_end = 1 + match inner.by_ref().take_while(|&(_, c)| c != b'=').last() {
                    Some((e, _)) => e,
                    None => name_start,
                };
                let name = sig.split_at(name_end).0.split_at(name_start).1;
                if name == "$module" && *kind == FunctionKind::FunctionPassModule {
                    continue;
                }
                let Some(param) = param_iter.next() else {
                    bail!("additional parameter '{name}' in text_signature")
                };
                if param.name != name.strip_prefix('$').unwrap_or(name) {
                    bail!(
                        "parameter names do not agree ('{}' vs. '{name}' in text_signature)",
                        param.name
                    )
                }

                if let Some((val_start, _)) = inner.next() {
                    let val_end = 1 + match inner.last() {
                        Some((e, _)) => e,
                        None => val_start,
                    };
                    let val = sig.split_at(val_end).0.split_at(val_start).1;
                    param.default_value = Some(val.to_string())
                }
            }
        }

        Ok(params)
    }
}

impl fmt::Display for Parameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)?;
        match (&self.ty, &self.default_value) {
            (None, None) => Ok(()),
            (Some(ty), None) => write!(f, ": {ty}"),
            (None, Some(val)) => write!(f, "={val}"),
            (Some(ty), Some(val)) => write!(f, ": {ty} = {val}"),
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Type {
    Any,
    Union(Vec<Self>),
    Other(OrdRc<String>, Vec<Self>),
}

#[derive(Default)]
pub struct TypeEnv {
    py: HashMap<String, OrdRc<String>>,
    rs: HashMap<String, OrdRc<String>>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_python_type(&mut self, py_name: String) -> Result<OrdRc<String>> {
        let name = OrdRc::new(py_name.clone());
        if self.py.insert(py_name, name.clone()).is_some() {
            bail!("Python type '{name}' already declared");
        }
        Ok(name)
    }

    pub fn register_rust_type(&mut self, rust_name: &str, py_name: OrdRc<String>) -> Result<()> {
        if self.rs.insert(rust_name.to_string(), py_name).is_some() {
            bail!("Rust type '{rust_name}' already declared");
        }
        Ok(())
    }

    fn get_py_type(&self, name: &str, args: Vec<Type>) -> Result<Type> {
        match self.py.get(name) {
            Some(info) => Ok(Type::Other(info.clone(), args)),
            None => bail!("Unknown Python type '{name}'"),
        }
    }

    fn parse_py(&self, string: &str) -> Result<Type> {
        #[derive(Clone, Copy)]
        enum Token<'a> {
            Ident(&'a str),
            LBrack,
            RBrack,
            Comma,
            Dot,
            Or,
        }

        let mut tokens = Vec::new();
        let mut remainder = string;
        while let Some(pos) = remainder.find([' ', '\t', '[', ']', ',', '.', '|']) {
            if pos != 0 {
                let (before, s) = remainder.split_at(pos);
                tokens.push(Token::Ident(before));
                remainder = s;
            }
            let (delim, after) = remainder.split_at(1);
            remainder = after;
            tokens.push(match delim {
                "[" => Token::LBrack,
                "]" => Token::RBrack,
                "," => Token::Comma,
                "." => Token::Dot,
                "|" => Token::Or,
                _ => continue,
            });
        }
        if !remainder.is_empty() {
            tokens.push(Token::Ident(remainder));
        }

        /// Split tokens into two parts such that the first part is the longest
        /// sequence from the start consisting only of `Token::Ident(..)` and
        /// `Token::Dot`
        fn get_identifier<'a>(tokens: &'a [Token<'a>]) -> (&'a [Token<'a>], &'a [Token<'a>]) {
            let mut i = 0;
            let mut second = tokens;
            while let [Token::Ident(_) | Token::Dot, r @ ..] = second {
                i += 1;
                second = r;
            }
            (&tokens[..i], second)
        }

        fn parse_args<'a>(
            env: &TypeEnv,
            tokens: &'a [Token<'a>],
        ) -> Result<(Vec<Type>, &'a [Token<'a>])> {
            let [Token::LBrack, tokens @ ..] = tokens else {
                return Ok((Vec::new(), tokens));
            };
            let mut args = Vec::new();
            let mut tokens = tokens;
            loop {
                let (arg, t) = parse(env, tokens)?;
                args.push(arg);
                match t {
                    [Token::RBrack, tokens @ ..] | [Token::Comma, Token::RBrack, tokens @ ..] => {
                        return Ok((args, tokens))
                    }
                    [Token::Comma, t @ ..] => tokens = t,
                    _ => bail!("expected ',' or ']'"),
                }
            }
        }

        fn parse_single<'a>(
            env: &TypeEnv,
            tokens: &'a [Token<'a>],
        ) -> Result<(Type, &'a [Token<'a>])> {
            let (path, tokens) = get_identifier(tokens);
            let [.., Token::Ident(ident)] = *path else {
                bail!("")
            };
            Ok(match ident {
                "Any" => (Type::Any, tokens),
                _ => {
                    let (args, tokens) = parse_args(env, tokens)?;
                    (env.get_py_type(ident, args)?, tokens)
                }
            })
        }

        fn parse<'a>(env: &TypeEnv, tokens: &'a [Token<'a>]) -> Result<(Type, &'a [Token<'a>])> {
            let (ty, tokens) = parse_single(env, tokens)?;
            let [Token::Or, tokens @ ..] = tokens else {
                return Ok((ty, tokens));
            };
            let mut union = vec![ty];
            let mut tokens = tokens;
            loop {
                let (ty, t) = parse_single(env, tokens)?;
                union.push(ty);
                match t {
                    [Token::Or, t @ ..] => tokens = t,
                    _ => return Ok((Type::Union(union), t)),
                }
            }
        }

        match parse(self, &tokens) {
            Ok((ty, [])) => Ok(ty),
            Ok(_) => bail!("Failed to parse Python type '{string}'"),
            Err(err) => Err(err.context(format!("Failed to parse Python type '{string}'"))),
        }
    }

    /// Read a function/method signature from a docstring
    ///
    /// `item_name` is used for error reporting only
    fn signature_from_doc(
        &self,
        doc: &str,
        item_name: &str,
        mut params: &mut [Parameter],
    ) -> Result<Type> {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum State {
            Init,
            Args,
            Returns,
        }
        let mut state = State::Init;
        let mut return_type = None;
        for line in doc.lines() {
            match line {
                "Args:" => {
                    state = State::Args;
                    continue;
                }
                "Returns:" => {
                    state = State::Returns;
                    continue;
                }
                "" => continue,
                _ if state == State::Init => continue,
                _ => {}
            }
            let Some(line) = line.strip_prefix("    ") else {
                state = State::Init;
                continue;
            };

            if state == State::Args {
                if line.starts_with(' ') {
                    continue;
                }
                if let Some((arg_name, after)) = line.split_once('(') {
                    let arg_name = arg_name.trim_ascii_end();
                    if !arg_name.is_empty() {
                        if let Some((ty, _)) = after.split_once(')') {
                            let ty = self.parse_py(ty)?;
                            loop {
                                let [p, pr @ ..] = params else {
                                    bail!("Additional parameter '{arg_name}' documented for '{item_name}'")
                                };
                                params = pr;
                                if arg_name == p.name {
                                    p.ty = Some(ty);
                                    break;
                                }
                            }
                            continue;
                        }
                    }
                }
                bail!("Failed to parse arguments in docstring for '{item_name}'");
            }

            debug_assert!(state == State::Returns);
            return_type = Some(if line == "None" {
                self.get_py_type("None", Vec::new())?
            } else if let Some((ty, _)) = line.split_once(':') {
                self.parse_py(ty)?
            } else {
                bail!("Failed to parse the return type in docstring for '{item_name}'");
            });
            state = State::Init;
        }

        if let Some(ret) = return_type {
            Ok(ret)
        } else {
            bail!("Missing return type documentation for '{item_name}'")
        }
    }

    /// Read an attribute/property type from a docstring
    ///
    /// `item_name` is used for error reporting only
    fn attr_type_from_doc(&self, doc: &str, item_name: &str) -> Result<Type> {
        if let Some((ty, _)) = doc.split_once(':') {
            self.parse_py(ty)
        } else {
            bail!("Type annotation missing in docstring for '{item_name}'")
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Any => f.write_str("Any"),
            Type::Union(ts) => {
                if let [tr @ .., t] = &ts[..] {
                    for t in tr {
                        write!(f, "{t} | ")?;
                    }
                    t.fmt(f)
                } else {
                    f.write_str("Never")
                }
            }
            Type::Other(name, args) => {
                f.write_str(name)?;
                if let [arg, args @ ..] = &args[..] {
                    write!(f, "[{arg}")?;
                    for arg in args {
                        write!(f, ", {arg}")?;
                    }
                    f.write_str("]")?;
                }
                Ok(())
            }
        }
    }
}

fn attr_matches(attribute: &syn::Attribute, name: &str) -> bool {
    if let Some(segment) = attribute.meta.path().segments.last() {
        if segment.ident == name {
            return true;
        }
    }
    false
}

fn get_doc(attributes: &[syn::Attribute]) -> String {
    let mut doc = String::new();
    for attr in attributes {
        let syn::Meta::NameValue(meta) = &attr.meta else {
            continue;
        };

        let path = &meta.path.segments;
        if path.len() != 1 || path[0].ident != "doc" {
            continue;
        }

        let syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Str(value),
            ..
        }) = &meta.value
        else {
            continue;
        };

        let value = value.value();
        doc.push_str(value.strip_prefix(" ").unwrap_or(&value));
        doc.push('\n');
    }

    doc
}

pub struct StubGen {
    items: Vec<Item>,
    class_to_item: HashMap<syn::Ident, usize>,
    type_env: TypeEnv,
}

impl StubGen {
    pub fn new(type_env: TypeEnv) -> Self {
        Self {
            items: Vec::new(),
            class_to_item: HashMap::new(),
            type_env,
        }
    }

    /// Process a Rust `struct`/`enum`, potentially annotated by `#[pyclass]`
    ///
    /// If a new Python class was successfully registered, then the return value
    /// is `Ok(id)`, where `id` denotes the class ID.
    ///
    /// `Err(..)` is used for actual processing errors (parsing etc.). If there
    /// is no `#[pyclass]` attribute, then the return value is `Ok(usize::MAX)`.
    fn process_class(
        &mut self,
        ident: &syn::Ident,
        attributes: &[syn::Attribute],
    ) -> Result<usize> {
        let Some(pyclass) = attributes.iter().find(|a| attr_matches(a, "pyclass")) else {
            return Ok(usize::MAX);
        };

        let doc = get_doc(attributes);

        let mut name = String::new();
        let mut bases = Vec::new();
        let mut subclassable = false;
        let mut impl_eq = false;
        let mut impl_eq_int = false;
        let mut impl_ord = false;
        let mut impl_hash = false;
        if let syn::Meta::List(syn::MetaList { tokens, .. }) = &pyclass.meta {
            util::parse::recognize_key_val(tokens.clone(), pyclass, |key, tokens| {
                if key == "eq" {
                    impl_eq = true;
                } else if key == "eq_int" {
                    impl_eq_int = true;
                } else if key == "ord" {
                    impl_ord = true;
                } else if key == "hash" {
                    impl_hash = true;
                } else if key == "name" {
                    util::parse::expect_punct_opt('=', tokens.next(), pyclass)?;
                    name = util::parse::expect_string_opt(tokens.next(), pyclass)?;
                } else if key == "extends" {
                    util::parse::expect_punct_opt('=', tokens.next(), pyclass)?;
                    let base_name =
                        util::parse::expect_ident_opt(tokens.next(), pyclass)?.to_string();
                    let Some(base) = self.type_env.rs.get(&base_name) else {
                        bail!("Unknown Rust type '{base_name}' as base class for '{ident}'")
                    };
                    bases.push(base.clone());
                } else if key == "subclass" {
                    subclassable = true;
                } else {
                    return Ok(false);
                }
                Ok(true)
            })?;
        }

        if name.is_empty() {
            name = identifier_to_string(ident);
        }
        self.type_env.register_python_type(name.clone())?;

        let id = self.items.len();
        if self.class_to_item.insert(ident.clone(), id).is_some() {
            bail!("Python class '{name}' (Rust identifier '{ident}') was declared twice")
        }

        self.items.push(Item {
            name,
            doc,
            kind: ItemKind::Class {
                items: Vec::new(),
                bases,
                constructible: false,
                subclassable,
                impl_eq,
                impl_eq_int,
                impl_ord,
                impl_hash,
            },
        });

        Ok(id)
    }

    fn process_method(env: &TypeEnv, item: &syn::ImplItemFn, last: Option<&Item>) -> Result<Item> {
        let mut kind = FunctionKind::Method;
        let mut name = String::new();
        let mut signature = None;
        let mut text_signature = None;
        for attr in &item.attrs {
            let Some(syn::PathSegment { ident, .. }) = attr.meta.path().segments.last() else {
                continue;
            };

            if ident == "pyo3" {
                let syn::Meta::List(syn::MetaList { tokens, .. }) = &attr.meta else {
                    continue;
                };
                util::parse::recognize_key_val(tokens.clone(), attr, |ident, tokens| {
                    if ident == "name" {
                        util::parse::expect_punct_opt('=', tokens.next(), attr)?;
                        name = util::parse::expect_string_opt(tokens.next(), attr)?;
                    } else if ident == "signature" {
                        util::parse::expect_punct_opt('=', tokens.next(), attr)?;
                        signature = Some(util::parse::expect_group_opt(tokens.next(), attr)?);
                    } else if ident == "text_signature" {
                        util::parse::expect_punct_opt('=', tokens.next(), attr)?;
                        text_signature = Some(util::parse::expect_string_opt(tokens.next(), attr)?);
                    } else {
                        return Ok(false);
                    }
                    Ok(true)
                })?;
            } else if ident == "getter" {
                kind = FunctionKind::Getter;
                if let syn::Meta::List(syn::MetaList { tokens, .. }) = &attr.meta {
                    name = util::parse::expect_ident_opt(tokens.clone().into_iter().next(), attr)?
                        .to_string();
                }
            } else if ident == "setter" {
                kind = FunctionKind::Setter(
                    if let syn::Meta::List(syn::MetaList { tokens, .. }) = &attr.meta {
                        util::parse::expect_ident_opt(tokens.clone().into_iter().next(), attr)?
                            .to_string()
                    } else {
                        ident.to_string()
                    },
                );
            } else if ident == "new" {
                kind = FunctionKind::Constructor;
                name = "__new__".into();
            } else if ident == "classmethod" {
                kind = FunctionKind::Classmethod;
            } else if ident == "staticmethod" {
                kind = FunctionKind::Staticmethod;
            }
        }
        let doc = get_doc(&item.attrs);

        let ident = &item.sig.ident;
        if name.is_empty() {
            name = identifier_to_string(ident);
        }

        let (params, return_type) = match &kind {
            FunctionKind::Getter => {
                let r = env.attr_type_from_doc(&doc, &name)?;
                let p = vec![Parameter::name_only("self"), Parameter::name_only("/")];
                (p, r)
            }
            FunctionKind::Setter(property_name) => {
                // we rely on the type associated with the getter
                let Some(ty) = last.and_then(|last| match &last.kind {
                    ItemKind::Function {
                        kind: FunctionKind::Getter,
                        return_type,
                        ..
                    } if last.name == *property_name => Some(return_type),
                    _ => None,
                }) else {
                    bail!("Expected the respective getter directly before setter '{name}' (Rust identifier '{ident}')")
                };

                let r = env.get_py_type("None", Vec::new())?;
                let p = vec![
                    Parameter::name_only("self"),
                    Parameter {
                        name: "value".into(),
                        ty: Some(ty.clone()),
                        default_value: None,
                    },
                    Parameter::name_only("/"),
                ];
                // TODO: should we check that this signature agrees with the Rust signature?

                (p, r)
            }
            _ => {
                let mut p = Parameter::params_for(
                    &name,
                    &kind,
                    &item.sig.inputs,
                    signature.as_ref(),
                    text_signature.as_deref(),
                )
                .map_err(|err| {
                    err.context(format!(
                        "Failed to process parameters for method '{name}' (Rust identifier '{ident}')"
                    ))
                })?;
                let r = env.signature_from_doc(&doc, &name, &mut p[..])?;
                (p, r)
            }
        };

        Ok(Item {
            name,
            doc,
            kind: ItemKind::Function {
                kind,
                params,
                return_type,
            },
        })
    }

    fn process_function(&mut self, function: &syn::ItemFn) -> Result<()> {
        let mut name = String::new();
        let mut signature = None;
        let mut text_signature = None;
        let mut pass_module = false;
        for attr in &function.attrs {
            let Some(syn::PathSegment { ident, .. }) = attr.meta.path().segments.last() else {
                continue;
            };

            if ident == "pyo3" {
                let syn::Meta::List(syn::MetaList { tokens, .. }) = &attr.meta else {
                    continue;
                };
                util::parse::recognize_key_val(tokens.clone(), attr, |ident, tokens| {
                    if ident == "name" {
                        util::parse::expect_punct_opt('=', tokens.next(), attr)?;
                        name = util::parse::expect_string_opt(tokens.next(), attr)?;
                    } else if ident == "signature" {
                        util::parse::expect_punct_opt('=', tokens.next(), attr)?;
                        signature = Some(util::parse::expect_group_opt(tokens.next(), attr)?);
                    } else if ident == "text_signature" {
                        util::parse::expect_punct_opt('=', tokens.next(), attr)?;
                        text_signature = Some(util::parse::expect_string_opt(tokens.next(), attr)?);
                    } else if ident == "pass_module" {
                        pass_module = true;
                    } else {
                        return Ok(false);
                    }
                    Ok(true)
                })?;
            }
        }
        let doc = get_doc(&function.attrs);

        let ident = &function.sig.ident;
        if name.is_empty() {
            name = identifier_to_string(ident);
        }

        let mut params = Parameter::params_for(
            &name,
            &if pass_module {
                FunctionKind::FunctionPassModule
            } else {
                FunctionKind::Function
            },
            &function.sig.inputs,
            signature.as_ref(),
            text_signature.as_deref(),
        )
        .map_err(|err| {
            err.context(format!(
                "Failed to process parameters for method '{name}' (Rust identifier '{ident}')"
            ))
        })?;
        let return_type = self
            .type_env
            .signature_from_doc(&doc, &name, &mut params[..])?;

        self.items.push(Item {
            name,
            doc,
            kind: ItemKind::Function {
                kind: FunctionKind::Function,
                params,
                return_type,
            },
        });

        Ok(())
    }

    /// Process the fields of a Rust `struct` annotated with `#[pyclass]`
    fn process_class_fields(&mut self, class_id: usize, fields: &syn::Fields) -> Result<()> {
        let class = &mut self.items[class_id];
        let class_name = &class.name;
        let ItemKind::Class {
            items: class_items, ..
        } = &mut class.kind
        else {
            panic!("class_id must refer to a class");
        };

        for field in fields {
            // attributes may look like: #[pyo3(get, set, name = "custom_name")]
            let mut get = false;
            let mut set = false;
            let mut name = String::new();
            for attr in &field.attrs {
                let Some(syn::PathSegment { ident, .. }) = attr.meta.path().segments.last() else {
                    continue;
                };

                if ident == "pyo3" {
                    let syn::Meta::List(syn::MetaList { tokens, .. }) = &attr.meta else {
                        continue;
                    };
                    util::parse::recognize_key_val(tokens.clone(), attr, |ident, tokens| {
                        if ident == "get" {
                            get = true;
                        } else if ident == "set" {
                            set = true;
                        } else if ident == "name" {
                            util::parse::expect_punct_opt('=', tokens.next(), attr)?;
                            name = util::parse::expect_string_opt(tokens.next(), attr)?;
                        } else {
                            return Ok(false);
                        }
                        Ok(true)
                    })?;
                }
            }

            if !get && !set {
                continue;
            }
            if name.is_empty() {
                let Some(ident) = &field.ident else {
                    bail!("Missing field name for getter/setter in '{class_name}'");
                };
                name = identifier_to_string(ident);
            }
            if set && !get {
                bail!("Setter-only fields are not supported ('{class_name}.{name}')");
            }

            let doc = get_doc(&field.attrs);
            let ty = self.type_env.attr_type_from_doc(&doc, &name)?;
            // TODO: should we check correspondence with the Rust type?

            debug_assert!(get);
            class_items.push(Item {
                name,
                doc,
                kind: if set {
                    ItemKind::Attribute { ty }
                } else {
                    ItemKind::Function {
                        kind: FunctionKind::Getter,
                        params: vec![Parameter::name_only("self"), Parameter::name_only("/")],
                        return_type: ty,
                    }
                },
            });
        }

        Ok(())
    }

    /// Process a `#[pymethods]` `impl` block for the Python class with the
    /// given `class_id`
    fn process_class_items(&mut self, class_id: usize, items: &[syn::ImplItem]) -> Result<()> {
        let class = &mut self.items[class_id];
        let class_name = &class.name;
        let ItemKind::Class {
            items: class_items,
            constructible,
            ..
        } = &mut class.kind
        else {
            panic!("class_id must refer to a class");
        };

        for item in items {
            class_items.push(match item {
                syn::ImplItem::Const(item) => {
                    if !item.attrs.iter().any(|a| attr_matches(a, "classattr")) {
                        continue;
                    }
                    let name = identifier_to_string(&item.ident);
                    let doc = get_doc(&item.attrs);
                    let ty = self.type_env.attr_type_from_doc(&doc, &name)?;

                    let kind = ItemKind::Attribute { ty };
                    Item { name, doc, kind }
                }
                syn::ImplItem::Fn(item) => {
                    let f = Self::process_method(&self.type_env, item, class_items.last())
                        .map_err(|err| {
                            err.context(format!(
                                "failed to process methods for Python class '{class_name}'"
                            ))
                        })?;

                    *constructible |= matches!(
                        f,
                        Item {
                            kind: ItemKind::Function {
                                kind: FunctionKind::Constructor,
                                ..
                            },
                            ..
                        }
                    );

                    f
                }
                _ => continue,
            });
        }

        Ok(())
    }

    fn process_exception(&mut self, tokens: &TokenStream) -> Result<()> {
        let mut iter = tokens.clone().into_iter();
        iter.by_ref()
            .take_while(|tok| !util::parse::is_punct(',', tok))
            .count();
        let Some(proc_macro2::TokenTree::Ident(name)) = iter.next() else {
            bail!("Missing identifier in create_exception!({tokens})")
        };
        iter.next(); // skip ','
        let mut base = None;
        for tok in &mut iter {
            if let proc_macro2::TokenTree::Ident(b) = tok {
                base = Some(b);
            } else if util::parse::is_punct(',', &tok) {
                break;
            }
        }
        let Some(base) = base else {
            bail!("Missing base in create_exception!({tokens})")
        };
        let Some(base) = self.type_env.rs.get(&base.to_string()) else {
            bail!("Unknown base '{base}' in create_exception!({tokens})")
        };
        let mut doc = String::new();
        if let Some(proc_macro2::TokenTree::Literal(l)) = iter.next() {
            match syn::Lit::new(l) {
                syn::Lit::Str(l) => doc = l.value(),
                _ => bail!(
                    "Expected a string literal as documentation in create_exception!({tokens})"
                ),
            }
        }

        let base = base.clone();
        self.type_env.register_python_type(name.to_string())?;
        self.items.push(Item {
            name: name.to_string(),
            doc,
            kind: ItemKind::Class {
                items: Vec::new(),
                bases: vec![base],
                constructible: true,
                subclassable: true,
                impl_eq: false,
                impl_eq_int: false,
                impl_ord: false,
                impl_hash: false,
            },
        });

        Ok(())
    }

    fn process_items(&mut self, items: &[syn::Item]) -> Result<()> {
        // Process all sorts of classes (without their methods) first to
        // register their names
        for item in items {
            match item {
                syn::Item::Mod(item_mod) => {
                    if let Some((_, items)) = &item_mod.content {
                        self.process_items(items)?;
                    }
                }
                syn::Item::Struct(item_struct) => {
                    let id = self.process_class(&item_struct.ident, &item_struct.attrs)?;
                    if id != usize::MAX {
                        self.process_class_fields(id, &item_struct.fields)?;
                    }
                }
                syn::Item::Enum(item_enum) => {
                    self.process_class(&item_enum.ident, &item_enum.attrs)?;
                    // TODO: handle variants `#[pyo3(constructor =
                    // (radius=1.0))]`
                }
                syn::Item::Macro(mac) => {
                    if let Some(last) = mac.mac.path.segments.last() {
                        if last.ident == "create_exception" {
                            self.process_exception(&mac.mac.tokens)?;
                        }
                    };
                }
                _ => {}
            }
        }

        for item in items {
            match item {
                syn::Item::Mod(item_mod) => {
                    if let Some((_, items)) = &item_mod.content {
                        self.process_items(items)?;
                    }
                }
                syn::Item::Fn(item_fn) => {
                    if item_fn.attrs.iter().any(|a| attr_matches(a, "pyfunction")) {
                        self.process_function(item_fn)?;
                    }
                }
                syn::Item::Impl(item_impl) => {
                    if !item_impl.attrs.iter().any(|a| attr_matches(a, "pymethods")) {
                        continue;
                    }

                    if let syn::Type::Path(path) = &*item_impl.self_ty {
                        if let Some(ident) = path.path.get_ident() {
                            if let Some(id) = self.class_to_item.get(ident) {
                                self.process_class_items(*id, &item_impl.items)?;
                                continue;
                            }

                            bail!("Did not find a #[pyclass] '{ident}'")
                        }
                    }

                    bail!(
                        "Unexpected type '{}' in #[pymethods] impl",
                        item_impl.self_ty.to_token_stream()
                    )
                }
                _ => {}
            }
        }

        Ok(())
    }

    pub fn process_files<P: AsRef<Path>>(
        &mut self,
        paths: impl IntoIterator<Item = P>,
    ) -> Result<()> {
        let mut items = Vec::new();
        for p in paths {
            items.extend(syn::parse_file(&fs::read_to_string(p)?)?.items);
        }
        self.process_items(&items)
    }

    fn write_doc<W: io::Write>(writer: &mut W, doc: &str, indent: Indent) -> io::Result<()> {
        let doc = doc.trim_ascii_end();
        let raw = if doc.contains('\\') { "r" } else { "" };
        let mut lines = doc.lines();
        let Some(line) = lines.next() else {
            return Ok(());
        };

        write!(writer, "{indent}{raw}\"\"\"{line}")?;
        let Some(line) = lines.next() else {
            return writeln!(writer, "\"\"\"");
        };
        writeln!(writer)?;
        match line {
            "" => writeln!(writer)?,
            _ => writeln!(writer, "{indent}{line}")?,
        }

        for line in lines {
            match line {
                "" => writeln!(writer)?,
                _ => writeln!(writer, "{indent}{line}")?,
            }
        }
        writeln!(writer, "{indent}\"\"\"")
    }

    fn write_items<W: io::Write>(
        w: &mut W,
        items: &[Item],
        indent: Indent,
        mut blanks_before: u32,
    ) -> Result<()> {
        for item in items {
            for _ in 0..blanks_before {
                writeln!(w)?;
            }

            let name = &item.name;
            let sub_indent = Indent(indent.0 + 1);
            match &item.kind {
                ItemKind::Attribute { ty } => {
                    writeln!(w, "{indent}{name}: {ty} = ...")?;
                    Self::write_doc(w, &item.doc, indent)?;
                    blanks_before = 0;
                }
                ItemKind::Function {
                    kind,
                    params,
                    return_type,
                } => {
                    match kind {
                        FunctionKind::Getter => writeln!(w, "{indent}@property")?,
                        FunctionKind::Setter(name) => writeln!(w, "{indent}@{name}.setter")?,
                        FunctionKind::Classmethod | FunctionKind::Constructor => {
                            writeln!(w, "{indent}@classmethod")?
                        }
                        FunctionKind::Staticmethod => writeln!(w, "{indent}@staticmethod")?,
                        _ => {}
                    }
                    write!(w, "{indent}def {name}(")?;
                    if let [param, params @ ..] = &params[..] {
                        write!(w, "{param}")?;
                        for param in params {
                            write!(w, ", {param}")?;
                        }
                    }
                    write!(w, ") -> {return_type}:")?;
                    if item.doc.is_empty() {
                        writeln!(w, " ...")?;
                        blanks_before = 0;
                    } else {
                        writeln!(w)?;
                        Self::write_doc(w, &item.doc, sub_indent)?;
                        blanks_before = 1;
                    }
                }
                ItemKind::Class {
                    items,
                    bases,
                    constructible,
                    subclassable,
                    impl_eq,
                    impl_eq_int,
                    impl_ord,
                    impl_hash,
                } => {
                    if !*subclassable {
                        writeln!(w, "{indent}@final")?;
                    }
                    write!(w, "{indent}class {name}")?;
                    if let [base, bases @ ..] = &bases[..] {
                        write!(w, "({base}")?;
                        for base in bases {
                            write!(w, ", {base}")?;
                        }
                        write!(w, ")")?;
                    }

                    if item.doc.is_empty() && items.is_empty() {
                        writeln!(w, ":")?;
                        blanks_before = 0;
                        continue;
                    }
                    writeln!(w, ":")?;
                    Self::write_doc(w, &item.doc, sub_indent)?;

                    if !*constructible {
                        writeln!(w)?;
                        writeln!(w, "{sub_indent}@classmethod")?;
                        writeln!(w, "{sub_indent}def __new__(cls, _: Never) -> Self:")?;
                        writeln!(w, "{sub_indent}    \"\"\"Private constructor.\"\"\"")?;
                    }

                    if !items.is_empty() {
                        Self::write_items(w, items, sub_indent, 1)?;
                    }

                    if *impl_eq || *impl_eq_int || *impl_ord || *impl_hash {
                        writeln!(w)?;
                    }
                    if *impl_eq || *impl_eq_int || *impl_ord {
                        writeln!(
                            w,
                            "{sub_indent}def __eq__(self, /, rhs: object) -> bool: ..."
                        )?;
                        writeln!(
                            w,
                            "{sub_indent}def __ne__(self, /, rhs: object) -> bool: ..."
                        )?;
                    }
                    if *impl_eq_int {
                        writeln!(w, "{sub_indent}def __int__(self, /) -> int: ...")?;
                    }
                    if *impl_ord {
                        for op in ["le", "lt", "ge", "gt"] {
                            writeln!(
                                w,
                                "{sub_indent}def __{op}__(self, /, rhs: Self) -> bool: ..."
                            )?;
                        }
                    }
                    if *impl_hash {
                        writeln!(w, "{sub_indent}def __hash__(self, /) -> int: ...")?;
                    }

                    blanks_before = 2;
                }
            }
        }

        Ok(())
    }

    pub fn write<W: io::Write>(&self, writer: &mut W) -> Result<()> {
        Self::write_items(writer, &self.items, Indent(0), 1)
    }
}
