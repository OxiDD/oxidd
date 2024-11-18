use std::fmt;
use std::rc::Rc;

// spell-checker:ignore punct

pub struct OrdRc<T>(Rc<T>);

impl<T> OrdRc<T> {
    pub fn new(value: T) -> Self {
        Self(Rc::new(value))
    }
}

impl<T> Clone for OrdRc<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T> PartialEq for OrdRc<T> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}
impl<T> Eq for OrdRc<T> {}
impl<T> PartialOrd for OrdRc<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<T> Ord for OrdRc<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        Rc::as_ptr(&self.0).cmp(&Rc::as_ptr(&other.0))
    }
}

impl<T> std::ops::Deref for OrdRc<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<T: fmt::Debug> fmt::Debug for OrdRc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl<T: fmt::Display> fmt::Display for OrdRc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

pub mod parse {
    use anyhow::{bail, Result};
    use proc_macro2::{token_stream, Ident, TokenStream, TokenTree};
    use quote::ToTokens;

    pub fn expect_ident(tok: TokenTree, context: impl ToTokens) -> Result<Ident> {
        match tok {
            TokenTree::Ident(ident) => Ok(ident),
            _ => bail!(
                "expected an identifier, got '{tok}' in '{}'",
                context.to_token_stream()
            ),
        }
    }

    pub fn expect_ident_opt(tok: Option<TokenTree>, context: impl ToTokens) -> Result<Ident> {
        let Some(tok) = tok else {
            bail!(
                "expected an identifier at the end of '{}'",
                context.to_token_stream()
            )
        };
        expect_ident(tok, context)
    }

    pub fn is_punct(punct: char, tok: &TokenTree) -> bool {
        if let TokenTree::Punct(p) = tok {
            if p.as_char() == punct {
                return true;
            }
        }
        false
    }

    pub fn expect_punct_opt(
        punct: char,
        tok: Option<TokenTree>,
        context: impl ToTokens,
    ) -> Result<()> {
        if let Some(tok) = tok {
            if is_punct(punct, &tok) {
                return Ok(());
            }
            bail!(
                "expected '{punct}', got '{tok}' at the end of '{}'",
                context.to_token_stream()
            )
        }
        bail!(
            "expected '{punct}' at the end of '{}'",
            context.to_token_stream()
        )
    }

    pub fn expect_string_opt(tok: Option<TokenTree>, context: impl ToTokens) -> Result<String> {
        if let Some(tok) = tok {
            if let TokenTree::Literal(literal) = &tok {
                let mut s = literal.to_string();
                if s.len() > 2 && s.as_bytes()[0] == b'"' && s.pop().unwrap() == '"' {
                    s.remove(0);
                    return Ok(s);
                }
            }
            bail!(
                "expected a string literal, got {tok} in {}",
                context.to_token_stream()
            )
        }
        bail!(
            "expected a string literal at the end of '{}'",
            context.to_token_stream()
        )
    }

    pub fn expect_group_opt(tok: Option<TokenTree>, context: impl ToTokens) -> Result<TokenStream> {
        if let Some(tok) = tok {
            if let TokenTree::Group(group) = &tok {
                return Ok(group.stream());
            }
            bail!(
                "expected a token group, got {tok} in {}",
                context.to_token_stream()
            )
        }
        bail!(
            "expected a token group at the end of '{}'",
            context.to_token_stream()
        )
    }

    /// Recognize single keys or key-value pairs
    ///
    /// `recognizer` should return `Ok(true)` whenever it recognized and fully
    /// consumed an entry (i.e., the next token is expected to be a ',' or the
    /// end). If the key is unknown, it should return `Ok(false)`, then the
    /// parser will skip to the next ','.
    pub fn recognize_key_val(
        tokens: TokenStream,
        context: impl ToTokens + Copy,
        mut recognizer: impl FnMut(Ident, &mut token_stream::IntoIter) -> Result<bool>,
    ) -> Result<()> {
        let mut tokens = tokens.into_iter();
        'outer: while let Some(tok) = tokens.next() {
            let recognized = recognizer(expect_ident(tok, context)?, &mut tokens)?;
            if recognized {
                if let Some(tok) = tokens.next() {
                    expect_punct_opt(',', Some(tok), context)?;
                    continue;
                }
                return Ok(());
            }

            // unknown, consume everything up to the next comma
            for tok in &mut tokens {
                if is_punct(',', &tok) {
                    continue 'outer;
                }
            }
            return Ok(());
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct Indent(pub u32);

impl fmt::Display for Indent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for _ in 0..self.0 {
            f.write_str("    ")?;
        }
        Ok(())
    }
}

pub fn identifier_to_string(ident: &syn::Ident) -> String {
    let mut s = ident.to_string();
    if s.starts_with("r#") {
        s = s.split_off(2);
    }
    s
}
