//! Convenience functions etc. to load a [`Problem`] from file

// spell-checker:ignore termcolor

use std::fmt;
use std::path::Path;

use codespan_reporting::diagnostic::{Diagnostic, Label};
use codespan_reporting::files::SimpleFile;
use codespan_reporting::term::termcolor::ColorChoice;
use codespan_reporting::term::termcolor::{StandardStream, WriteColor};
use codespan_reporting::term::{emit, Config};
use nom::error::{ContextError, ErrorKind, FromExternalError, ParseError};
use nom::Offset;

use crate::ParseOptions;
use crate::Problem;
use crate::{aiger, dimacs};

/// File types
///
/// Every file type is associated with a parser and (possibly multiple)
/// extensions
#[non_exhaustive]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FileType {
    /// DIMACS CNF/SAT formats
    ///
    /// Extensions: `.cnf`, `.sat`, `.dimacs`
    DIMACS,
    /// AIGER
    ///
    /// Extensions: `.aag`, `.aig`
    AIGER,
}

impl FileType {
    /// Guess the file type based on the given path
    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?;
        match ext.as_encoded_bytes() {
            b"cnf" | b"sat" | b"dimacs" => Some(FileType::DIMACS),
            b"aag" | b"aig" => Some(FileType::AIGER),
            _ => None,
        }
    }
}

struct ParserReport<I>(Vec<(I, ParserError)>);

enum ParserError {
    Nom(ErrorKind),
    Char(char),
    Context(&'static str),
    External(String),
}

impl<I> ParseError<I> for ParserReport<I> {
    fn from_error_kind(input: I, kind: ErrorKind) -> Self {
        ParserReport(vec![(input, ParserError::Nom(kind))])
    }

    fn append(input: I, kind: ErrorKind, mut other: Self) -> Self {
        other.0.push((input, ParserError::Nom(kind)));
        other
    }

    fn from_char(input: I, c: char) -> Self {
        ParserReport(vec![(input, ParserError::Char(c))])
    }
}

impl<I> ContextError<I> for ParserReport<I> {
    fn add_context(input: I, ctx: &'static str, mut other: Self) -> Self {
        match other.0[0].1 {
            ParserError::Context(_) => {}
            // Assume that the context is a better description
            _ => other.0.clear(),
        }
        other.0.push((input, ParserError::Context(ctx)));
        other
    }
}

impl<I, S: ToString> FromExternalError<I, S> for ParserReport<I> {
    fn from_external_error(input: I, _kind: ErrorKind, e: S) -> Self {
        Self(vec![(input, ParserError::External(e.to_string()))])
    }
}

/// Parse `input` using the parser for the given `file_type`, emitting errors or
/// warnings to `writer`
///
/// `file_id` is an identifier for the file used for error reporting. `config`
/// configures how diagnostics are rendered.
///
/// If you simply want to parse a file with error reporting to stderr, you are
/// probably looking for [`load_file()`].
pub fn parse<S: AsRef<str> + Clone + fmt::Display>(
    input: &[u8],
    file_type: FileType,
    parse_options: &ParseOptions,
    file_id: S,
    writer: &mut dyn WriteColor,
    config: &Config,
) -> Option<Problem> {
    let errors = match file_type {
        FileType::DIMACS => match dimacs::parse::<ParserReport<_>>(parse_options)(input) {
            Ok((rest, problem)) => {
                debug_assert!(rest.is_empty());
                return Some(problem);
            }
            Err(e) => e,
        },
        FileType::AIGER => match aiger::parse::<ParserReport<_>>(parse_options)(input) {
            Ok((rest, aig)) => {
                debug_assert!(rest.is_empty());
                return Some(Problem::AIG(aig));
            }
            Err(e) => e,
        },
    };
    let errors = match errors {
        nom::Err::Error(e) | nom::Err::Failure(e) => e.0,
        nom::Err::Incomplete(_) => unreachable!("only using complete parsers"),
    };

    let range = move |span: &[u8]| {
        let offset = input.offset(span);
        let end = offset + span.len();
        if end >= input.len() {
            offset..offset
        } else {
            offset..end
        }
    };

    let mut labels = Vec::with_capacity(errors.len());
    let mut errors = errors.into_iter().map(|(span, err)| {
        (
            span,
            match err {
                ParserError::Nom(e) => format!("Expected {}", e.description()),
                ParserError::Char(c) => format!("Expected '{c}'"),
                ParserError::Context(msg) => msg.to_string(),
                ParserError::External(s) => s,
            },
        )
    });
    let (span, error) = errors.next().unwrap();
    labels.push(Label::primary((), range(span)).with_message(error.to_string()));
    for (span, error) in errors {
        labels.push(Label::secondary((), range(span)).with_message(error.to_string()));
    }

    let diagnostic = Diagnostic::error()
        .with_message("parsing failed")
        .with_labels(labels);

    let file = SimpleFile::new(file_id, String::from_utf8_lossy(input));
    emit(writer, config, &file, &diagnostic).ok();

    None
}

/// Load and parse the file at `path`, reporting errors to stderr
///
/// Returns `Some(problem)` on success, `None` on error.
pub fn load_file(path: impl AsRef<Path>, options: &ParseOptions) -> Option<Problem> {
    let path = path.as_ref();
    let file_type = match FileType::from_path(path) {
        Some(t) => t,
        None => {
            eprintln!(
                "error: could not determine file type of '{}'",
                path.display()
            );
            return None;
        }
    };

    let src = match std::fs::read(path) {
        Ok(src) => src,
        Err(err) => {
            eprintln!("error: could not read '{}' ({err})", path.display());
            return None;
        }
    };

    let config = codespan_reporting::term::Config::default();
    let writer = StandardStream::stderr(ColorChoice::Auto);
    let mut write_lock = writer.lock();

    parse(
        &src,
        file_type,
        options,
        path.to_string_lossy(),
        &mut write_lock,
        &config,
    )
}
