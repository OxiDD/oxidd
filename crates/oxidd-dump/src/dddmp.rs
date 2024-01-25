//! Im- and export to the [DDDMP] format used by CUDD
//!
//! [DDDMP]: https://github.com/ivmai/cudd/tree/release/dddmp

use std::fmt;
use std::io;
use std::str::FromStr;

use bitvec::prelude::*;
use is_sorted::IsSorted;

use oxidd_core::function::Function;
use oxidd_core::util::AllocResult;
use oxidd_core::util::Borrowed;
use oxidd_core::util::EdgeDropGuard;
use oxidd_core::util::EdgeHashMap;
use oxidd_core::util::EdgeVecDropGuard;
use oxidd_core::util::OutOfMemory;
use oxidd_core::DiagramRules;
use oxidd_core::Edge;
use oxidd_core::HasLevel;
use oxidd_core::InnerNode;
use oxidd_core::LevelNo;
use oxidd_core::Manager;
use oxidd_core::Node;

// spell-checker:ignore varinfo,suppvar,suppvars,suppvarnames
// spell-checker:ignore ordervar,orderedvarnames
// spell-checker:ignore permid,permids,auxids,rootids,rootnames
// spell-checker:ignore nnodes,nvars,nsuppvars,nroots

type FxBuildHasher = std::hash::BuildHasherDefault<rustc_hash::FxHasher>;

/// Encoding of the variable ID and then/else edges in binary format
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
enum Code {
    Terminal,
    AbsoluteID,
    RelativeID,
    Relative1,
}

impl From<u8> for Code {
    fn from(value: u8) -> Self {
        match value {
            0 => Code::Terminal,
            1 => Code::AbsoluteID,
            2 => Code::RelativeID,
            3 => Code::Relative1,
            _ => panic!(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum VarInfo {
    VariableID,
    PermutationID,
    AuxiliaryID,
    VariableName,
    None,
}

/// Information from the header of a DDDMP file
#[derive(Debug)]
pub struct DumpHeader {
    ascii: bool, // .mode A|B
    varinfo: VarInfo,
    dd: String, // optional
    nnodes: usize,
    nvars: u32,
    suppvarnames: Vec<String>,    // optional
    orderedvarnames: Vec<String>, // optional
    ids: Vec<u32>,
    permids: Vec<u32>,
    auxids: Vec<u32>, // optional
    rootids: Vec<isize>,
    rootnames: Vec<String>, // optional

    lines: usize, // number of header lines (including `.nodes`)
}

impl DumpHeader {
    /// Load the DDDMP header of the given input
    ///
    /// This always needs to be called before [`import()`] to retrieve some
    /// necessary information and to position the `input`'s cursor.
    pub fn load(mut input: impl io::BufRead) -> io::Result<Self> {
        let mut header = DumpHeader {
            ascii: true,
            varinfo: VarInfo::None,
            dd: String::new(),
            nnodes: 0,
            nvars: 0,
            suppvarnames: Vec::new(),
            orderedvarnames: Vec::new(),
            ids: Vec::new(),
            permids: Vec::new(),
            auxids: Vec::new(),
            rootids: Vec::new(),
            rootnames: Vec::new(),
            lines: 1,
        };
        let mut nsuppvars = 0;
        let mut nroots = 0;

        let mut line = Vec::new();
        let mut line_no = 1usize;

        loop {
            match input.read_until(b'\n', &mut line) {
                Ok(0) => return err("unexpected end of file"),
                Ok(_) => {}
                Err(e) => return Err(e),
            }
            while let Some(b'\n' | b'\r') = line.last() {
                line.pop();
            }

            let (key, value) = match memchr::memchr2(b' ', b'\t', &line) {
                Some(p) => (&line[..p], &line[p + 1..]),
                None => (&line[..], [].as_slice()),
            };
            let value = trim(value);

            // we don't check for duplicate entries; the last one counts
            match key {
                b".ver" => {
                    if value != b"DDDMP-2.0" {
                        return err(format!(
                            "unsupported version '{}' (line {line_no})",
                            String::from_utf8_lossy(value)
                        ));
                    }
                }
                b".mode" => {
                    header.ascii = match value {
                        b"A" => true,
                        b"B" => false,
                        _ => {
                            return err(format!(
                                "unknown value '{}' for key '.mode' (line {line_no})",
                                String::from_utf8_lossy(value),
                            ))
                        }
                    };
                }
                b".varinfo" => {
                    header.varinfo = match value {
                        b"0" => VarInfo::VariableID,
                        b"1" => VarInfo::PermutationID,
                        b"2" => VarInfo::AuxiliaryID,
                        b"3" => VarInfo::VariableName,
                        b"4" => VarInfo::None,
                        _ => {
                            return err(format!(
                                "unknown value '{}' for key '.varinfo' (line {line_no})",
                                String::from_utf8_lossy(value),
                            ))
                        }
                    };
                }
                b".dd" => header.dd = String::from_utf8_lossy(value).to_string(),
                b".nnodes" => header.nnodes = parse_single_usize(value, line_no)?,
                b".nvars" => header.nvars = parse_single_u32(value, line_no)?,
                b".nsuppvars" => nsuppvars = parse_single_u32(value, line_no)?,
                b".suppvarnames" => {
                    header.suppvarnames = parse_str_list(value, header.nvars as usize);
                }
                b".orderedvarnames" => {
                    header.orderedvarnames = parse_str_list(value, header.nvars as usize);
                }
                b".ids" => {
                    header.ids = parse_u32_list(value, nsuppvars as usize, line_no)?;
                }
                b".permids" => {
                    header.permids = parse_u32_list(value, nsuppvars as usize, line_no)?;
                }
                b".auxids" => {
                    header.auxids = parse_u32_list(value, nsuppvars as usize, line_no)?;
                }
                b".nroots" => nroots = parse_single_usize(value, line_no)?,
                b".rootids" => {
                    header.rootids.clear();
                    parse_edge_list(value, &mut header.rootids, line_no)?;
                }
                b".rootnames" => header.rootnames = parse_str_list(value, nroots),
                b".nodes" => break,
                _ => {
                    return err(format!(
                        "unknown key '{}' (line {line_no})",
                        String::from_utf8_lossy(key),
                    ))
                }
            }

            line_no += 1;
            line.clear();
        }

        header.lines = line_no;

        // validation

        if nsuppvars > header.nvars {
            return err(format!(
                ".nsuppvars ({nsuppvars}) must not be greater than .nvars ({})",
                header.nvars,
            ));
        }

        if header.ids.len() != nsuppvars as usize {
            return err(format!(
            "number of support variables in .ids entry ({}) does not match .nsuppvars ({nsuppvars})",
            header.ids.len(),
        ));
        }
        if header.permids.len() != nsuppvars as usize {
            return err(format!(
            "number of support variables in .permids entry ({}) does not match .nsuppvars ({nsuppvars})",
            header.permids.len(),
        ));
        }
        if !header.auxids.is_empty() && header.auxids.len() != nsuppvars as usize {
            return err(format!(
            "number of support variables in .auxids entry ({}) does not match .nsuppvars ({nsuppvars})",
            header.auxids.len(),
        ));
        }

        if !header.ids.is_empty() {
            if !IsSorted::is_sorted_by(&mut header.ids.iter(), cmp_strict) {
                return err("support variables in .ids must be ascending");
            }
            if *header.ids.last().unwrap() >= header.nvars {
                return err(format!(
                    "support variables in .ids must be less than .nvars ({})",
                    header.nvars,
                ));
            }
        }

        let mut permids_present: BitVec = bitvec![0; header.nvars as usize];
        for &id in &header.permids {
            if id >= header.nvars {
                return err(format!(
                    "support variables in .permids must be less than .nvars ({})",
                    header.nvars,
                ));
            }
            if permids_present[id as usize] {
                return err(format!("variable ({id}) occurs twice in .permids"));
            }
            permids_present.set(id as usize, true);
        }

        if !header.orderedvarnames.is_empty()
            && header.orderedvarnames.len() != header.nvars as usize
        {
            return err(format!(
                "number of variables in .orderedvarnames entry ({}) does not match .nvars ({})",
                header.orderedvarnames.len(),
                header.nvars
            ));
        }
        if !header.suppvarnames.is_empty() {
            if header.suppvarnames.len() != nsuppvars as usize {
                return err(format!(
                "number of variables in .suppvarnames entry ({}) does not match .nsuppvars ({nsuppvars})",
                header.suppvarnames.len(),
            ));
            }

            if !header.orderedvarnames.is_empty() {
                for (i, suppvar) in header.suppvarnames.iter().enumerate() {
                    let permid = header.permids[i];
                    let ordervar = &header.orderedvarnames[permid as usize];
                    if suppvar != ordervar {
                        return err(format!(
                            ".suppvarnames and .orderedvarnames do not match \
                        (entry {i} of .suppvarnames is '{suppvar}' \
                        but the permuted ID is {permid} with name '{ordervar}')",
                        ));
                    }
                }
            }
        }

        if header.rootids.len() != nroots {
            return err(format!(
                "number of roots in .rootids entry ({}) does not match .nroots ({nroots})",
                header.rootids.len(),
            ));
        }
        for &id in &header.rootids {
            if id == 0 {
                return err(".rootids must not be 0");
            }
            if id.unsigned_abs() > header.nnodes {
                return err(format!(
                    "entry in .rootids out of range ({id} > {})",
                    header.nnodes,
                ));
            }
        }
        if !header.rootnames.is_empty() && header.rootnames.len() != nroots {
            return err(format!(
                "number of roots in .rootnames entry ({}) does not match .nroots ({nroots})",
                header.rootnames.len(),
            ));
        }

        Ok(header)
    }

    /// Name of the decision diagram
    ///
    /// Corresponds to the DDDMP `.dd` field.
    pub fn diagram_name(&self) -> Option<&str> {
        if self.dd.is_empty() {
            None
        } else {
            Some(&self.dd)
        }
    }

    /// Number of nodes in the dumped decision diagram
    ///
    /// Corresponds to the DDDMP `.nnodes` field.
    pub fn num_nodes(&self) -> usize {
        self.nnodes
    }

    /// Number of all variables in the exported decision diagram
    ///
    /// Corresponds to the DDDMP `.nvars` field.
    pub fn num_vars(&self) -> u32 {
        self.nvars
    }

    /// Number of variables in the true support of the decision diagram
    ///
    /// Corresponds to the DDDMP `.nsuppvars` field.
    pub fn num_support_vars(&self) -> u32 {
        self.ids.len() as _
    }

    /// Variables in the true support of the decision diagram. Concretely, these
    /// are indices of the original variable numbering. Hence, the returned
    /// slice contains [`DumpHeader::num_support_vars()`] integers in strictly
    /// ascending order.
    ///
    /// Example: Consider a decision diagram that was created with the variables
    /// `x`, `y` and `z`, in this order (`x` is the top-most variable). Suppose
    /// that only `y` and `z` are used by the dumped functions. Then this
    /// function returns `[1, 2]` regardless of any subsequent reordering.
    ///
    /// Corresponds to the DDDMP `.ids` field.
    pub fn support_vars(&self) -> &[u32] {
        &self.ids
    }

    /// Permutation of the variables in the true support of the decision
    /// diagram. The returned slice is always [`DumpHeader::num_support_vars()`]
    /// elements long. If the value at the `i`th index is `l`, then the `i`th
    /// support variable is at level `l` in the dumped decision diagram. By the
    /// `i`th support variable, we mean the variable `header.support_vars()[i]`
    /// in the original numbering.
    ///
    /// Example: Consider a decision diagram that was created with the variables
    /// `x`, `y` and `z` (`x` is the top-most variable). The variables were
    /// re-ordered to `z`, `x`, `y`. Suppose that only `y` and `z` are used by
    /// the dumped functions. Then this function returns `[2, 0]`.
    ///
    /// Corresponds to the DDDMP `.permids` field.
    pub fn support_var_permutation(&self) -> &[u32] {
        &self.permids
    }

    /// Auxiliary variable IDs. The returned slice contains
    /// [`DumpHeader::num_support_vars()`] elements.
    ///
    /// Corresponds to the DDDMP `.auxids` field.
    pub fn auxiliary_var_ids(&self) -> &[u32] {
        &self.auxids
    }

    /// Names of variables in the true support of the decision diagram. If
    /// present, the returned slice contains [`DumpHeader::num_support_vars()`]
    /// many elements. The order is the "original" variable order.
    ///
    /// Example: Consider a decision diagram that was created with the variables
    /// `x`, `y` and `z`, in this order (`x` is the top-most variable). Suppose
    /// that only `y` and `z` are used by the dumped functions. Then this
    /// function returns `["y", "z"]` regardless of any subsequent reordering.
    ///
    /// Corresponds to the DDDMP `.suppvarnames` field.
    pub fn support_var_names(&self) -> Option<&[String]> {
        if self.suppvarnames.is_empty() {
            None
        } else {
            Some(&self.suppvarnames[..])
        }
    }

    /// Names of all variables in the exported decision diagram. If present, the
    /// returned slice contains [`DumpHeader::num_vars()`] many elements. The
    /// order is the "current" variable order, i.e. the first name corresponds
    /// to the top-most variable in the dumped decision diagram.
    ///
    /// Example: Consider a decision diagram that was created with the variables
    /// `x`, `y` and `z` (`x` is the top-most variable). The variables were
    /// re-ordered to `z`, `x`, `y`. In this case, the function returns
    /// `["z", "x", "y"]` (even if some of the variables are unused by the
    /// dumped functions).
    ///
    /// Corresponds to the DDDMP `.nvars` field.
    pub fn ordered_var_names(&self) -> Option<&[String]> {
        if self.orderedvarnames.is_empty() {
            None
        } else {
            Some(&self.orderedvarnames[..])
        }
    }

    /// Number of roots
    ///
    /// [`import()`] returns this number of roots on success.
    /// Corresponds to the DDDMP `.nroots` field.
    pub fn num_roots(&self) -> usize {
        self.rootids.len()
    }

    /// Names of roots, if present, in the same order as returned by
    /// [`import()`]
    ///
    /// Corresponds to the DDDMP `.rootnames` field.
    pub fn root_names(&self) -> Option<&[String]> {
        if self.rootnames.is_empty() {
            None
        } else {
            Some(&self.rootnames[..])
        }
    }
}

/// Helper function to return a parse error
fn err<T>(msg: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidData, msg))
}

/// Compare, mapping `Equal` to `Greater`
fn cmp_strict<T: Ord>(a: &T, b: &T) -> Option<std::cmp::Ordering> {
    Some(a.cmp(b).then(std::cmp::Ordering::Greater))
}

/// Like [`std::fmt::Display`], but the format should use ASCII characters only
pub trait AsciiDisplay {
    /// Format the value with the given formatter
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error>;
}

struct Ascii<T>(T);
impl<T: AsciiDisplay> fmt::Display for Ascii<&T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Export the decision diagram in `manager` to `file`
///
/// `ascii` indicates whether to use the ASCII or binary format.
///
/// `dd_name` is the name that is output to the `.dd` field, unless it is an
/// empty string.
///
/// `vars` are edges representing *all* variables in the decision diagram. The
/// order does not matter. `var_names` are the names of these variables
/// (optional). If given, there must be `vars.len()` names in the same order as
/// in `vars`.
///
/// `functions` are edges pointing to the root nodes of functions.
/// `function_names` are the corresponding names (optional). If given, there
/// must be `functions.len()` names in the same order as in `function_names`.
///
/// `is_complemented` is a function that returns whether an edge is
/// complemented.
#[allow(clippy::too_many_arguments)] // FIXME: use a builder pattern
pub fn export<'id, F: Function>(
    mut file: impl io::Write,
    manager: &F::Manager<'id>,
    ascii: bool,
    dd_name: &str,
    vars: &[&F],
    var_names: Option<&[&str]>,
    functions: &[&F],
    function_names: Option<&[&str]>,
    is_complemented: impl Fn(&<F::Manager<'id> as Manager>::Edge) -> bool,
) -> io::Result<()>
where
    <F::Manager<'id> as Manager>::InnerNode: HasLevel,
    <F::Manager<'id> as Manager>::Terminal: AsciiDisplay,
{
    assert!(var_names.is_none() || var_names.unwrap().len() == vars.len());
    assert!(function_names.is_none() || function_names.unwrap().len() == functions.len());
    assert!(
        ascii || <F::Manager<'id> as Manager>::InnerNode::ARITY == 2,
        "binary mode is (currently) only supported for binary nodes"
    );

    writeln!(file, ".ver DDDMP-2.0")?;
    writeln!(file, ".mode {}", if ascii { 'A' } else { 'B' })?;

    // TODO: other .varinfo modes?
    writeln!(file, ".varinfo {}", VarInfo::None as u32)?;

    if !dd_name.is_empty() {
        let invalid = |c: char| !c.is_ascii() || c.is_ascii_control();
        if dd_name.chars().any(invalid) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "decision diagram name must be ASCII text without control characters",
            ));
        }
        writeln!(file, ".dd {dd_name}")?;
    }

    let nvars = manager.num_levels();
    assert!(nvars as usize == vars.len());

    // Map from the current level number to its internal var index (almost the
    // same numbering, but vars not in the support removed, i.e. range
    // `0..nsuppvars`), and the nodes (as edges) contained at this level
    // together with their indexes.
    let mut node_map: Vec<(LevelNo, EdgeHashMap<F::Manager<'id>, usize, FxBuildHasher>)> =
        (0..nvars).map(|_| (0, EdgeHashMap::new(manager))).collect();
    let mut terminal_map: EdgeHashMap<F::Manager<'id>, usize, FxBuildHasher> =
        EdgeHashMap::new(manager);

    fn rec_add_map<M: Manager>(
        manager: &M,
        node_map: &mut Vec<(u32, EdgeHashMap<M, usize, FxBuildHasher>)>,
        terminal_map: &mut EdgeHashMap<M, usize, FxBuildHasher>,
        e: Borrowed<M::Edge>,
    ) where
        M::InnerNode: HasLevel,
    {
        match manager.get_node(&e) {
            Node::Inner(node) => {
                let (_, map) = &mut node_map[node.level() as usize];
                // Map to 0 -> we assign the indexes below
                let res = map.insert(&e.with_tag(Default::default()), 0);
                if res.is_none() {
                    for e in node.children() {
                        rec_add_map::<M>(manager, node_map, terminal_map, e);
                    }
                }
            }
            Node::Terminal(_) => {
                terminal_map.insert(&e.with_tag(Default::default()), 0);
            }
        }
    }

    for &func in functions {
        rec_add_map::<F::Manager<'id>>(
            manager,
            &mut node_map,
            &mut terminal_map,
            func.as_edge(manager).borrowed(),
        );
    }

    let mut nnodes = 0;
    for (_, idx) in &mut terminal_map {
        nnodes += 1; // pre increment -> numbers in range 1..=#nodes
        *idx = nnodes;
    }
    let nsuppvars = node_map.iter().filter(|(_, m)| !m.is_empty()).count() as u32;
    let mut suppvars: BitVec = bitvec![0; nvars as usize];
    let mut suppvar_idx = nsuppvars;
    // rev() -> assign node IDs bottom-up
    for (i, (var_idx, level)) in node_map.iter_mut().enumerate().rev() {
        if level.is_empty() {
            continue;
        }
        suppvar_idx -= 1;
        *var_idx = suppvar_idx;
        suppvars.set(i, true);
        for (_, idx) in level.iter_mut() {
            nnodes += 1; // pre increment -> numbers in range 1..=#nodes
            *idx = nnodes;
        }
    }
    writeln!(file, ".nnodes {nnodes}")?;
    writeln!(file, ".nvars {nvars}")?;
    writeln!(file, ".nsuppvars {nsuppvars}")?;

    if let Some(var_names) = var_names {
        let mut ordered_var_names = Vec::new();
        ordered_var_names.resize(var_names.len(), "");

        write!(file, ".suppvarnames")?;
        for (&f, &name) in vars.iter().zip(var_names) {
            let invalid =
                |c: char| !c.is_ascii() || c.is_ascii_control() || c.is_ascii_whitespace();
            if name.is_empty() || name.chars().any(invalid) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "variable name must be ASCII text without control characters and spaces",
                ));
            }
            let node = manager
                .get_node(f.as_edge(manager))
                .expect_inner("variables must not be terminals");
            let level = node.level() as usize;
            assert!(
                ordered_var_names[level].is_empty(),
                "variables must not be given multiple times"
            );
            ordered_var_names[level] = name;
            if suppvars[level] {
                write!(file, " {name}")?;
            }
        }
        writeln!(file)?;

        write!(file, ".orderedvarnames")?;
        for name in ordered_var_names {
            write!(file, " {name}")?;
        }
        writeln!(file)?;
    }

    write!(file, ".ids")?;
    for (id, &f) in vars.iter().enumerate() {
        let node = manager
            .get_node(f.as_edge(manager))
            .expect_inner("variables must not be terminals");
        if suppvars[node.level() as usize] {
            write!(file, " {id}")?;
        }
    }
    writeln!(file)?;

    write!(file, ".permids")?;
    for &f in vars {
        let level = manager.get_node(f.as_edge(manager)).unwrap_inner().level();
        if suppvars[level as usize] {
            write!(file, " {level}")?;
        }
    }
    writeln!(file)?;

    // TODO: .auxids?

    let idx = |e: &<F::Manager<'id> as Manager>::Edge| {
        let idx = match manager.get_node(e) {
            Node::Inner(node) => {
                let (_, map) = &node_map[node.level() as usize];
                *map.get(&e.with_tag(Default::default())).unwrap() as isize
            }
            Node::Terminal(_) => {
                *terminal_map.get(&e.with_tag(Default::default())).unwrap() as isize
            }
        };
        if is_complemented(e) {
            -idx
        } else {
            idx
        }
    };
    let bin_idx = |e: &<F::Manager<'id> as Manager>::Edge, node_id: usize| {
        let idx = match manager.get_node(e) {
            Node::Inner(node) => {
                let (_, map) = &node_map[node.level() as usize];
                *map.get(&e.with_tag(Default::default())).unwrap()
            }
            Node::Terminal(_) => {
                if manager.num_terminals() == 1 {
                    // TODO: This may be bad in case there could be more
                    // terminals for the decision diagram type (i.e. the value
                    // returned by `manager.num_terminals()` is not always 1),
                    // but currently, there is only a single one.
                    return (Code::Terminal, 0);
                }
                todo!();
            }
        };
        if idx == node_id - 1 {
            (Code::Relative1, 0)
        } else if node_id - idx < idx {
            (Code::RelativeID, node_id - idx)
        } else {
            (Code::AbsoluteID, idx)
        }
    };

    writeln!(file, ".nroots {}", functions.len())?;
    write!(file, ".rootids")?;
    for &func in functions {
        write!(file, " {}", idx(func.as_edge(manager)))?;
    }
    writeln!(file)?;
    if let Some(function_names) = function_names {
        write!(file, ".rootnames")?;
        for &name in function_names {
            let invalid =
                |c: char| !c.is_ascii() || c.is_ascii_control() || c.is_ascii_whitespace();
            if name.is_empty() || name.chars().any(invalid) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "function name must be ASCII text without control characters and spaces",
                ));
            }
            write!(file, " {}", name)?;
        }
        writeln!(file)?;
    }

    writeln!(file, ".nodes")?;

    #[inline]
    const fn node_code(var: Code, t: Code, e_complement: bool, e: Code) -> u8 {
        (var as u8) << 5 | (t as u8) << 3 | (e_complement as u8) << 2 | e as u8
    }

    let mut exported_nodes = 0;
    for (edge, &node_id) in terminal_map.iter() {
        // This implementation relies on that the iteration order for
        // `EdgeHashMap`s stays the same as long as no elements are
        // inserted/removed. To be sure, we assert that this assumption holds.
        assert_eq!(exported_nodes + 1, node_id);
        if ascii {
            // <Node-index> [<Var-extra-info>] <Var-internal-index> <Then-index>
            // <Else-index>
            let node = manager.get_node(edge);
            let desc = Ascii(node.unwrap_terminal());
            writeln!(file, "{node_id} {desc} 0 0")?;
        } else {
            let terminal = node_code(Code::Terminal, Code::Terminal, false, Code::Terminal);
            write_escaped(&mut file, &[terminal])?;
        }
        exported_nodes += 1;
    }
    // Work from bottom to top
    for &(var_idx, ref level) in node_map.iter().rev() {
        for (e, &node_id) in level.iter() {
            assert_eq!(exported_nodes + 1, node_id);
            let node = manager.get_node(e).unwrap_inner();
            if ascii {
                // <Node-index> [<Var-extra-info>] <Var-internal-index> <Then-index>
                // <Else-index>
                write!(file, "{node_id} {var_idx}")?;
                for child in node.children() {
                    write!(file, " {}", idx(&child))?;
                }
                writeln!(file)?;
            } else {
                let mut iter = node.children();
                let t = iter.next().unwrap();
                let t_lvl = manager.get_node(&t).level();
                let e = iter.next().unwrap();
                let e_lvl = manager.get_node(&e).level();
                debug_assert!(iter.next().is_none());

                let mut var_code = Code::AbsoluteID;
                let mut var_idx = var_idx;
                let min_lvl = std::cmp::min(t_lvl, e_lvl);
                if min_lvl != LevelNo::MAX {
                    let (min_var_idx, _) = node_map[min_lvl as usize];
                    if var_idx == min_var_idx - 1 {
                        var_code = Code::Relative1;
                    } else if min_var_idx - var_idx < var_idx {
                        var_code = Code::RelativeID;
                        var_idx = min_var_idx - var_idx;
                    }
                }

                let (t_code, t_idx) = bin_idx(&t, node_id);
                let (e_code, e_idx) = bin_idx(&e, node_id);

                debug_assert!(!is_complemented(&t));
                write_escaped(
                    &mut file,
                    &[node_code(var_code, t_code, is_complemented(&e), e_code)],
                )?;
                if var_code == Code::AbsoluteID || var_code == Code::RelativeID {
                    encode_7bit(&mut file, var_idx as usize)?;
                }
                if t_code == Code::AbsoluteID || t_code == Code::RelativeID {
                    encode_7bit(&mut file, t_idx)?;
                }
                if e_code == Code::AbsoluteID || e_code == Code::RelativeID {
                    encode_7bit(&mut file, e_idx)?;
                }
                assert!(u64::BITS >= usize::BITS);
            }
            exported_nodes += 1;
        }
    }

    writeln!(file, ".end")?;

    Ok(())
}

/// 7-bit encode an integer and write it (escaped) via `writer`
fn encode_7bit(writer: impl io::Write, mut value: usize) -> io::Result<()> {
    let mut buf = [0u8; (usize::BITS as usize + 8 - 1) / 7];
    let mut idx = buf.len() - 1;
    buf[idx] = (value as u8).wrapping_shl(1);
    value >>= 7;
    while value != 0 {
        idx -= 1;
        buf[idx] = (value as u8).wrapping_shl(1) | 1;
        value >>= 7;
    }
    write_escaped(writer, &buf[idx..])
}

/// Write an DDDMP-style escaped byte
fn write_escaped(mut writer: impl io::Write, buf: &[u8]) -> io::Result<()> {
    for &c in buf {
        match c {
            0x00 => writer.write_all(&[0x00, 0x00])?,
            0x0a => writer.write_all(&[0x00, 0x01])?,
            0x0d => writer.write_all(&[0x00, 0x02])?,
            0x1a => writer.write_all(&[0x00, 0x03])?,
            _ => writer.write_all(&[c])?,
        }
    }
    Ok(())
}

/// Read and unescape a byte. Counterpart of [`write_escaped()`]
fn read_unescape(input: impl io::Read) -> io::Result<u8> {
    let mut bytes = input.bytes();
    let eof_msg = "unexpected end of file";
    match bytes.next() {
        None => return err(eof_msg),
        Some(Ok(0)) => {}
        Some(res) => return res,
    };
    match bytes.next() {
        None => err(eof_msg),
        Some(Ok(0x00)) => Ok(0x00),
        Some(Ok(0x01)) => Ok(0x0a),
        Some(Ok(0x02)) => Ok(0x0d),
        Some(Ok(0x03)) => Ok(0x1a),
        Some(Ok(b)) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid escape sequence 0x00 0x{b:x}"),
        )),
        Some(e) => e,
    }
}

/// 7-bit decode a number from `input` (unescaping the bytes via
/// [`read_unescape()`])
fn decode_7bit(mut input: impl io::Read) -> io::Result<usize> {
    let mut res = 0usize;
    loop {
        let b = read_unescape(&mut input)?;
        match res.checked_shl(7) {
            Some(v) => res = v,
            None => return err("integer too large"),
        }
        res |= (b >> 1) as usize;
        if b & 1 == 0 {
            return Ok(res);
        }
    }
}

/// Import the decision diagram from `input` into `manager` after loading
/// `header` from `input`
///
/// Important: there must not be any read/seek/... operations on `input` after
/// reading `header` using [`DumpHeader::load()`].
///
/// `support_vars` contains edges representing the variables in the true support
/// of the decision diagram in `input`. The variables must be ordered by their
/// current level (lower level numbers first).
///
/// `complement` is a function that returns the complemented edge for a given
/// edge.
pub fn import<'id, F: Function>(
    mut input: impl io::BufRead,
    header: &DumpHeader,
    manager: &F::Manager<'id>,
    support_vars: &[F],
    complement: impl Fn(
        &F::Manager<'id>,
        <F::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<F::Manager<'id> as Manager>::Edge>,
) -> io::Result<Vec<F>>
where
    <F::Manager<'id> as Manager>::InnerNode: HasLevel,
    <F::Manager<'id> as Manager>::Terminal: FromStr,
{
    assert_eq!(support_vars.len(), header.ids.len());
    let suppvar_level_map: Vec<LevelNo> = support_vars
        .iter()
        .map(|f| {
            let nw = manager
                .get_node(f.as_edge(manager))
                .expect_inner("variables must not be terminals");
            nw.level()
        })
        .collect();
    assert!(IsSorted::is_sorted_by(
        &mut suppvar_level_map.iter(),
        cmp_strict
    ));

    let nodes = if header.ascii {
        import_ascii::<F::Manager<'id>>(
            &mut input,
            header,
            manager,
            &suppvar_level_map,
            &complement,
        )
    } else {
        let mut level_suppvar_map = vec![u32::MAX; manager.num_levels() as usize];
        for (i, &level) in suppvar_level_map.iter().enumerate() {
            level_suppvar_map[level as usize] = i as u32;
        }
        import_bin::<F::Manager<'id>>(
            &mut input,
            header,
            manager,
            &level_suppvar_map,
            &suppvar_level_map,
            &complement,
        )
    }?;
    let nodes = EdgeVecDropGuard::new(manager, nodes);
    debug_assert_eq!(nodes.len(), header.nnodes);

    let mut buf = Vec::with_capacity(8);
    input.read_to_end(&mut buf)?;
    if &buf[..4] != b".end" || !buf[4..].iter().all(|b| b.is_ascii_whitespace()) {
        return err("file must end with '.end'");
    }

    let mut roots = Vec::with_capacity(header.rootids.len());
    for &root in &header.rootids {
        debug_assert_ne!(root, 0);
        let node_index = root.unsigned_abs() - 1;
        let e = manager.clone_edge(&nodes[node_index]);
        roots.push(F::from_edge(
            manager,
            if root > 0 {
                e
            } else {
                match complement(manager, e) {
                    Ok(e) => e,
                    Err(OutOfMemory) => return Err(io::ErrorKind::OutOfMemory.into()),
                }
            },
        ));
    }

    Ok(roots)
}

fn import_ascii<M: Manager>(
    mut input: impl io::BufRead,
    header: &DumpHeader,
    manager: &M,
    suppvar_level_map: &[LevelNo],
    complement: impl Fn(&M, M::Edge) -> AllocResult<M::Edge>,
) -> io::Result<Vec<M::Edge>>
where
    M::InnerNode: HasLevel,
    M::Terminal: FromStr,
{
    let mut nodes: Vec<M::Edge> = Vec::with_capacity(header.nnodes);
    let mut line = Vec::new();
    let mut line_no = header.lines + 1;
    let mut children = Vec::with_capacity(M::InnerNode::ARITY);
    for node_id in 1..=header.nnodes {
        match input.read_until(b'\n', &mut line) {
            Ok(0) => return err("unexpected end of file"),
            Ok(_) => {}
            Err(e) => return Err(e),
        }
        while let Some(b'\n' | b'\r') = line.last() {
            line.pop();
        }

        let (rest, node_id_lineno) = parse_usize(&line, line_no)?;
        if node_id_lineno != node_id {
            return err(format!("expected node ID {node_id} on line {line_no}"));
        }

        let rest = trim_start(rest);
        let rest = match header.varinfo {
            VarInfo::VariableID
            | VarInfo::PermutationID
            | VarInfo::AuxiliaryID
            | VarInfo::VariableName => match memchr::memchr2(b' ', b'\t', rest) {
                Some(pos) => &rest[pos + 1..],
                None => {
                    return err(format!(
                        "expected a space after variable extra info (line {line_no})"
                    ))
                }
            },
            VarInfo::None => rest,
        };

        let rest = trim_start(rest);
        let (rest, var_id) = match memchr::memchr2(b' ', b'\t', rest) {
            Some(pos) => (&rest[pos + 1..], &rest[..pos]),
            None => {
                return err(format!(
                    "expected a space after variable internal index (line {line_no})"
                ))
            }
        };

        parse_edge_list(rest, &mut children, line_no)?;
        if children.len() != M::InnerNode::ARITY {
            return err(format!(
                "expected {} children, got {} (line {line_no})",
                M::InnerNode::ARITY,
                children.len()
            ));
        }

        let res = if children.iter().any(|&c| c == 0) {
            // terminal
            let string = match std::str::from_utf8(var_id) {
                Ok(s) => s,
                Err(_) => {
                    return err(format!(
                        "terminal description must be valid UTF-8 (line {line_no})"
                    ));
                }
            };
            let terminal: M::Terminal = match string.parse() {
                Ok(t) => t,
                Err(_) => return err(format!("invalid terminal description (line {line_no})")),
            };
            manager.get_terminal(terminal)
        } else {
            let (_, var_id) = parse_u32(var_id, line_no)?;
            let Some(&level) = suppvar_level_map.get(var_id as usize) else {
                return err(format!("variable out of range (line {line_no})"));
            };

            for &child in &children {
                let child = child.unsigned_abs();
                if child >= node_id {
                    return err(format!(
                        "children ids must be less than node ({child} >= {node_id}, line {line_no})",
                    ));
                }
                let child_level = manager.get_node(&nodes[child - 1]).level();
                if level >= child_level {
                    return err(format!(
                        "node level must be less than the children's levels ({level} >= {child_level}, line {line_no})",
                    ));
                }
            }

            <M::Rules as DiagramRules<_, _, _>>::reduce(
                manager,
                level,
                children.iter().map(|&child| {
                    debug_assert_ne!(child, 0);
                    let e = manager.clone_edge(&nodes[child.unsigned_abs() - 1]);
                    if child < 0 {
                        complement(manager, e).unwrap()
                    } else {
                        e
                    }
                }),
            )
            .then_insert(manager, level)
        };
        let node = match res {
            Ok(e) => e,
            Err(OutOfMemory) => {
                for e in nodes {
                    manager.drop_edge(e);
                }
                return Err(io::ErrorKind::OutOfMemory.into());
            }
        };

        nodes.push(node);

        children.clear();
        line.clear();
        line_no += 1;
    }

    Ok(nodes)
}

fn import_bin<M: Manager>(
    mut input: impl io::BufRead,
    header: &DumpHeader,
    manager: &M,
    level_suppvar_map: &[u32],
    suppvar_level_map: &[LevelNo],
    complement: impl Fn(&M, M::Edge) -> AllocResult<M::Edge>,
) -> io::Result<Vec<M::Edge>>
where
    M::InnerNode: HasLevel,
{
    assert_eq!(
        M::InnerNode::ARITY,
        2,
        "binary mode is (currently) only supported for binary nodes"
    );
    let terminal = {
        let msg = "binary mode is (currently) only supported for diagrams with a single terminal";
        let mut it = manager.terminals();
        let t = EdgeDropGuard::new(manager, it.next().expect(msg));
        assert!(it.next().is_none(), "{msg}");
        t
    };

    fn idx(input: impl io::BufRead, node_id: usize, code: Code) -> io::Result<usize> {
        debug_assert!(node_id >= 1);
        let id = match code {
            Code::Terminal => 1,
            Code::AbsoluteID => decode_7bit(input)?,
            Code::RelativeID => node_id - decode_7bit(input)?,
            Code::Relative1 => node_id - 1,
        };
        if id == 0 {
            return err("then/else ID must not be 0");
        }
        if id >= node_id {
            return err("then/else ID too large");
        }
        Ok((id - 1) as usize)
    }

    let mut nodes = EdgeVecDropGuard::new(manager, Vec::with_capacity(header.nnodes));
    for node_id in 1..=header.nnodes {
        let node_code = read_unescape(&mut input)?;
        let var_code = Code::from((node_code >> 5) & 0b11);
        let t_code = Code::from((node_code >> 3) & 0b11);
        let e_complement = ((node_code >> 2) & 0b1) != 0;
        let e_code = Code::from(node_code & 0b11);

        let vid = match var_code {
            Code::Terminal => {
                nodes.push(manager.clone_edge(&terminal));
                continue;
            }
            Code::AbsoluteID | Code::RelativeID => decode_7bit(&mut input)?,
            Code::Relative1 => 1,
        };

        let t = manager.clone_edge(&nodes[idx(&mut input, node_id, t_code)?]);
        let t_level = manager.get_node(&t).level();
        let e = manager.clone_edge(&nodes[idx(&mut input, node_id, e_code)?]);
        let e_level = manager.get_node(&e).level();
        let e = if e_complement {
            match complement(manager, e) {
                Ok(e) => e,
                Err(OutOfMemory) => {
                    return Err(io::ErrorKind::OutOfMemory.into());
                }
            }
        } else {
            e
        };

        let vid = match var_code {
            Code::Terminal => unreachable!(),
            Code::AbsoluteID if vid >= suppvar_level_map.len() => {
                return err("variable ID out of range")
            }
            Code::AbsoluteID => vid,
            Code::RelativeID | Code::Relative1 => {
                let min_level = std::cmp::min(t_level, e_level);
                let child_min_suppvar = if min_level == LevelNo::MAX {
                    level_suppvar_map.len()
                } else {
                    level_suppvar_map[min_level as usize] as usize
                };
                match child_min_suppvar.checked_sub(vid) {
                    Some(v) => v,
                    None => return err("variable ID out of range"),
                }
            }
        };

        let level = suppvar_level_map[vid as usize];
        if level >= t_level || level >= e_level {
            return err("node level must be less than the children's levels");
        }

        match <M::Rules as DiagramRules<_, _, _>>::reduce(manager, level, [t, e])
            .then_insert(manager, level)
        {
            Ok(e) => nodes.push(e),
            Err(OutOfMemory) => {
                return Err(io::ErrorKind::OutOfMemory.into());
            }
        }
    }

    Ok(nodes.into_vec())
}

/// Remove leading spaces and tabs
const fn trim_start(mut s: &[u8]) -> &[u8] {
    while let [b' ' | b'\t', rest @ ..] = s {
        s = rest;
    }
    s
}

/// Remove trailing spaces and tabs
const fn trim_end(mut s: &[u8]) -> &[u8] {
    while let [rest @ .., b' ' | b'\t'] = s {
        s = rest;
    }
    s
}

/// Remove leading and trailing spaces and tabs
const fn trim(s: &[u8]) -> &[u8] {
    trim_end(trim_start(s))
}

/// Parse a list of strings
fn parse_str_list(input: &[u8], capacity: usize) -> Vec<String> {
    let mut res = Vec::with_capacity(capacity);
    let mut start = 0;
    for pos in memchr::memchr2_iter(b' ', b'\t', input).chain([input.len()]) {
        // skip empty strings
        if pos != start {
            res.push(String::from_utf8_lossy(&input[start..pos]).to_string())
        }
        start = pos + 1;
    }
    res
}

macro_rules! parse_unsigned {
    ($f:ident, $t:ty) => {
        /// Parse an unsigned integer
        ///
        /// Returns the remaining input and the parsed value on success. Fails
        /// if no integer is present, the integer is too large for the return
        /// type, or the input contains non-digit characters (after the leading
        /// spaces/tabs and before the first space/tab after the number).
        fn $f(input: &[u8], line_no: usize) -> io::Result<(&[u8], $t)> {
            let mut res: $t = 0;
            let mut num = false;
            let mut consumed = 0;
            for &c in input {
                match c {
                    b'0'..=b'9' => {
                        num = true;
                        match res
                            .checked_mul(10)
                            .and_then(|v| v.checked_add((c - b'0') as _))
                        {
                            Some(v) => res = v,
                            None => {
                                return err(format!(
                                    "integer '{}' too large (line {line_no})",
                                    String::from_utf8_lossy(input),
                                ));
                            }
                        }
                    }
                    b' ' | b'\t' if num => break,
                    b' ' | b'\t' => {}
                    _ => {
                        return err(format!(
                            "unexpected char '{}' in integer (line {line_no})",
                            c as char,
                        ));
                    }
                }
                consumed += 1;
            }
            if !num {
                return err(format!(
                    "expected an integer, got an empty string (line {line_no})"
                ));
            }
            Ok((&input[consumed..], res))
        }
    };
}
parse_unsigned!(parse_u32, u32);
parse_unsigned!(parse_usize, usize);

macro_rules! parse_single_unsigned {
    ($f:ident, $t:ty) => {
        /// Parse a single unsigned integer (no spaces etc. allowed)
        fn $f(input: &[u8], line_no: usize) -> io::Result<$t> {
            let mut res: $t = 0;
            let mut num = false;
            for &c in input {
                match c {
                    b'0'..=b'9' => {
                        num = true;
                        match res
                            .checked_mul(10)
                            .and_then(|v| v.checked_add((c - b'0') as _))
                        {
                            Some(v) => res = v,
                            None => {
                                return err(format!(
                                    "integer '{}' too large (line {line_no})",
                                    String::from_utf8_lossy(input),
                                ));
                            }
                        }
                    }
                    _ => {
                        return err(format!(
                            "unexpected char '{}' in integer (line {line_no})",
                            c as char,
                        ));
                    }
                }
            }
            if !num {
                return err(format!("expected an integer (line {line_no})"));
            }
            Ok(res)
        }
    };
}
parse_single_unsigned!(parse_single_u32, u32);
parse_single_unsigned!(parse_single_usize, usize);

/// Parse a space (or tab) separated list of integers
fn parse_u32_list(input: &[u8], capacity: usize, line_no: usize) -> io::Result<Vec<u32>> {
    let mut res = Vec::with_capacity(capacity);
    let mut i = 0u32;
    let mut num = false;

    for &c in input {
        match c {
            b'0'..=b'9' => {
                num = true;
                match i
                    .checked_mul(10)
                    .and_then(|v| v.checked_add((c - b'0') as _))
                {
                    Some(v) => i = v,
                    None => {
                        return err(format!("integer too large (line {line_no})"));
                    }
                }
            }
            b' ' | b'\t' => {
                if num {
                    res.push(i);
                    num = false;
                    i = 0;
                }
            }
            _ => {
                return err(format!(
                    "unexpected char '{}' in space-separated list of integers (line {line_no})",
                    c as char,
                ));
            }
        }
    }
    if num {
        res.push(i);
    }

    Ok(res)
}

/// Parse a space (or tab) separated list of edges into `dst`
///
/// The input consists of non-zero decimal integers, possibly negative. If the
/// value is negative, this indicates a complemented edge. In this case, the
/// boolean in the resulting entry is `true`.
///
/// `line_no` is only used for error reporting.
fn parse_edge_list(input: &[u8], dst: &mut Vec<isize>, line_no: usize) -> io::Result<()> {
    let mut i = 0isize;
    let mut neg = false;
    let mut num = false;

    for &c in input {
        match c {
            b'0'..=b'9' => {
                num = true;
                match i
                    .checked_mul(10)
                    .and_then(|v| v.checked_add((c - b'0') as isize))
                {
                    Some(v) => i = v,
                    None => {
                        return err(format!("integer too large (line {line_no})"));
                    }
                }
            }
            b'-' => {
                if neg {
                    return err(format!("expected a digit after '-' (line {line_no})"));
                }
                if num {
                    return err(format!("expected a space before '-' (line {line_no})"));
                }
                neg = true;
            }
            b' ' | b'\t' => {
                if num {
                    dst.push(if neg { -i } else { i });
                    num = false;
                    neg = false;
                    i = 0;
                }
            }
            _ => {
                return err(format!(
                    "unexpected char '{}' in space-separated list of integers (line {line_no})",
                    c as char,
                ));
            }
        }
    }
    if num {
        dst.push(if neg { -i } else { i });
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_encode_7bit() -> io::Result<()> {
        let mut buf: Vec<u8> = Vec::new();

        if usize::BITS == 64 {
            encode_7bit(&mut buf, usize::MAX)?;
            assert_eq!(
                &buf[..],
                &[
                    0b0000_0011, //  1
                    0b1111_1111, //  8
                    0b1111_1111, // 15
                    0b1111_1111, // 22
                    0b1111_1111, // 29
                    0b1111_1111, // 36
                    0b1111_1111, // 43
                    0b1111_1111, // 50
                    0b1111_1111, // 57
                    0b1111_1110, // 64
                ]
            );
        }

        buf.clear();
        encode_7bit(&mut buf, 0)?;
        assert_eq!(&buf[..], &[0, 0]); // 2 times 0 due to escaping

        buf.clear();
        encode_7bit(&mut buf, 1)?;
        assert_eq!(&buf[..], &[0b10]);

        Ok(())
    }

    #[test]
    fn test_decode_7bit() -> io::Result<()> {
        if usize::BITS == 64 {
            let b = [
                0b0000_0011, //  1
                0b1111_1111, //  8
                0b1111_1111, // 15
                0b1111_1111, // 22
                0b1111_1111, // 29
                0b1111_1111, // 36
                0b1111_1111, // 43
                0b1111_1111, // 50
                0b1111_1111, // 57
                0b1111_1110, // 64
            ];
            assert_eq!(decode_7bit(&b[..])?, usize::MAX);

            let b = [
                0b0000_0101, //  1
                0b0000_0001, //  8
                0b0000_0001, // 15
                0b0000_0001, // 22
                0b0000_0001, // 29
                0b0000_0001, // 36
                0b0000_0001, // 43
                0b0000_0001, // 50
                0b0000_0001, // 57
                0b0000_0000, // 64
            ];
            assert!(decode_7bit(&b[..]).is_err());
        }

        // 2 times 0 due to escaping
        assert_eq!(decode_7bit([0, 0].as_slice())?, 0);

        assert_eq!(decode_7bit([0b10].as_slice())?, 1);

        Ok(())
    }

    // spell-checker:disable

    #[test]
    fn test_trim_start() {
        assert_eq!(trim_start(b""), b"");
        assert_eq!(trim_start(b"abc"), b"abc");
        assert_eq!(trim_start(b"abc 123"), b"abc 123");
        assert_eq!(trim_start(b" abc 123"), b"abc 123");
        assert_eq!(trim_start(b"abc 123 "), b"abc 123 ");
        assert_eq!(trim_start(b"\t foo bar \t"), b"foo bar \t");
    }

    #[test]
    fn test_trim_end() {
        assert_eq!(trim_end(b""), b"");
        assert_eq!(trim_end(b"abc"), b"abc");
        assert_eq!(trim_end(b"abc 123"), b"abc 123");
        assert_eq!(trim_end(b" abc 123"), b" abc 123");
        assert_eq!(trim_end(b"abc 123 "), b"abc 123");
        assert_eq!(trim_end(b"\t foo bar \t"), b"\t foo bar");
    }

    #[test]
    fn test_trim() {
        assert_eq!(trim(b""), b"");
        assert_eq!(trim(b"abc"), b"abc");
        assert_eq!(trim(b"abc 123"), b"abc 123");
        assert_eq!(trim(b" abc 123"), b"abc 123");
        assert_eq!(trim(b"abc 123 "), b"abc 123");
        assert_eq!(trim(b"\t foo bar \t"), b"foo bar");
    }

    #[test]
    fn test_parse_str_list() {
        assert_eq!(parse_str_list(b"", 0), Vec::<String>::new());
        assert_eq!(&parse_str_list(b"abc 123", 0)[..], ["abc", "123"]);
        assert_eq!(&parse_str_list(b" abc 123 ", 0)[..], ["abc", "123"]);
        assert_eq!(
            &parse_str_list(" a.-bc\tUniversität ".as_bytes(), 2)[..],
            ["a.-bc", "Universität"]
        );
    }

    #[test]
    fn test_parse_single_u32() -> io::Result<()> {
        assert_eq!(parse_single_u32(b"0", 42)?, 0);
        assert_eq!(parse_single_u32(b"00", 42)?, 0);
        assert_eq!(parse_single_u32(b"123", 42)?, 123);
        assert_eq!(
            parse_single_u32(u32::MAX.to_string().as_bytes(), 42)?,
            u32::MAX
        );

        assert!(parse_single_u32(b"", 42).is_err());
        assert!(parse_single_u32(b"42a", 42).is_err());

        assert!(parse_single_u32((u32::MAX as u64 + 1).to_string().as_bytes(), 42).is_err());

        Ok(())
    }

    #[test]
    fn test_parse_single_usize() -> io::Result<()> {
        assert_eq!(parse_single_usize(b"0", 42)?, 0);
        assert_eq!(parse_single_usize(b"00", 42)?, 0);
        assert_eq!(parse_single_usize(b"123", 42)?, 123);
        assert_eq!(
            parse_single_usize(usize::MAX.to_string().as_bytes(), 42)?,
            usize::MAX
        );

        assert!(parse_single_usize(b"", 42).is_err());
        assert!(parse_single_usize(b"42a", 42).is_err());

        if u128::BITS > usize::BITS {
            assert!(
                parse_single_usize((usize::MAX as u128 + 1).to_string().as_bytes(), 42).is_err()
            );
        }

        Ok(())
    }

    #[test]
    fn test_parse_u32() -> io::Result<()> {
        assert_eq!(parse_u32(b"0", 42)?, (&b""[..], 0));
        assert_eq!(parse_u32(b"00", 42)?, (&b""[..], 0));
        assert_eq!(parse_u32(b"123", 42)?, (&b""[..], 123));
        assert_eq!(
            parse_u32(u32::MAX.to_string().as_bytes(), 42)?,
            (&b""[..], u32::MAX)
        );

        let (rest, v) = parse_u32(b" 023 9 \t432\t", 42)?;
        assert_eq!(rest, b" 9 \t432\t");
        assert_eq!(v, 23);
        let (rest, v) = parse_u32(rest, 42)?;
        assert_eq!(rest, b" \t432\t");
        assert_eq!(v, 9);
        let (rest, v) = parse_u32(rest, 42)?;
        assert_eq!(rest, b"\t");
        assert_eq!(v, 432);
        assert!(parse_u32(rest, 42).is_err());

        assert!(parse_u32(b"", 42).is_err());
        assert!(parse_u32(b"42a", 42).is_err());

        assert!(parse_u32((u32::MAX as u64 + 1).to_string().as_bytes(), 42).is_err());

        Ok(())
    }

    #[test]
    fn test_parse_usize() -> io::Result<()> {
        assert_eq!(parse_usize(b"0", 42)?, (&b""[..], 0));
        assert_eq!(parse_usize(b"00", 42)?, (&b""[..], 0));
        assert_eq!(parse_usize(b"123", 42)?, (&b""[..], 123));
        assert_eq!(
            parse_usize(usize::MAX.to_string().as_bytes(), 42)?,
            (&b""[..], usize::MAX)
        );

        let (rest, v) = parse_usize(b" 023 9 \t432\t", 42)?;
        assert_eq!(rest, b" 9 \t432\t");
        assert_eq!(v, 23);
        let (rest, v) = parse_usize(rest, 42)?;
        assert_eq!(rest, b" \t432\t");
        assert_eq!(v, 9);
        let (rest, v) = parse_usize(rest, 42)?;
        assert_eq!(rest, b"\t");
        assert_eq!(v, 432);
        assert!(parse_usize(rest, 42).is_err());

        assert!(parse_usize(b"", 42).is_err());
        assert!(parse_usize(b"42a", 42).is_err());

        if u128::BITS > usize::BITS {
            assert!(parse_usize((usize::MAX as u128 + 1).to_string().as_bytes(), 42).is_err());
        }

        Ok(())
    }

    #[test]
    fn test_parse_u32_list() -> io::Result<()> {
        assert_eq!(&parse_u32_list(b"", 0, 42)?[..], []);
        assert_eq!(&parse_u32_list(b" \t ", 1, 42)?[..], []);
        assert_eq!(&parse_u32_list(b" 123 \t 432 ", 2, 42)?[..], [123, 432]);
        assert_eq!(&parse_u32_list(b"0 00", 1, 42)?[..], [0, 0]);
        assert_eq!(
            &parse_u32_list(u32::MAX.to_string().as_bytes(), 1, 42)?[..],
            [u32::MAX]
        );

        assert!(parse_u32_list(b"123 42a", 2, 42).is_err());
        assert!(parse_u32_list(b"123 -42", 2, 42).is_err());

        assert!(parse_u32_list((u32::MAX as u64 + 1).to_string().as_bytes(), 1, 42).is_err());

        Ok(())
    }

    #[test]
    fn test_parse_edge_list() -> io::Result<()> {
        let mut res = Vec::with_capacity(32);

        parse_edge_list(b"", &mut res, 42)?;
        assert_eq!(&res[..], []);
        parse_edge_list(b" \t ", &mut res, 42)?;
        assert_eq!(&res[..], []);
        parse_edge_list(b" 123 \t -432 ", &mut res, 42)?;
        assert_eq!(&res[..], [123, -432,]);
        res.clear();
        parse_edge_list(b"01 -02", &mut res, 42)?;
        assert_eq!(&res[..], [1, -2,]);
        res.clear();
        parse_edge_list(b"0 -0", &mut res, 42)?;
        assert_eq!(&res[..], [0, 0,]);
        res.clear();
        parse_edge_list(isize::MAX.to_string().as_bytes(), &mut res, 42)?;
        assert_eq!(&res[..], [isize::MAX]);
        res.clear();
        parse_edge_list(format!("-{}", isize::MAX).as_bytes(), &mut res, 42)?;
        assert_eq!(&res[..], [-isize::MAX]);

        assert!(parse_edge_list(b"123 42a", &mut res, 42).is_err());
        assert!(parse_edge_list(b"123 4-2", &mut res, 42).is_err());

        let isize_min = isize::MIN.to_string();
        assert!(parse_edge_list(isize_min.as_bytes(), &mut res, 42).is_err());
        assert!(parse_edge_list(isize_min[1..].as_bytes(), &mut res, 42).is_err());

        Ok(())
    }

    // spell-checker:enable
}
