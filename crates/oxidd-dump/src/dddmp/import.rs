use std::io;

use is_sorted::IsSorted;

use oxidd_core::error::OutOfMemory;
use oxidd_core::function::{ETagOfFunc, EdgeOfFunc, Function, INodeOfFunc, TermOfFunc};
use oxidd_core::util::{AllocResult, EdgeDropGuard, EdgeVecDropGuard};
use oxidd_core::{DiagramRules, Edge, HasLevel, InnerNode, LevelNo, Manager, VarNo};

use crate::ParseTagged;

use super::{Code, VarInfo};

// spell-checker:dictionaries dddmp

/// Compare, mapping `Equal` to `Greater`
fn cmp_strict<T: Ord>(a: &T, b: &T) -> Option<std::cmp::Ordering> {
    Some(a.cmp(b).then(std::cmp::Ordering::Greater))
}

/// Helper function to return a parse error
fn err<T>(msg: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidData, msg))
}

/// Information from the header of a DDDMP file
#[derive(Debug)]
pub struct DumpHeader {
    /// Whether this is ASCII or binary mode (from `.mode A|B`)
    ascii: bool,
    varinfo: VarInfo,
    /// Decision diagram name (optional)
    dd: String,
    nnodes: usize,
    nvars: u32,
    /// Support variable IDs (in strictly ascending order)
    ids: Vec<VarNo>,
    /// Mapping from positions to support variables
    support_var_order: Vec<VarNo>,
    /// Mapping from support variables to levels
    permids: Vec<LevelNo>,
    auxids: Vec<u32>, // optional
    /// Variable names in the original order (optional)
    varnames: Vec<String>,
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
            ids: Vec::new(),
            support_var_order: Vec::new(),
            permids: Vec::new(),
            auxids: Vec::new(),
            varnames: Vec::new(),
            rootids: Vec::new(),
            rootnames: Vec::new(),
            lines: 1,
        };
        let mut nsuppvars = 0;
        let mut nroots = 0;
        let mut suppvarnames = Vec::new();
        let mut orderedvarnames = Vec::new();

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
                b".ver" => match value {
                    b"DDDMP-2.0" | b"DDDMP-3.0" => {}
                    _ => {
                        return err(format!(
                            "unsupported version '{}' (line {line_no})",
                            String::from_utf8_lossy(value)
                        ))
                    }
                },
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
                b".varnames" => header.varnames = parse_str_list(value, header.nvars as usize),
                b".suppvarnames" => suppvarnames = parse_str_list(value, nsuppvars as usize),
                b".orderedvarnames" => {
                    orderedvarnames = parse_str_list(value, header.nvars as usize)
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
                    header.rootids.reserve(nroots);
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

        let mut level_count = vec![0u32; header.nvars as usize];
        for &level in &header.permids {
            let Some(count) = level_count.get_mut(level as usize) else {
                return err(format!(
                    "levels in .permids must be less than .nvars ({})",
                    header.nvars,
                ));
            };
            if *count != 0 {
                return err(format!("level ({level}) occurs twice in .permids"));
            }
            *count = 1;
        }
        // accumulate the counts such that level_count maps from the level
        // number to the position
        let mut count = 0;
        for i in level_count.iter_mut() {
            let present = *i;
            *i = count;
            count += present;
        }
        header.support_var_order.resize(nsuppvars as usize, 0);
        for (&var, &level) in header.ids.iter().zip(&header.permids) {
            header.support_var_order[level_count[level as usize] as usize] = var;
        }
        drop(level_count);

        if !orderedvarnames.is_empty() && orderedvarnames.len() != header.nvars as usize {
            return err(format!(
                "number of variables in .orderedvarnames entry ({}) does not match .nvars ({})",
                orderedvarnames.len(),
                header.nvars
            ));
        }
        if !suppvarnames.is_empty() && suppvarnames.len() != nsuppvars as usize {
            return err(format!(
                "number of variables in .suppvarnames entry ({}) does not match .nsuppvars ({nsuppvars})",
                suppvarnames.len(),
            ));
        }

        'var_names: {
            if header.varnames.is_empty() {
                if orderedvarnames.is_empty() {
                    if suppvarnames.is_empty() {
                        break 'var_names;
                    }
                    debug_assert!(!suppvarnames.is_empty());
                    header.varnames = vec![String::new(); header.nvars as usize];
                    for (name, &target) in suppvarnames.into_iter().zip(&header.ids) {
                        header.varnames[target as usize] = name;
                    }
                    break 'var_names;
                }

                // Note that `.ids` and `.permids` only define the positions of
                // support variables. For all other variables, we use the
                // relative ordering from `.orderedvarnames`.
                header.varnames = vec![String::new(); header.nvars as usize];
                for (&id, &permid) in header.ids.iter().zip(&header.permids) {
                    header.varnames[id as usize] =
                        std::mem::take(&mut orderedvarnames[permid as usize]);
                }

                let mut non_suppvarnames = orderedvarnames.into_iter().filter(|s| !s.is_empty());
                for name in &mut header.varnames {
                    if name.is_empty() {
                        *name = non_suppvarnames.next().unwrap();
                    }
                }
            } else {
                if header.varnames.len() != header.nvars as usize {
                    return err(format!(
                        "number of variables in .varnames entry ({}) does not match .nvars ({})",
                        header.varnames.len(),
                        header.nvars
                    ));
                }

                if !orderedvarnames.is_empty() {
                    // Check that the names for the support agree in `.varnames`
                    // and `.orderedvarnames`. For all remaining variables, we
                    // could only perform a matching provided that the names are
                    // actually unique. However, since the variables are unused,
                    // a mismatch would not lead to wrong semantics of the
                    // imported decision diagram functions.
                    for (&id, &permid) in header.ids.iter().zip(&header.permids) {
                        let name = header.varnames[id as usize].as_str();
                        let order_name = orderedvarnames[permid as usize].as_str();
                        if name != order_name {
                            return err(format!(
                                ".varnames and .orderedvarnames do not match \
                            (variable {id} has name '{name}', but the entry at \
                            position {permid} of .orderedvarnames is \
                            '{order_name}'"
                            ));
                        }
                    }
                }
            }

            // no special check whether suppvarnames is empty needed here
            for (i, (name, &id)) in suppvarnames.into_iter().zip(&header.ids).enumerate() {
                let expected = header.varnames[id as usize].as_str();
                if name != expected {
                    return err(format!(
                        ".suppvarnames and .varnames/.orderedvarnames do not \
                        match (entry {i} of .suppvarnames is '{name}' \
                        but the variable ID is {id} with name '{expected}')"
                    ));
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
    /// `x`, `y`, and `z`, in this order (`x` is the top-most variable). Suppose
    /// that only `y` and `z` are used by the dumped functions. Then, the
    /// returned slice is `[1, 2]`, regardless of any subsequent reordering.
    ///
    /// Corresponds to the DDDMP `.ids` field.
    pub fn support_vars(&self) -> &[VarNo] {
        &self.ids
    }

    /// Order of the support variables
    ///
    /// The returned slice is always [`DumpHeader::num_support_vars()`] elements
    /// long and represents a mapping from positions to variable numbers.
    ///
    /// Example: Consider a decision diagram that was created with the variables
    /// `x`, `y`, and `z` (`x` is the top-most variable). The variables were
    /// re-ordered to `z`, `x`, `y`. Suppose that only `y` and `z` are used by
    /// the dumped functions. Then, the returned slice is `[2, 1]`.
    pub fn support_var_order(&self) -> &[VarNo] {
        &self.support_var_order
    }

    /// Mapping from the variables in the true support of the decision diagram
    /// to their levels.
    ///
    /// The returned slice is always [`DumpHeader::num_support_vars()`]
    /// elements long. If the value at the `i`th index is `l`, then the `i`th
    /// support variable is at level `l` in the dumped decision diagram. By the
    /// `i`th support variable, we mean the variable `header.support_vars()[i]`
    /// in the original numbering.
    ///
    /// Example: Consider a decision diagram that was created with the variables
    /// `x`, `y`, and `z` (`x` is the top-most variable). The variables were
    /// re-ordered to `z`, `x`, `y`. Suppose that only `y` and `z` are used by
    /// the dumped functions. Then, the returned slice is `[2, 0]`.
    ///
    /// Corresponds to the DDDMP `.permids` field.
    pub fn support_var_to_level(&self) -> &[LevelNo] {
        &self.permids
    }
    /// Deprecated alias for [`Self::support_var_to_level()`]
    #[deprecated(since = "0.11.0", note = "use support_var_to_level instead")]
    pub fn support_var_permutation(&self) -> &[LevelNo] {
        &self.permids
    }

    /// Auxiliary variable IDs. The returned slice contains
    /// [`DumpHeader::num_support_vars()`] elements.
    ///
    /// Corresponds to the DDDMP `.auxids` field.
    pub fn auxiliary_var_ids(&self) -> &[u32] {
        &self.auxids
    }

    /// Names of all variables in the decision diagram. If
    /// present, the returned slice contains [`DumpHeader::num_vars()`]
    /// many elements. The order is the "original" variable order.
    ///
    /// Corresponds to the DDDMP `.varnames` field, but `.orderedvarnames` and
    /// `.suppvarnames` are also considered if one of the fields is missing. All
    /// variable names are non-empty unless only `.suppvarnames` is given in the
    /// input (in which case only the names of support variables are non-empty).
    /// The return value is only `None` if neither of `.varnames`,
    /// `.orderedvarnames`, and `.suppvarnames` is present in the input.
    pub fn var_names(&self) -> Option<&[String]> {
        if self.varnames.is_empty() {
            None
        } else {
            Some(&self.varnames[..])
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

/// Import the decision diagram from `input` into `manager` after loading
/// `header` from `input`
///
/// Important: there must not be any read/seek/... operations on `input` after
/// reading `header` using [`DumpHeader::load()`].
///
/// `support_vars` can be used to adjust the mapping from support variables in
/// the DDDMP file to variables in the manager. The iterator must yield
/// [`header.num_support_vars()`][DumpHeader::num_support_vars()] variable
/// numbers valid in the manager. If the `i`-th value is `v`, then the `i`-th
/// support variable (see [`DumpHeader::support_vars()`]) will be mapped to
/// variable `v` in `manager`. In the simplest case,
/// [`header.support_var_order()`][DumpHeader::support_var_order()] can be used.
///
/// Note that the support variables must also be ordered by their current level
/// (lower level numbers first). To this end, you can use
/// [`oxidd_reorder::set_var_order()`][set_var_order] with `support_vars`.
///
/// `complement` is a function that returns the complemented edge for a given
/// edge.
///
/// [set_var_order]: https://docs.rs/oxidd-reorder/latest/oxidd_reorder/fn.set_var_order.html
pub fn import<'id, F: Function>(
    mut input: impl io::BufRead,
    header: &DumpHeader,
    manager: &F::Manager<'id>,
    support_vars: impl IntoIterator<Item = VarNo>,
    complement: impl Fn(&F::Manager<'id>, EdgeOfFunc<'id, F>) -> AllocResult<EdgeOfFunc<'id, F>>,
) -> io::Result<Vec<F>>
where
    INodeOfFunc<'id, F>: HasLevel,
    TermOfFunc<'id, F>: ParseTagged<ETagOfFunc<'id, F>>,
{
    let suppvar_level_map =
        Vec::from_iter(support_vars.into_iter().map(|v| manager.var_to_level(v)));
    assert_eq!(
        suppvar_level_map.len(),
        header.ids.len(),
        "`support_vars` must provide one target variable per support variable in the DDDMP file",
    );
    assert!(
        IsSorted::is_sorted_by(&mut suppvar_level_map.iter(), cmp_strict),
        "`support_vars` must be sorted by the variables' current level",
    );

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

    if !reads_expected(input, b".end")? {
        return err("file must end with '.end'");
    }

    let mut roots = Vec::with_capacity(header.rootids.len());
    for &root in &header.rootids {
        debug_assert_ne!(root, 0);
        let node_index = root.unsigned_abs() - 1;
        let e = manager.clone_edge(&nodes[node_index]);
        roots.push(F::from_edge(
            manager,
            if root > 0 { e } else { complement(manager, e)? },
        ));
    }

    Ok(roots)
}

/// Check if the remaining bytes in `file` are `expected` plus whitespace
/// characters only
fn reads_expected(mut file: impl io::Read, expected: &[u8]) -> io::Result<bool> {
    let mut buf = Vec::with_capacity(expected.len() + 2);
    file.read_to_end(&mut buf)?;
    Ok(buf.starts_with(expected) && buf[expected.len()..].iter().all(u8::is_ascii_whitespace))
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
    M::Terminal: ParseTagged<M::EdgeTag>,
{
    let mut nodes = EdgeVecDropGuard::new(manager, Vec::with_capacity(header.nnodes));
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

        let node = if children.contains(&0) {
            // terminal
            let string = match std::str::from_utf8(var_id) {
                Ok(s) => s,
                Err(_) => {
                    return err(format!(
                        "terminal description must be valid UTF-8 (line {line_no})"
                    ));
                }
            };
            let Some((terminal, tag)) = M::Terminal::parse(string) else {
                return err(format!(
                    "invalid terminal description '{string}' (line {line_no})"
                ));
            };
            manager.get_terminal(terminal)?.with_tag_owned(tag)
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
            .then_insert(manager, level)?
        };
        nodes.push(node);

        children.clear();
        line.clear();
        line_no += 1;
    }

    Ok(nodes.into_vec())
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
        if let Some(t) = it.next() {
            manager.drop_edge(t);
            panic!("{msg}");
        }
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

        let t = EdgeDropGuard::new(
            manager,
            manager.clone_edge(&nodes[idx(&mut input, node_id, t_code)?]),
        );
        let t_level = manager.get_node(&t).level();
        let e = EdgeDropGuard::new(
            manager,
            manager.clone_edge(&nodes[idx(&mut input, node_id, e_code)?]),
        );
        let e_level = manager.get_node(&e).level();
        let e = if e_complement {
            match complement(manager, e.into_edge()) {
                Ok(e) => EdgeDropGuard::new(manager, e),
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

        let children = [t.into_edge(), e.into_edge()];
        match <M::Rules as DiagramRules<_, _, _>>::reduce(manager, level, children)
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

/// Read and unescape a byte. Counterpart of [`write_escaped()`]
fn read_unescape(input: impl io::BufRead) -> io::Result<u8> {
    // In principle, `io::Read` (instead of `io::BufRead`) is enough, but not
    // passing a buffered reader would be very bad for performance.
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
fn decode_7bit(mut input: impl io::BufRead) -> io::Result<usize> {
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

/// Parse a list of space- or tab-separated  strings
///
/// All strings in the returned vector are guaranteed to be non-empty.
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
        assert!(parse_edge_list(&isize_min.as_bytes()[1..], &mut res, 42).is_err());

        Ok(())
    }

    // spell-checker:enable
}
