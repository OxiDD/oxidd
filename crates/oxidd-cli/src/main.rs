#![forbid(unsafe_code)]

use std::borrow::Cow;
use std::fmt;
use std::fs;
use std::hash::{BuildHasherDefault, Hash};
use std::io;
use std::io::{Seek, Write};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::{Duration, Instant};

use bitvec::prelude::*;
use clap::{Parser, ValueEnum};
use num_bigint::BigUint;
use oxidd::util::{AllocResult, SatCountCache};
use oxidd::{BooleanFunction, Edge, LevelNo, Manager, WorkerManager};
use oxidd_core::{ApplyCache, HasApplyCache, HasLevel, ManagerRef};
use oxidd_dump::{dddmp, dot};
use oxidd_parser::load_file::load_file;
use oxidd_parser::{ParseOptionsBuilder, Problem, Prop, Tree, VarSet};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rustc_hash::{FxHashMap, FxHasher};

mod progress;
use progress::PROGRESS;
mod profiler;
use profiler::Profiler;
mod util;
use util::{handle_oom, HDuration};

// spell-checker:ignore mref,funcs,dotfile,dmpfile

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// The decision diagram type
    #[arg(value_enum, short = 't', long, default_value_t = DDType::BCDD)]
    dd_type: DDType,

    /// Number of inner nodes to reserve memory for
    ///
    /// Currently, this is only relevant for the index-based manager. A value of
    /// 0 means to guess the number based on the available RAM. Note that the
    /// index-based manager cannot reserve more space on demand.
    #[arg(long, default_value_t = 0)]
    inner_node_capacity: usize,

    /// Capacity of the apply cache in entries (lower bound)
    #[arg(long, default_value_t = 32 * 1024 * 1024)]
    apply_cache_capacity: usize,

    /// Dump the final decision diagram to a dot file
    #[arg(long)]
    dot_output: Option<PathBuf>,

    /// Dump the final decision diagram to a dddmp file
    #[arg(long, short = 'e')]
    dddmp_export: Option<PathBuf>,

    /// Dump in ASCII format
    #[arg(long)]
    dddmp_ascii: bool,

    /// Import a dddmp file (may be given multiple times)
    #[arg(long, short = 'i')]
    dddmp_import: Vec<PathBuf>,

    /// Do not prune unreachable nodes before dumping the diagram
    #[arg(long)]
    no_prune_unreachable: bool,

    /// Read the variable order from the input file
    #[arg(long)]
    read_var_order: bool,

    /// Order in which to apply operations when building a CNF
    #[arg(value_enum, long, default_value_t = CNFBuildOrder::Balanced)]
    cnf_build_order: CNFBuildOrder,

    /// For every DD operation of the problem(s), compute and print the size of
    /// the resulting DD function to the given CSV file
    ///
    /// For CNFs, we only consider the conjunction operations. If multiple input
    /// problems are given, then the values for the i-th problem are written
    /// to the i-th CSV file (if present).
    #[arg(long)]
    size_profile: Vec<PathBuf>,

    /// Perform model counting
    #[arg(long)]
    count_models: bool,

    /// Run multiple operations concurrently
    #[arg(long)]
    parallel: bool,

    /// Split recursive operations into multiple tasks up to the specified
    /// recursion depth
    ///
    /// If omitted, a reasonable value is determined automatically
    #[arg(long)]
    operation_split_depth: Option<u32>,

    /// Number of threads to use for concurrent operations
    ///
    /// A value of 0 means automatic detection
    #[arg(long, default_value_t = 0)]
    threads: u32,

    /// Report progress
    #[arg(long, short = 'p')]
    progress: bool,

    /// Interval between each progress report in seconds
    #[arg(long, default_value_t = 1.0)]
    progress_interval: f32,

    /// Problem input file(s)
    file: Vec<PathBuf>,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, ValueEnum)]
enum DDType {
    /// Binary decision diagram
    BDD,
    /// Binary decision diagram with complement edges
    BCDD,
    /// Zero-suppressed decision diagram
    ZBDD,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, ValueEnum)]
enum CNFBuildOrder {
    /// left-deep bracketing of the conjunctions in the input
    /// order. Incompatible with --parallel.
    LeftDeep,
    /// Approximately balanced bracketing of the conjunctions in the input order
    Balanced,
    /// Non-deterministic bracketing as a result of work stealing (using Rayon's
    /// reduce method) but the order is kept as in the input. Requires
    /// --parallel.
    WorkStealing,
    /// Bracketing tree and order from the comment lines in the input file
    Tree,
}

impl CNFBuildOrder {
    /// Whether an externally supplied clause order is necessary
    fn needs_clause_order(self) -> bool {
        self == CNFBuildOrder::Tree
    }
}

fn make_vars<'id, B: BooleanFunction>(
    manager: &mut B::Manager<'id>,
    var_set: &VarSet,
    use_order: bool,
    name_map: &mut FxHashMap<String, B>,
) -> Vec<B>
where
    B::Manager<'id>: WorkerManager,
    <B::Manager<'id> as Manager>::InnerNode: HasLevel,
{
    let num_vars = var_set.len();
    if num_vars >= LevelNo::MAX as usize {
        eprintln!("error: too many variables");
        std::process::exit(1);
    }

    if use_order {
        let Some(var_order) = var_set.order() else {
            eprintln!("error: variable order not given");
            std::process::exit(1);
        };

        let mut vars: Vec<Option<B>> = vec![None; var_order.len()];
        let mut order = Vec::with_capacity(num_vars);
        for &var in var_order {
            let name = match var_set.name(var) {
                Some(s) => s.to_string(),
                None => var.to_string(),
            };
            let f = name_map
                .entry(name)
                .or_insert_with(|| handle_oom!(B::new_var(manager)))
                .clone();
            vars[var] = Some(f.clone());
            order.push(f);
        }
        let vars: Vec<B> = vars
            .into_iter()
            .map(|x| x.expect("`var_order` must contain every variable id exactly once"))
            .collect();
        oxidd_reorder::set_var_order(manager, &order);
        vars
    } else {
        (0..num_vars)
            .map(|i| {
                name_map
                    .entry(i.to_string())
                    .or_insert_with(|| handle_oom!(B::new_var(manager)))
                    .clone()
            })
            .collect()
    }
}

fn make_bool_dd<B>(
    mref: &B::ManagerRef,
    problem: Problem,
    problem_no: usize,
    cli: &Cli,
    var_name_map: &mut FxHashMap<String, B>,
) -> B
where
    B: BooleanFunction + Send + Sync + 'static,
    for<'id> B::Manager<'id>: WorkerManager,
    for<'id> <B::Manager<'id> as Manager>::InnerNode: HasLevel,
{
    fn prop_rec<B: BooleanFunction>(
        manager: &B::Manager<'_>,
        prop: &Prop,
        vars: &[B],
        profiler: &Profiler,
    ) -> B {
        let fold = |ps: &[Prop], init: B, op: fn(&B, &B) -> AllocResult<B>| {
            ps.iter().fold(init, |acc, p| {
                let rhs = prop_rec(manager, p, vars, profiler);
                let op_start = profiler.start_op();
                let res = handle_oom!(op(&acc, &rhs));
                profiler.finish_op(op_start, &res);
                res
            })
        };

        match prop {
            Prop::Lit(l) if l.positive() => vars[l.variable()].clone(),
            Prop::Lit(l) => handle_oom!(vars[l.variable()].not()),
            Prop::Neg(p) => handle_oom!(prop_rec(manager, p, vars, profiler).not()),
            Prop::And(ps) => fold(ps, B::t(manager), B::and),
            Prop::Or(ps) => fold(ps, B::f(manager), B::or),
            Prop::Xor(ps) => fold(ps, B::f(manager), B::xor),
            Prop::Eq(ps) => fold(ps, B::t(manager), B::and),
        }
    }

    fn prop_inner_nodes(prop: &Prop) -> usize {
        match prop {
            Prop::Lit(_) => 0,
            Prop::Neg(p) => 1 + prop_inner_nodes(p),
            Prop::And(ps) | Prop::Or(ps) | Prop::Xor(ps) | Prop::Eq(ps) => {
                ps.len() + ps.iter().map(prop_inner_nodes).sum::<usize>()
            }
        }
    }

    fn balanced_reduce<T>(
        iter: impl IntoIterator<Item = T>,
        mut f: impl for<'a> FnMut(&'a T, &'a T) -> T,
    ) -> Option<T> {
        // We use `Option<T>` such that we can drop the values as soon as possible
        let mut leaves: Vec<Option<T>> = iter.into_iter().map(Some).collect();

        fn rec<T, F: for<'a> FnMut(&'a T, &'a T) -> T>(
            leaves: &mut [Option<T>],
            f: &mut F,
        ) -> Option<T> {
            match leaves {
                [] => None,
                [l] => l.take(),
                _ => {
                    let (l, r) = leaves.split_at_mut(leaves.len() / 2);
                    let l = rec(l, f);
                    let r = rec(r, f);
                    match (l, r) {
                        (None, v) | (v, None) => v,
                        (Some(l), Some(r)) => Some(f(&l, &r)),
                    }
                }
            }
        }

        rec(&mut leaves, &mut f)
    }

    fn balanced_reduce_par<T: Send>(
        iter: impl IntoParallelIterator<Item = T>,
        f: impl for<'a> Fn(&'a T, &'a T) -> T + Send + Copy,
    ) -> Option<T> {
        // We use `Option<T>` such that we can drop the values as soon as possible
        let mut leaves: Vec<Option<T>> = iter.into_par_iter().map(Some).collect();

        fn rec<T: Send>(
            leaves: &mut [Option<T>],
            f: impl for<'a> Fn(&'a T, &'a T) -> T + Send + Copy,
        ) -> Option<T> {
            match leaves {
                [] => None,
                [l] => l.take(),
                _ => {
                    let (l, r) = leaves.split_at_mut(leaves.len() / 2);
                    match rayon::join(move || rec(l, f), move || rec(r, f)) {
                        (None, v) | (v, None) => v,
                        (Some(l), Some(r)) => Some(f(&l, &r)),
                    }
                }
            }
        }

        rec(&mut leaves, f)
    }

    fn clause_tree_rec<B: BooleanFunction>(
        clauses: &[B],
        clause_order: &Tree<usize>,
        profiler: &Profiler,
    ) -> Option<B> {
        match clause_order {
            Tree::Leaf(n) => Some(clauses[*n].clone()),
            Tree::Inner(sub) => balanced_reduce(
                sub.iter()
                    .filter_map(|t| clause_tree_rec(clauses, t, profiler)),
                |lhs: &B, rhs: &B| {
                    let op_start = profiler.start_op();
                    let conj = handle_oom!(lhs.and(rhs));
                    profiler.finish_op(op_start, &conj);
                    conj
                },
            ),
        }
    }

    fn clause_tree_par_rec<B: BooleanFunction + Send + Sync>(
        clauses: &[B],
        clause_order: &Tree<usize>,
        profiler: &Profiler,
    ) -> Option<B> {
        match clause_order {
            Tree::Leaf(n) => Some(clauses[*n].clone()),
            Tree::Inner(sub) => balanced_reduce_par(
                sub.into_par_iter()
                    .filter_map(|t| clause_tree_par_rec(clauses, t, profiler)),
                |lhs: &B, rhs: &B| {
                    let op_start = profiler.start_op();
                    let conj = handle_oom!(lhs.and(rhs));
                    profiler.finish_op(op_start, &conj);
                    conj
                },
            ),
        }
    }

    let profiler = Profiler::new(cli.size_profile.get(problem_no));

    let func = match problem {
        Problem::CNF(mut cnf) => {
            if cli.cnf_build_order.needs_clause_order() && cnf.clause_order().is_none() {
                eprintln!("error: clause order not given");
                std::process::exit(1);
            }

            let vars = mref.with_manager_exclusive(|manager| {
                make_vars::<B>(manager, cnf.vars(), cli.read_var_order, var_name_map)
            });

            mref.with_manager_shared(|manager| {
                manager.set_split_depth(Some(0));
                let clauses = cnf.clauses_mut();
                PROGRESS.set_task(
                    format!("build clauses (problem {problem_no})"),
                    clauses.len(),
                );
                let mut bdd_clauses = Vec::with_capacity(clauses.len());
                let mut i = 0;
                while let Some(clause) = clauses.get_mut(i) {
                    if clause.is_empty() {
                        println!("clause {i} is empty");
                        return B::f(manager);
                    }

                    // Build clause bottom-up
                    clause.sort_unstable_by_key(|lit| std::cmp::Reverse(lit.variable()));
                    let l = clause[0];
                    let init = &vars[l.variable()];
                    let init = if l.positive() {
                        init.clone()
                    } else {
                        handle_oom!(init.not())
                    };
                    bdd_clauses.push(clause[1..].iter().fold(init, |acc, l| {
                        let var = &vars[l.variable()];
                        if l.positive() {
                            handle_oom!(acc.or(var))
                        } else {
                            handle_oom!(acc.or(&handle_oom!(var.not())))
                        }
                    }));

                    i += 1;
                }
                let clauses = bdd_clauses;
                println!(
                    "all {} clauses built after {}",
                    clauses.len(),
                    HDuration(profiler.elapsed_time())
                );

                manager.set_split_depth(cli.operation_split_depth);
                PROGRESS.set_task(
                    format!("conjoin clauses (problem {problem_no})"),
                    clauses.len() - 1,
                );
                if clauses.is_empty() {
                    B::t(manager)
                } else {
                    match (cli.cnf_build_order, cli.parallel) {
                        (CNFBuildOrder::LeftDeep, false) => {
                            let init = clauses[0].clone();
                            clauses[1..].iter().fold(init, |acc, f| {
                                let op_start = profiler.start_op();
                                let res = handle_oom!(acc.and(f));
                                profiler.finish_op(op_start, &res);
                                res
                            })
                        }
                        (CNFBuildOrder::LeftDeep, true) => unreachable!(),
                        (CNFBuildOrder::Balanced, false) => {
                            balanced_reduce(clauses, |lhs: &B, rhs: &B| {
                                let op_start = profiler.start_op();
                                let conj = handle_oom!(lhs.and(rhs));
                                profiler.finish_op(op_start, &conj);
                                conj
                            })
                            .unwrap()
                        }
                        (CNFBuildOrder::Balanced, true) => {
                            balanced_reduce_par(clauses, |lhs: &B, rhs: &B| {
                                let op_start = profiler.start_op();
                                let conj = handle_oom!(lhs.and(rhs));
                                profiler.finish_op(op_start, &conj);
                                conj
                            })
                            .unwrap()
                        }
                        (CNFBuildOrder::WorkStealing, false) => unreachable!(),
                        (CNFBuildOrder::WorkStealing, true) => ParallelIterator::reduce(
                            clauses.into_par_iter().map(|c| Some(c)),
                            || None,
                            |lhs: Option<B>, rhs: Option<B>| match (lhs, rhs) {
                                (None, v) | (v, None) => v,
                                (Some(lhs), Some(rhs)) => {
                                    let op_start = profiler.start_op();
                                    let res = handle_oom!(lhs.and(&rhs));
                                    profiler.finish_op(op_start, &res);
                                    Some(res)
                                }
                            },
                        )
                        .unwrap(),
                        (CNFBuildOrder::Tree, false) => {
                            clause_tree_rec(&clauses, cnf.clause_order().unwrap(), &profiler)
                                .unwrap()
                        }
                        (CNFBuildOrder::Tree, true) => {
                            clause_tree_par_rec(&clauses, cnf.clause_order().unwrap(), &profiler)
                                .unwrap()
                        }
                    }
                }
            })
        }

        Problem::Prop(prop) => {
            PROGRESS.set_task(
                "build propositional formula",
                prop_inner_nodes(prop.formula()),
            );
            let vars = mref.with_manager_exclusive(|manager| {
                make_vars::<B>(manager, prop.vars(), cli.read_var_order, var_name_map)
            });
            mref.with_manager_shared(|manager| {
                manager.set_split_depth(cli.operation_split_depth);
                prop_rec(manager, prop.formula(), &vars, &profiler)
            })
        }

        _ => todo!(),
    };
    println!(
        "BDD building done within {}",
        HDuration(profiler.elapsed_time())
    );
    func
}

fn bool_dd_main<B, O>(cli: &Cli, mref: B::ManagerRef)
where
    B: BooleanFunction + Send + Sync + 'static,
    B::ManagerRef: Send + 'static,
    for<'id> B: dot::DotStyle<<B::Manager<'id> as Manager>::EdgeTag>,
    for<'id> B::Manager<'id>: WorkerManager + HasApplyCache<B::Manager<'id>, O>,
    for<'id> <B::Manager<'id> as Manager>::InnerNode: HasLevel,
    for<'id> <B::Manager<'id> as Manager>::EdgeTag: fmt::Debug,
    for<'id> <B::Manager<'id> as Manager>::Terminal: FromStr + fmt::Display + dddmp::AsciiDisplay,
    O: Copy + Ord + Hash,
{
    let parse_options = ParseOptionsBuilder::default()
        .orders(cli.read_var_order || cli.cnf_build_order.needs_clause_order())
        .build()
        .unwrap();

    let mut vars: FxHashMap<String, B> = Default::default();
    let mut funcs: Vec<(B, String)> = Vec::new();

    let progress_interval = match Duration::try_from_secs_f32(cli.progress_interval) {
        Ok(int) if cli.progress_interval > f32::EPSILON => int,
        _ => {
            eprintln!("Invalid progress interval (must be positive)");
            std::process::exit(1);
        }
    };

    let handle = if cli.progress {
        Some(progress::start_progress_report::<B::ManagerRef, O>(
            mref.clone(),
            progress_interval,
        ))
    } else {
        None
    };

    let report_dd_node_count = || {
        mref.with_manager_shared(|manager| {
            if !cli.no_prune_unreachable {
                println!("node count before pruning: {}", manager.num_inner_nodes());
                manager.apply_cache().clear(manager);
                let start = Instant::now();
                manager.gc();
                println!("garbage collection took {}", HDuration(start.elapsed()));
            }
            let count = manager.num_inner_nodes();
            println!("node count: {count}");
        })
    };

    // Import dddmp files
    if !cli.dddmp_import.is_empty() {
        PROGRESS.set_task("import from dddmp", cli.dddmp_import.len());
    }
    for path in &cli.dddmp_import {
        PROGRESS.start_op();
        let start = Instant::now();
        print!("importing '{}' ...", path.display());
        io::stdout().flush().unwrap();

        let file = match fs::File::open(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("\nerror: could not open '{}' ({e})", path.display());
                std::process::exit(1);
            }
        };
        let mut reader = io::BufReader::new(file);

        let header = match dddmp::DumpHeader::load(&mut reader) {
            Ok(h) => h,
            Err(e) => {
                eprintln!(
                    "\nerror: failed to load header of '{}' ({e})",
                    path.display()
                );
                std::process::exit(1);
            }
        };

        let mut support_vars = Vec::with_capacity(header.num_support_vars() as usize);
        mref.with_manager_exclusive(|manager| {
            if let Some(var_names) = header.ordered_var_names() {
                let mut filter = bitvec![0; header.num_vars() as usize];
                for &i in header.support_var_permutation() {
                    filter.set(i as usize, true);
                }
                for (name, in_support) in var_names.iter().zip(filter) {
                    let f = vars
                        .entry(name.clone())
                        .or_insert_with(|| handle_oom!(B::new_var(manager)));
                    if in_support {
                        support_vars.push(f.clone());
                    }
                }
            } else {
                todo!()
            }

            oxidd_reorder::set_var_order(manager, &support_vars);
        });

        mref.with_manager_shared(|manager| {
            match dddmp::import(reader, &header, manager, &support_vars, B::not_edge_owned) {
                Ok(roots) => {
                    let name_prefix = match header.diagram_name() {
                        Some(n) => Cow::Borrowed(n),
                        None => path.file_name().unwrap().to_string_lossy(),
                    };

                    if let Some(names) = header.root_names() {
                        for (root, name) in roots.into_iter().zip(names) {
                            funcs.push((root, format!("{name_prefix}:{name}")));
                        }
                    } else {
                        for (i, root) in roots.into_iter().enumerate() {
                            funcs.push((root, format!("{name_prefix}:{i}")));
                        }
                    }
                }
                Err(e) => {
                    eprintln!("\nerror: failed to import '{}' ({e})", path.display());
                    std::process::exit(1);
                }
            }
        });

        PROGRESS.finish_op();
        println!(" done ({})", HDuration(start.elapsed()));

        report_dd_node_count();
    }

    // Construct DDs for input problems (e.g., from DIMACS files)
    for (i, file) in cli.file.iter().enumerate() {
        let start = Instant::now();
        let Some(problem) = load_file(file, &parse_options) else {
            std::process::exit(1)
        };
        println!("parsing done within {}", HDuration(start.elapsed()));

        funcs.push((
            make_bool_dd(&mref, problem, i, cli, &mut vars),
            file.file_name().unwrap().to_string_lossy().to_string(),
        ));

        report_dd_node_count();
    }

    let pause_handle = PROGRESS.pause_progress_report();
    // Identify equivalent functions
    let mut equivalences: FxHashMap<&B, Vec<usize>> = Default::default();
    for (i, (func, _)) in funcs.iter().enumerate() {
        equivalences
            .entry(func)
            .and_modify(|v| v.push(i))
            .or_insert_with(|| vec![i]);
    }

    // Count nodes and satisfying assignments
    let mut model_count_cache: SatCountCache<BigUint, BuildHasherDefault<FxHasher>> =
        SatCountCache::default();
    for (f, equiv) in equivalences.into_iter() {
        print!("- {}", funcs[equiv[0]].1);
        for &i in &equiv[1..] {
            print!(", {}", funcs[i].1);
        }
        println!();

        print!("  nodes: ");
        io::stdout().flush().unwrap();
        let start = Instant::now();
        let count = f.node_count();
        println!("{count} ({})", HDuration(start.elapsed()));

        if cli.count_models {
            print!("  model count: ");
            io::stdout().flush().unwrap();
            let start = Instant::now();
            let count = f.sat_count(vars.len() as LevelNo, &mut model_count_cache);
            println!("{count} ({})", HDuration(start.elapsed()));
        }
    }
    drop(pause_handle);

    // Export (dot, dddmp)
    mref.with_manager_shared(|manager| {
        if let Some(dotfile) = &cli.dot_output {
            PROGRESS.set_task("dot export", 1);
            fs::File::create(dotfile)
                .and_then(|file| {
                    dot::dump_all(
                        std::io::BufWriter::new(file),
                        manager,
                        vars.iter().map(|(n, f)| (f, n.as_str())),
                        funcs.iter().map(|(f, n)| (f, n.as_str())),
                    )
                })
                .unwrap_or_else(|e| {
                    eprintln!("error: could not write '{}' ({e})", dotfile.display())
                });
        }

        if let Some(dmpfile) = &cli.dddmp_export {
            PROGRESS.set_task("dddmp export", 1);
            fs::File::create(dmpfile)
                .and_then(|file| {
                    let mut var_edges = Vec::with_capacity(vars.len());
                    let mut var_names = Vec::with_capacity(vars.len());
                    for (name, var) in vars.iter() {
                        var_edges.push(var);
                        var_names.push(name.as_str());
                    }
                    let functions: Vec<_> = funcs.iter().map(|(f, _)| f).collect();
                    let function_names: Vec<&str> = funcs.iter().map(|(_, n)| n.as_str()).collect();

                    let mut writer = std::io::BufWriter::new(file);
                    let start = Instant::now();
                    dddmp::export(
                        &mut writer,
                        manager,
                        cli.dddmp_ascii,
                        "",
                        &var_edges,
                        Some(&var_names),
                        &functions,
                        Some(&function_names),
                        |e| e.tag() != Default::default(),
                    )?;
                    println!(
                        "exported BDD ({} bytes) in {}",
                        writer.stream_position().unwrap_or_default(),
                        HDuration(start.elapsed())
                    );

                    Ok(())
                })
                .unwrap_or_else(|e| {
                    eprintln!("error: could not write '{}' ({e})", dmpfile.display())
                });
        }
    });

    if let Some(handle) = handle {
        handle.join();
    }
}

fn main() {
    let cli = Cli::parse();

    match (cli.cnf_build_order, cli.parallel) {
        (CNFBuildOrder::LeftDeep, true) => {
            eprintln!("--cnf-build-order=left-deep and --parallel are incompatible");
            std::process::exit(1);
        }
        (CNFBuildOrder::WorkStealing, false) => {
            eprintln!("--cnf-build-order=work-stealing requires --parallel");
            std::process::exit(1);
        }
        _ => {}
    }

    let mut inner_node_capacity = cli.inner_node_capacity;

    #[cfg(miri)]
    if inner_node_capacity == 0 {
        inner_node_capacity = 4096;
    }
    #[cfg(not(miri))]
    if inner_node_capacity == 0 {
        use sysinfo::System;
        let mut sys = System::new();
        sys.refresh_memory();
        let mem = sys.available_memory();
        let apply_cache_size = 4 * 4 * cli.apply_cache_capacity.next_power_of_two();
        let terminals = if cli.dd_type == DDType::BCDD { 1 } else { 2 };
        inner_node_capacity = std::cmp::min(
            (mem - apply_cache_size as u64) / (4 * 8),
            (1 << 32) - terminals,
        ) as usize;
    }

    println!("inner node capacity: {inner_node_capacity}");
    println!("apply cache capacity: {}", cli.apply_cache_capacity);

    match cli.dd_type {
        DDType::BDD => {
            let mref =
                oxidd::bdd::new_manager(inner_node_capacity, cli.apply_cache_capacity, cli.threads);
            bool_dd_main::<oxidd::bdd::BDDFunction, _>(&cli, mref);
        }
        DDType::BCDD => {
            let mref = oxidd::bcdd::new_manager(
                inner_node_capacity,
                cli.apply_cache_capacity,
                cli.threads,
            );
            bool_dd_main::<oxidd::bcdd::BCDDFunction, _>(&cli, mref);
        }
        DDType::ZBDD => todo!(),
    }
}
