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
use oxidd::util::SatCountCache;
use oxidd::{BooleanFunction, Edge, HasWorkers, LevelNo, Manager, WorkerPool};
use oxidd_core::{ApplyCache, HasApplyCache, HasLevel, ManagerRef};
use oxidd_dump::{dddmp, dot};
use oxidd_parser::{load_file, ParseOptionsBuilder, VarSet};
use rustc_hash::{FxHashMap, FxHasher};

mod progress;
mod scheduler;
use progress::PROGRESS;
mod profiler;
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

    /// Read the CNF clause tree
    #[arg(long)]
    read_clause_tree: bool,

    /// Order in which to apply operations when constructing a gate with more
    /// than two inputs
    #[arg(value_enum, long, default_value_t = GateBuildScheme::Balanced)]
    gate_build_scheme: GateBuildScheme,

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

    /// Always output durations as seconds (floating point)
    #[arg(long)]
    durations_as_secs: bool,

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
enum GateBuildScheme {
    /// left-deep bracketing of the operands in the input order
    LeftDeep,
    /// approximately balanced bracketing of the operands in the input order
    Balanced,
    /// Non-deterministic bracketing as a result of work stealing, but the order
    /// is kept as in the input. Requires --parallel.
    WorkStealing,
}

fn make_vars<'id, B: BooleanFunction>(
    manager: &mut B::Manager<'id>,
    var_set: &VarSet,
    use_order: bool,
    name_map: &mut FxHashMap<String, B>,
) -> Vec<B>
where
    B::Manager<'id>: HasWorkers,
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

fn bool_dd_main<B, O>(cli: &Cli, mref: B::ManagerRef)
where
    B: BooleanFunction + Send + Sync + 'static,
    B::ManagerRef: HasWorkers + Send + 'static,
    for<'id> B: dot::DotStyle<<B::Manager<'id> as Manager>::EdgeTag>,
    for<'id> B::Manager<'id>: HasWorkers + HasApplyCache<B::Manager<'id>, O>,
    for<'id> <B::Manager<'id> as Manager>::InnerNode: HasLevel,
    for<'id> <B::Manager<'id> as Manager>::EdgeTag: fmt::Debug,
    for<'id> <B::Manager<'id> as Manager>::Terminal: FromStr + fmt::Display + dddmp::AsciiDisplay,
    O: Copy + Ord + Hash,
{
    let parse_options = ParseOptionsBuilder::default()
        .var_order(cli.read_var_order)
        .clause_tree(cli.read_clause_tree)
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

    let report_dd_node_count = |gc_count_before_construction: Option<u64>| {
        mref.with_manager_shared(|manager| {
            if let Some(gc_count) = gc_count_before_construction {
                println!(
                    "garbage collections during construction: {}",
                    manager.gc_count() - gc_count
                );
            }
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

        report_dd_node_count(None);
    }

    // Construct DDs for input problems (e.g., from DIMACS files)
    for (problem_no, file) in cli.file.iter().enumerate() {
        PROGRESS.set_problem_no(problem_no);
        let file_name = file.file_name().unwrap().to_string_lossy();

        let start = Instant::now();
        let Some(problem) = load_file(file, &parse_options) else {
            std::process::exit(1)
        };
        println!("parsing done within {}", HDuration(start.elapsed()));

        let problem = {
            let simplified = problem.simplify().unwrap().0;
            drop(problem);
            println!("simplified after {}", HDuration(start.elapsed()));
            simplified
        };

        let (vars, gc_count) = mref.with_manager_exclusive(|manager| {
            let vars = make_vars::<B>(
                manager,
                problem.circuit.inputs(),
                cli.read_var_order,
                &mut vars,
            );
            (vars, manager.gc_count())
        });

        match problem.details {
            oxidd_parser::ProblemDetails::Root(root) => {
                let mut result = [None];
                scheduler::construct_bool_circuit(
                    &mref,
                    problem.circuit,
                    cli,
                    &vars,
                    &[root],
                    &mut result,
                    cli.size_profile.get(problem_no),
                );
                let [result] = result;
                funcs.push((result.unwrap(), file_name.to_string()));
            }
            oxidd_parser::ProblemDetails::AIGER(_aig) => todo!(),
        }

        report_dd_node_count(Some(gc_count));
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
    util::DURATIONS_AS_SECS.store(cli.durations_as_secs, std::sync::atomic::Ordering::Relaxed);

    if let (GateBuildScheme::WorkStealing, false) = (cli.gate_build_scheme, cli.parallel) {
        eprintln!("--gate-build-order=work-stealing requires --parallel");
        std::process::exit(1);
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
            // Run all operations from within the worker pool to reduce the number of
            // context switches
            mref.clone()
                .workers()
                .install(move || bool_dd_main::<oxidd::bdd::BDDFunction, _>(&cli, mref))
        }
        DDType::BCDD => {
            let mref = oxidd::bcdd::new_manager(
                inner_node_capacity,
                cli.apply_cache_capacity,
                cli.threads,
            );
            mref.clone()
                .workers()
                .install(move || bool_dd_main::<oxidd::bcdd::BCDDFunction, _>(&cli, mref))
        }
        DDType::ZBDD => todo!(),
    }
}
