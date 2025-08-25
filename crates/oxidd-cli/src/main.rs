#![forbid(unsafe_code)]

use std::borrow::Cow;
use std::fmt;
use std::fs;
use std::hash::{BuildHasherDefault, Hash};
use std::io;
use std::io::{Seek, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use num_bigint::BigUint;
use oxidd::util::SatCountCache;
use oxidd::{BooleanFunction, HasLevel, HasWorkers, Manager, ManagerRef, VarNo, WorkerPool};
use oxidd_core::function::{ETagOfFunc, INodeOfFunc, TermOfFunc};
use oxidd_core::util::VarNameMap;
use oxidd_core::{ApplyCache, HasApplyCache};
use oxidd_dump::{dddmp, dot};
use oxidd_parser::Literal;
use oxidd_parser::{load_file, ParseOptionsBuilder};
use rustc_hash::{FxHashMap, FxHasher};

mod progress;
mod scheduler;
use progress::PROGRESS;
mod profiler;
mod util;
use util::HDuration;

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

struct Inputs<'a> {
    dddmp: Vec<(io::BufReader<fs::File>, dddmp::DumpHeader)>,
    dddmp_paths: &'a [PathBuf],
    /// root literal plus
    problems: Vec<(oxidd_parser::Literal, oxidd_parser::Circuit)>,
    problem_paths: &'a [PathBuf],
}

impl<'a> Inputs<'a> {
    fn load(cli: &'a Cli) -> Self {
        let dddmp = Vec::from_iter(cli.dddmp_import.iter().map(|path| {
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
            (reader, header)
        }));

        let parse_options = ParseOptionsBuilder::default()
            .var_order(cli.read_var_order)
            .clause_tree(cli.read_clause_tree)
            .build()
            .unwrap();

        let problems = Vec::from_iter(cli.file.iter().map(|path| {
            let Some(problem) = load_file(path, &parse_options) else {
                std::process::exit(1)
            };

            let simplified = problem.simplify().unwrap().0;

            match problem.details {
                oxidd_parser::ProblemDetails::Root(literal) => (literal, simplified.circuit),
                oxidd_parser::ProblemDetails::AIGER(_) => {
                    eprintln!("\nerror: AIGER inputs are not yet supported by oxidd-cli");
                    std::process::exit(1);
                }
            }
        }));

        Self {
            dddmp,
            dddmp_paths: &cli.dddmp_import,
            problems,
            problem_paths: &cli.file,
        }
    }

    fn vars(&mut self) -> (VarNameMap, Vec<Vec<VarNo>>) {
        #[cold]
        fn too_many_vars(path: &Path) -> ! {
            eprintln!(
                "\nerror while loading {}: too many variables",
                path.display()
            );
            std::process::exit(1);
        }
        #[inline]
        fn check_too_many_vars(var_no: VarNo, path: &Path) {
            if var_no == VarNo::MAX {
                too_many_vars(path)
            }
        }

        fn add<'a>(
            var_name_map: &mut VarNameMap,
            path: &Path,
            num_vars: VarNo,
            var_names: Option<impl IntoIterator<Item = &'a str>>,
        ) -> Vec<VarNo> {
            if var_name_map.is_empty() {
                var_name_map.reserve(num_vars);

                if let Some(var_names) = var_names {
                    if let Err(err) = var_name_map.add_named(var_names) {
                        eprintln!("\nerror while loading {}: {err}", path.display());
                        std::process::exit(1);
                    }
                } else {
                    var_name_map.add_unnamed(num_vars);
                }
                return Vec::new();
            }

            let Some(var_names) = var_names else {
                if num_vars > var_name_map.len() {
                    var_name_map.add_unnamed(num_vars - var_name_map.len());
                }
                return Vec::new();
            };

            let previously_defined = var_name_map.len();
            let mut unnamed_search = 0;
            let mut next_unnamed = |var_name_map: &mut VarNameMap| {
                while unnamed_search != previously_defined {
                    if var_name_map.var_name(unnamed_search).is_empty() {
                        let id = unnamed_search;
                        unnamed_search += 1;
                        return id;
                    }
                    unnamed_search += 1;
                }
                let id = var_name_map.len();
                check_too_many_vars(id, path);
                var_name_map.add_unnamed(1);
                id
            };

            let mut vars = Vec::with_capacity(num_vars as usize);
            for name in var_names {
                if name.is_empty() {
                    vars.push(next_unnamed(var_name_map));
                } else {
                    let (id, found) = var_name_map.get_or_add(name);
                    if !found {
                        check_too_many_vars(id, path);
                    } else if id >= previously_defined {
                        eprintln!("\nerror while loading {}: the variable name '{name}' is used for two variables", path.display());
                    }
                }
            }
            vars
        }

        let mut var_name_map = VarNameMap::new();
        // mappings from problem variable number to DD variable number
        let mut var_maps = Vec::with_capacity(self.dddmp.len() + self.problems.len());

        var_maps.extend(
            self.dddmp
                .iter()
                .zip(self.dddmp_paths)
                .map(|((_, header), path)| {
                    add(
                        &mut var_name_map,
                        path,
                        header.num_vars(),
                        header
                            .var_names()
                            .map(|names| names.iter().map(String::as_str)),
                    )
                }),
        );
        var_maps.extend(self.problems.iter().zip(self.problem_paths).map(
            |((_, circuit), path)| {
                let inputs = circuit.inputs();
                let Ok(num_inputs) = VarNo::try_from(inputs.len()) else {
                    eprintln!(
                        "\nerror while loading {}: the problem requires {} \
                    variables, but OxiDD only supports up to {}",
                        path.display(),
                        inputs.len(),
                        VarNo::MAX,
                    );
                    std::process::exit(1);
                };

                let var_names = if inputs.has_names() {
                    Some((0..inputs.len()).map(|i| inputs.name(i).unwrap_or_default()))
                } else {
                    None
                };
                add(&mut var_name_map, path, num_inputs, var_names)
            },
        ));

        (var_name_map, var_maps)
    }
}

fn bool_dd_main<B, O>(cli: &Cli, mref: B::ManagerRef)
where
    B: BooleanFunction + Send + Sync + 'static,
    B::ManagerRef: HasWorkers + Send + 'static,
    for<'id> B: dot::DotStyle<ETagOfFunc<'id, B>>,
    for<'id> B::Manager<'id>: HasWorkers + HasApplyCache<B::Manager<'id>, O>,
    for<'id> INodeOfFunc<'id, B>: HasLevel,
    for<'id> ETagOfFunc<'id, B>: fmt::Debug,
    for<'id> TermOfFunc<'id, B>:
        oxidd_dump::ParseTagged<ETagOfFunc<'id, B>> + fmt::Display + oxidd_dump::AsciiDisplay,
    O: Copy + Ord + Hash,
{
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

    let mut inputs = Inputs::load(cli);
    let (var_name_map, var_maps) = inputs.vars();
    let var_count = var_name_map.len();

    mref.with_manager_exclusive(|manager| manager.add_named_vars_from_map(var_name_map))
        .unwrap();

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

    let mut var_maps: std::vec::IntoIter<Vec<VarNo>> = var_maps.into_iter();

    // Import dddmp files
    if !inputs.dddmp.is_empty() {
        PROGRESS.set_task("import from dddmp", cli.dddmp_import.len());
    }
    for ((path, (reader, header)), var_map) in inputs
        .dddmp_paths
        .iter()
        .zip(inputs.dddmp)
        .zip(var_maps.by_ref())
    {
        PROGRESS.start_op();
        let start = Instant::now();
        print!("importing '{}' ...", path.display());
        io::stdout().flush().unwrap();

        let var_map: &[VarNo] = &var_map[..]; // to help rust-analyzer
        let support_vars = Vec::from_iter(
            header
                .support_var_permutation()
                .iter()
                .map(|&i| var_map.get(i as usize).copied().unwrap_or(i)),
        );
        mref.with_manager_exclusive(|manager| {
            oxidd_reorder::set_var_order(manager, &support_vars);
        });

        mref.with_manager_shared(|manager| {
            let support_vars = support_vars.iter().copied();
            match dddmp::import(reader, &header, manager, support_vars, B::not_edge_owned) {
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
    for (problem_no, ((path, (mut root, mut circuit)), var_map)) in inputs
        .problem_paths
        .iter()
        .zip(inputs.problems)
        .zip(var_maps)
        .enumerate()
    {
        PROGRESS.set_problem_no(problem_no);
        let file_name = path.file_name().unwrap().to_string_lossy();

        if let Some(order) = circuit.inputs().order() {
            let start = Instant::now();
            let var_map: &[VarNo] = &var_map[..]; // to help rust-analyzer
            let order = Vec::from_iter(
                order
                    .iter()
                    .map(|&i| var_map.get(i).copied().unwrap_or(i as VarNo)),
            );
            mref.with_manager_exclusive(|manager| {
                oxidd_reorder::set_var_order(manager, &order);
            });
            println!("reordering took {}", HDuration(start.elapsed()));
        }

        if !var_map.is_empty() {
            for gate_no in 0..circuit.num_gates() {
                for l in circuit.gate_inputs_mut_for_no(gate_no).unwrap() {
                    if let Some(var) = l.get_input() {
                        *l = Literal::from_input(l.is_negative(), var_map[var] as usize);
                    }
                }
            }
            if let Some(var) = root.get_input() {
                root = Literal::from_input(root.is_negative(), var_map[var] as usize);
            }
        }
        drop(var_map);

        let gc_count = mref.with_manager_shared(|manager| manager.gc_count());

        let mut result = [None];
        scheduler::construct_bool_circuit(
            &mref,
            circuit,
            cli,
            &[root],
            &mut result,
            cli.size_profile.get(problem_no),
        );
        let [result] = result;
        funcs.push((result.unwrap(), file_name.to_string()));

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
            let count = f.sat_count(var_count, &mut model_count_cache);
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
                    let mut writer = std::io::BufWriter::new(file);
                    let start = Instant::now();
                    let mut export = dddmp::ExportSettings::default();
                    if cli.dddmp_ascii {
                        export = export.ascii();
                    }
                    export.export_with_names(
                        &mut writer,
                        manager,
                        funcs.iter().map(|(a, b)| (a, b)),
                    )?;
                    println!(
                        "exported decision diagram ({} bytes) in {}",
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
        DDType::ZBDD => {
            let mref = oxidd::zbdd::new_manager(
                inner_node_capacity,
                cli.apply_cache_capacity,
                cli.threads,
            );
            mref.clone()
                .workers()
                .install(move || bool_dd_main::<oxidd::zbdd::ZBDDFunction, _>(&cli, mref))
        }
    }
}
