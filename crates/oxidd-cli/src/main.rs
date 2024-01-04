#![forbid(unsafe_code)]

use std::borrow::Cow;
use std::fmt;
use std::fs;
use std::io;
use std::io::Seek;
use std::io::Write;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;
use std::time::Instant;

use bitvec::prelude::*;
use clap::Parser;
use clap::ValueEnum;
use num_bigint::BigUint;
use oxidd::BooleanFunction;
use oxidd_core::ApplyCache;
use oxidd_core::Edge;
use oxidd_core::HasApplyCache;
use oxidd_core::HasLevel;
use oxidd_core::LevelNo;
use oxidd_core::LevelView;
use oxidd_core::Manager;
use oxidd_core::ManagerRef;
use oxidd_core::NodeID;
use oxidd_core::WorkerManager;
use oxidd_dump::dddmp;
use oxidd_dump::dot;
use oxidd_parser::load_file::load_file;
use oxidd_parser::ClauseOrderNode;
use oxidd_parser::ParseOptionsBuilder;
use oxidd_parser::Problem;
use oxidd_parser::Prop;
use oxidd_parser::Var;
use rustc_hash::FxHashMap;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// The decision diagram type
    #[arg(value_enum, short = 't', long, default_value_t = DDType::CBDD)]
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
    #[arg(value_enum, long, default_value_t = CNFBuildOrder::Seq)]
    cnf_build_order: CNFBuildOrder,

    /// Perform model counting
    #[arg(long)]
    count_models: bool,

    /// Number of threads to use for concurrent operations
    ///
    /// A value of 0 means automatic detection
    #[arg(long, default_value_t = 0)]
    threads: u32,

    /// Report statistics every STATS_SECS seconds
    ///
    /// A value of 0 disables the reports.
    #[arg(long, default_value_t = 0)]
    stats_secs: u64,

    /// Problem input file(s)
    file: Vec<PathBuf>,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, ValueEnum)]
enum DDType {
    /// Binary decision diagram
    BDD,
    /// Binary decision diagram with complement edges
    CBDD,
    /// Zero-suppressed decision diagram
    ZBDD,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, ValueEnum)]
enum CNFBuildOrder {
    /// Sequential, as specified in the file
    Seq,
    /// Read the order from the comment lines in the input file
    File,
}

/// Human-readable durations
struct HDuration(Duration);

impl fmt::Display for HDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let d = self.0;
        let s = d.as_secs();
        if s >= 60 {
            let (m, s) = (s / 60, s % 60);
            let (h, m) = (m / 60, m % 60);
            if h == 0 {
                return write!(f, "{m} m {s} s");
            }
            let (d, h) = (h / 60, h % 60);
            if d == 0 {
                return write!(f, "{h} h {m} m {s} s");
            }
            return write!(f, "{d} d {h} h {m} m {s} s");
        }
        if s != 0 {
            return write!(f, "{:.3} s", d.as_secs_f32());
        }
        let ms = d.subsec_millis();
        if ms != 0 {
            return write!(f, "{ms} ms");
        }
        let us = d.subsec_micros();
        if us != 0 {
            return write!(f, "{us} us");
        }
        return write!(f, "{} ns", d.subsec_nanos());
    }
}

const OOM_MSG: &'static str = "Out of memory";

fn make_vars<'id, B: BooleanFunction>(
    manager: &mut B::Manager<'id>,
    num_vars: u32,
    var_order: Vec<(Var, String)>,
    name_map: &mut FxHashMap<String, B>,
) -> Vec<B>
where
    B::Manager<'id>: WorkerManager,
    <B::Manager<'id> as Manager>::InnerNode: HasLevel,
{
    if !var_order.is_empty() {
        debug_assert_eq!(var_order.len(), num_vars as usize);
        let mut vars: Vec<Option<B>> = vec![None; var_order.len()];
        let mut order = Vec::with_capacity(num_vars as usize);
        for (var, name) in var_order {
            let f = name_map
                .entry(name)
                .or_insert_with(|| B::new_var(manager).expect(OOM_MSG))
                .clone();
            vars[(var.get() - 1) as usize] = Some(f.clone());
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
                    .or_insert_with(|| B::new_var(manager).expect(OOM_MSG))
                    .clone()
            })
            .collect()
    }
}

fn make_bool_dd<B>(
    mref: &B::ManagerRef,
    problem: Problem,
    cli: &Cli,
    vars: &mut FxHashMap<String, B>,
) -> B
where
    B: BooleanFunction + Send + 'static,
    for<'id> B::Manager<'id>: WorkerManager,
    for<'id> <B::Manager<'id> as Manager>::InnerNode: HasLevel,
{
    fn prop_rec<'id, B: BooleanFunction>(manager: &B::Manager<'id>, prop: &Prop, vars: &[B]) -> B {
        match prop {
            Prop::Lit(v, false) => vars[(v.get() - 1) as usize].clone(),
            Prop::Lit(v, true) => vars[(v.get() - 1) as usize].not().expect(OOM_MSG),
            Prop::Neg(p) => prop_rec(manager, &*p, vars).not().expect(OOM_MSG),
            Prop::And(ps) => ps.iter().fold(B::t(manager), |b, p| {
                b.and(&prop_rec(manager, p, vars)).expect(OOM_MSG)
            }),
            Prop::Or(ps) => ps.iter().fold(B::f(manager), |b, p| {
                b.or(&prop_rec(manager, p, vars)).expect(OOM_MSG)
            }),
            Prop::Xor(ps) => ps.iter().fold(B::f(manager), |b, p| {
                b.xor(&prop_rec(manager, p, vars)).expect(OOM_MSG)
            }),
            Prop::Eq(ps) => ps.iter().fold(B::t(manager), |b, p| {
                b.equiv(&prop_rec(manager, p, vars)).expect(OOM_MSG)
            }),
        }
    }

    fn clause_tree_rec<B: BooleanFunction>(
        clauses: &[B],
        clause_order: &[ClauseOrderNode],
    ) -> (B, usize) {
        match clause_order[0] {
            ClauseOrderNode::Clause(n) => (clauses[(n.get() - 1) as usize].clone(), 1),
            ClauseOrderNode::Conj => {
                let mut consumed = 1;
                let (lhs, c) = clause_tree_rec(clauses, &clause_order[consumed..]);
                consumed += c;
                let (rhs, c) = clause_tree_rec(clauses, &clause_order[consumed..]);
                consumed += c;
                (lhs.and(&rhs).expect(OOM_MSG), consumed)
            }
        }
    }

    let start = Instant::now();

    let func = match problem {
        Problem::CNF {
            num_vars,
            mut var_order,
            clauses,
            clause_order,
        } => {
            if !cli.read_var_order {
                var_order.clear();
            } else if var_order.is_empty() {
                eprintln!("error: variable order not given");
                std::process::exit(1);
            }
            if cli.cnf_build_order == CNFBuildOrder::File && clause_order.is_empty() {
                eprintln!("error: clause order not given");
                std::process::exit(1);
            }

            let vars = mref.with_manager_exclusive(|manager| {
                make_vars::<B>(manager, num_vars, var_order, vars)
            });

            mref.with_manager_shared(|manager| {
                let clauses: Vec<B> = clauses
                    .into_iter()
                    .map(|mut clause| {
                        if clause.len() == 0 {
                            return B::f(manager);
                        }

                        // Build clause bottom-up
                        clause.sort_unstable_by_key(|lit| lit.0);
                        let (var, neg) = clause.pop().unwrap();
                        let init = &vars[(var.get() - 1) as usize];
                        let init = if neg {
                            init.not().expect(OOM_MSG)
                        } else {
                            init.clone()
                        };
                        clause.iter().rev().fold(init, |acc, &(var, neg)| {
                            let var = &vars[(var.get() - 1) as usize];
                            if neg {
                                acc.or(&var.not().expect(OOM_MSG)).expect(OOM_MSG)
                            } else {
                                acc.or(var).expect(OOM_MSG)
                            }
                        })
                    })
                    .collect();
                println!(
                    "all {} clauses built after {}",
                    clauses.len(),
                    HDuration(start.elapsed())
                );

                if clauses.len() == 0 {
                    B::t(manager)
                } else {
                    match cli.cnf_build_order {
                        CNFBuildOrder::Seq => {
                            let init = clauses[0].clone();
                            clauses[1..]
                                .iter()
                                .fold(init, |acc, f| acc.and(f).expect(OOM_MSG))
                        }
                        CNFBuildOrder::File => {
                            let (func, _consumed) = clause_tree_rec(&clauses, &clause_order);
                            debug_assert_eq!(_consumed, clause_order.len());
                            func
                        }
                    }
                }
            })
        }

        Problem::Prop {
            num_vars,
            var_order,
            ast,
            ..
        } => {
            if cli.read_var_order && var_order.is_empty() {
                eprintln!("error: variable order not given");
                std::process::exit(1);
            }

            let vars = mref.with_manager_exclusive(|manager| {
                make_vars::<B>(manager, num_vars, var_order, vars)
            });
            mref.with_manager_shared(|manager| prop_rec(manager, &ast, &vars))
        }

        _ => todo!(),
    };
    println!("BDD building done within {}", HDuration(start.elapsed()));
    func
}

fn background_stats<MR: ManagerRef + Send + 'static>(mref: MR, interval: Duration)
where
    for<'id> MR::Manager<'id>: HasApplyCache<MR::Manager<'id>>,
{
    if interval.is_zero() {
        return; // statistics report disabled
    }

    std::thread::spawn(move || {
        let mut scheduled = Instant::now();
        loop {
            scheduled += interval;
            if let Some(d) = scheduled.checked_duration_since(Instant::now()) {
                std::thread::sleep(d);
            }

            mref.with_manager_shared(|manager| {
                println!("[stat] {} nodes", manager.num_inner_nodes());
                //manager.apply_cache().print_stats();
            });
        }
    });
}

fn bool_dd_main<B: BooleanFunction>(cli: &Cli, mref: B::ManagerRef)
where
    B: Send + 'static,
    B::ManagerRef: Send + 'static,
    for<'id> B: dot::DotStyle<<B::Manager<'id> as Manager>::EdgeTag>,
    for<'id> B::Manager<'id>: WorkerManager + HasApplyCache<B::Manager<'id>>,
    for<'id> <B::Manager<'id> as Manager>::InnerNode: HasLevel,
    for<'id> <B::Manager<'id> as Manager>::EdgeTag: fmt::Debug,
    for<'id> <B::Manager<'id> as Manager>::Terminal: FromStr + fmt::Display + dddmp::AsciiDisplay,
{
    let parse_options = ParseOptionsBuilder::default()
        .orders(cli.read_var_order | (cli.cnf_build_order == CNFBuildOrder::File))
        .build()
        .unwrap();

    let mut vars: FxHashMap<String, B> = Default::default();
    let mut funcs: Vec<(B, String)> = Vec::new();

    background_stats::<B::ManagerRef>(mref.clone(), Duration::from_secs(cli.stats_secs));

    let report_dd_node_count = || {
        mref.with_manager_shared(|manager| {
            if !cli.no_prune_unreachable {
                println!("node count before pruning: {}", manager.num_inner_nodes());
                assert_eq!(
                    manager.num_inner_nodes(),
                    manager.levels().map(|l| l.len()).sum::<usize>()
                );
                manager.apply_cache().clear(manager);
                assert_eq!(
                    manager.num_inner_nodes(),
                    manager.levels().map(|l| l.len()).sum::<usize>()
                );
                manager.gc();
                assert_eq!(
                    manager.num_inner_nodes(),
                    manager.levels().map(|l| l.len()).sum::<usize>()
                );
            }
            let count = manager.num_inner_nodes();
            println!("node count: {count}");
            assert_eq!(count, manager.levels().map(|l| l.len()).sum::<usize>());
        })
    };

    for path in &cli.dddmp_import {
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
                        .or_insert_with(|| B::new_var(manager).expect(OOM_MSG));
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

        println!(" done ({})", HDuration(start.elapsed()));

        report_dd_node_count();
    }

    for file in &cli.file {
        let start = Instant::now();
        let Some(problem) = load_file(file, &parse_options) else {
            std::process::exit(1)
        };
        println!("parsing done within {}", HDuration(start.elapsed()));

        funcs.push((
            make_bool_dd(&mref, problem, &cli, &mut vars),
            file.file_name().unwrap().to_string_lossy().to_string(),
        ));

        report_dd_node_count();
    }

    let mut equivalences: FxHashMap<&B, Vec<usize>> = Default::default();
    for (i, (func, _)) in funcs.iter().enumerate() {
        equivalences
            .entry(func)
            .and_modify(|v| v.push(i))
            .or_insert_with(|| vec![i]);
    }

    let mut model_count_cache: FxHashMap<NodeID, BigUint> = Default::default();
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

    mref.with_manager_shared(|manager| {
        if let Some(dotfile) = &cli.dot_output {
            fs::File::create(&dotfile)
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
            fs::File::create(&dmpfile)
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
}

fn main() {
    let cli = Cli::parse();

    let mut inner_node_capacity = cli.inner_node_capacity;

    #[cfg(miri)]
    if inner_node_capacity == 0 {
        inner_node_capacity = 4096;
    }
    #[cfg(not(miri))]
    if inner_node_capacity == 0 {
        use sysinfo::{System, SystemExt};
        let mut sys = System::new();
        sys.refresh_memory();
        let mem = sys.available_memory();
        let apply_cache_size = 4 * 4 * cli.apply_cache_capacity.next_power_of_two();
        inner_node_capacity =
            std::cmp::min((mem - apply_cache_size as u64) / (4 * 8), (1 << 32) - 1) as usize;
    }

    println!("inner node capacity: {inner_node_capacity}");
    println!("apply cache capacity: {}", cli.apply_cache_capacity);

    match cli.dd_type {
        DDType::BDD => {
            let mref =
                oxidd::bdd::new_manager(inner_node_capacity, cli.apply_cache_capacity, cli.threads);
            bool_dd_main::<oxidd::bdd::BDDFunction>(&cli, mref);
        }
        DDType::CBDD => {
            let mref = oxidd::cbdd::new_manager(
                inner_node_capacity,
                cli.apply_cache_capacity,
                cli.threads,
            );
            bool_dd_main::<oxidd::cbdd::CBDDFunction>(&cli, mref);
        }
        DDType::ZBDD => todo!(),
    }
}
