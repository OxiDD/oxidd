use std::{collections::VecDeque, path::PathBuf};

use oxidd::{
    util::AllocResult, BooleanFunction, HasLevel, HasWorkers, Manager, ManagerRef, VarNo,
    WorkerPool,
};
use oxidd_parser::{Circuit, GateKind, Literal, Var, Vec2d};
use parking_lot::Mutex;

use crate::profiler::Profiler;
use crate::progress::PROGRESS;
use crate::util::{handle_oom, HDuration};
use crate::{Cli, GateBuildScheme};

// spell-checker:ignore mref

// Parallel construction of a DD from a circuit cannot simply be done using a
// top-down traversal and Rayon's join or reduce methods. The problem are
// diamond-shaped dependencies:
//
//         1
//        / \
//       2   3
//      / \ /|\
//     4   5 6 7
//
// In this example, gate 1 depends on gate 2 and 3, which in turn depend on
// gate 5. In a top-down approach, we would schedule the construction of gates 2
// and 3 concurrently, which would then schedule building gate 5. However, we
// only want to construct gate 5 once. While, e.g., worker A responsible for
// constructing gate 2 is computing gate 5, worker B responsible for gate 3
// could work on something else (and in particular should not busy-wait).
// Neither a lock nor Rayon's yield methods solve the problem.
//
// Instead, we use a bottom-up approach. We pre-compute the users of each gate
// ([`compute_users()`]), and propagate the DD for a gate to all its users once
// we finish the computation ([`ExecutorContext::finish_task()`]). Then, we try
// to schedule a binary operation involving the result ([`try_schedule()`]).
//
// The construction of a gate may involve multiple binary operations, and in
// this case we support multiple processing schemes ([`GateBuildScheme`]). In
// the work-stealing scheme, we allow an operand to be processed with either its
// predecessor or successor. Suppose we have a gate with three inputs/operands,
// and operand 0 is already computed, but the others are not. If operand 1
// becomes ready, then `0 op 1` can be scheduled. Suppose that instead only
// operand 2 is ready. Then `1 op 2` can be scheduled when operand 1 becomes
// ready. I.e., work-stealing determines the bracketing of the formula
// `0 op 1 op 2` (`(0 op 1) op 2` or `0 op (1 op 2)`) based on how quickly
// operands can be computed. To implement this scheme, we also need a slot to
// store the intermediate result (be it `0 op 1` or `1 op 2`) as well as an
// efficient way to lookup the predecessor or successor. We achieve this using
// a linked list of [`OperandSlot`]s. When scheduling `0 op 1`, we simply remove
// the slot for operand 1 and store the result in slot 0.
//
// For the left-deep construction scheme, the bracketing is fixed in advance. In
// the example with three operands, it is `(0 op 1) op 2`, i.e., the formula's
// syntax tree is as deep as possible on the left. We implement this scheme
// using the same data structures as the work-stealing scheme, but require that
// one of the operands comes from slot 0 when scheduling a binary operation.
//
// For the balanced construction scheme, the bracketing is also fixed in advance
// and the corresponding syntax tree is approximately balanced. For five
// operands, we have `((0 op 1) op 2) op (3 op 4)` (the first half is as large
// as the second or one larger). Here, the linked list structure does not really
// help. Instead, we note the heights of each node in syntax tree, i.e., the
// maximum depth minus the node's depth. For `n` operands, we reserve
// `n.next_power_of_two()` slots (leaving `n.next_power_of_two() - n` slots
// empty). So in the example, we initially have:
//
//      0 1 2 3 4 5 6 7
//     +-+-+-+-+-+-+-+-+
//     |0|0|1| |1| |1| |
//     +-+-+-+-+-+-+-+-+
//
// Now, the operand at index `i` and depth `d_i` can be combined with an operand
// at index `i ^ (1 << d_i)` and the same depth (`d_i`). E.g., operands 4 and 6
// can be combined, operands `2` and `0` cannot. After `0 op 1` and `3 op 4`
// have been computed, the slots would look like this:
//
//      0 1 2 3 4 5 6 7
//     +-+-+-+-+-+-+-+-+
//     |1| |1| |2| | | |
//     +-+-+-+-+-+-+-+-+
//

/// Construct a DD from a Boolean circuit
///
/// Important: `circuit` must be simplified ([Circuit::simplify()]).
pub(crate) fn construct_bool_circuit<B>(
    mref: &B::ManagerRef,
    circuit: Circuit, // could take it by reference, but this way we can drop it
    cli: &Cli,
    roots: &[Literal],
    root_results: &mut [Option<B>],
    profiler_csv_out: Option<&PathBuf>,
) where
    B: BooleanFunction + Send + Sync + 'static,
    for<'id> B::Manager<'id>: HasWorkers,
    for<'id> <B::Manager<'id> as Manager>::InnerNode: HasLevel,
    B::ManagerRef: HasWorkers,
{
    let scheme = cli.gate_build_scheme;
    PROGRESS.set_task("process circuit inputs", 1);
    let (users, mut slots, inner_ops) = mref.with_manager_shared(|manager| {
        manager.workers().set_split_depth(Some(0));

        let mut literal_cache = LiteralCache::<B>::new(manager);

        let mut all_done = true;
        for (i, &root) in roots.iter().enumerate() {
            root_results[i] = Some(if root.is_input() {
                literal_cache.get(manager, root).clone()
            } else if root == Literal::TRUE {
                B::t(manager)
            } else if root == Literal::FALSE {
                B::f(manager)
            } else {
                all_done = false;
                continue;
            });
        }

        let res = if all_done {
            (Vec2d::new(), Vec::new(), 0)
        } else {
            prepare(
                manager,
                circuit,
                &mut literal_cache,
                roots,
                root_results,
                scheme,
            )
        };
        manager.workers().set_split_depth(cli.operation_split_depth);
        res
    });

    let mut executor = QueueExecutor(VecDeque::new());
    for i in 0..slots.len() {
        if slots[i].operand.is_some() {
            try_schedule(i, &mut slots, &mut executor, scheme);
        }
    }

    PROGRESS.set_task("construct gates", inner_ops);
    let ctx = ExecutorContext {
        profiler: Profiler::new(profiler_csv_out),
        gate_users: users,
        scheme,
    };
    if !cli.parallel {
        while let Some(task) = executor.0.pop_front() {
            let (result, target) = ctx.run_task(task);
            ctx.finish_task(result, target, &mut slots, root_results, &mut executor);
        }
    } else {
        let state = Mutex::new((&mut slots[..], root_results));
        let queue = executor.0;
        rayon::scope_fifo(|scope| {
            let mut executor = RayonExecutor {
                state: &state,
                ctx: &ctx,
                scope,
            };
            for task in queue {
                executor.enqueue_task(task);
            }
        });
    }
    debug_assert!(slots.iter().all(|s| s.operand.is_none()));

    println!(
        "DD building done within {}",
        HDuration(ctx.profiler.elapsed_time())
    );
}

struct LiteralCache<B>(Vec<(Option<B>, Option<B>)>);

impl<B: BooleanFunction> LiteralCache<B> {
    fn new(manager: &B::Manager<'_>) -> Self {
        Self(vec![(None, None); manager.num_vars() as usize])
    }

    fn get(&mut self, manager: &B::Manager<'_>, literal: Literal) -> &B {
        let var = literal.get_input().unwrap();
        let slot = &mut self.0[var];
        if literal.is_positive() {
            slot.0
                .get_or_insert_with(|| handle_oom!(B::var(manager, var as VarNo)))
        } else {
            slot.1
                .get_or_insert_with(|| handle_oom!(B::not_var(manager, var as VarNo)))
        }
    }
}

/// Divide `x` into two halves
///
/// If `x` is even, both halves are equal in size, otherwise the first is one
/// larger than the second.
#[inline(always)]
const fn split_half(x: usize) -> (usize, usize) {
    let snd = x / 2;
    (x - snd, snd)
}

type OpFn<B> = fn(&B, &B) -> AllocResult<B>;

fn choose_operator<B>(gate_kind: GateKind, neg_lhs: bool, neg_rhs: bool) -> OpFn<B>
where
    B: BooleanFunction,
{
    const {
        assert!(GateKind::And as usize == 0);
        assert!(GateKind::Or as usize == 1);
        assert!(GateKind::Xor as usize == 2);
    }

    fn rev_imp_strict<B: BooleanFunction>(lhs: &B, rhs: &B) -> AllocResult<B> {
        B::imp_strict(rhs, lhs)
    }
    fn rev_imp<B: BooleanFunction>(lhs: &B, rhs: &B) -> AllocResult<B> {
        B::imp(rhs, lhs)
    }

    let tbl = &[
        [[B::and, B::imp_strict], [rev_imp_strict, B::nor]],
        [[B::or, B::imp], [rev_imp, B::nand]],
        [[B::xor, B::equiv], [B::equiv, B::xor]],
    ];
    tbl[gate_kind as usize][neg_rhs as usize][neg_lhs as usize]
}

struct Task<B> {
    lhs: B,
    rhs: B,
    operator: OpFn<B>,
    target: usize,
}

/// User of a gate
#[derive(Clone, Copy, PartialEq, Eq)]
struct User(usize);

impl User {
    const POLARITY_BIT: u32 = 0;
    const ROOT_BIT: u32 = 1;
    const VALUE_LSB: u32 = 2;

    #[inline(always)]
    const fn is_positive(self) -> bool {
        self.0 & (1 << Self::POLARITY_BIT) == 0
    }
    #[inline(always)]
    const fn is_negative(self) -> bool {
        !self.is_positive()
    }

    #[inline(always)]
    const fn is_root(self) -> bool {
        self.0 & (1 << Self::ROOT_BIT) != 0
    }

    #[inline(always)]
    const fn index(self) -> usize {
        self.0 >> Self::VALUE_LSB
    }

    #[inline]
    const fn from_root(negative: bool, root: usize) -> Self {
        Self(
            (root << Self::VALUE_LSB)
                | (1 << Self::ROOT_BIT)
                | ((negative as usize) << Self::POLARITY_BIT),
        )
    }

    #[inline]
    const fn from_gate_input(negative: bool, index: usize) -> Self {
        Self((index << Self::VALUE_LSB) | ((negative as usize) << Self::POLARITY_BIT))
    }
}

struct OperandSlot<B> {
    operand: Option<B>,
    /// The gate's operator
    operator: GateKind, // included here because it has no
    /// Whether the operand must be negated on use
    neg: bool,
    /// Gate number
    gate: usize,
    /// For [`GateBuildScheme::WorkStealing`] and [`GateBuildScheme::LeftDeep`]:
    /// index of the previous slot in the linked list, or [`usize::MAX`] if this
    /// is the first slot of the gate.
    ///
    /// For [`GateBuildScheme::Balanced`]: Index of the first slot for the gate.
    prev: usize,
    /// For [`GateBuildScheme::WorkStealing`] and [`GateBuildScheme::LeftDeep`]:
    /// index of the next slot in the linked list, or [`usize::MAX`] if this is
    /// the last slot of the gate.
    ///
    /// For [`GateBuildScheme::Balanced`]: Height of the node in the gate
    /// formula's syntax tree (i.e., maximum depth - depth of the node).
    next: usize,
}

impl<B> OperandSlot<B> {
    #[track_caller]
    #[inline]
    fn store_operand(&mut self, operand: B) {
        debug_assert!(self.operand.is_none());
        self.operand = Some(operand);
    }
}

/// Count the users of each gate reachable from `root` in `circuit`
fn count_users(circuit: &Circuit, root: Var, use_counts: &mut [usize], total_users: &mut usize) {
    let inputs = circuit.gate_for_no(root).unwrap().inputs;
    for &input in inputs {
        if let Some(g) = input.get_gate_no() {
            let prev = use_counts[g];
            use_counts[g] += 1;
            *total_users += 1;
            if prev == 0 {
                count_users(circuit, g, use_counts, total_users);
            }
        }
    }
}

/// Decrement `x` by 1, returning its new value
#[inline(always)]
fn sub_fetch(x: &mut usize) -> usize {
    *x -= 1;
    *x
}

/// Compute the users of each gate and the length of the [`OperandSlot`] array
///
/// If a [`User`] `u` is a root, then [`u.index()`][User::index] will be the
/// index in `roots`. Otherwise the index will refer to a target
/// [`OperandSlot`]. The sequence of `OperandSlot`s is to be created by the
/// caller, its length is given by the second component of the return value.
fn compute_users(
    circuit: &Circuit,
    roots: &[Literal],
    scheme: GateBuildScheme,
) -> (Vec2d<User>, usize) {
    // count the users of each gate reachable from `roots`
    let mut use_counts = vec![0usize; circuit.num_gates()];
    let mut total_users = 0;
    for &root in roots {
        if let Some(g) = root.get_gate_no() {
            let prev = use_counts[g];
            use_counts[g] += 1;
            total_users += 1;
            if prev == 0 {
                count_users(circuit, g, &mut use_counts, &mut total_users);
            }
        }
    }

    // create the user vectors filled with zeros
    let mut users = Vec2d::with_capacity(circuit.num_gates(), total_users);
    for &count in use_counts.iter() {
        users.push_vec();
        users.resize_last(count, User(0));
    }

    // Fill in the users. We do this from back to front for each gate, decrementing
    // the respective element of `use_counts`.

    for (i, &root) in roots.iter().enumerate() {
        if let Some(g) = root.get_gate_no() {
            let c = sub_fetch(&mut use_counts[g]);
            users[g][c] = User::from_root(root.is_negative(), i);
        }
    }

    let mut target = 0; // index of the `TaskOperand` corresponding to a gate input
    let mut offset_buf = Vec::new();
    for (i, gate) in circuit.iter_gates().enumerate() {
        if users[i].is_empty() {
            continue;
        }
        let gate_input_count = gate.inputs.iter().filter(|input| input.is_gate()).count();
        if gate_input_count == 0 {
            continue;
        }

        let has_circuit_inputs = (gate_input_count != gate.inputs.len()) as usize;
        let operand_count = gate_input_count + has_circuit_inputs;
        match scheme {
            GateBuildScheme::LeftDeep | GateBuildScheme::WorkStealing => {
                target += has_circuit_inputs;
                for &input in gate.inputs {
                    if let Some(g) = input.get_gate_no() {
                        let c = sub_fetch(&mut use_counts[g]);
                        users[g][c] = User::from_gate_input(input.is_negative(), target);
                        target += 1;
                    }
                }
            }
            GateBuildScheme::Balanced => {
                debug_assert!(offset_buf.is_empty());
                offset_buf.reserve(operand_count);

                fn write_offsets(
                    offset_buf: &mut Vec<usize>,
                    operand_count: usize,
                    pow2: usize,
                    offset: usize,
                ) {
                    debug_assert!(pow2 >= operand_count);
                    debug_assert!(pow2.is_power_of_two());
                    if operand_count == 1 {
                        offset_buf.push(offset);
                    } else {
                        let (h1, h2) = split_half(operand_count);
                        write_offsets(offset_buf, h1, pow2 / 2, offset);
                        write_offsets(offset_buf, h2, pow2 / 2, offset + pow2 / 2);
                    }
                }

                let pow2 = operand_count.next_power_of_two();
                write_offsets(&mut offset_buf, operand_count, pow2, target);
                debug_assert_eq!(offset_buf.len(), operand_count);
                target += pow2;

                let gate_inputs = gate.inputs.iter().filter(|i| i.is_gate());
                for (&input, &target) in gate_inputs.zip(&offset_buf[has_circuit_inputs..]) {
                    let g = input.get_gate_no().unwrap();
                    let c = sub_fetch(&mut use_counts[g]);
                    users[g][c] = User::from_gate_input(input.is_negative(), target);
                }
                offset_buf.clear();
            }
        }
    }

    (users, target)
}

fn prepare<B>(
    manager: &B::Manager<'_>,
    circuit: Circuit,
    literal_cache: &mut LiteralCache<B>,
    roots: &[Literal],
    root_results: &mut [Option<B>],
    scheme: GateBuildScheme,
) -> (Vec2d<User>, Vec<OperandSlot<B>>, usize)
where
    B: BooleanFunction,
    for<'id> <B::Manager<'id> as Manager>::InnerNode: HasLevel,
{
    debug_assert_eq!(roots.len(), root_results.len());

    let (users, slots_len) = compute_users(&circuit, roots, scheme);

    let mut input_buf: Vec<Literal> = Vec::new();
    let mut slots: Vec<OperandSlot<B>> = Vec::from_iter((0..slots_len).map(|_| OperandSlot {
        operand: None,
        operator: GateKind::And,
        neg: false,
        gate: 0,
        prev: 0,
        next: 0,
    }));

    for &user in users.all_elements() {
        if !user.is_root() && user.is_negative() {
            slots[user.index()].neg = true;
        }
    }

    let mut target = 0;
    let mut inner_ops = 0; // for progress reporting only
    for (gate_no, gate) in circuit.iter_gates().enumerate() {
        debug_assert!(input_buf.is_empty());
        input_buf.reserve(gate.inputs.len());

        for &input in gate.inputs.iter().filter(|input| !input.is_gate()) {
            input_buf.push(input);
        }
        let circuit_input_count = input_buf.len();

        if !input_buf.is_empty() {
            input_buf
                .sort_unstable_by_key(|l| l.get_input().map(|v| manager.var_to_level(v as VarNo)));
            // input_buf has no constants due to simplification
            let mut acc = literal_cache.get(manager, input_buf.pop().unwrap()).clone();
            while let Some(l) = input_buf.pop() {
                let f = literal_cache.get(manager, l);
                acc = handle_oom!(match gate.kind {
                    GateKind::And => acc.and(f),
                    GateKind::Or => acc.or(f),
                    GateKind::Xor => acc.xor(f),
                });
            }

            if circuit_input_count == gate.inputs.len() {
                let mut neg_cache = None;
                for user in &users[gate_no] {
                    if user.is_root() {
                        let r = negate_with_cache(&acc, user.is_negative(), &mut neg_cache);
                        debug_assert!(root_results[user.index()].is_none());
                        root_results[user.index()] = Some(r);
                    } else {
                        slots[user.index()].store_operand(acc.clone());
                    }
                }
                continue;
            }
            slots[target].store_operand(acc);
        }

        let gate_input_count = gate.inputs.len() - circuit_input_count;
        let gate_input_offset = (circuit_input_count != 0) as usize;
        let op_count = gate_input_count + gate_input_offset;
        inner_ops += op_count - 1;
        debug_assert!(op_count >= 2); // due to simplification
        let next_target = target
            + match scheme {
                GateBuildScheme::LeftDeep | GateBuildScheme::WorkStealing => op_count,
                GateBuildScheme::Balanced => op_count.next_power_of_two(),
            };
        let gate_slots = &mut slots[target..next_target]; // slots of the current gate

        match scheme {
            GateBuildScheme::LeftDeep | GateBuildScheme::WorkStealing => {
                for op in gate_slots.iter_mut() {
                    op.gate = gate_no;
                    op.operator = gate.kind;
                }

                gate_slots.first_mut().unwrap().prev = usize::MAX;
                gate_slots.last_mut().unwrap().next = usize::MAX;
                for i in 0..op_count - 1 {
                    gate_slots[i + 1].prev = target + i;
                    gate_slots[i].next = target + i + 1;
                }
            }
            GateBuildScheme::Balanced => {
                for op in gate_slots.iter_mut() {
                    op.gate = gate_no;
                    op.operator = gate.kind;
                    op.prev = target;
                }

                fn write_heights<B>(slots: &mut [OperandSlot<B>], op_count: usize, height: usize) {
                    debug_assert!(slots.len().is_power_of_two());
                    debug_assert!(slots.len() >= op_count);
                    if op_count == 1 {
                        slots[0].next = height;
                    } else {
                        let slots_mid = slots.len() / 2;
                        let (h1, h2) = split_half(op_count);
                        write_heights(&mut slots[..slots_mid], h1, height - 1);
                        write_heights(&mut slots[slots_mid..], h2, height - 1);
                    }
                }

                let height = gate_slots.len().trailing_zeros() as usize;
                // Example: If op_count is 2, then height is 1 and the height of
                // the two operands is 0. So we can get the other operand's
                // index in gate_slots
                write_heights(gate_slots, op_count, height);
            }
        }
        target = next_target;
    }

    (users, slots, inner_ops)
}

fn negate_with_cache<B: BooleanFunction>(f: &B, negate: bool, cache: &mut Option<B>) -> B {
    if negate {
        if let Some(negated) = &cache {
            negated.clone()
        } else {
            let negated = handle_oom!(f.not());
            *cache = Some(negated.clone());
            negated
        }
    } else {
        f.clone()
    }
}

/// Try to schedule an operation involving the operand at `index`
fn try_schedule<B: BooleanFunction, E: Executor<B>>(
    index: usize,
    slots: &mut [OperandSlot<B>],
    executor: &mut E,
    scheme: GateBuildScheme,
) {
    let slot = &slots[index];
    debug_assert!(slot.operand.is_some());

    match scheme {
        GateBuildScheme::LeftDeep | GateBuildScheme::WorkStealing => {
            let mut check_and_enqueue = |slots: &mut [OperandSlot<B>], first, second| {
                if first >= second || second >= slots.len() {
                    debug_assert!(first == usize::MAX || second == usize::MAX);
                    return false;
                }
                let [s1, .., s2] = &mut slots[first..=second] else {
                    unreachable!();
                };
                debug_assert_eq!(s1.next, second);
                debug_assert_eq!(s2.prev, first);
                debug_assert_eq!(s1.gate, s2.gate);
                debug_assert_eq!(s1.operator, s2.operator);
                if scheme == GateBuildScheme::LeftDeep && s1.prev != usize::MAX {
                    return false;
                }
                if s1.operand.is_none() || s2.operand.is_none() {
                    return false;
                }

                executor.enqueue_task(Task {
                    lhs: Option::take(&mut s1.operand).unwrap(),
                    rhs: Option::take(&mut s2.operand).unwrap(),
                    operator: choose_operator::<B>(s1.operator, s1.neg, s2.neg),
                    target: first,
                });
                s1.neg = false;
                let next = s2.next;
                s1.next = next;
                if let Some(s3) = slots.get_mut(next) {
                    s3.prev = first;
                }
                true
            };

            let prev = slot.prev;
            let next = slot.next;
            debug_assert!(prev != usize::MAX || next != usize::MAX); // due to preprocessing (simplify)

            if check_and_enqueue(slots, prev, index) {
                return;
            }
            check_and_enqueue(slots, index, next);
        }
        GateBuildScheme::Balanced => {
            let first = slot.prev;
            let height = slot.next;
            let i = index - first;
            let j = i ^ (1 << height);

            let (i, j) = if i < j { (i, j) } else { (j, i) };
            let target = first + i;
            let [s1, .., s2] = &mut slots[target..=first + j] else {
                unreachable!()
            };

            // heights agree and both operands are present as DDs
            if s1.next == s2.next && s1.operand.is_some() && s2.operand.is_some() {
                executor.enqueue_task(Task {
                    lhs: Option::take(&mut s1.operand).unwrap(),
                    rhs: Option::take(&mut s2.operand).unwrap(),
                    operator: choose_operator::<B>(s1.operator, s1.neg, s2.neg),
                    target,
                });
                s1.neg = false;
                s1.next += 1;
            }
        }
    }
}

struct ExecutorContext {
    profiler: Profiler,
    gate_users: Vec2d<User>,
    scheme: GateBuildScheme,
}

impl ExecutorContext {
    fn run_task<B: BooleanFunction>(&self, task: Task<B>) -> (B, usize) {
        let op_start = self.profiler.start_op();
        let result = handle_oom!((task.operator)(&task.lhs, &task.rhs));
        self.profiler.finish_op(op_start, &result);
        (result, task.target)
    }

    fn finish_task<B: BooleanFunction, E: Executor<B>>(
        &self,
        result: B,
        target: usize,
        slots: &mut [OperandSlot<B>],
        root_results: &mut [Option<B>],
        executor: &mut E,
    ) {
        let slot = &slots[target];

        let gate_finished = match self.scheme {
            GateBuildScheme::LeftDeep | GateBuildScheme::WorkStealing => {
                slot.prev == usize::MAX && slot.next == usize::MAX
            }
            GateBuildScheme::Balanced => {
                let height = slot.next;
                target == slot.prev
                    && match slots.get(target + (1 << height)) {
                        None => true,
                        Some(op) => op.prev != target,
                    }
            }
        };

        if gate_finished {
            let mut neg_cache: Option<B> = None;
            for user in &self.gate_users[slot.gate] {
                if user.is_root() {
                    let r = negate_with_cache(&result, user.is_negative(), &mut neg_cache);
                    debug_assert!(root_results[user.index()].is_none());
                    root_results[user.index()] = Some(r);
                } else {
                    let i = user.index();
                    slots[i].store_operand(result.clone());
                    try_schedule(i, slots, executor, self.scheme);
                }
            }
        } else {
            debug_assert!(!slots[target].neg);
            slots[target].store_operand(result);
            try_schedule(target, slots, executor, self.scheme);
        }
    }
}

trait Executor<B> {
    fn enqueue_task(&mut self, task: Task<B>);
}

struct QueueExecutor<B>(VecDeque<Task<B>>);

impl<B> Executor<B> for QueueExecutor<B> {
    #[inline(always)]
    fn enqueue_task(&mut self, task: Task<B>) {
        self.0.push_back(task);
    }
}

type State<'a, B> = (&'a mut [OperandSlot<B>], &'a mut [Option<B>]);

struct RayonExecutor<'scope, 'a, B> {
    state: &'scope Mutex<State<'scope, B>>,
    ctx: &'scope ExecutorContext,
    scope: &'a rayon::ScopeFifo<'scope>,
}

impl<B: BooleanFunction + Send> Executor<B> for RayonExecutor<'_, '_, B> {
    fn enqueue_task(&mut self, task: Task<B>) {
        let state = self.state;
        let ctx = self.ctx;
        self.scope.spawn_fifo(move |scope| {
            let (result, target) = ctx.run_task(task);

            let mut exec = RayonExecutor { state, ctx, scope };
            let mut state_guard = state.lock();
            let state = &mut *state_guard;
            ctx.finish_task(result, target, state.0, state.1, &mut exec)
        });
    }
}
