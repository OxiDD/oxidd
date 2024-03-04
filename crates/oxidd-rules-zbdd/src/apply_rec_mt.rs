//! Recursive, multi-threaded apply algorithms

use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::Hash;

use oxidd_core::function::BooleanFunction;
use oxidd_core::function::BooleanVecSet;
use oxidd_core::function::Function;
use oxidd_core::util::AllocResult;
use oxidd_core::util::Borrowed;
use oxidd_core::util::EdgeDropGuard;
use oxidd_core::util::OptBool;
use oxidd_core::util::SatCountNumber;
use oxidd_core::ApplyCache;
use oxidd_core::Edge;
use oxidd_core::HasApplyCache;
use oxidd_core::HasLevel;
use oxidd_core::InnerNode;
use oxidd_core::LevelNo;
use oxidd_core::LevelView;
use oxidd_core::Manager;
use oxidd_core::Node;
use oxidd_core::NodeID;
use oxidd_core::Tag;
use oxidd_derive::Function;
use oxidd_dump::dot::DotStyle;

use super::apply_rec_st;
use super::collect_children;
use super::reduce;
use super::reduce_borrowed;
use super::singleton_level;
use super::stat;
use super::HasZBDDCache;
use super::ZBDDCache;
use super::ZBDDOp;
use super::ZBDDTerminal;

// spell-checker:ignore fnode,gnode,hnode,flevel,glevel,hlevel,ghlevel
// spell-checker:ignore hitask,symm

/// Recursively compute the subset with `var` set to `VAL`, or change `var` if
/// `VAL == -1`
fn subset<M, const VAL: i8>(
    manager: &M,
    depth: u32,
    f: Borrowed<M::Edge>,
    var: Borrowed<M::Edge>,
    var_level: LevelNo,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = ZBDDTerminal> + HasApplyCache<M, Operator = ZBDDOp> + WorkerManager,
    M::InnerNode: HasLevel,
    M::Edge: Send + Sync,
{
    if depth == 0 {
        return apply_rec_st::subset::<M, VAL>(manager, f, var, var_level);
    }
    let op = match VAL {
        -1 => ZBDDOp::Change,
        0 => ZBDDOp::Subset0,
        1 => ZBDDOp::Subset1,
        _ => unreachable!(),
    };
    stat!(call op);

    let Node::Inner(node) = manager.get_node(&f) else {
        return Ok(manager.get_terminal(ZBDDTerminal::Empty).unwrap());
    };
    let level = node.level();
    match level.cmp(&var_level) {
        Ordering::Less => {}
        Ordering::Equal => {
            if op == ZBDDOp::Change {
                // The swap of `hi` and `lo` below is intentional
                let (lo, hi) = collect_children(node);
                return reduce_borrowed(manager, level, hi, manager.clone_edge(&lo), op);
            }
            return Ok(manager.clone_edge(&node.child(VAL as usize)));
        }
        Ordering::Greater => {
            return Ok(manager.get_terminal(ZBDDTerminal::Empty).unwrap());
        }
    }

    // Query apply cache
    stat!(cache_query op);
    if let Some(h) = manager
        .apply_cache()
        .get(manager, op, &[f.borrowed(), var.borrowed()])
    {
        stat!(cache_hit op);
        return Ok(h);
    }

    let (fhi, flo) = collect_children(node);
    let d = depth - 1;
    let (hi, lo) = manager.join(
        || {
            let hi = subset::<M, VAL>(manager, d, fhi, var.borrowed(), var_level)?;
            Ok(EdgeDropGuard::new(manager, hi))
        },
        || {
            let lo = subset::<M, VAL>(manager, d, flo, var.borrowed(), var_level)?;
            Ok(EdgeDropGuard::new(manager, lo))
        },
    );
    let (hi, lo) = (hi?, lo?);
    let h = reduce(manager, level, hi.into_edge(), lo.into_edge(), op)?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, op, &[f, var], h.borrowed());

    Ok(h)
}

/// Recursively apply the union operator to `f` and `g`
fn apply_union<M>(
    manager: &M,
    depth: u32,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = ZBDDTerminal> + HasApplyCache<M, Operator = ZBDDOp> + WorkerManager,
    M::InnerNode: HasLevel,
    M::Edge: Send + Sync,
{
    use ZBDDOp::Union;
    if depth == 0 {
        return apply_rec_st::apply_union(manager, f, g);
    }

    stat!(call Union);
    let empty = EdgeDropGuard::new(manager, manager.get_terminal(ZBDDTerminal::Empty).unwrap());
    if f == g || *g == *empty {
        return Ok(manager.clone_edge(&*f));
    }
    if *f == *empty {
        return Ok(manager.clone_edge(&*g));
    }

    // Union is commutative, make the set `{f, g}` unique
    let (f, g) = if f > g { (g, f) } else { (f, g) };

    // Query apply cache
    stat!(cache_query Union);
    if let Some(h) = manager
        .apply_cache()
        .get(manager, Union, &[f.borrowed(), g.borrowed()])
    {
        stat!(cache_hit Union);
        return Ok(h);
    }

    let fnode = manager.get_node(&*f);
    let gnode = manager.get_node(&*g);
    let flevel = fnode.level();
    let glevel = gnode.level();

    let h = match flevel.cmp(&glevel) {
        Ordering::Less => {
            let (hi, flo) = collect_children(fnode.unwrap_inner());
            let lo = apply_union(manager, depth, flo, g.borrowed())?;
            reduce_borrowed(manager, flevel, hi, lo, Union)
        }
        Ordering::Equal => {
            let (fhi, flo) = collect_children(fnode.unwrap_inner());
            let (ghi, glo) = collect_children(gnode.unwrap_inner());
            let d = depth - 1;
            let (hi, lo) = manager.join(
                || {
                    let hi = apply_union(manager, d, fhi, ghi)?;
                    Ok(EdgeDropGuard::new(manager, hi))
                },
                || {
                    let lo = apply_union(manager, d, flo, glo)?;
                    Ok(EdgeDropGuard::new(manager, lo))
                },
            );
            let (hi, lo) = (hi?, lo?);
            reduce(manager, flevel, hi.into_edge(), lo.into_edge(), Union)
        }
        Ordering::Greater => {
            let (hi, glo) = collect_children(gnode.unwrap_inner());
            let lo = apply_union(manager, depth, f.borrowed(), glo)?;
            reduce_borrowed(manager, glevel, hi, lo, Union)
        }
    }?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, Union, &[f, g], h.borrowed());

    Ok(h)
}

/// Recursively apply the intersection operator to `f` and `g`
fn apply_intsec<M>(
    manager: &M,
    depth: u32,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = ZBDDTerminal> + HasApplyCache<M, Operator = ZBDDOp> + WorkerManager,
    M::InnerNode: HasLevel,
    M::Edge: Send + Sync,
{
    use ZBDDOp::Intsec;
    if depth == 0 {
        return apply_rec_st::apply_intsec(manager, f, g);
    }

    stat!(call Intsec);
    if f == g {
        return Ok(manager.clone_edge(&*f));
    }
    let empty = EdgeDropGuard::new(manager, manager.get_terminal(ZBDDTerminal::Empty).unwrap());
    if *f == *empty || *g == *empty {
        return Ok(empty.into_edge());
    }

    // Intersection is commutative, make the set `{f, g}` unique
    let (f, g) = if f > g { (g, f) } else { (f, g) };

    // Query apply cache
    stat!(cache_query Intsec);
    if let Some(h) = manager
        .apply_cache()
        .get(manager, Intsec, &[f.borrowed(), g.borrowed()])
    {
        stat!(cache_hit Intsec);
        return Ok(h);
    }

    let fnode = manager.get_node(&*f);
    let gnode = manager.get_node(&*g);
    let flevel = fnode.level();
    let glevel = gnode.level();

    let h = match flevel.cmp(&glevel) {
        Ordering::Less => {
            // f above g
            let flo = fnode.unwrap_inner().child(1);
            apply_intsec(manager, depth, flo.borrowed(), g.borrowed())
        }
        Ordering::Equal => {
            let (fhi, flo) = collect_children(fnode.unwrap_inner());
            let (ghi, glo) = collect_children(gnode.unwrap_inner());
            let d = depth - 1;
            let (hi, lo) = manager.join(
                || {
                    let hi = apply_intsec(manager, d, fhi, ghi)?;
                    Ok(EdgeDropGuard::new(manager, hi))
                },
                || {
                    let lo = apply_intsec(manager, d, flo, glo)?;
                    Ok(EdgeDropGuard::new(manager, lo))
                },
            );
            let (hi, lo) = (hi?, lo?);
            reduce(manager, flevel, hi.into_edge(), lo.into_edge(), Intsec)
        }
        Ordering::Greater => {
            let glo = gnode.unwrap_inner().child(1);
            apply_intsec(manager, depth, f.borrowed(), glo.borrowed())
        }
    }?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, Intsec, &[f, g], h.borrowed());

    Ok(h)
}

/// Recursively apply the intersection operator to `f` and `g`
fn apply_diff<M>(
    manager: &M,
    depth: u32,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = ZBDDTerminal> + HasApplyCache<M, Operator = ZBDDOp> + WorkerManager,
    M::InnerNode: HasLevel,
    M::Edge: Send + Sync,
{
    use ZBDDOp::Diff;
    if depth == 0 {
        return apply_rec_st::apply_diff(manager, f, g);
    }

    stat!(call Diff);
    let empty = EdgeDropGuard::new(manager, manager.get_terminal(ZBDDTerminal::Empty).unwrap());
    if f == g || *f == *empty {
        return Ok(empty.into_edge());
    }
    if *g == *empty {
        return Ok(manager.clone_edge(&*f));
    }

    // Query apply cache
    stat!(cache_query Diff);
    if let Some(h) = manager
        .apply_cache()
        .get(manager, Diff, &[f.borrowed(), g.borrowed()])
    {
        stat!(cache_hit Diff);
        return Ok(h);
    }

    let fnode = manager.get_node(&*f);
    let gnode = manager.get_node(&*g);
    let flevel = fnode.level();
    let glevel = gnode.level();

    let h = match flevel.cmp(&glevel) {
        Ordering::Less => {
            let (hi, flo) = collect_children(fnode.unwrap_inner());
            let lo = apply_diff(manager, depth, flo, g.borrowed())?;
            reduce_borrowed(manager, flevel, hi, lo, Diff)
        }
        Ordering::Equal => {
            let (fhi, flo) = collect_children(fnode.unwrap_inner());
            let (ghi, glo) = collect_children(gnode.unwrap_inner());
            let d = depth - 1;
            let (hi, lo) = manager.join(
                || {
                    let hi = apply_diff(manager, d, fhi, ghi)?;
                    Ok(EdgeDropGuard::new(manager, hi))
                },
                || {
                    let lo = apply_diff(manager, d, flo, glo)?;
                    Ok(EdgeDropGuard::new(manager, lo))
                },
            );
            let (hi, lo) = (hi?, lo?);
            reduce(manager, flevel, hi.into_edge(), lo.into_edge(), Diff)
        }
        Ordering::Greater => {
            let glo = gnode.unwrap_inner().child(1);
            apply_diff(manager, depth, f.borrowed(), glo.borrowed())
        }
    }?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, Diff, &[f, g], h.borrowed());

    Ok(h)
}

/// Recursively apply the symmetric difference operator to `f` and `g`
fn apply_symm_diff<M>(
    manager: &M,
    depth: u32,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = ZBDDTerminal> + HasApplyCache<M, Operator = ZBDDOp> + WorkerManager,
    M::InnerNode: HasLevel,
    M::Edge: Send + Sync,
{
    use ZBDDOp::SymmDiff;
    if depth == 0 {
        return apply_rec_st::apply_symm_diff(manager, f, g);
    }

    stat!(call SymmDiff);
    let empty = EdgeDropGuard::new(manager, manager.get_terminal(ZBDDTerminal::Empty).unwrap());
    if f == g {
        return Ok(empty.into_edge());
    }
    if *f == *empty {
        return Ok(manager.clone_edge(&*g));
    }
    if *g == *empty {
        return Ok(manager.clone_edge(&*f));
    }

    // Symmetric difference is commutative, make the set `{f, g}` unique
    let (f, g) = if f > g { (g, f) } else { (f, g) };

    // Query apply cache
    stat!(cache_query SymmDiff);
    if let Some(h) = manager
        .apply_cache()
        .get(manager, SymmDiff, &[f.borrowed(), g.borrowed()])
    {
        stat!(cache_hit SymmDiff);
        return Ok(h);
    }

    let fnode = manager.get_node(&*f);
    let gnode = manager.get_node(&*g);
    let flevel = fnode.level();
    let glevel = gnode.level();

    let h = match flevel.cmp(&glevel) {
        Ordering::Less => {
            let (hi, flo) = collect_children(fnode.unwrap_inner());
            let lo = apply_symm_diff(manager, depth, flo, g.borrowed())?;
            reduce_borrowed(manager, flevel, hi, lo, SymmDiff)
        }
        Ordering::Equal => {
            let (fhi, flo) = collect_children(fnode.unwrap_inner());
            let (ghi, glo) = collect_children(gnode.unwrap_inner());
            let d = depth - 1;
            let (hi, lo) = manager.join(
                || {
                    let hi = apply_symm_diff(manager, d, fhi, ghi)?;
                    Ok(EdgeDropGuard::new(manager, hi))
                },
                || {
                    let lo = apply_symm_diff(manager, d, flo, glo)?;
                    Ok(EdgeDropGuard::new(manager, lo))
                },
            );
            let (hi, lo) = (hi?, lo?);
            reduce(manager, flevel, hi.into_edge(), lo.into_edge(), SymmDiff)
        }
        Ordering::Greater => {
            let (hi, glo) = collect_children(gnode.unwrap_inner());
            let lo = apply_symm_diff(manager, depth, f.borrowed(), glo.borrowed())?;
            reduce_borrowed(manager, glevel, hi, lo, SymmDiff)
        }
    }?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, SymmDiff, &[f, g], h.borrowed());

    Ok(h)
}

/// Recursively apply the if-then-else operator (`if f { g } else { h }`)
fn apply_ite<M>(
    manager: &M,
    depth: u32,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
    h: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = ZBDDTerminal>
        + HasApplyCache<M, Operator = ZBDDOp>
        + HasZBDDCache<M::Edge>
        + WorkerManager,
    M::InnerNode: HasLevel,
    M::Edge: Send + Sync,
{
    use ZBDDOp::Ite;
    use ZBDDTerminal::*;
    if depth == 0 {
        return apply_rec_st::apply_ite(manager, f, g, h);
    }
    stat!(call Ite);

    // Terminal cases
    if g == h {
        return Ok(manager.clone_edge(&*g));
    }
    if f == g {
        return apply_union(manager, depth, f, h);
    }
    if f == h {
        return apply_intsec(manager, depth, f, g);
    }

    let fnode = manager.get_node(&*f);
    let flevel = fnode.level();
    if fnode.is_terminal(&Empty) {
        return Ok(manager.clone_edge(&*h));
    }

    let gnode = manager.get_node(&*g);
    let glevel = gnode.level();
    if gnode.is_terminal(&Empty) {
        // f < h = h \ f
        return apply_diff(manager, depth, h, f);
    }

    let hnode = manager.get_node(&*h);
    let hlevel = hnode.level();
    if hnode.is_terminal(&Empty) {
        return apply_intsec(manager, depth, f, g);
    }

    let ghlevel = std::cmp::min(glevel, hlevel);
    let level = std::cmp::min(flevel, ghlevel);
    let tautology = manager.zbdd_cache().tautology(level);
    if *f == *tautology {
        return Ok(manager.clone_edge(&*g));
    }
    if *g == *tautology {
        return apply_union(manager, depth, f, h);
    }
    // if &*h == &*tautology { f → g }; we cannot handle this properly

    // Query apply cache
    stat!(cache_query Ite);
    if let Some(res) =
        manager
            .apply_cache()
            .get(manager, Ite, &[f.borrowed(), g.borrowed(), h.borrowed()])
    {
        stat!(cache_hit Ite);
        return Ok(res);
    }

    let res = match Ord::cmp(&flevel, &ghlevel) {
        Ordering::Greater => {
            debug_assert!(hlevel < flevel || glevel < flevel);
            if glevel < hlevel {
                let glo = gnode.unwrap_inner().child(1);
                apply_ite(manager, depth, f.borrowed(), glo.borrowed(), h.borrowed())
            } else {
                let (hi, hlo) = collect_children(hnode.unwrap_inner());
                let g = if glevel == hlevel {
                    gnode.unwrap_inner().child(1)
                } else {
                    g.borrowed()
                };
                let lo = apply_ite(manager, depth, f.borrowed(), g, hlo)?;
                reduce_borrowed(manager, level, hi, lo, Ite)
            }
        }
        Ordering::Less => {
            let flo = fnode.unwrap_inner().child(1);
            apply_ite(manager, depth, flo.borrowed(), g.borrowed(), h.borrowed())
        }
        Ordering::Equal => {
            debug_assert!(flevel == glevel || flevel == hlevel);

            let (fhi, flo) = collect_children(fnode.unwrap_inner());

            enum HiTask<E> {
                Intsec(E, E),
                Diff(E, E),
                Ite(E, E, E),
            }
            let (hitask, glo, hlo) = if hlevel > flevel {
                let (ghi, glo) = collect_children(gnode.unwrap_inner());
                (HiTask::Intsec(fhi, ghi), glo, h.borrowed())
            } else if glevel > flevel {
                let (hhi, hlo) = collect_children(hnode.unwrap_inner());
                (HiTask::Diff(hhi, fhi), g.borrowed(), hlo)
            } else {
                debug_assert!(flevel == glevel && flevel == hlevel);
                let (ghi, glo) = collect_children(gnode.unwrap_inner());
                let (hhi, hlo) = collect_children(hnode.unwrap_inner());
                (HiTask::Ite(fhi, ghi, hhi), glo, hlo)
            };

            let d = depth - 1;
            let (hi, lo) = manager.join(
                || {
                    let hi = match hitask {
                        HiTask::Intsec(f, g) => apply_intsec(manager, d, f, g),
                        HiTask::Diff(f, g) => apply_diff(manager, d, f, g),
                        HiTask::Ite(f, g, h) => apply_ite(manager, d, f, g, h),
                    }?;
                    Ok(EdgeDropGuard::new(manager, hi))
                },
                || {
                    let lo = apply_ite(manager, d, flo, glo, hlo)?;
                    Ok(EdgeDropGuard::new(manager, lo))
                },
            );
            let (hi, lo) = (hi?, lo?);
            reduce(manager, level, hi.into_edge(), lo.into_edge(), Ite)
        }
    }?;

    manager
        .apply_cache()
        .add(manager, Ite, &[f, g, h], res.borrowed());

    Ok(res)
}

// --- Function Interface ------------------------------------------------------

use oxidd_core::WorkerManager;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Function, Debug)]
#[repr(transparent)]
pub struct ZBDDSetMT<F: Function>(F);

impl<F: Function> From<F> for ZBDDSetMT<F> {
    #[inline(always)]
    fn from(value: F) -> Self {
        ZBDDSetMT(value)
    }
}

impl<F: Function> ZBDDSetMT<F>
where
    for<'id> F::Manager<'id>: WorkerManager,
{
    /// Convert `self` into the underlying [`Function`]
    #[inline(always)]
    pub fn into_inner(self) -> F {
        self.0
    }

    fn init_depth(manager: &F::Manager<'_>) -> u32 {
        let n = manager.current_num_threads();
        if n > 1 {
            (4096 * n).ilog2()
        } else {
            0
        }
    }
}

impl<F: Function> BooleanVecSet for ZBDDSetMT<F>
where
    for<'id> F::Manager<'id>: Manager<Terminal = ZBDDTerminal>
        + super::HasZBDDOpApplyCache<F::Manager<'id>>
        + super::HasZBDDCache<<F::Manager<'id> as Manager>::Edge>
        + WorkerManager,
    for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
    for<'id> <F::Manager<'id> as Manager>::Edge: Send + Sync,
{
    #[inline]
    fn new_singleton<'id>(manager: &mut Self::Manager<'id>) -> AllocResult<Self> {
        let hi = manager.get_terminal(ZBDDTerminal::Base).unwrap();
        let lo = manager.get_terminal(ZBDDTerminal::Empty).unwrap();
        let edge = manager.add_level(|level| InnerNode::new(level, [hi, lo]))?;
        ZBDDCache::rebuild(manager);
        Ok(Self::from_edge(manager, edge))
    }

    #[inline]
    fn empty_edge<'id>(manager: &Self::Manager<'id>) -> <Self::Manager<'id> as Manager>::Edge {
        manager.get_terminal(ZBDDTerminal::Empty).unwrap()
    }

    #[inline]
    fn base_edge<'id>(manager: &Self::Manager<'id>) -> <Self::Manager<'id> as Manager>::Edge {
        manager.get_terminal(ZBDDTerminal::Base).unwrap()
    }

    #[inline]
    fn subset0_edge<'id>(
        manager: &Self::Manager<'id>,
        set: &<Self::Manager<'id> as Manager>::Edge,
        var: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        let var_level = singleton_level(manager, var);
        let depth = Self::init_depth(manager);
        subset::<_, 0>(manager, depth, set.borrowed(), var.borrowed(), var_level)
    }

    #[inline]
    fn subset1_edge<'id>(
        manager: &Self::Manager<'id>,
        set: &<Self::Manager<'id> as Manager>::Edge,
        var: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        let depth = Self::init_depth(manager);
        let var_level = singleton_level(manager, var);
        subset::<_, 1>(manager, depth, set.borrowed(), var.borrowed(), var_level)
    }

    #[inline]
    fn change_edge<'id>(
        manager: &Self::Manager<'id>,
        set: &<Self::Manager<'id> as Manager>::Edge,
        var: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        let depth = Self::init_depth(manager);
        let var_level = singleton_level(manager, var);
        subset::<_, -1>(manager, depth, set.borrowed(), var.borrowed(), var_level)
    }

    #[inline]
    fn union_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        let depth = Self::init_depth(manager);
        apply_union(manager, depth, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn intsec_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        let depth = Self::init_depth(manager);
        apply_intsec(manager, depth, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn diff_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        let depth = Self::init_depth(manager);
        apply_diff(manager, depth, lhs.borrowed(), rhs.borrowed())
    }
}

impl<F: Function> BooleanFunction for ZBDDSetMT<F>
where
    for<'id> F::Manager<'id>: Manager<Terminal = ZBDDTerminal>
        + super::HasZBDDOpApplyCache<F::Manager<'id>>
        + super::HasZBDDCache<<F::Manager<'id> as Manager>::Edge>
        + WorkerManager,
    for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
    for<'id> <F::Manager<'id> as Manager>::Edge: Send + Sync,
{
    fn new_var<'id>(manager: &mut Self::Manager<'id>) -> AllocResult<Self> {
        let hi = manager.get_terminal(ZBDDTerminal::Base).unwrap();
        let lo = manager.get_terminal(ZBDDTerminal::Empty).unwrap();
        let mut edge = manager.add_level(|level| InnerNode::new(level, [hi, lo]))?;

        // Build the chain bottom up. We need to skip the newly created level.
        let mut levels = manager.levels().rev();
        levels.next().unwrap();
        for mut view in levels {
            let level = view.level_no();
            let edge2 = manager.clone_edge(&edge);
            edge = view.get_or_insert(<F::Manager<'id> as Manager>::InnerNode::new(
                level,
                [edge, edge2],
            ))?;
        }

        ZBDDCache::rebuild(manager);

        Ok(Self::from_edge(manager, edge))
    }

    fn f_edge<'id>(manager: &Self::Manager<'id>) -> <Self::Manager<'id> as Manager>::Edge {
        manager.get_terminal(ZBDDTerminal::Empty).unwrap()
    }

    fn t_edge<'id>(manager: &Self::Manager<'id>) -> <Self::Manager<'id> as Manager>::Edge {
        manager.clone_edge(manager.zbdd_cache().tautology(0))
    }

    fn not_edge<'id>(
        manager: &Self::Manager<'id>,
        edge: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        let taut = manager.zbdd_cache().tautology(0);
        let depth = Self::init_depth(manager);
        apply_diff(manager, depth, taut.borrowed(), edge.borrowed())
    }

    fn and_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        let depth = Self::init_depth(manager);
        apply_intsec(manager, depth, lhs.borrowed(), rhs.borrowed())
    }
    fn or_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        let depth = Self::init_depth(manager);
        apply_union(manager, depth, lhs.borrowed(), rhs.borrowed())
    }
    fn nand_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        let and = Self::and_edge(manager, lhs, rhs)?;
        Self::not_edge(manager, &*EdgeDropGuard::new(manager, and))
    }
    fn nor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        let or = Self::or_edge(manager, lhs, rhs)?;
        Self::not_edge(manager, &*EdgeDropGuard::new(manager, or))
    }
    fn xor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        let depth = Self::init_depth(manager);
        apply_symm_diff(manager, depth, lhs.borrowed(), rhs.borrowed())
    }
    fn equiv_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        let xor = Self::xor_edge(manager, lhs, rhs)?;
        Self::not_edge(manager, &*EdgeDropGuard::new(manager, xor))
    }
    fn imp_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        Self::ite_edge(manager, lhs, rhs, manager.zbdd_cache().tautology(0))
    }
    fn imp_strict_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &<Self::Manager<'id> as Manager>::Edge,
        rhs: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        let depth = Self::init_depth(manager);
        apply_diff(manager, depth, rhs.borrowed(), lhs.borrowed())
    }

    fn ite_edge<'id>(
        manager: &Self::Manager<'id>,
        f: &<Self::Manager<'id> as Manager>::Edge,
        g: &<Self::Manager<'id> as Manager>::Edge,
        h: &<Self::Manager<'id> as Manager>::Edge,
    ) -> AllocResult<<Self::Manager<'id> as Manager>::Edge> {
        let depth = Self::init_depth(manager);
        apply_ite(manager, depth, f.borrowed(), g.borrowed(), h.borrowed())
    }

    #[inline]
    fn sat_count_edge<'id, N: SatCountNumber, S: std::hash::BuildHasher>(
        manager: &Self::Manager<'id>,
        edge: &<Self::Manager<'id> as Manager>::Edge,
        vars: LevelNo,
        cache: &mut HashMap<NodeID, N, S>,
    ) -> N {
        apply_rec_st::ZBDDSet::<F>::sat_count_edge(manager, edge, vars, cache)
    }

    #[inline]
    fn pick_cube_edge<'id, 'a, I>(
        manager: &'a Self::Manager<'id>,
        edge: &'a <Self::Manager<'id> as Manager>::Edge,
        order: impl IntoIterator<IntoIter = I>,
        choice: impl FnMut(&Self::Manager<'id>, &<Self::Manager<'id> as Manager>::Edge) -> bool,
    ) -> Option<Vec<OptBool>>
    where
        I: ExactSizeIterator<Item = &'a <Self::Manager<'id> as Manager>::Edge>,
    {
        apply_rec_st::ZBDDSet::<F>::pick_cube_edge(manager, edge, order, choice)
    }

    #[inline]
    fn eval_edge<'id, 'a>(
        manager: &'a Self::Manager<'id>,
        edge: &'a <Self::Manager<'id> as Manager>::Edge,
        env: impl IntoIterator<Item = (&'a <Self::Manager<'id> as Manager>::Edge, bool)>,
    ) -> bool {
        apply_rec_st::ZBDDSet::<F>::eval_edge(manager, edge, env)
    }
}

impl<F: Function, T: Tag> DotStyle<T> for ZBDDSetMT<F> {}
