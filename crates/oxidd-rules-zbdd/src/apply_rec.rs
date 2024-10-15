//! Recursive single-threaded apply algorithms

use std::borrow::Borrow;
use std::cmp::{Ord, Ordering};
use std::hash::{BuildHasher, Hash};

use bitvec::slice::BitSlice;
use bitvec::vec::BitVec;

use oxidd_core::{
    function::{BooleanFunction, BooleanVecSet, EdgeOfFunc, Function},
    util::{AllocResult, Borrowed, EdgeDropGuard, OptBool, SatCountCache, SatCountNumber},
    ApplyCache, Edge, HasApplyCache, HasLevel, InnerNode, LevelNo, Manager, Node, Tag,
};
use oxidd_derive::Function;
use oxidd_dump::dot::DotStyle;

use crate::recursor::{Recursor, SequentialRecursor};

#[cfg(feature = "statistics")]
use super::STAT_COUNTERS;
use super::{
    collect_children, reduce, reduce_borrowed, singleton_level, stat, HasZBDDCache, ZBDDCache,
    ZBDDOp, ZBDDTerminal,
};

// spell-checker:ignore fnode,gnode,hnode,flevel,glevel,hlevel,ghlevel
// spell-checker:ignore symm

/// Recursively compute the subset with `var` set to `VAL`, or change `var` if
/// `VAL == -1`
fn subset<M, R: Recursor<M>, const VAL: i8>(
    manager: &M,
    rec: R,
    f: Borrowed<M::Edge>,
    var: Borrowed<M::Edge>,
    var_level: LevelNo,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = ZBDDTerminal> + HasApplyCache<M, ZBDDOp>,
    M::InnerNode: HasLevel,
{
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
    let (hi, lo) = rec.subset(
        subset::<M, R, VAL>,
        manager,
        (fhi, var.borrowed(), var_level),
        (flo, var.borrowed(), var_level),
    )?;
    let h = reduce(manager, level, hi.into_edge(), lo.into_edge(), op)?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, op, &[f, var], h.borrowed());

    Ok(h)
}

/// Recursively apply the union operator to `f` and `g`
fn apply_union<M, R: Recursor<M>>(
    manager: &M,
    rec: R,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = ZBDDTerminal> + HasApplyCache<M, ZBDDOp>,
    M::InnerNode: HasLevel,
{
    if rec.should_switch_to_sequential() {
        return apply_union(manager, SequentialRecursor, f, g);
    }
    use ZBDDOp::Union;
    stat!(call Union);

    let empty = EdgeDropGuard::new(manager, manager.get_terminal(ZBDDTerminal::Empty).unwrap());
    if f == g || *g == *empty {
        return Ok(manager.clone_edge(&f));
    }
    if *f == *empty {
        return Ok(manager.clone_edge(&g));
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

    let fnode = manager.get_node(&f);
    let gnode = manager.get_node(&g);
    let flevel = fnode.level();
    let glevel = gnode.level();

    let h = match flevel.cmp(&glevel) {
        Ordering::Less => {
            let (hi, flo) = collect_children(fnode.unwrap_inner());
            let lo = apply_union(manager, rec, flo, g.borrowed())?;
            reduce_borrowed(manager, flevel, hi, lo, Union)
        }
        Ordering::Equal => {
            let (fhi, flo) = collect_children(fnode.unwrap_inner());
            let (ghi, glo) = collect_children(gnode.unwrap_inner());
            let (hi, lo) = rec.binary(apply_union, manager, (fhi, ghi), (flo, glo))?;
            reduce(manager, flevel, hi.into_edge(), lo.into_edge(), Union)
        }
        Ordering::Greater => {
            let (hi, glo) = collect_children(gnode.unwrap_inner());
            let lo = apply_union(manager, rec, f.borrowed(), glo)?;
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
fn apply_intsec<M, R: Recursor<M>>(
    manager: &M,
    rec: R,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = ZBDDTerminal> + HasApplyCache<M, ZBDDOp>,
    M::InnerNode: HasLevel,
{
    if rec.should_switch_to_sequential() {
        return apply_intsec(manager, SequentialRecursor, f, g);
    }
    use ZBDDOp::Intsec;
    stat!(call Intsec);

    if f == g {
        return Ok(manager.clone_edge(&f));
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

    let fnode = manager.get_node(&f);
    let gnode = manager.get_node(&g);
    let flevel = fnode.level();
    let glevel = gnode.level();

    let h = match flevel.cmp(&glevel) {
        Ordering::Less => {
            // f above g
            let flo = fnode.unwrap_inner().child(1);
            apply_intsec(manager, rec, flo.borrowed(), g.borrowed())
        }
        Ordering::Equal => {
            let (fhi, flo) = collect_children(fnode.unwrap_inner());
            let (ghi, glo) = collect_children(gnode.unwrap_inner());
            let (hi, lo) = rec.binary(apply_intsec, manager, (fhi, ghi), (flo, glo))?;
            reduce(manager, flevel, hi.into_edge(), lo.into_edge(), Intsec)
        }
        Ordering::Greater => {
            let glo = gnode.unwrap_inner().child(1);
            apply_intsec(manager, rec, f.borrowed(), glo.borrowed())
        }
    }?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, Intsec, &[f, g], h.borrowed());

    Ok(h)
}

/// Recursively apply the difference operator to `f` and `g`
fn apply_diff<M, R: Recursor<M>>(
    manager: &M,
    rec: R,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = ZBDDTerminal> + HasApplyCache<M, ZBDDOp>,
    M::InnerNode: HasLevel,
{
    if rec.should_switch_to_sequential() {
        return apply_diff(manager, SequentialRecursor, f, g);
    }
    use ZBDDOp::Diff;
    stat!(call Diff);

    let empty = EdgeDropGuard::new(manager, manager.get_terminal(ZBDDTerminal::Empty).unwrap());
    if f == g || *f == *empty {
        return Ok(empty.into_edge());
    }
    if *g == *empty {
        return Ok(manager.clone_edge(&f));
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

    let fnode = manager.get_node(&f);
    let gnode = manager.get_node(&g);
    let flevel = fnode.level();
    let glevel = gnode.level();

    let h = match flevel.cmp(&glevel) {
        Ordering::Less => {
            let (hi, flo) = collect_children(fnode.unwrap_inner());
            let lo = apply_diff(manager, rec, flo, g.borrowed())?;
            reduce_borrowed(manager, flevel, hi, lo, Diff)
        }
        Ordering::Equal => {
            let (fhi, flo) = collect_children(fnode.unwrap_inner());
            let (ghi, glo) = collect_children(gnode.unwrap_inner());
            let (hi, lo) = rec.binary(apply_diff, manager, (fhi, ghi), (flo, glo))?;
            reduce(manager, flevel, hi.into_edge(), lo.into_edge(), Diff)
        }
        Ordering::Greater => {
            let glo = gnode.unwrap_inner().child(1);
            apply_diff(manager, rec, f.borrowed(), glo.borrowed())
        }
    }?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, Diff, &[f, g], h.borrowed());

    Ok(h)
}

fn apply_not<M, R: Recursor<M>>(manager: &M, rec: R, f: Borrowed<M::Edge>) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = ZBDDTerminal> + HasApplyCache<M, ZBDDOp> + HasZBDDCache<M::Edge>,
    M::InnerNode: HasLevel,
{
    let taut = manager.zbdd_cache().tautology(0);
    apply_diff(manager, rec, taut.borrowed(), f)
}

/// Recursively apply the symmetric difference operator to `f` and `g`
fn apply_symm_diff<M, R: Recursor<M>>(
    manager: &M,
    rec: R,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = ZBDDTerminal> + HasApplyCache<M, ZBDDOp>,
    M::InnerNode: HasLevel,
{
    if rec.should_switch_to_sequential() {
        return apply_symm_diff(manager, SequentialRecursor, f, g);
    }
    use ZBDDOp::SymmDiff;
    stat!(call SymmDiff);

    let empty = EdgeDropGuard::new(manager, manager.get_terminal(ZBDDTerminal::Empty).unwrap());
    if f == g {
        return Ok(empty.into_edge());
    }
    if *f == *empty {
        return Ok(manager.clone_edge(&g));
    }
    if *g == *empty {
        return Ok(manager.clone_edge(&f));
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

    let fnode = manager.get_node(&f);
    let gnode = manager.get_node(&g);
    let flevel = fnode.level();
    let glevel = gnode.level();

    let h = match flevel.cmp(&glevel) {
        Ordering::Less => {
            let (hi, flo) = collect_children(fnode.unwrap_inner());
            let lo = apply_symm_diff(manager, rec, flo, g.borrowed())?;
            reduce_borrowed(manager, flevel, hi, lo, SymmDiff)
        }
        Ordering::Equal => {
            let (fhi, flo) = collect_children(fnode.unwrap_inner());
            let (ghi, glo) = collect_children(gnode.unwrap_inner());
            let (hi, lo) = rec.binary(apply_symm_diff, manager, (fhi, ghi), (flo, glo))?;
            reduce(manager, flevel, hi.into_edge(), lo.into_edge(), SymmDiff)
        }
        Ordering::Greater => {
            let (hi, glo) = collect_children(gnode.unwrap_inner());
            let lo = apply_symm_diff(manager, rec, f.borrowed(), glo.borrowed())?;
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
fn apply_ite<M, R: Recursor<M>>(
    manager: &M,
    rec: R,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
    h: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = ZBDDTerminal> + HasApplyCache<M, ZBDDOp> + HasZBDDCache<M::Edge>,
    M::InnerNode: HasLevel,
{
    if rec.should_switch_to_sequential() {
        return apply_ite(manager, SequentialRecursor, f, g, h);
    }
    use ZBDDOp::Ite;
    use ZBDDTerminal::*;
    stat!(call Ite);

    // Terminal cases
    if g == h {
        return Ok(manager.clone_edge(&g));
    }
    if f == g {
        return apply_union(manager, rec, f, h);
    }
    if f == h {
        return apply_intsec(manager, rec, f, g);
    }

    let fnode = manager.get_node(&f);
    let flevel = fnode.level();
    if fnode.is_terminal(&Empty) {
        return Ok(manager.clone_edge(&h));
    }

    let gnode = manager.get_node(&g);
    let glevel = gnode.level();
    if gnode.is_terminal(&Empty) {
        // f < h = h \ f
        return apply_diff(manager, rec, h, f);
    }

    let hnode = manager.get_node(&h);
    let hlevel = hnode.level();
    if hnode.is_terminal(&Empty) {
        return apply_intsec(manager, rec, f, g);
    }

    let ghlevel = std::cmp::min(glevel, hlevel);
    let level = std::cmp::min(flevel, ghlevel);
    let tautology = manager.zbdd_cache().tautology(level);
    if *f == *tautology {
        return Ok(manager.clone_edge(&g));
    }
    if *g == *tautology {
        return apply_union(manager, rec, f, h);
    }
    // if *h == *tautology { f → g }; we cannot handle this properly

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
                apply_ite(manager, rec, f.borrowed(), glo.borrowed(), h.borrowed())
            } else {
                let (hi, hlo) = collect_children(hnode.unwrap_inner());
                let g = if glevel == hlevel {
                    gnode.unwrap_inner().child(1)
                } else {
                    g.borrowed()
                };
                let lo = apply_ite(manager, rec, f.borrowed(), g, hlo)?;
                reduce_borrowed(manager, level, hi, lo, Ite)
            }
        }
        Ordering::Less => {
            let flo = fnode.unwrap_inner().child(1);
            apply_ite(manager, rec, flo.borrowed(), g.borrowed(), h.borrowed())
        }
        Ordering::Equal => {
            debug_assert!(flevel == glevel || flevel == hlevel);
            let (fhi, flo) = collect_children(fnode.unwrap_inner());
            let (hi, lo) = if hlevel > flevel {
                let (ghi, glo) = collect_children(gnode.unwrap_inner());
                rec.binary_ternary(
                    manager,
                    apply_intsec,
                    (fhi, ghi),
                    apply_ite,
                    (flo, glo, h.borrowed()),
                )
            } else if glevel > flevel {
                let (hhi, hlo) = collect_children(hnode.unwrap_inner());
                rec.binary_ternary(
                    manager,
                    apply_diff,
                    (hhi, fhi),
                    apply_ite,
                    (flo, g.borrowed(), hlo),
                )
            } else {
                debug_assert!(flevel == glevel && flevel == hlevel);
                let (ghi, glo) = collect_children(gnode.unwrap_inner());
                let (hhi, hlo) = collect_children(hnode.unwrap_inner());
                rec.ternary(apply_ite, manager, (fhi, ghi, hhi), (flo, glo, hlo))
            }?;
            reduce(manager, level, hi.into_edge(), lo.into_edge(), Ite)
        }
    }?;

    manager
        .apply_cache()
        .add(manager, Ite, &[f, g, h], res.borrowed());

    Ok(res)
}

// --- Function Interface ------------------------------------------------------

/// Workaround for https://github.com/rust-lang/rust/issues/49601
trait HasZBDDOpApplyCache<M: Manager>: HasApplyCache<M, ZBDDOp> {}
impl<M: Manager + HasApplyCache<M, ZBDDOp>> HasZBDDOpApplyCache<M> for M {}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Function, Debug)]
#[repr(transparent)]
pub struct ZBDDFunction<F: Function>(F);

impl<F: Function> From<F> for ZBDDFunction<F> {
    #[inline(always)]
    fn from(value: F) -> Self {
        ZBDDFunction(value)
    }
}

impl<F: Function> ZBDDFunction<F> {
    /// Convert `self` into the underlying [`Function`]
    #[inline(always)]
    pub fn into_inner(self) -> F {
        self.0
    }
}

impl<F: Function> BooleanVecSet for ZBDDFunction<F>
where
    for<'id> F::Manager<'id>: Manager<Terminal = ZBDDTerminal>
        + HasZBDDOpApplyCache<F::Manager<'id>>
        + HasZBDDCache<<F::Manager<'id> as Manager>::Edge>,
    for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
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
    fn empty_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
        manager.get_terminal(ZBDDTerminal::Empty).unwrap()
    }

    #[inline]
    fn base_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
        manager.get_terminal(ZBDDTerminal::Base).unwrap()
    }

    #[inline]
    fn subset0_edge<'id>(
        manager: &Self::Manager<'id>,
        set: &EdgeOfFunc<'id, Self>,
        var: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        let var_level = singleton_level(manager, var);
        subset::<_, _, 0>(manager, rec, set.borrowed(), var.borrowed(), var_level)
    }

    #[inline]
    fn subset1_edge<'id>(
        manager: &Self::Manager<'id>,
        set: &EdgeOfFunc<'id, Self>,
        var: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        let var_level = singleton_level(manager, var);
        subset::<_, _, 1>(manager, rec, set.borrowed(), var.borrowed(), var_level)
    }

    #[inline]
    fn change_edge<'id>(
        manager: &Self::Manager<'id>,
        set: &EdgeOfFunc<'id, Self>,
        var: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        let var_level = singleton_level(manager, var);
        subset::<_, _, -1>(manager, rec, set.borrowed(), var.borrowed(), var_level)
    }

    #[inline]
    fn union_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_union(manager, SequentialRecursor, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn intsec_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_intsec(manager, SequentialRecursor, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn diff_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_diff(manager, SequentialRecursor, lhs.borrowed(), rhs.borrowed())
    }
}

impl<F: Function> BooleanFunction for ZBDDFunction<F>
where
    for<'id> F::Manager<'id>: Manager<Terminal = ZBDDTerminal>
        + HasZBDDOpApplyCache<F::Manager<'id>>
        + HasZBDDCache<<F::Manager<'id> as Manager>::Edge>,
    for<'id> <F::Manager<'id> as Manager>::InnerNode: HasLevel,
{
    fn new_var<'id>(manager: &mut Self::Manager<'id>) -> AllocResult<Self> {
        let hi = manager.get_terminal(ZBDDTerminal::Base).unwrap();
        let lo = manager.get_terminal(ZBDDTerminal::Empty).unwrap();
        let mut edge = manager.add_level(|level| InnerNode::new(level, [hi, lo]))?;

        // Build the chain bottom up. We need to skip the newly created level.
        let mut levels = manager.levels().rev();
        levels.next().unwrap();
        for mut view in levels {
            // only use `oxidd_core::LevelView` here to mitigate confusion of Rust Analyzer
            use oxidd_core::LevelView;

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

    #[inline]
    fn f_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
        manager.get_terminal(ZBDDTerminal::Empty).unwrap()
    }

    #[inline]
    fn t_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
        manager.clone_edge(manager.zbdd_cache().tautology(0))
    }

    #[inline]
    fn not_edge<'id>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_not(manager, SequentialRecursor, edge.borrowed())
    }

    #[inline]
    fn and_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_intsec(manager, SequentialRecursor, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn or_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_union(manager, SequentialRecursor, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn nand_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let and = Self::and_edge(manager, lhs, rhs)?;
        Self::not_edge(manager, &EdgeDropGuard::new(manager, and))
    }
    #[inline]
    fn nor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let or = Self::or_edge(manager, lhs, rhs)?;
        Self::not_edge(manager, &EdgeDropGuard::new(manager, or))
    }
    #[inline]
    fn xor_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_symm_diff(manager, SequentialRecursor, lhs.borrowed(), rhs.borrowed())
    }
    #[inline]
    fn equiv_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let xor = Self::xor_edge(manager, lhs, rhs)?;
        Self::not_edge(manager, &EdgeDropGuard::new(manager, xor))
    }
    #[inline]
    fn imp_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        Self::ite_edge(manager, lhs, rhs, manager.zbdd_cache().tautology(0))
    }
    #[inline]
    fn imp_strict_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_diff(manager, SequentialRecursor, rhs.borrowed(), lhs.borrowed())
    }

    #[inline]
    fn ite_edge<'id>(
        manager: &Self::Manager<'id>,
        f: &EdgeOfFunc<'id, Self>,
        g: &EdgeOfFunc<'id, Self>,
        h: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let rec = SequentialRecursor;
        apply_ite(manager, rec, f.borrowed(), g.borrowed(), h.borrowed())
    }

    #[inline]
    fn sat_count_edge<'id, N: SatCountNumber, S: BuildHasher>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
        vars: LevelNo,
        cache: &mut SatCountCache<N, S>,
    ) -> N {
        fn inner<M, N, S>(manager: &M, e: Borrowed<M::Edge>, cache: &mut SatCountCache<N, S>) -> N
        where
            M: Manager<Terminal = ZBDDTerminal>,
            N: SatCountNumber,
            S: BuildHasher,
        {
            match manager.get_node(&e) {
                Node::Inner(node) => {
                    let node_id = e.node_id();
                    if let Some(n) = cache.map.get(&node_id) {
                        return n.clone();
                    }
                    let (e0, e1) = collect_children(node);
                    let mut n = inner(manager, e0, cache);
                    n += &inner(manager, e1, cache);
                    cache.map.insert(node_id, n.clone());
                    n
                }
                Node::Terminal(t) => N::from(if *t.borrow() == ZBDDTerminal::Empty {
                    0u32
                } else {
                    1u32
                }),
            }
        }

        cache.clear_if_invalid(manager, vars);

        let mut n = inner(manager, edge.borrowed(), cache);
        n >>= manager.num_levels() - vars;
        n
    }

    fn pick_cube_edge<'id, 'a, I>(
        manager: &'a Self::Manager<'id>,
        edge: &'a EdgeOfFunc<'id, Self>,
        order: impl IntoIterator<IntoIter = I>,
        choice: impl FnMut(&Self::Manager<'id>, &EdgeOfFunc<'id, Self>, LevelNo) -> bool,
    ) -> Option<Vec<OptBool>>
    where
        I: ExactSizeIterator<Item = &'a EdgeOfFunc<'id, Self>>,
    {
        #[inline]
        fn inner<M: Manager<Terminal = ZBDDTerminal>>(
            manager: &M,
            edge: Borrowed<M::Edge>,
            cube: &mut [OptBool],
            mut choice: impl FnMut(&M, &M::Edge, LevelNo) -> bool,
        ) where
            M::InnerNode: HasLevel,
        {
            let Node::Inner(node) = manager.get_node(&edge) else {
                return;
            };
            let level = node.level();
            let (hi, lo) = collect_children(node);
            let (val, next_edge) = if hi == lo {
                (OptBool::None, hi)
            } else {
                let c = if manager.get_node(&lo).is_terminal(&ZBDDTerminal::Empty) {
                    true
                } else {
                    choice(manager, &edge, level)
                };
                (OptBool::from(c), if c { hi } else { lo })
            };
            cube[level as usize] = val;
            inner(manager, next_edge, cube, choice);
        }

        let order = order.into_iter();
        debug_assert!(
            order.len() == 0 || order.len() == manager.num_levels() as usize,
            "order must be empty or contain all variables"
        );

        match manager.get_node(edge) {
            Node::Inner(_) => {}
            Node::Terminal(t) => {
                return match *t.borrow() {
                    ZBDDTerminal::Empty => None,
                    ZBDDTerminal::Base => Some(vec![OptBool::False; manager.num_levels() as usize]),
                }
            }
        }

        let mut cube = vec![OptBool::False; manager.num_levels() as usize];
        inner(manager, edge.borrowed(), &mut cube, choice);

        Some(if order.len() == 0 {
            cube
        } else {
            order
                .map(|e| cube[manager.get_node(e).unwrap_inner().level() as usize])
                .collect()
        })
    }

    #[inline]
    fn pick_cube_dd_edge<'id>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
        choice: impl FnMut(&Self::Manager<'id>, &EdgeOfFunc<'id, Self>, LevelNo) -> bool,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        fn inner<M: Manager<Terminal = ZBDDTerminal>>(
            manager: &M,
            edge: Borrowed<M::Edge>,
            mut choice: impl FnMut(&M, &M::Edge, LevelNo) -> bool,
        ) -> AllocResult<M::Edge>
        where
            M::InnerNode: HasLevel,
        {
            let Node::Inner(node) = manager.get_node(&edge) else {
                return Ok(manager.clone_edge(&edge));
            };

            let level = node.level();
            let (hi, lo) = collect_children(node);
            let do_not_care = hi == lo;
            let c = if hi == lo || manager.get_node(&lo).is_terminal(&ZBDDTerminal::Empty) {
                true
            } else {
                choice(manager, &edge, level)
            };

            let sub = inner(manager, if c { hi } else { lo }, choice);
            if !c {
                return sub;
            }

            let hi = EdgeDropGuard::new(manager, sub?);
            let lo = if do_not_care {
                manager.clone_edge(&hi)
            } else {
                manager.get_terminal(ZBDDTerminal::Empty)?
            };
            oxidd_core::LevelView::get_or_insert(
                &mut manager.level(level),
                M::InnerNode::new(level, [hi.into_edge(), lo]),
            )
        }

        inner(manager, edge.borrowed(), choice)
    }

    #[inline]
    fn pick_cube_dd_set_edge<'id>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
        literal_set: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        #[inline] // tail-recursive
        fn set_pop<'a, M: Manager>(
            manager: &'a M,
            edge: Borrowed<'a, M::Edge>,
            until: LevelNo,
        ) -> (Borrowed<'a, M::Edge>, Option<&'a M::InnerNode>)
        where
            M::InnerNode: HasLevel,
        {
            match manager.get_node(&edge) {
                Node::Terminal(_) => (edge, None),
                Node::Inner(node) => match node.level().cmp(&until) {
                    Ordering::Less => set_pop(manager, node.child(0), until),
                    Ordering::Equal => (edge, Some(node)),
                    Ordering::Greater => (edge, None),
                },
            }
        }

        fn inner<M: Manager<Terminal = ZBDDTerminal>>(
            manager: &M,
            edge: Borrowed<M::Edge>,
            literal_set: Borrowed<M::Edge>,
        ) -> AllocResult<M::Edge>
        where
            M::InnerNode: HasLevel,
        {
            let Node::Inner(node) = manager.get_node(&edge) else {
                return Ok(manager.clone_edge(&edge));
            };
            let level = node.level();

            let (literal_set, set_node) = set_pop(manager, literal_set, level);

            let (hi, lo) = collect_children(node);
            let mut do_not_care = false;
            let c = if manager.get_node(&lo).is_terminal(&ZBDDTerminal::Empty) {
                true // enforced (otherwise `cube → function` would not hold)
            } else if let Some(node) = set_node {
                let (shi, slo) = collect_children(node);
                if shi == slo {
                    do_not_care = hi == lo;
                } else {
                    debug_assert!(manager.get_node(&slo).is_terminal(&ZBDDTerminal::Empty));
                }
                true // either don't care or selected
            } else {
                false // selected
            };

            let sub = inner(manager, if c { hi } else { lo }, literal_set);
            if !c {
                return sub;
            }

            let hi = EdgeDropGuard::new(manager, sub?);
            let lo = if do_not_care {
                manager.clone_edge(&hi)
            } else {
                manager.get_terminal(ZBDDTerminal::Empty)?
            };
            oxidd_core::LevelView::get_or_insert(
                &mut manager.level(level),
                M::InnerNode::new(level, [hi.into_edge(), lo]),
            )
        }

        inner(manager, edge.borrowed(), literal_set.borrowed())
    }

    fn eval_edge<'id, 'a>(
        manager: &'a Self::Manager<'id>,
        edge: &'a EdgeOfFunc<'id, Self>,
        args: impl IntoIterator<Item = (Borrowed<'a, EdgeOfFunc<'id, Self>>, bool)>,
    ) -> bool {
        let mut values = BitVec::new();
        values.resize(manager.num_levels() as usize, false);
        for (edge, val) in args {
            let node = manager
                .get_node(&edge)
                .expect_inner("edges in `args` must refer to inner nodes");
            values.set(node.level() as usize, val);
        }

        #[inline] // this function is tail-recursive
        fn inner<M>(manager: &M, edge: Borrowed<M::Edge>, mut values: BitVec) -> Option<BitVec>
        where
            M: Manager<Terminal = ZBDDTerminal>,
            M::InnerNode: HasLevel,
        {
            match manager.get_node(&edge) {
                Node::Inner(node) => {
                    let level = node.level() as usize;
                    let edge = node.child((!values[level]) as usize);
                    values.set(level, false);
                    inner(manager, edge, values)
                }
                Node::Terminal(t) if *t.borrow() == ZBDDTerminal::Base => Some(values),
                Node::Terminal(_) => None,
            }
        }

        if let Some(values) = inner(manager, edge.borrowed(), values) {
            BitSlice::not_any(&values)
        } else {
            false
        }
    }
}

impl<F: Function, T: Tag> DotStyle<T> for ZBDDFunction<F> {}

#[cfg(feature = "multi-threading")]
pub mod mt {
    use oxidd_core::HasWorkers;

    use crate::recursor::mt::ParallelRecursor;

    use super::*;

    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Function, Debug)]
    #[repr(transparent)]
    pub struct ZBDDFunctionMT<F: Function>(F);

    impl<F: Function> From<F> for ZBDDFunctionMT<F> {
        #[inline(always)]
        fn from(value: F) -> Self {
            ZBDDFunctionMT(value)
        }
    }

    impl<F: Function> ZBDDFunctionMT<F>
    where
        for<'id> F::Manager<'id>: HasWorkers,
    {
        /// Convert `self` into the underlying [`Function`]
        #[inline(always)]
        pub fn into_inner(self) -> F {
            self.0
        }
    }

    impl<F: Function> BooleanVecSet for ZBDDFunctionMT<F>
    where
        for<'id> F::Manager<'id>: Manager<Terminal = ZBDDTerminal>
            + super::HasZBDDOpApplyCache<F::Manager<'id>>
            + super::HasZBDDCache<<F::Manager<'id> as Manager>::Edge>
            + HasWorkers,
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
        fn empty_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
            manager.get_terminal(ZBDDTerminal::Empty).unwrap()
        }

        #[inline]
        fn base_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
            manager.get_terminal(ZBDDTerminal::Base).unwrap()
        }

        #[inline]
        fn subset0_edge<'id>(
            manager: &Self::Manager<'id>,
            set: &EdgeOfFunc<'id, Self>,
            var: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (set, var) = (set.borrowed(), var.borrowed());
            let rec = ParallelRecursor::new(manager);
            let var_level = singleton_level(manager, &var);
            subset::<_, _, 0>(manager, rec, set, var, var_level)
        }

        #[inline]
        fn subset1_edge<'id>(
            manager: &Self::Manager<'id>,
            set: &EdgeOfFunc<'id, Self>,
            var: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (set, var) = (set.borrowed(), var.borrowed());
            let rec = ParallelRecursor::new(manager);
            let var_level = singleton_level(manager, &var);
            subset::<_, _, 1>(manager, rec, set, var, var_level)
        }

        #[inline]
        fn change_edge<'id>(
            manager: &Self::Manager<'id>,
            set: &EdgeOfFunc<'id, Self>,
            var: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (set, var) = (set.borrowed(), var.borrowed());
            let rec = ParallelRecursor::new(manager);
            let var_level = singleton_level(manager, &var);
            subset::<_, _, -1>(manager, rec, set, var, var_level)
        }

        #[inline]
        fn union_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            apply_union(manager, ParallelRecursor::new(manager), lhs, rhs)
        }

        #[inline]
        fn intsec_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            apply_intsec(manager, ParallelRecursor::new(manager), lhs, rhs)
        }

        #[inline]
        fn diff_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            apply_diff(manager, ParallelRecursor::new(manager), lhs, rhs)
        }
    }

    impl<F: Function> BooleanFunction for ZBDDFunctionMT<F>
    where
        for<'id> F::Manager<'id>: Manager<Terminal = ZBDDTerminal>
            + super::HasZBDDOpApplyCache<F::Manager<'id>>
            + super::HasZBDDCache<<F::Manager<'id> as Manager>::Edge>
            + HasWorkers,
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
                // only use `oxidd_core::LevelView` here to mitigate confusion of Rust Analyzer
                use oxidd_core::LevelView;

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

        #[inline]
        fn f_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
            manager.get_terminal(ZBDDTerminal::Empty).unwrap()
        }

        #[inline]
        fn t_edge<'id>(manager: &Self::Manager<'id>) -> EdgeOfFunc<'id, Self> {
            manager.clone_edge(manager.zbdd_cache().tautology(0))
        }

        #[inline]
        fn not_edge<'id>(
            manager: &Self::Manager<'id>,
            edge: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let edge = edge.borrowed();
            let rec = ParallelRecursor::new(manager);
            let taut = manager.zbdd_cache().tautology(0);
            apply_diff(manager, rec, taut.borrowed(), edge)
        }

        #[inline]
        fn and_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            apply_intsec(manager, ParallelRecursor::new(manager), lhs, rhs)
        }
        #[inline]
        fn or_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            apply_union(manager, ParallelRecursor::new(manager), lhs, rhs)
        }
        #[inline]
        fn nand_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            let rec = ParallelRecursor::new(manager);
            let and = EdgeDropGuard::new(manager, apply_intsec(manager, rec, lhs, rhs)?);
            apply_not(manager, rec, and.borrowed())
        }
        #[inline]
        fn nor_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            let rec = ParallelRecursor::new(manager);
            let or = EdgeDropGuard::new(manager, apply_union(manager, rec, lhs, rhs)?);
            apply_not(manager, rec, or.borrowed())
        }
        #[inline]
        fn xor_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            apply_symm_diff(manager, ParallelRecursor::new(manager), lhs, rhs)
        }
        #[inline]
        fn equiv_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            let rec = ParallelRecursor::new(manager);
            let xor = EdgeDropGuard::new(manager, apply_symm_diff(manager, rec, lhs, rhs)?);
            apply_not(manager, rec, xor.borrowed())
        }
        #[inline]
        fn imp_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            let rec = ParallelRecursor::new(manager);
            let taut = manager.zbdd_cache().tautology(0);
            apply_ite(manager, rec, lhs, rhs, taut.borrowed())
        }
        #[inline]
        fn imp_strict_edge<'id>(
            manager: &Self::Manager<'id>,
            lhs: &EdgeOfFunc<'id, Self>,
            rhs: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (lhs, rhs) = (lhs.borrowed(), rhs.borrowed());
            apply_diff(manager, ParallelRecursor::new(manager), rhs, lhs)
        }

        #[inline]
        fn ite_edge<'id>(
            manager: &Self::Manager<'id>,
            f: &EdgeOfFunc<'id, Self>,
            g: &EdgeOfFunc<'id, Self>,
            h: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            let (f, g, h) = (f.borrowed(), g.borrowed(), h.borrowed());
            apply_ite(manager, ParallelRecursor::new(manager), f, g, h)
        }

        #[inline]
        fn sat_count_edge<'id, N: SatCountNumber, S: std::hash::BuildHasher>(
            manager: &Self::Manager<'id>,
            edge: &EdgeOfFunc<'id, Self>,
            vars: LevelNo,
            cache: &mut SatCountCache<N, S>,
        ) -> N {
            ZBDDFunction::<F>::sat_count_edge(manager, edge, vars, cache)
        }

        #[inline]
        fn pick_cube_edge<'id, 'a, I>(
            manager: &'a Self::Manager<'id>,
            edge: &'a EdgeOfFunc<'id, Self>,
            order: impl IntoIterator<IntoIter = I>,
            choice: impl FnMut(&Self::Manager<'id>, &EdgeOfFunc<'id, Self>, LevelNo) -> bool,
        ) -> Option<Vec<OptBool>>
        where
            I: ExactSizeIterator<Item = &'a EdgeOfFunc<'id, Self>>,
        {
            ZBDDFunction::<F>::pick_cube_edge(manager, edge, order, choice)
        }
        #[inline]
        fn pick_cube_dd_edge<'id>(
            manager: &Self::Manager<'id>,
            edge: &EdgeOfFunc<'id, Self>,
            choice: impl FnMut(&Self::Manager<'id>, &EdgeOfFunc<'id, Self>, LevelNo) -> bool,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            ZBDDFunction::<F>::pick_cube_dd_edge(manager, edge, choice)
        }
        #[inline]
        fn pick_cube_dd_set_edge<'id>(
            manager: &Self::Manager<'id>,
            edge: &EdgeOfFunc<'id, Self>,
            literal_set: &EdgeOfFunc<'id, Self>,
        ) -> AllocResult<EdgeOfFunc<'id, Self>> {
            ZBDDFunction::<F>::pick_cube_dd_set_edge(manager, edge, literal_set)
        }

        #[inline]
        fn eval_edge<'id, 'a>(
            manager: &'a Self::Manager<'id>,
            edge: &'a EdgeOfFunc<'id, Self>,
            args: impl IntoIterator<Item = (Borrowed<'a, EdgeOfFunc<'id, Self>>, bool)>,
        ) -> bool {
            ZBDDFunction::<F>::eval_edge(manager, edge, args)
        }
    }

    impl<F: Function, T: Tag> DotStyle<T> for ZBDDFunctionMT<F> {}
}
