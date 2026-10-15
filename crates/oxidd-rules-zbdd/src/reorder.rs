use oxidd_core::restricted::*;
use oxidd_core::util::{AllocResult, Ref};
use oxidd_core::{HasLevel, InnerNode, Manager};

use crate::{HI, LO, ZBDDRules, ZBDDTerminal};

impl<M: Manager<Terminal = ZBDDTerminal>> SafeLevelSwapper<M> for ZBDDRules
where
    M::InnerNode: HasLevel,
{
    #[inline]
    fn swap_fixup<'a, 'v>(
        manager: RestrictedManager<'a, M>,
        mut main: MutNode<'a, M>,
        upper: &mut UpperLevelView<'v, M>,
        lower: &mut LowerLevelView<'a, 'v, M>,
    ) -> AllocResult<Inserted> {
        let main_node = main.node();
        debug_assert_eq!(main_node.level(), lower.previous_level_no());

        let hi = main_node.child(HI);
        let lo = main_node.child(LO);
        let hi_node = upper.node_checked(manager, hi);
        let lo_node = upper.node_checked(manager, lo);
        let hi_node_at_upper = hi_node.is_some();
        let lo_node_at_upper = lo_node.is_some();

        if !hi_node_at_upper && !lo_node_at_upper {
            // All children are below the lower level, we just move the node
            // to the lower level.
            return Ok(main.insert_at_lower_level(manager, lower));
        }

        let empty = EdgeDropGuard::new(manager, manager.get_terminal(ZBDDTerminal::Empty)?);
        let mut reduce = |hi: Ref<M::Edge>, lo: Ref<M::Edge>| {
            if hi == empty.borrowed() {
                Ok(manager.clone_edge(lo))
            } else {
                lower.get_or_insert(manager, &[hi, lo])
            }
        };

        let (hi_hi, hi_lo) = match hi_node {
            Some(n) => (n.child(HI), n.child(LO)),
            None => (empty.borrowed(), hi),
        };
        let (lo_hi, lo_lo) = match lo_node {
            Some(n) => (n.child(HI), n.child(LO)),
            None => (empty.borrowed(), lo),
        };
        let new_hi = EdgeDropGuard::new(manager, reduce(hi_hi, lo_hi)?);
        let new_lo = EdgeDropGuard::new(manager, reduce(hi_lo, lo_lo)?);

        // If `tt` and `et` were pointing to ∅, then `t` and `e` would be
        // referring to nodes below `lower`.
        debug_assert!(*new_hi != *empty);

        let mut try_remove = |edge, at_upper| {
            if at_upper {
                upper.try_remove(edge);
            } else {
                manager.drop_edge(edge);
            }
        };

        try_remove(main.set_child(HI, new_hi.into_edge()), hi_node_at_upper);
        try_remove(main.set_child(LO, new_lo.into_edge()), lo_node_at_upper);

        Ok(main.insert_at_upper_level(manager, upper))
    }
}
