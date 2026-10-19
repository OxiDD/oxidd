use oxidd_core::restricted::*;
use oxidd_core::util::{AllocResult, Ref};
use oxidd_core::{HasLevel, InnerNode, Manager};

use super::{BCDDRules, BCDDTerminal, EdgeTag};

impl<M> SafeLevelSwapper<M> for BCDDRules
where
    M: Manager<Terminal = BCDDTerminal, EdgeTag = EdgeTag, InnerNodeValue = ()>,
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

        let t = main_node.child(0);
        let e = main_node.child(1);
        let t_node = upper.node_checked(manager, t);
        let e_node = upper.node_checked(manager, e);
        let t_node_at_upper = t_node.is_some();
        let e_node_at_upper = e_node.is_some();

        if !t_node_at_upper && !e_node_at_upper {
            // All children are below the lower level, we just move the node
            // to the lower level.
            return Ok(main.insert_at_lower_level(manager, lower));
        }

        let mut reduce = |t: Ref<M::Edge>, e: Ref<M::Edge>| {
            if t == e {
                Ok(manager.clone_edge(t))
            } else {
                lower.get_or_insert(manager, &[t, e], &())
            }
        };

        let (tt, te_untagged, te_tag) = match t_node {
            Some(n) => {
                let te = n.child(1);
                (n.child(0), te.with_tag(EdgeTag::None), te.tag())
            }
            None => (t, t, EdgeTag::None),
        };
        let (et, ee, ee_neg) = match e_node {
            Some(n) if e.tag() == EdgeTag::None => (n.child(0), n.child(1), te_tag),
            Some(n) /* if complemented */ => (n.child(0).with_tag(EdgeTag::Complemented), n.child(1), !te_tag),
            None => (e, e, te_tag),
        };
        let new_t = EdgeDropGuard::new(manager, reduce(tt, et)?);
        let new_e = EdgeDropGuard::new(
            manager,
            reduce(te_untagged, ee.with_tag(ee.tag() ^ ee_neg))?.with_tag(te_tag),
        );

        let mut try_remove = |edge, at_upper| {
            if at_upper {
                upper.try_remove(edge);
            } else {
                manager.drop_edge(edge);
            }
        };

        try_remove(main.set_child(0, new_t.into_edge()), t_node_at_upper);
        try_remove(main.set_child(1, new_e.into_edge()), e_node_at_upper);

        Ok(main.insert_at_upper_level(manager, upper))
    }
}
