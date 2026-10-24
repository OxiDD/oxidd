use oxidd_core::restricted::*;
use oxidd_core::util::{AllocResult, Ref};
use oxidd_core::{HasLevel, InnerNode, Manager};

use super::TDDRules;

// spell-checker:ignore unreduce
#[inline]
fn unreduce<'a, M: Manager>(
    edge: Ref<'a, M::Edge>,
    node: Option<&'a M::InnerNode>,
) -> [Ref<'a, M::Edge>; 3] {
    match node {
        Some(n) => [n.child(0), n.child(1), n.child(2)],
        None => [edge, edge, edge],
    }
}

impl<M: Manager<InnerNodeValue = ()>> SafeLevelSwapper<M> for TDDRules
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

        let t = main_node.child(0);
        let u = main_node.child(1);
        let e = main_node.child(2);
        let t_node = upper.node_checked(manager, t);
        let u_node = upper.node_checked(manager, u);
        let e_node = upper.node_checked(manager, e);
        let t_node_at_upper = t_node.is_some();
        let u_node_at_upper = u_node.is_some();
        let e_node_at_upper = e_node.is_some();

        if !t_node_at_upper && !u_node_at_upper && !e_node_at_upper {
            // All children are below the lower level, we just move the node
            // to the lower level.
            return Ok(main.insert_at_lower_level(manager, lower));
        }

        let mut reduce = |t: Ref<M::Edge>, u: Ref<M::Edge>, e: Ref<M::Edge>| {
            if t == u && u == e {
                Ok(manager.clone_edge(t))
            } else {
                lower.get_or_insert(manager, &[t, u, e], &())
            }
        };

        let [tt, tu, te] = unreduce::<M>(t, t_node);
        let [ut, uu, ue] = unreduce::<M>(u, u_node);
        let [et, eu, ee] = unreduce::<M>(e, e_node);
        let new_t = EdgeDropGuard::new(manager, reduce(tt, ut, et)?);
        let new_u = EdgeDropGuard::new(manager, reduce(tu, uu, eu)?);
        let new_e = EdgeDropGuard::new(manager, reduce(te, ue, ee)?);

        let mut try_remove = |edge, at_upper| {
            if at_upper {
                upper.try_remove(edge);
            } else {
                manager.drop_edge(edge);
            }
        };

        try_remove(main.set_child(0, new_t.into_edge()), t_node_at_upper);
        try_remove(main.set_child(1, new_u.into_edge()), u_node_at_upper);
        try_remove(main.set_child(2, new_e.into_edge()), e_node_at_upper);

        Ok(main.insert_at_upper_level(manager, upper))
    }
}
