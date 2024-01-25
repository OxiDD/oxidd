//! Binary decision diagrams with complemented edges

use std::convert::Infallible;
use std::fmt;
use std::hash::Hash;
use std::iter::FusedIterator;
use std::marker::PhantomData;

use oxidd_core::util::AllocResult;
use oxidd_core::util::Borrowed;
use oxidd_core::DiagramRules;
use oxidd_core::Edge;
use oxidd_core::HasApplyCache;
use oxidd_core::InnerNode;
use oxidd_core::LevelNo;
use oxidd_core::Manager;
use oxidd_core::Node;
use oxidd_core::ReducedOrNew;
use oxidd_derive::Countable;
use oxidd_dump::dddmp::AsciiDisplay;

use crate::stat;

// spell-checker:ignore fnode,gnode

#[cfg(feature = "multi-threading")]
mod apply_rec_mt;
mod apply_rec_st;

// --- Edge Tag ----------------------------------------------------------------

/// Edge tag in complement edge BDDs
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug, Countable)]
#[repr(u8)]
pub enum EdgeTag {
    /// The edge's semantics is the semantics of the referenced node
    #[default]
    None,
    /// The edge's semantics is the negation of the referenced node
    Complemented,
}

impl std::ops::Not for EdgeTag {
    type Output = EdgeTag;

    #[inline]
    fn not(self) -> EdgeTag {
        match self {
            EdgeTag::None => EdgeTag::Complemented,
            EdgeTag::Complemented => EdgeTag::None,
        }
    }
}

#[inline]
#[must_use]
fn not_owned<E: Edge<Tag = EdgeTag>>(e: E) -> E {
    let tag = e.tag();
    e.with_tag_owned(!tag)
}

#[inline]
#[must_use]
fn not<E: Edge<Tag = EdgeTag>>(e: &E) -> Borrowed<E> {
    let tag = e.tag();
    e.with_tag(!tag)
}

// --- Reduction Rules ---------------------------------------------------------

/// [`DiagramRules`] for complement edge binary decision diagrams
pub struct CBDDRules;

impl<E: Edge<Tag = EdgeTag>, N: InnerNode<E>> DiagramRules<E, N, CBDDTerminal> for CBDDRules {
    type Cofactors<'a> = Cofactors<'a, E, N::ChildrenIter<'a>> where N: 'a, E: 'a;

    #[inline]
    #[must_use]
    fn reduce<M: Manager<Edge = E, InnerNode = N>>(
        manager: &M,
        level: LevelNo,
        children: impl IntoIterator<Item = E>,
    ) -> ReducedOrNew<E, N> {
        let mut it = children.into_iter();
        let t = it.next().unwrap();
        let e = it.next().unwrap();
        debug_assert!(it.next().is_none());

        if t == e {
            manager.drop_edge(e);
            return ReducedOrNew::Reduced(t);
        }

        let tt = t.tag();
        let et = e.tag();
        if tt == EdgeTag::Complemented {
            let node = N::new(
                level,
                [t.with_tag_owned(EdgeTag::None), e.with_tag_owned(!et)],
            );
            ReducedOrNew::New(node, EdgeTag::Complemented)
        } else {
            let node = N::new(level, [t, e]);
            ReducedOrNew::New(node, EdgeTag::None)
        }
    }

    #[inline]
    #[must_use]
    fn cofactors(tag: E::Tag, node: &N) -> Self::Cofactors<'_> {
        Cofactors {
            it: node.children(),
            tag,
            phantom: PhantomData,
        }
    }
}

/// Iterator over the cofactors of a node in a complement edge BDD
pub struct Cofactors<'a, E, I> {
    it: I,
    tag: EdgeTag,
    phantom: PhantomData<Borrowed<'a, E>>,
}

impl<'a, E: Edge<Tag = EdgeTag> + 'a, I: Iterator<Item = Borrowed<'a, E>>> Iterator
    for Cofactors<'a, E, I>
{
    type Item = Borrowed<'a, E>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match (self.it.next(), self.tag) {
            (Some(e), EdgeTag::None) => Some(e),
            (Some(e), EdgeTag::Complemented) => {
                let tag = !e.tag();
                Some(Borrowed::edge_with_tag(e, tag))
            }
            (None, _) => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, E: Edge<Tag = EdgeTag>, I: FusedIterator<Item = Borrowed<'a, E>>> FusedIterator
    for Cofactors<'a, E, I>
{
}

impl<'a, E: Edge<Tag = EdgeTag>, I: ExactSizeIterator<Item = Borrowed<'a, E>>> ExactSizeIterator
    for Cofactors<'a, E, I>
{
    #[inline]
    fn len(&self) -> usize {
        self.it.len()
    }
}

/// Collect the two cofactors of a binary node
#[inline]
#[must_use]
fn collect_cofactors<E: Edge<Tag = EdgeTag>, N: InnerNode<E>>(
    tag: EdgeTag,
    node: &N,
) -> (Borrowed<E>, Borrowed<E>) {
    debug_assert_eq!(N::ARITY, 2);
    let mut it = CBDDRules::cofactors(tag, node);
    let f0 = it.next().unwrap();
    let f1 = it.next().unwrap();
    debug_assert!(it.next().is_none());
    (f0, f1)
}

#[inline(always)]
fn reduce<M>(
    manager: &M,
    level: LevelNo,
    t: M::Edge,
    e: M::Edge,
    _op: CBDDOp,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = CBDDTerminal, EdgeTag = EdgeTag>,
{
    let tmp = <CBDDRules as DiagramRules<_, _, _>>::reduce(manager, level, [t, e]);
    if let ReducedOrNew::Reduced(..) = &tmp {
        stat!(reduced _op);
    }
    tmp.then_insert(manager, level)
}

// --- Terminal Type -----------------------------------------------------------

/// Terminal nodes in complement edge binary decision diagrams
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Countable, Debug)]
pub struct CBDDTerminal;

impl std::str::FromStr for CBDDTerminal {
    type Err = Infallible;

    fn from_str(_s: &str) -> Result<Self, Self::Err> {
        Ok(CBDDTerminal)
    }
}

impl AsciiDisplay for CBDDTerminal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.write_str("T")
    }
}

impl fmt::Display for CBDDTerminal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("⊤")
    }
}

#[inline]
#[must_use]
fn get_terminal<M: Manager<EdgeTag = EdgeTag, Terminal = CBDDTerminal>>(
    manager: &M,
    val: bool,
) -> M::Edge {
    let t = manager.get_terminal(CBDDTerminal).unwrap();
    if val {
        t
    } else {
        t.with_tag_owned(EdgeTag::Complemented)
    }
}

/// Terminal case for 'and'
#[inline]
#[must_use]
fn terminal_and<'a, M>(
    manager: &'a M,
    f: &'a M::Edge,
    g: &'a M::Edge,
) -> NodesOrDone<'a, M::Edge, M::InnerNode>
where
    M: Manager<EdgeTag = EdgeTag, Terminal = CBDDTerminal>,
{
    use EdgeTag::*;
    use Node::*;
    use NodesOrDone::*;

    let ft = f.tag();
    let gt = g.tag();
    let fu = f.with_tag(None);
    let gu = g.with_tag(None);
    if *fu == *gu {
        if ft == gt {
            return Done(manager.clone_edge(g));
        }
        return Done(get_terminal(manager, false));
    }

    let (h, tag) = match (manager.get_node(f), manager.get_node(g)) {
        (Inner(fnode), Inner(gnode)) => return Nodes(fnode, gnode),
        (Inner(_), Terminal(_)) => (f, gt),
        (Terminal(_), Inner(_)) => (g, ft),
        (Terminal(_), Terminal(_)) => return Done(get_terminal(manager, ft == None && gt == None)),
    };
    Done(if tag == Complemented {
        get_terminal(manager, false)
    } else {
        manager.clone_edge(h)
    })
}

/// Terminal case for 'xor'
#[inline]
#[must_use]
fn terminal_xor<'a, M>(
    manager: &'a M,
    f: &'a M::Edge,
    g: &'a M::Edge,
) -> NodesOrDone<'a, M::Edge, M::InnerNode>
where
    M: Manager<EdgeTag = EdgeTag, Terminal = CBDDTerminal>,
{
    use EdgeTag::*;
    use Node::*;
    use NodesOrDone::*;

    let ft = f.tag();
    let gt = g.tag();
    let fu = f.with_tag(None);
    let gu = g.with_tag(None);
    if *fu == *gu {
        return Done(get_terminal(manager, !(ft == gt)));
    }
    let (h, tag) = match (manager.get_node(f), manager.get_node(g)) {
        (Inner(fnode), Inner(gnode)) => return Nodes(fnode, gnode),
        (Inner(_), Terminal(_)) => (f, gt),
        (Terminal(_), Inner(_)) => (g, ft),
        (Terminal(_), Terminal(_)) => return Done(get_terminal(manager, !(ft == gt))),
    };
    let h = manager.clone_edge(h);
    if tag == Complemented {
        Done(h)
    } else {
        Done(not_owned(h))
    }
}

// --- Operations & Apply Implementation ---------------------------------------

/// Native operators of this BDD implementation
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
#[allow(missing_docs)]
pub enum CBDDOp {
    And,
    Xor,

    /// If-then-else
    Ite,

    /// Forall quantification
    Forall,
    /// Existential quantification
    Exist,
    /// Unique quantification
    Unique,
}

enum NodesOrDone<'a, E, N> {
    Nodes(&'a N, &'a N),
    Done(E),
}

#[cfg(feature = "statistics")]
static STAT_COUNTERS: [crate::StatCounters; 6] = [crate::StatCounters::INIT; 6];

#[cfg(feature = "statistics")]
/// Print statistics to stderr
pub fn print_stats() {
    eprintln!("[oxidd_rules_bdd::complement_edge]");
    // FIXME: we should auto generate the labels
    crate::StatCounters::print(
        &STAT_COUNTERS,
        &["And", "Xor", "Ite", "Forall", "Exist", "Unique"],
    );
}

// --- Function Interface ------------------------------------------------------

/// Workaround for https://github.com/rust-lang/rust/issues/49601
trait HasCBDDOpApplyCache<M: Manager>: HasApplyCache<M, Operator = CBDDOp> {}
impl<M: Manager + HasApplyCache<M, Operator = CBDDOp>> HasCBDDOpApplyCache<M> for M {}

#[cfg(feature = "multi-threading")]
pub use apply_rec_mt::CBDDFunctionMT;
pub use apply_rec_st::CBDDFunction;
