"""
The protocols classes declared in this module allow abstracting away the
concrete decision diagram kind in a type-safe fashion.
"""

__all__ = [
    "Manager",
    "BooleanFunctionManager",
    "Function",
    "FunctionSubst",
    "BooleanFunction",
    "BooleanFunctionQuant",
    "HasLevel",
]

import collections.abc
from abc import abstractmethod
from typing import Generic, Optional, Protocol, TypeVar

from typing_extensions import Self

from .util import Assignment, BooleanOperator


class Function(Protocol):
    """Function represented as decision diagram

    A function is the combination of a reference to a :class:`Manager` and a
    (possibly tagged) edge pointing to a node. Obtaining the manager reference
    is possible via the :attr:`manager` property.
    """

    @property
    @abstractmethod
    def manager(self) -> "Manager[Self]":
        """The associated manager"""
        raise NotImplementedError

    @abstractmethod
    def node_count(self) -> int:
        """Get the number of descendant nodes

        The returned number includes the root node itself and terminal nodes.

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    def __lt__(self, other: Self) -> bool:
        """Check if ``self`` is less than ``other`` according to an arbitrary
        total order

        The following restrictions apply: Assuming two functions ``f``, ``g``
        with ``f < g``, then if either ``f`` or ``g`` is deleted and recreated
        later on, ``f < g`` does not necessarily hold anymore. Moreover, assume
        `f < g` and two structurally equivalent functions ``f2``, ``g2`` in
        different managers (i.e., ``f ≅ f2`` but ``f != f2``, and ``g ≅ g2`` but
        ``g != g2``), then ``f2 < g2`` does not necessarily hold.
        """
        raise NotImplementedError

    def __gt__(self, other: Self) -> bool:
        """Same as ``other < self``"""
        raise NotImplementedError

    def __le__(self, other: Self) -> bool:
        """Same as ``not self > other``"""
        raise NotImplementedError

    def __ge__(self, other: Self) -> bool:
        """Same as ``not self < other``"""
        raise NotImplementedError


S = TypeVar("S")


class FunctionSubst(Function, Generic[S], Protocol):
    """Substitution extension for :class:`Function`"""

    @classmethod
    @abstractmethod
    def make_substitution(cls, pairs: collections.abc.Iterable[tuple[Self, Self]]) -> S:
        """Create a new substitution object from a collection of pairs
        ``(var, replacement)``

        The intent behind substitution objects is to optimize the case where the
        same substitution is applied multiple times. We would like to re-use
        apply cache entries across these operations, and therefore, we need a
        compact identifier for the substitution (provided by the returned
        substitution object).

        All variables of must be distinct. Furthermore, variables must be
        handles for the respective decision diagram levels, e.g., the respective
        Boolean function for B(C)DDs, and a singleton set for ZBDDs. The order
        of the pairs is irrelevant.
        """
        raise NotImplementedError

    @abstractmethod
    def substitute(self, substitution: S) -> Self:
        """Substitute variables in ``self`` according to ``substitution``, which
        can be created using :meth:`make_substitution`

        The substitution is performed in a parallel fashion, e.g.:
        ``(¬x ∧ ¬y)[x ↦ ¬x ∧ ¬y, y ↦ ⊥] = ¬(¬x ∧ ¬y) ∧ ¬⊥ = x ∨ y``

        Acquires the manager's lock for shared access.

        ``self`` and all functions in ``substitution`` must belong to the same
        manager.
        """
        raise NotImplementedError


class HasLevel(Function, Protocol):
    """Function whose decision diagram node is associated with a level"""

    @abstractmethod
    def level(self) -> Optional[int]:
        """Get the level of the underlying node (``None`` for terminals)

        Locking behavior: acquires the manager's lock for shared access.

        Runtime complexity: O(1)
        """
        raise NotImplementedError


class BooleanFunction(Function, Protocol):
    """Boolean function represented as decision diagram"""

    @abstractmethod
    def cofactors(self) -> tuple[Self, Self]:
        r"""Get the cofactors ``(f_true, f_false)`` of ``self``

        Let f(x₀, …, xₙ) be represented by ``self``, where x₀ is (currently) the
        top-most variable. Then f\ :sub:`true`\ (x₁, …, xₙ) = f(⊤, x₁, …, xₙ)
        and f\ :sub:`false`\ (x₁, …, xₙ) = f(⊥, x₁, …, xₙ).

        Note that the domain of f is 𝔹\ :sup:`n+1` while the domain of
        f\ :sub:`true` and f\ :sub:`false` is 𝔹\ :sup:`n`. This is irrelevant in
        case of BDDs and BCDDs, but not for ZBDDs: For instance, g(x₀) = x₀ and
        g'(x₀, x₁) = x₀ have the same representation as BDDs or BCDDs, but
        different representations as ZBDDs.

        Structurally, the cofactors are simply the children in case of BDDs and
        ZBDDs. (For BCDDs, the edge tags are adjusted accordingly.) On these
        representations, runtime is thus in O(1).

        Returns ``None`` iff ``self`` references a terminal node. If you only
        need ``f_true`` or ``f_false``, :meth:`cofactor_true` or
        :meth:`cofactor_false` are slightly more efficient.

        Locking behavior: acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def cofactor_true(self) -> Self:
        """Get the cofactor ``f_true`` of ``self``

        This method is slightly more efficient than :meth:`Self::cofactors` in
        case ``f_false`` is not needed.

        For a more detailed description, see :meth:`cofactors`.

        Returns ``None`` iff ``self`` references a terminal node.

        Locking behavior: acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def cofactor_false(self) -> Self:
        """Get the cofactor ``f_true`` of ``self``

        This method is slightly more efficient than :meth:`Self::cofactors` in
        case ``f_true`` is not needed.

        For a more detailed description, see :meth:`cofactors`.

        Returns ``None`` iff ``self`` references a terminal node.

        Locking behavior: acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def __invert__(self) -> Self:
        """Compute the negation ``¬self``

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def __and__(self, rhs: Self) -> Self:
        """Compute the conjunction ``self ∧ rhs``

        ``self`` and ``rhs`` must belong to the same manager.

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def __or__(self, rhs: Self) -> Self:
        """Compute the disjunction ``self ∨ rhs``

        ``self`` and ``rhs`` must belong to the same manager.

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def __xor__(self, rhs: Self) -> Self:
        """Compute the exclusive disjunction ``self ⊕ rhs``

        ``self`` and ``rhs`` must belong to the same manager.

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def nand(self, rhs: Self) -> Self:
        """Compute the negated conjunction ``self ⊼ rhs``

        ``self`` and ``rhs`` must belong to the same manager.

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def nor(self, rhs: Self) -> Self:
        """Compute the negated disjunction ``self ⊽ rhs``

        ``self`` and ``rhs`` must belong to the same manager.

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def equiv(self, rhs: Self) -> Self:
        """Compute the equivalence ``self ↔ rhs``

        ``self`` and ``rhs`` must belong to the same manager.

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def imp(self, rhs: Self) -> Self:
        """Compute the implication ``self → rhs`` (or ``f ≤ g``)

        ``self`` and ``rhs`` must belong to the same manager.

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def imp_strict(self, rhs: Self) -> Self:
        """Compute the strict implication ``self < rhs``

        ``self`` and ``rhs`` must belong to the same manager.

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def ite(self, t: Self, e: Self) -> Self:
        """Compute the BDD for the conditional ``t if self else e``

        ``self``, ``t``, and ``e`` must belong to the same manager.

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def satisfiable(self) -> bool:
        """Check for satisfiability (i.e., at least one satisfying assignment)

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def valid(self) -> bool:
        """Check for validity (i.e., that all assignments are satisfying)

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def sat_count_float(self, vars: int) -> float:
        """Count the number of satisfying assignments

        This method assumes that the function's domain of has `vars` many
        variables.

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def pick_cube(self) -> Optional[Assignment]:
        """Pick a satisfying assignment

        Returns ``None`` iff ``self`` is unsatisfiable.

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def pick_cube_symbolic(self) -> Self:
        """Pick a satisfying assignment, represented as decision diagram

        Returns ``⊥`` iff ``self`` is unsatisfiable.

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def pick_cube_symbolic_set(self, literal_set: Self) -> Self:
        """Pick a satisfying assignment, represented as BDD, using the
        literals in ``literal_set`` if there is a choice

        ``literal_set`` is represented as a conjunction of literals. Whenever
        there is a choice for a variable, it will be set to true if the variable
        has a positive occurrence in ``literal_set``, and set to false if it
        occurs negated in ``literal_set``. If the variable does not occur in
        ``literal_set``, then it will be left as don't care if possible,
        otherwise an arbitrary (not necessarily random) choice will be
        performed.

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def eval(self, args: collections.abc.Collection[tuple[Self, bool]]) -> bool:
        """Evaluate this Boolean function with arguments ``args``

        ``args`` determines the valuation for all variables. Missing values are
        assumed to be false. However, note that the arguments may also determine
        the domain, e.g., in case of ZBDDs. If values are specified multiple
        times, the last one counts. Panics if any function in `args` refers to a
        terminal node.

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError


class BooleanFunctionQuant(BooleanFunction, Protocol):
    """Quantification extension for :class:`BooleanFunction`"""

    @abstractmethod
    def forall(self, vars: Self) -> Self:
        """Compute the universal quantification over ``vars``

        ``vars`` is a set of variables, which in turn is just the conjunction of
        the variables. This operation removes all occurrences of the variables
        by universal quantification. Universal quantification ∀x. f(…, x, …)
        of a boolean function f(…, x, …) over a single variable x is
        f(…, 0, …) ∧ f(…, 1, …).

        ``self`` and ``vars`` must belong to the same manager.

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def exist(self, vars: Self) -> Self:
        """Compute the existential quantification over ``vars``

        ``vars`` is a set of variables, which in turn is just the conjunction of
        the variables. This operation removes all occurrences of the variables
        by existential quantification. Existential quantification ∃x. f(…, x, …)
        of a boolean function f(…, x, …) over a single variable x is
        f(…, 0, …) ∨ f(…, 1, …).

        ``self`` and ``vars`` must belong to the same manager.

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def unique(self, vars: Self) -> Self:
        """Compute the unique quantification over ``vars``

        ``vars`` is a set of variables, which in turn is just the conjunction of
        the variables. This operation removes all occurrences of the variables
        by unique quantification. Unique quantification ∃!x. f(…, x, …) of a
        boolean function f(…, x, …) over a single variable x is
        f(…, 0, …) ⊕ f(…, 1, …).

        Unique quantification is also known as the
        `Boolean difference <https://en.wikipedia.org/wiki/Boole%27s_expansion_theorem#Operations_with_cofactors>`_ or
        `Boolean derivative <https://en.wikipedia.org/wiki/Boolean_differential_calculus>`_.

        ``self`` and ``vars`` must belong to the same manager.

        Acquires the manager's lock for shared access.
        """  # noqa: E501
        raise NotImplementedError

    @abstractmethod
    def apply_forall(self, op: BooleanOperator, rhs: Self, vars: Self) -> Self:
        """Combined application of ``op`` and :meth:`forall()`:
        ``∀ vars. self <op> rhs``

        ``self``, ``rhs``, and ``vars`` must belong to the same manager.

        Acquires the manager's lock for shared access."""
        raise NotImplementedError

    @abstractmethod
    def apply_exist(self, op: BooleanOperator, rhs: Self, vars: Self) -> Self:
        """Combined application of ``op`` and :meth:`exist()`:
        ``∃ vars. self <op> rhs``

        ``self``, ``rhs``, and ``vars`` must belong to the same manager.

        Acquires the manager's lock for shared access."""
        raise NotImplementedError

    @abstractmethod
    def apply_unique(self, op: BooleanOperator, rhs: Self, vars: Self) -> Self:
        """Combined application of ``op`` and :meth:`unique()`:
        ``∃! vars. self <op> rhs``

        ``self``, ``rhs``, and ``vars`` must belong to the same manager.

        Acquires the manager's lock for shared access."""
        raise NotImplementedError


F = TypeVar("F", bound=Function, covariant=True)


class Manager(Generic[F], Protocol):
    """Manager storing nodes and ensuring their uniqueness

    A manager is the data structure responsible for storing nodes and ensuring
    their uniqueness. It also defines the variable order.

    Implementations supporting concurrency have an internal read/write lock.
    Many operations acquire this lock for reading (shared) or writing
    (exclusive).
    """

    @abstractmethod
    def num_inner_nodes(self) -> int:
        """Get the number of inner nodes

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError


BF = TypeVar("BF", bound=BooleanFunction, covariant=True)


class BooleanFunctionManager(Manager[BF], Protocol):
    """Manager whose nodes represent Boolean functions"""

    @abstractmethod
    def new_var(self) -> BF:
        """Get a fresh variable, i.e., a function that is true if and only if
        the variable is true. This adds a new level to a decision diagram.

        Acquires the manager's lock for exclusive access.
        """
        raise NotImplementedError

    @abstractmethod
    def true(self) -> BF:
        """Get the constant true Boolean function ``⊤``

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def false(self) -> BF:
        """Get the constant false Boolean function ``⊥``

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError
