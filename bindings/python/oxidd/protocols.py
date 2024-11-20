"""Protocols that allow abstracting away the concrete DD kind in a type-safe fashion."""

from __future__ import annotations

__all__ = [
    "Manager",
    "BooleanFunctionManager",
    "Function",
    "FunctionSubst",
    "BooleanFunction",
    "BooleanFunctionQuant",
    "HasLevel",
]

from abc import abstractmethod
from collections.abc import Iterable
from os import PathLike
from typing import Generic, Protocol, TypeVar

from typing_extensions import Self

from .util import BooleanOperator


class Function(Protocol):
    """Function represented as decision diagram.

    A function is the combination of a reference to a :class:`Manager` and a
    (possibly tagged) edge pointing to a node. Obtaining the manager reference
    is possible via the :attr:`manager` property.
    """

    @property
    @abstractmethod
    def manager(self, /) -> Manager[Self]:
        """The associated manager."""
        raise NotImplementedError

    @abstractmethod
    def node_count(self, /) -> int:
        """Get the number of descendant nodes.

        Locking behavior: acquires the manager's lock for shared access.

        Returns:
            The count of descendant nodes including the node referenced by
            ``self`` and terminal nodes.
        """
        raise NotImplementedError

    def __lt__(self, other: Self, /) -> bool:
        """Check if ``self < other`` according to an arbitrary total order.

        The following restrictions apply: Assuming two functions ``f``, ``g``
        with ``f < g``, then if either ``f`` or ``g`` is deleted and recreated
        later on, ``f < g`` does not necessarily hold anymore. Moreover, assume
        `f < g` and two structurally equivalent functions ``f2``, ``g2`` in
        different managers (i.e., ``f ≅ f2`` but ``f != f2``, and ``g ≅ g2`` but
        ``g != g2``), then ``f2 < g2`` does not necessarily hold.
        """
        raise NotImplementedError

    def __gt__(self, other: Self, /) -> bool:
        """Same as ``other < self``."""
        raise NotImplementedError

    def __le__(self, other: Self, /) -> bool:
        """Same as ``not self > other``."""
        raise NotImplementedError

    def __ge__(self, other: Self, /) -> bool:
        """Same as ``not self < other``."""
        raise NotImplementedError


S = TypeVar("S")


class FunctionSubst(Function, Generic[S], Protocol):
    """Substitution extension for :class:`Function`."""

    @classmethod
    @abstractmethod
    def make_substitution(cls, pairs: Iterable[tuple[Self, Self]], /) -> S:
        """Create a new substitution object from pairs ``(var, replacement)``.

        The intent behind substitution objects is to optimize the case where the
        same substitution is applied multiple times. We would like to re-use
        apply cache entries across these operations, and therefore, we need a
        compact identifier for the substitution. This identifier is provided by
        the returned substitution object.

        Args:
            pairs: ``(variable, replacement)`` pairs, where all variables are
                distinct. Furthermore, variables must be handles for the
                respective decision diagram levels, i.e., the Boolean function
                representing the variable for B(C)DDs, and a singleton set for
                ZBDDs. The order of the pairs is irrelevant.

        Returns:
            The substitution to be used with :meth:`substitute`
        """
        raise NotImplementedError

    @abstractmethod
    def substitute(self, substitution: S, /) -> Self:
        """Substitute variables in ``self`` according to ``substitution``.

        The substitution is performed in a parallel fashion, e.g.:
        ``(¬x ∧ ¬y)[x ↦ ¬x ∧ ¬y, y ↦ ⊥] = ¬(¬x ∧ ¬y) ∧ ¬⊥ = x ∨ y``

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            substitution: A substitution object created using
                :meth:`make_substitution`. All contained DD functions must
                belong to the same manager as ``self``.
        """
        raise NotImplementedError


class HasLevel(Function, Protocol):
    """Function whose decision diagram node is associated with a level."""

    @abstractmethod
    def level(self, /) -> int | None:
        """Get the level of the underlying node.

        Locking behavior: acquires the manager's lock for shared access.

        Time complexity: O(1)

        Returns:
            The level number or ``None`` for terminals
        """
        raise NotImplementedError


class BooleanFunction(Function, Protocol):
    """Boolean function represented as decision diagram."""

    @abstractmethod
    def cofactors(self, /) -> tuple[Self, Self] | None:
        r"""Get the cofactors ``(f_true, f_false)`` of ``self``.

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
        representations, the running time is thus in O(1).

        Locking behavior: acquires the manager's lock for shared access.

        Returns:
            The cofactors ``(f_true, f_false)``, or ``None`` if ``self``
            references a terminal node

        See Also:
            :meth:`cofactor_true`, :meth:`cofactor_false` if you only need one
            of the cofactors
        """
        raise NotImplementedError

    @abstractmethod
    def cofactor_true(self, /) -> Self | None:
        """Get the cofactor ``f_true`` of ``self``.

        Locking behavior: acquires the manager's lock for shared access.

        Returns:
            The cofactor ``f_true``, or ``None`` if ``self`` references a
            terminal node

        See Also:
            :meth:`cofactors`, also for a more detailed description
        """
        raise NotImplementedError

    @abstractmethod
    def cofactor_false(self, /) -> Self | None:
        """Get the cofactor ``f_false`` of ``self``.

        Locking behavior: acquires the manager's lock for shared access.

        Returns:
            The cofactor ``f_false``, or ``None`` if ``self`` references a
            terminal node

        See Also:
            :meth:`cofactors`, also for a more detailed description
        """
        raise NotImplementedError

    @abstractmethod
    def __invert__(self, /) -> Self:
        """Compute the negation ``¬self``.

        Locking behavior: acquires the manager's lock for shared access.

        Raises:
            DDMemoryError: If the operation runs out of memory
        """
        raise NotImplementedError

    @abstractmethod
    def __and__(self, rhs: Self, /) -> Self:
        """Compute the conjunction ``self ∧ rhs``.

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            rhs: Right-hand side operand. Must belong to the same manager as
                ``self``.

        Raises:
            DDMemoryError: If the operation runs out of memory
        """
        raise NotImplementedError

    @abstractmethod
    def __or__(self, rhs: Self, /) -> Self:
        """Compute the disjunction ``self ∨ rhs``.

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            rhs: Right-hand side operand. Must belong to the same manager as
                ``self``.

        Raises:
            DDMemoryError: If the operation runs out of memory
        """
        raise NotImplementedError

    @abstractmethod
    def __xor__(self, rhs: Self, /) -> Self:
        """Compute the exclusive disjunction ``self ⊕ rhs``.

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            rhs: Right-hand side operand. Must belong to the same manager as
                ``self``.

        Raises:
            DDMemoryError: If the operation runs out of memory
        """
        raise NotImplementedError

    @abstractmethod
    def nand(self, rhs: Self, /) -> Self:
        """Compute the negated conjunction ``self ⊼ rhs``.

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            rhs: Right-hand side operand. Must belong to the same manager as
                ``self``.

        Raises:
            DDMemoryError: If the operation runs out of memory
        """
        raise NotImplementedError

    @abstractmethod
    def nor(self, rhs: Self, /) -> Self:
        """Compute the negated disjunction ``self ⊽ rhs``.

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            rhs: Right-hand side operand. Must belong to the same manager as
                ``self``.

        Raises:
            DDMemoryError: If the operation runs out of memory
        """
        raise NotImplementedError

    @abstractmethod
    def equiv(self, rhs: Self, /) -> Self:
        """Compute the equivalence ``self ↔ rhs``.

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            rhs: Right-hand side operand. Must belong to the same manager as
                ``self``.

        Raises:
            DDMemoryError: If the operation runs out of memory
        """
        raise NotImplementedError

    @abstractmethod
    def imp(self, rhs: Self, /) -> Self:
        """Compute the implication ``self → rhs`` (or ``f ≤ g``).

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            rhs: Right-hand side operand. Must belong to the same manager as
                ``self``.

        Raises:
            DDMemoryError: If the operation runs out of memory
        """
        raise NotImplementedError

    @abstractmethod
    def imp_strict(self, rhs: Self, /) -> Self:
        """Compute the strict implication ``self < rhs``.

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            rhs: Right-hand side operand. Must belong to the same manager as
                ``self``.

        Raises:
            DDMemoryError: If the operation runs out of memory
        """
        raise NotImplementedError

    @abstractmethod
    def ite(self, /, t: Self, e: Self) -> Self:
        """Compute the decision diagram for the conditional ``t if self else e``.

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            t: Then-case; must belong to the same manager as ``self``
            e: Else-case; must belong to the same manager as ``self``

        Raises:
            DDMemoryError: If the operation runs out of memory
        """
        raise NotImplementedError

    @abstractmethod
    def satisfiable(self, /) -> bool:
        """Check for satisfiability.

        Locking behavior: acquires the manager's lock for shared access.

        Returns:
            Whether the Boolean function has at least one satisfying assignment
        """
        raise NotImplementedError

    @abstractmethod
    def valid(self, /) -> bool:
        """Check for validity.

        Locking behavior: acquires the manager's lock for shared access.

        Returns:
            Whether all assignments satisfy the Boolean function
        """
        raise NotImplementedError

    @abstractmethod
    def sat_count(self, /, vars: int) -> int:
        """Count the exact number of satisfying assignments.

        Uses arbitrary precision integers.

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            vars: Assume that the function's domain has this many variables.
        """
        raise NotImplementedError

    @abstractmethod
    def sat_count_float(self, /, vars: int) -> float:
        """Count the (approximate) number of satisfying assignments.

        Uses floating point numbers.

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            vars: Assume that the function's domain has this many variables.
        """
        raise NotImplementedError

    @abstractmethod
    def pick_cube(self, /) -> list[bool | None] | None:
        """Pick a satisfying assignment.

        Locking behavior: acquires the manager's lock for shared access.

        Returns:
            The satisfying assignment where the i-th value means that the i-th
            variable is false, true, or don't care, respectively, or ``None`` if
            ``self`` is unsatisfiable
        """
        raise NotImplementedError

    @abstractmethod
    def pick_cube_dd(self, /) -> Self:
        """Pick a satisfying assignment, represented as decision diagram.

        Locking behavior: acquires the manager's lock for shared access.

        Returns:
            The satisfying assignment as decision diagram, or ``⊥`` if ``self``
            is unsatisfiable

        Raises:
            DDMemoryError: If the operation runs out of memory
        """
        raise NotImplementedError

    @abstractmethod
    def pick_cube_dd_set(self, /, literal_set: Self) -> Self:
        """Pick a satisfying assignment as DD, with choices as of ``literal_set``.

        ``literal_set`` is a conjunction of literals. Whenever there is a choice
        for a variable, it will be set to true if the variable has a positive
        occurrence in ``literal_set``, and set to false if it occurs negated in
        ``literal_set``. If the variable does not occur in ``literal_set``, then
        it will be left as don't care if possible, otherwise an arbitrary (not
        necessarily random) choice will be performed.

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            literal_set: Conjunction of literals to determine the choice for
                variables

        Returns:
            The satisfying assignment as decision diagram, or ``⊥`` if ``self``
            is unsatisfiable

        Raises:
            DDMemoryError: If the operation runs out of memory
        """
        raise NotImplementedError

    @abstractmethod
    def eval(self, /, args: Iterable[tuple[Self, bool]]) -> bool:
        """Evaluate this Boolean function with arguments ``args``.

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            args: ``(variable, value)`` pairs where variables must be handles
                for the respective decision diagram levels, i.e., the Boolean
                function representing the variable for B(C)DDs, and a singleton
                set for ZBDDs.
                Missing variables are assumed to be false. However, note that
                the arguments may also determine the domain, e.g., in case of
                ZBDDs. If variables are given multiple times, the last value
                counts. Besides that, the order is irrelevant.
                All variable handles must belong to the same manager as ``self``
                and must reference inner nodes.

        Returns:
            The result of applying the function ``self`` to ``args``
        """
        raise NotImplementedError


class BooleanFunctionQuant(BooleanFunction, Protocol):
    """Quantification extension for :class:`BooleanFunction`."""

    @abstractmethod
    def forall(self, /, vars: Self) -> Self:
        """Compute the universal quantification over ``vars``.

        This operation removes all occurrences of variables in ``vars`` by
        universal quantification. Universal quantification ∀x. f(…, x, …) of a
        Boolean function f(…, x, …) over a single variable x is
        f(…, 0, …) ∧ f(…, 1, …).

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            vars: Set of variables represented as conjunction thereof. Must
                belong to the same manager as ``self``.

        Raises:
            DDMemoryError: If the operation runs out of memory
        """
        raise NotImplementedError

    @abstractmethod
    def exist(self, /, vars: Self) -> Self:
        """Compute the existential quantification over ``vars``.

        This operation removes all occurrences of variables in ``vars`` by
        existential quantification. Existential quantification ∃x. f(…, x, …) of
        a Boolean function f(…, x, …) over a single variable x is
        f(…, 0, …) ∨ f(…, 1, …).

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            vars: Set of variables represented as conjunction thereof. Must
                belong to the same manager as ``self``.

        Raises:
            DDMemoryError: If the operation runs out of memory
        """
        raise NotImplementedError

    @abstractmethod
    def unique(self, /, vars: Self) -> Self:
        """Compute the unique quantification over ``vars``.

        This operation removes all occurrences of variables in ``vars`` by
        unique quantification. Unique quantification ∃!x. f(…, x, …) of a
        Boolean function f(…, x, …) over a single variable x is
        f(…, 0, …) ⊕ f(…, 1, …). Unique quantification is also known as the
        `Boolean difference <https://en.wikipedia.org/wiki/Boole%27s_expansion_theorem#Operations_with_cofactors>`_ or
        `Boolean derivative <https://en.wikipedia.org/wiki/Boolean_differential_calculus>`_.

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            vars: Set of variables represented as conjunction thereof. Must
                belong to the same manager as ``self``.

        Raises:
            DDMemoryError: If the operation runs out of memory
        """  # noqa: E501
        raise NotImplementedError

    @abstractmethod
    def apply_forall(self, /, op: BooleanOperator, rhs: Self, vars: Self) -> Self:
        """Combined application of ``op`` and :meth:`forall()`.

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            op: Binary Boolean operator to apply to ``self`` and ``rhs``
            rhs: Right-hand side of the operator. Must belong to the same
                manager as ``self``.
            vars: Set of variables to quantify over. Represented as conjunction
                of variables. Must belong to the same manager as ``self``.

        Returns:
            ``∀ vars. self <op> rhs``

        Raises:
            DDMemoryError: If the operation runs out of memory
        """
        raise NotImplementedError

    @abstractmethod
    def apply_exist(self, /, op: BooleanOperator, rhs: Self, vars: Self) -> Self:
        """Combined application of ``op`` and :meth:`exist()`.

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            op: Binary Boolean operator to apply to ``self`` and ``rhs``
            rhs: Right-hand side of the operator. Must belong to the same
                manager as ``self``.
            vars: Set of variables to quantify over. Represented as conjunction
                of variables. Must belong to the same manager as ``self``.

        Returns:
            ``∃ vars. self <op> rhs``

        Raises:
            DDMemoryError: If the operation runs out of memory
        """
        raise NotImplementedError

    @abstractmethod
    def apply_unique(self, /, op: BooleanOperator, rhs: Self, vars: Self) -> Self:
        """Combined application of ``op`` and :meth:`unique()`.

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            op: Binary Boolean operator to apply to ``self`` and ``rhs``
            rhs: Right-hand side of the operator. Must belong to the same
                manager as ``self``.
            vars: Set of variables to quantify over. Represented as conjunction
                of variables. Must belong to the same manager as ``self``.

        Returns:
            ``∃! vars. self <op> rhs``

        Raises:
            DDMemoryError: If the operation runs out of memory
        """
        raise NotImplementedError


F = TypeVar("F", bound=Function, contravariant=True)


class Manager(Generic[F], Protocol):
    """Manager storing nodes and ensuring their uniqueness.

    A manager is the data structure responsible for storing nodes and ensuring
    their uniqueness. It also defines the variable order.

    Implementations supporting concurrency have an internal read/write lock.
    Many operations acquire this lock for reading (shared) or writing
    (exclusive).
    """

    @abstractmethod
    def num_inner_nodes(self, /) -> int:
        """Get the number of inner nodes.

        Locking behavior: acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def dump_all_dot_file(
        self,
        /,
        path: str | PathLike[str],
        functions: Iterable[tuple[F, str]] = [],
        variables: Iterable[tuple[F, str]] = [],
    ) -> None:
        """Dump the entire decision diagram in this manager as Graphviz DOT code.

        The output may also include nodes that are not reachable from
        ``functions``.

        Locking behavior: acquires the manager's lock for shared access.

        Args:
            path: Path of the output file. If a file at ``path`` exists, it will
                be truncated, otherwise a new one will be created.
            functions: Optional names for DD functions
            variables: Optional names for variables. The variables must be
                handles for the respective decision diagram levels, i.e., the
                Boolean function representing the variable for B(C)DDs, and a
                singleton set for ZBDDs.
        """
        raise NotImplementedError


BF = TypeVar("BF", bound=BooleanFunction)


class BooleanFunctionManager(Manager[BF], Protocol):
    """Manager whose nodes represent Boolean functions."""

    @abstractmethod
    def new_var(self, /) -> BF:
        """Get a fresh variable, adding a new level to a decision diagram.

        Acquires the manager's lock for exclusive access.

        Returns:
            A Boolean function that is true if and only if the variable is true
        """
        raise NotImplementedError

    @abstractmethod
    def true(self, /) -> BF:
        """Get the constant true Boolean function ``⊤``.

        Locking behavior: acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def false(self, /) -> BF:
        """Get the constant false Boolean function ``⊥``.

        Locking behavior: acquires the manager's lock for shared access.
        """
        raise NotImplementedError
