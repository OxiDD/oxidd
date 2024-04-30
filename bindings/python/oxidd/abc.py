"""
The abstract base classes declared in this module allow abstracting away the
concrete decision diagram kind in a type-safe fashion.
"""

__all__ = [
    "Manager",
    "BooleanFunctionManager",
    "Function",
    "BooleanFunction",
    "BooleanFunctionQuant",
]

from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Generic, Optional, TypeVar

from typing_extensions import Self

from .util import Assignment


class Manager(ABC):
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


class BooleanFunctionManager(Manager):
    """Manager whose nodes represent Boolean functions"""

    @abstractmethod
    def new_var(self) -> "BooleanFunction[Self]":
        """Get a fresh variable, i.e., a function that is true if and only if
        the variable is true. This adds a new level to a decision diagram.

        Acquires the manager's lock for exclusive access.
        """
        raise NotImplementedError

    @abstractmethod
    def true(self) -> "BooleanFunction[Self]":
        """Get the constant true Boolean function ``⊤``

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError

    @abstractmethod
    def false(self) -> "BooleanFunction[Self]":
        """Get the constant false Boolean function ``⊥``

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError


M = TypeVar("M", covariant=True, bound=Manager)


class Function(ABC, Generic[M]):
    """Function represented as decision diagram

    A function is the combination of a reference to a :class:`Manager` and a
    (possibly tagged) edge pointing to a node. Obtaining the manager reference
    is possible via the :attr:`manager` property.
    """

    @property
    @abstractmethod
    def manager(self) -> M:
        """The associated manager"""
        raise NotImplementedError

    @abstractmethod
    def node_count(self) -> int:
        """Get the number of descendant nodes

        The returned number includes the root node itself and terminal nodes.

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError


class BooleanFunction(Function[M]):
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
    def eval(self, args: Collection[tuple[Self, bool]]) -> bool:
        """Evaluate this Boolean function with arguments ``args``

        ``args`` determines the valuation for all variables. Missing values are
        assumed to be false. However, note that the arguments may also determine
        the domain, e.g., in case of ZBDDs. If values are specified multiple
        times, the last one counts. Panics if any function in `args` refers to a
        terminal node.

        Acquires the manager's lock for shared access.
        """
        raise NotImplementedError


class BooleanFunctionQuant(BooleanFunction[M]):
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
