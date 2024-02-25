from typing import Protocol, Self, TypeVar

__all__ = ["Manager", "BooleanFunction"]


class Manager(Protocol):
    """Manager"""

    ...


class BooleanFunctionManager(Manager):
    def true(self) -> "BooleanFunction[Self]":
        """Get the constant true Boolean function `⊤`"""
        ...

    def false(self) -> "BooleanFunction[Self]":
        """Get the constant false Boolean function `⊥`"""
        ...

    def new_var(self) -> "BooleanFunction[Self]":
        """Get a fresh variable, i.e. a function that is true if and only if the
        variable is true. This adds a new level to a decision diagram."""
        ...


M = TypeVar("M", covariant=True, bound=Manager)


class Function(Protocol[M]):
    """Function represented as decision diagram"""

    @property
    def manager(self) -> M:
        """Get the associated manager"""
        ...

    def node_count(self) -> int:
        """Get the number of descendant nodes

        The returned number includes the root node itself and terminal nodes."""
        ...


class BooleanFunction(Function[M], Protocol):
    """Boolean function"""

    def __neg__(self) -> Self:
        """Compute the negation `¬self`"""
        ...

    def __and__(self, rhs: Self) -> Self:
        """Compute the conjunction `self ∧ rhs`

        `self` and `rhs` must belong to the same manager."""
        ...

    def __or__(self, rhs: Self) -> Self:
        """Compute the disjunction `self ∨ rhs`

        `self` and `rhs` must belong to the same manager."""
        ...

    def __xor__(self, rhs: Self) -> Self:
        """Compute the exclusive disjunction `self ⊕ rhs`

        `self` and `rhs` must belong to the same manager."""
        ...
