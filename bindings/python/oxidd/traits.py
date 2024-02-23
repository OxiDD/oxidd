from abc import ABC
from typing import Self

__all__ = ["Manager", "BooleanFunction"]


class Manager(ABC):
    """Manager"""


class BooleanFunction(ABC):
    """Boolean function"""

    @staticmethod
    def true(manager: Manager) -> "BooleanFunction":
        """Get the constant true Boolean function `⊤`"""
        raise NotImplementedError

    @staticmethod
    def false(manager: Manager) -> "BooleanFunction":
        """Get the constant false Boolean function `⊥`"""
        raise NotImplementedError

    @staticmethod
    def new_var(manager: Manager) -> "BooleanFunction":
        """Get a fresh variable, i.e. a function that is true if and only if the
        variable is true. This adds a new level to a decision diagram."""
        raise NotImplementedError

    def __neg__(self) -> Self:
        """Compute the negation `¬self`"""
        raise NotImplementedError

    def __and__(self, rhs: Self) -> Self:
        """Compute the conjunction `self ∧ rhs`

        `self` and `rhs` must belong to the same manager."""
        raise NotImplementedError

    def __or__(self, rhs: Self) -> Self:
        """Compute the disjunction `self ∨ rhs`

        `self` and `rhs` must belong to the same manager."""
        raise NotImplementedError

    def __xor__(self, rhs: Self) -> Self:
        """Compute the exclusive disjunction `self ⊕ rhs`

        `self` and `rhs` must belong to the same manager."""
        raise NotImplementedError

    def node_count(self) -> int:
        """Get the number of inner nodes"""
        raise NotImplementedError
