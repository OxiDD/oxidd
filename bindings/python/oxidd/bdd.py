from typing import Any

from _oxidd import lib
from typing_extensions import Self, override

from . import traits

__all__ = ["BDDManager", "BDDFunction"]


class BDDManager(traits.BooleanFunctionManager):
    def __init__(self, inner: Any):
        """Private constructor"""
        self._manager = inner

    @staticmethod
    def new(
        inner_node_capacity: int, apply_cache_size: int, threads: int
    ) -> "BDDManager":
        return BDDManager(
            lib.oxidd_bdd_manager_new(inner_node_capacity, apply_cache_size, threads)
        )

    def __del__(self):
        lib.oxidd_bdd_manager_unref(self._manager)

    @override
    def true(self) -> "BDDFunction":
        return BDDFunction(lib.oxidd_bdd_true(self._manager))

    @override
    def false(self) -> "BDDFunction":
        return BDDFunction(lib.oxidd_bdd_false(self._manager))

    @override
    def new_var(self) -> "BDDFunction":
        return BDDFunction(lib.oxidd_bdd_new_var(self._manager))

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, BDDManager) and self._manager._p == other._manager._p


class BDDFunction(traits.BooleanFunction[BDDManager]):
    """Boolean function represented as a simple binary decision diagram (BDD)"""

    def __init__(self, inner: Any):
        """Private constructor"""
        self._function = inner

    def __del__(self):
        lib.oxidd_bdd_unref(self._function)

    @property
    @override
    def manager(self) -> BDDManager:
        raise NotImplementedError

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BDDFunction)
            and self._function._p == other._function._p
            and self._function._i == other._function._i
        )

    @override
    def __neg__(self) -> "BDDFunction":
        return BDDFunction(lib.oxidd_bdd_not(self._function))

    @override
    def __and__(self, rhs: Self) -> "BDDFunction":
        assert (
            self._function._p == rhs._function._p
        ), "`self` and `rhs` must belong to the same manager"
        return BDDFunction(lib.oxidd_bdd_and(self._function, rhs._function))

    @override
    def __or__(self, rhs: Self) -> "BDDFunction":
        assert (
            self._function._p == rhs._function._p
        ), "`self` and `rhs` must belong to the same manager"
        return BDDFunction(lib.oxidd_bdd_or(self._function, rhs._function))

    @override
    def __xor__(self, rhs: Self) -> "BDDFunction":
        assert (
            self._function._p == rhs._function._p
        ), "`self` and `rhs` must belong to the same manager"
        return BDDFunction(lib.oxidd_bdd_xor(self._function, rhs._function))

    @override
    def node_count(self) -> int:
        return lib.oxidd_bdd_node_count(self._function)
