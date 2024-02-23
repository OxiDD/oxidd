from .oxidd import lib, ffi
from . import traits
from typing import Self, Any, override

__all__ = ["BDDManager", "BDDFunction"]


class BDDManager(traits.Manager):
    def __init__(self, inner_node_capacity: int, apply_cache_size: int, threads: int):
        self._manager: Any = lib.oxidd_bdd_manager_new(
            inner_node_capacity, apply_cache_size, threads
        )

    def __del__(self):
        lib.oxidd_bdd_manager_unref(self._manager)


class BDDFunction(traits.BooleanFunction):
    """Boolean function represented as a simple binary decision diagram (BDD)"""

    def __init__(self, inner: Any):
        """Private constructor"""
        self._function = inner

    def __del__(self):
        lib.oxidd_bdd_unref(self._function)

    @override
    @staticmethod
    def true(manager: traits.Manager) -> "BDDFunction":
        assert isinstance(manager, BDDManager)
        return BDDFunction(lib.oxidd_bdd_true(manager._manager))

    @override
    @staticmethod
    def false(manager: traits.Manager) -> "BDDFunction":
        assert isinstance(manager, BDDManager)
        return BDDFunction(lib.oxidd_bdd_false(manager._manager))

    @override
    @staticmethod
    def new_var(manager: traits.Manager) -> "BDDFunction":
        assert isinstance(manager, BDDManager)
        return BDDFunction(lib.oxidd_bdd_new_var(manager._manager))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BDDFunction)
            and self._function._p == other._function._p
            and self._function._i == other._function._i
        )

    @override
    def __neg__(self) -> Self:
        return BDDFunction(lib.oxidd_bdd_not(self._function))

    @override
    def __and__(self, rhs: Self) -> Self:
        # TODO: assert that `self` and `rhs` belong to the same manager
        return BDDFunction(lib.oxidd_bdd_and(self._function, rhs._function))

    @override
    def __or__(self, rhs: Self) -> Self:
        # TODO: assert that `self` and `rhs` belong to the same manager
        return BDDFunction(lib.oxidd_bdd_or(self._function, rhs._function))

    @override
    def __xor__(self, rhs: Self) -> Self:
        # TODO: assert that `self` and `rhs` belong to the same manager
        return BDDFunction(lib.oxidd_bdd_xor(self._function, rhs._function))

    @override
    def node_count(self) -> int:
        return lib.oxidd_bdd_node_count(self._function)
