"""Binary decision diagrams (BDDs, without complement edges)"""

__all__ = ["BDDManager", "BDDFunction"]

from collections.abc import Collection
from typing import Optional

from _oxidd import ffi as _ffi
from _oxidd import lib as _lib
from typing_extensions import Never, Self, override

from . import abc, util


class BDDManager(abc.BooleanFunctionManager):
    """Manager for binary decision diagrams (without complement edges)"""

    _mgr: ...  #: Wrapped FFI object

    def __init__(self, inner_node_capacity: int, apply_cache_size: int, threads: int):
        """Create a new manager

        :param inner_node_capacity: Maximum count of inner nodes
        :param apply_cache_capacity: Maximum count of apply cache entries
        :param threads: Worker thread count for the internal thread pool
        """
        self._mgr = _lib.oxidd_bdd_manager_new(
            inner_node_capacity, apply_cache_size, threads
        )

    @classmethod
    def _from_raw(cls, raw) -> Self:
        """Create a manager from a raw FFI object (``oxidd_bdd_manager_t``)"""
        manager = cls.__new__(cls)
        manager._mgr = raw
        return manager

    def __del__(self):
        _lib.oxidd_bdd_manager_unref(self._mgr)

    @override
    def __eq__(self, other: object) -> bool:
        """Check for referential equality"""
        return isinstance(other, BDDManager) and self._mgr._p == other._mgr._p

    @override
    def __hash__(self) -> int:
        return hash(self._mgr._p)

    @override
    def new_var(self) -> "BDDFunction":
        return BDDFunction._from_raw(_lib.oxidd_bdd_new_var(self._mgr))

    @override
    def true(self) -> "BDDFunction":
        return BDDFunction._from_raw(_lib.oxidd_bdd_true(self._mgr))

    @override
    def false(self) -> "BDDFunction":
        return BDDFunction._from_raw(_lib.oxidd_bdd_false(self._mgr))

    @override
    def num_inner_nodes(self) -> int:
        return _lib.oxidd_bdd_num_inner_nodes(self._mgr)


class BDDFunction(abc.BooleanFunctionQuant[BDDManager]):
    """Boolean function represented as a simple binary decision diagram (BDD)

    All operations constructing BDDs may throw a :exc:`MemoryError` in case they
    run out of memory.
    """

    _func: ...  #: Wrapped FFI object

    def __init__(self, _: Never):
        """Private constructor

        Functions cannot be instantiated directly and must be created from the
        :class:`BDDManager` or by combining existing functions instead.
        """
        raise RuntimeError(
            "Functions cannot be instantiated directly and must be created "
            "from the BDDManager and by combining existing functions instead."
        )

    @classmethod
    def _from_raw(cls, raw) -> Self:
        """Create a BDD function from a raw FFI object (``oxidd_bdd_t``)"""
        if raw._p == _ffi.NULL:
            raise MemoryError("OxiDD BDD operation ran out of memory")
        function = cls.__new__(cls)
        function._func = raw
        return function

    def __del__(self):
        _lib.oxidd_bdd_unref(self._func)

    @override
    def __eq__(self, other: object) -> bool:
        """Check if ``other`` references the same node in the same manager

        Since BDDs are a strong canonical form, ``self`` and ``other`` are
        semantically equal (i.e., represent the same Boolean functions) if and
        only if this method returns ``True``.
        """
        return (
            isinstance(other, BDDFunction)
            and self._func._p == other._func._p
            and self._func._i == other._func._i
        )

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
        return (self._func._p, self._func._i) < (other._func._p, other._func._i)

    def __gt__(self, other: Self) -> bool:
        """Same as ``other < self``"""
        return (self._func._p, self._func._i) > (other._func._p, other._func._i)

    def __le__(self, other: Self) -> bool:
        """Same as ``not self > other``"""
        return (self._func._p, self._func._i) <= (other._func._p, other._func._i)

    def __ge__(self, other: Self) -> bool:
        """Same as ``not self < other``"""
        return (self._func._p, self._func._i) >= (other._func._p, other._func._i)

    @override
    def __hash__(self) -> int:
        return hash((self._func._p, self._func._i))

    @property
    @override
    def manager(self) -> BDDManager:
        return BDDManager._from_raw(_lib.oxidd_bdd_containing_manager(self._func))

    @override
    def cofactors(self) -> tuple[Self, Self]:
        raw_pair = _lib.oxidd_bdd_cofactors(self._func)
        return (
            self.__class__._from_raw(raw_pair.first),
            self.__class__._from_raw(raw_pair.second),
        )

    @override
    def cofactor_true(self) -> Self:
        return self.__class__._from_raw(_lib.oxidd_bdd_cofactor_true(self._func))

    @override
    def cofactor_false(self) -> Self:
        return self.__class__._from_raw(_lib.oxidd_bdd_cofactor_false(self._func))

    @override
    def __invert__(self) -> Self:
        return self.__class__._from_raw(_lib.oxidd_bdd_not(self._func))

    @override
    def __and__(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bdd_and(self._func, rhs._func))

    @override
    def __or__(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bdd_or(self._func, rhs._func))

    @override
    def __xor__(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bdd_xor(self._func, rhs._func))

    @override
    def nand(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bdd_nand(self._func, rhs._func))

    @override
    def nor(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bdd_nor(self._func, rhs._func))

    @override
    def equiv(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bdd_equiv(self._func, rhs._func))

    @override
    def imp(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bdd_imp(self._func, rhs._func))

    @override
    def imp_strict(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(
            _lib.oxidd_bdd_imp_strict(self._func, rhs._func)
        )

    @override
    def ite(self, t: Self, e: Self) -> Self:
        assert (
            self._func._p == t._func._p == e._func._p
        ), "`self` and `t` and `e` must belong to the same manager"
        return self.__class__._from_raw(
            _lib.oxidd_bdd_ite(self._func, t._func, e._func)
        )

    @override
    def forall(self, vars: Self) -> Self:
        assert (
            self._func._p == vars._func._p
        ), "`self` and `vars` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bdd_forall(self._func, vars._func))

    @override
    def exist(self, vars: Self) -> Self:
        assert (
            self._func._p == vars._func._p
        ), "`self` and `vars` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bdd_exist(self._func, vars._func))

    @override
    def unique(self, vars: Self) -> Self:
        assert (
            self._func._p == vars._func._p
        ), "`self` and `vars` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bdd_unique(self._func, vars._func))

    @override
    def node_count(self) -> int:
        return int(_lib.oxidd_bdd_node_count(self._func))

    @override
    def satisfiable(self) -> bool:
        return bool(_lib.oxidd_bdd_satisfiable(self._func))

    @override
    def valid(self) -> bool:
        return bool(_lib.oxidd_bdd_valid(self._func))

    @override
    def sat_count_float(self, vars: int) -> float:
        return float(_lib.oxidd_bdd_sat_count_double(self._func, vars))

    @override
    def pick_cube(self) -> Optional[util.Assignment]:
        raw = _lib.oxidd_bdd_pick_cube(self._func)
        return util.Assignment._from_raw(raw) if raw.len > 0 else None

    @override
    def eval(self, args: Collection[tuple[Self, bool]]) -> bool:
        n = len(args)
        c_args = util._alloc("oxidd_bdd_bool_pair_t[]", n)
        for c_arg, (f, v) in zip(c_args, args):
            c_arg.func = f._func
            c_arg.val = v

        return bool(_lib.oxidd_bdd_eval(self._func, c_args, n))
