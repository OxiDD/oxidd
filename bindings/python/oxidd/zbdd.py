"""Zero-suppressed binary decision diagrams (ZBDDs)"""

__all__ = ["ZBDDManager", "ZBDDFunction"]

import collections.abc
from typing import Optional

from _oxidd import ffi as _ffi
from _oxidd import lib as _lib
from typing_extensions import Never, Self, override

from . import protocols, util


class ZBDDManager(protocols.BooleanFunctionManager["ZBDDFunction"]):
    """Manager for zero-suppressed binary decision diagrams"""

    _mgr: ...  #: Wrapped FFI object (``oxidd_zbdd_manager_t``)

    def __init__(self, inner_node_capacity: int, apply_cache_size: int, threads: int):
        """Create a new manager

        :param inner_node_capacity: Maximum count of inner nodes
        :param apply_cache_capacity: Maximum count of apply cache entries
        :param threads: Worker thread count for the internal thread pool
        """
        self._mgr = _lib.oxidd_zbdd_manager_new(
            inner_node_capacity, apply_cache_size, threads
        )

    @classmethod
    def _from_raw(cls, raw) -> Self:
        """Create a manager from a raw FFI object (``oxidd_zbdd_manager_t``)"""
        manager = cls.__new__(cls)
        manager._mgr = raw
        return manager

    def __del__(self):
        _lib.oxidd_zbdd_manager_unref(self._mgr)

    @override
    def __eq__(self, other: object) -> bool:
        """Check for referential equality"""
        return isinstance(other, ZBDDManager) and self._mgr._p == other._mgr._p

    @override
    def __hash__(self) -> int:
        return hash(self._mgr._p)

    def new_singleton(self) -> "ZBDDFunction":
        """Get a fresh variable in the form of a singleton set

        This adds a new level to a decision diagram.

        Acquires the manager's lock for exclusive access.
        """
        return ZBDDFunction._from_raw(_lib.oxidd_zbdd_new_singleton(self._mgr))

    @override
    def new_var(self) -> "ZBDDFunction":
        return ZBDDFunction._from_raw(_lib.oxidd_zbdd_new_var(self._mgr))

    def empty(self) -> "ZBDDFunction":
        """Get the ZBDD set ∅

        Acquires the manager's lock for shared access.
        """
        return ZBDDFunction._from_raw(_lib.oxidd_zbdd_empty(self._mgr))

    def base(self) -> "ZBDDFunction":
        """Get the ZBDD set {∅}

        Acquires the manager's lock for shared access.
        """
        return ZBDDFunction._from_raw(_lib.oxidd_zbdd_base(self._mgr))

    @override
    def true(self) -> "ZBDDFunction":
        return ZBDDFunction._from_raw(_lib.oxidd_zbdd_true(self._mgr))

    @override
    def false(self) -> "ZBDDFunction":
        return ZBDDFunction._from_raw(_lib.oxidd_zbdd_false(self._mgr))

    @override
    def num_inner_nodes(self) -> int:
        return _lib.oxidd_zbdd_num_inner_nodes(self._mgr)

    @override
    def dump_all_dot_file(
        self,
        path: str,
        functions: collections.abc.Iterable[tuple["ZBDDFunction", str]] = [],
        variables: collections.abc.Iterable[tuple["ZBDDFunction", str]] = [],
    ) -> bool:
        fs = []
        f_names = []
        for f, name in functions:
            fs.append(f._func)
            f_names.append(_ffi.new("char[]", name.encode()))

        vars = []
        var_names = []
        for f, name in variables:
            vars.append(f._func)
            var_names.append(_ffi.new("char[]", name.encode()))

        return bool(
            _lib.oxidd_zbdd_manager_dump_all_dot_file(
                self._mgr,
                path.encode(),
                fs,
                f_names,
                len(fs),
                vars,
                var_names,
                len(vars),
            )
        )


class ZBDDFunction(protocols.BooleanFunction, protocols.HasLevel):
    """Boolean function 𝔹ⁿ → 𝔹 (or set of Boolean vectors 𝔹ⁿ) represented as
    zero-suppressed binary decision diagram

    All operations constructing ZBDDs may throw a :exc:`MemoryError` in case they
    run out of memory.
    """

    _func: ...  #: Wrapped FFI object (``oxidd_zbdd_t``)

    def __init__(self, _: Never):
        """Private constructor

        Functions cannot be instantiated directly and must be created from the
        :class:`ZBDDManager` or by combining existing functions instead.
        """
        raise RuntimeError(
            "Functions cannot be instantiated directly and must be created "
            "from the ZBDDManager and by combining existing functions instead."
        )

    @classmethod
    def _from_raw(cls, raw) -> Self:
        """Create a ZBDD function from a raw FFI object (``oxidd_zbdd_t``)"""
        if raw._p == _ffi.NULL:
            raise MemoryError("OxiDD ZBDD operation ran out of memory")
        function = cls.__new__(cls)
        function._func = raw
        return function

    def __del__(self):
        _lib.oxidd_zbdd_unref(self._func)

    @override
    def __eq__(self, other: object) -> bool:
        """Check if ``other`` references the same node in the same manager

        Since ZBDDs are a strong canonical form, ``self`` and ``other`` are
        semantically equal (i.e., represent the same Boolean functions) if and
        only if this method returns ``True``.
        """
        return (
            isinstance(other, ZBDDFunction)
            and self._func._p == other._func._p
            and self._func._i == other._func._i
        )

    @override
    def __lt__(self, other: Self) -> bool:
        return (self._func._p, self._func._i) < (other._func._p, other._func._i)

    @override
    def __gt__(self, other: Self) -> bool:
        return (self._func._p, self._func._i) > (other._func._p, other._func._i)

    @override
    def __le__(self, other: Self) -> bool:
        return (self._func._p, self._func._i) <= (other._func._p, other._func._i)

    @override
    def __ge__(self, other: Self) -> bool:
        return (self._func._p, self._func._i) >= (other._func._p, other._func._i)

    @override
    def __hash__(self) -> int:
        return hash((self._func._p, self._func._i))

    @property
    @override
    def manager(self) -> ZBDDManager:
        return ZBDDManager._from_raw(_lib.oxidd_zbdd_containing_manager(self._func))

    @override
    def cofactors(self) -> Optional[tuple[Self, Self]]:
        raw_pair = _lib.oxidd_zbdd_cofactors(self._func)
        if raw_pair.first._p == _ffi.NULL:
            return None
        assert raw_pair.second._p != _ffi.NULL
        return (
            self.__class__._from_raw(raw_pair.first),
            self.__class__._from_raw(raw_pair.second),
        )

    @override
    def cofactor_true(self) -> Optional[Self]:
        raw = _lib.oxidd_zbdd_cofactor_true(self._func)
        return self.__class__._from_raw(raw) if raw._p != _ffi.NULL else None

    @override
    def cofactor_false(self) -> Optional[Self]:
        raw = _lib.oxidd_zbdd_cofactor_false(self._func)
        return self.__class__._from_raw(raw) if raw._p != _ffi.NULL else None

    @override
    def level(self) -> Optional[int]:
        val = _lib.oxidd_zbdd_level(self._func)
        return val if val != util._LEVEL_NO_MAX else None

    def var_boolean_function(self) -> Self:
        """Get the ZBDD Boolean function v for the singleton set {v}

        Acquires the manager's lock for shared access.
        """
        return self.__class__._from_raw(
            _lib.oxidd_zbdd_var_boolean_function(self._func)
        )

    @override
    def __invert__(self) -> Self:
        return self.__class__._from_raw(_lib.oxidd_zbdd_not(self._func))

    @override
    def __and__(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_zbdd_and(self._func, rhs._func))

    @override
    def __or__(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_zbdd_or(self._func, rhs._func))

    @override
    def __xor__(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_zbdd_xor(self._func, rhs._func))

    def __sub__(self, rhs: Self) -> Self:
        """Compute the ZBDD for the set difference ``self ∖ rhs``

        This is equivalent to the strict implication ``rhs < lhs`` on Boolean
        functions.

        ``self`` and ``rhs`` must belong to the same manager.

        Acquires the manager's lock for shared access.
        """
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_zbdd_diff(self._func, rhs._func))

    @override
    def nand(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_zbdd_nand(self._func, rhs._func))

    @override
    def nor(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_zbdd_nor(self._func, rhs._func))

    @override
    def equiv(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_zbdd_equiv(self._func, rhs._func))

    @override
    def imp(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_zbdd_imp(self._func, rhs._func))

    @override
    def imp_strict(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(
            _lib.oxidd_zbdd_imp_strict(self._func, rhs._func)
        )

    @override
    def ite(self, t: Self, e: Self) -> Self:
        assert (
            self._func._p == t._func._p == e._func._p
        ), "`self` and `t` and `e` must belong to the same manager"
        return self.__class__._from_raw(
            _lib.oxidd_zbdd_ite(self._func, t._func, e._func)
        )

    def make_node(self, hi: Self, lo: Self) -> Self:
        """Create a new ZBDD node at the level of `this` with the given `hi` and
        `lo` edges

        `var` must be a singleton set.

        Acquires the manager's lock for shared access.
        """
        assert (
            self._func._p == hi._func._p == lo._func._p
        ), "`self` and `hi` and `lo` must belong to the same manager"
        # `oxidd_zbdd_make_node()` takes ownership of `hi` and `lo`, but we
        # cannot give up ownership in Python, so we need to increment their
        # reference counts.
        _lib.oxidd_zbdd_ref(hi._func)
        _lib.oxidd_zbdd_ref(lo._func)
        return self.__class__._from_raw(
            _lib.oxidd_zbdd_make_node(self._func, hi._func, lo._func)
        )

    @override
    def node_count(self) -> int:
        return int(_lib.oxidd_zbdd_node_count(self._func))

    @override
    def satisfiable(self) -> bool:
        return bool(_lib.oxidd_zbdd_satisfiable(self._func))

    @override
    def valid(self) -> bool:
        return bool(_lib.oxidd_zbdd_valid(self._func))

    @override
    def sat_count_float(self, vars: int) -> float:
        return float(_lib.oxidd_zbdd_sat_count_double(self._func, vars))

    @override
    def pick_cube(self) -> Optional[util.Assignment]:
        raw = _lib.oxidd_zbdd_pick_cube(self._func)
        return util.Assignment._from_raw(raw) if raw.len > 0 else None

    @override
    def pick_cube_dd(self) -> Self:
        return self.__class__._from_raw(_lib.oxidd_zbdd_pick_cube_dd(self._func))

    @override
    def pick_cube_dd_set(self, literal_set: Self) -> Self:
        assert (
            self._func._p == literal_set._func._p
        ), "`self` and `literal_set` must belong to the same manager"
        return self.__class__._from_raw(
            _lib.oxidd_zbdd_pick_cube_dd_set(self._func, literal_set._func)
        )

    @override
    def eval(self, args: collections.abc.Collection[tuple[Self, bool]]) -> bool:
        n = len(args)
        c_args = util._alloc("oxidd_zbdd_bool_pair_t[]", n)
        for c_arg, (f, v) in zip(c_args, args):
            c_arg.func = f._func
            c_arg.val = v

        return bool(_lib.oxidd_zbdd_eval(self._func, c_args, n))
