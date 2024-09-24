"""Binary decision diagrams with complement edges (BCDDs)"""

__all__ = ["BCDDManager", "BCDDFunction"]

import collections.abc
from typing import Optional

from _oxidd import ffi as _ffi
from _oxidd import lib as _lib
from typing_extensions import Never, Self, override

from . import protocols, util


class BCDDManager(protocols.BooleanFunctionManager["BCDDFunction"]):
    """Manager for binary decision diagrams with complement edges"""

    _mgr: ...  #: Wrapped FFI object (``oxidd_bcdd_manager_t``)

    def __init__(self, inner_node_capacity: int, apply_cache_size: int, threads: int):
        """Create a new manager

        :param inner_node_capacity: Maximum count of inner nodes
        :param apply_cache_capacity: Maximum count of apply cache entries
        :param threads: Worker thread count for the internal thread pool
        """
        self._mgr = _lib.oxidd_bcdd_manager_new(
            inner_node_capacity, apply_cache_size, threads
        )

    @classmethod
    def _from_raw(cls, raw) -> Self:
        """Create a manager from a raw FFI object (``oxidd_bcdd_manager_t``)"""
        manager = cls.__new__(cls)
        manager._mgr = raw
        return manager

    def __del__(self):
        _lib.oxidd_bcdd_manager_unref(self._mgr)

    @override
    def __eq__(self, other: object) -> bool:
        """Check for referential equality"""
        return isinstance(other, BCDDManager) and self._mgr._p == other._mgr._p

    @override
    def __hash__(self) -> int:
        return hash(self._mgr._p)

    @override
    def new_var(self) -> "BCDDFunction":
        return BCDDFunction._from_raw(_lib.oxidd_bcdd_new_var(self._mgr))

    @override
    def true(self) -> "BCDDFunction":
        return BCDDFunction._from_raw(_lib.oxidd_bcdd_true(self._mgr))

    @override
    def false(self) -> "BCDDFunction":
        return BCDDFunction._from_raw(_lib.oxidd_bcdd_false(self._mgr))

    @override
    def num_inner_nodes(self) -> int:
        return _lib.oxidd_bcdd_num_inner_nodes(self._mgr)


class BCDDSubstitution:
    """Substitution mapping variables to replacement functions"""

    _subst: ...  #: Wrapped FFI object (``oxidd_bcdd_substitution_t *``)

    def __init__(
        self, pairs: collections.abc.Iterable[tuple["BCDDFunction", "BCDDFunction"]]
    ):
        """Create a new substitution object for BCDDs

        See :meth:`abc.FunctionSubst.make_substitution` fore more details.
        """
        self._subst = _lib.oxidd_bcdd_substitution_new(
            len(pairs) if isinstance(pairs, collections.abc.Sized) else 0
        )
        for v, r in pairs:
            _lib.oxidd_bcdd_substitution_add_pair(self._subst, v._func, r._func)

    def __del__(self):
        _lib.oxidd_bcdd_substitution_free(self._subst)


class BCDDFunction(
    protocols.BooleanFunctionQuant,
    protocols.FunctionSubst[BCDDSubstitution],
    protocols.HasLevel,
):
    """Boolean function represented as a binary decision diagram with complement
    edges (BCDD)

    All operations constructing BCDDs may throw a :exc:`MemoryError` in case
    they run out of memory.
    """

    _func: ...  #: Wrapped FFI object (``oxidd_bcdd_t``)

    def __init__(self, _: Never):
        """Private constructor

        Functions cannot be instantiated directly and must be created from the
        :class:`BCDDManager` or by combining existing functions instead.
        """
        raise RuntimeError(
            "Functions cannot be instantiated directly and must be created "
            "from the BCDDManager and by combining existing functions instead."
        )

    @classmethod
    def _from_raw(cls, raw) -> Self:
        """Create a BCDD function from a raw FFI object (``oxidd_bcdd_t``)"""
        if raw._p == _ffi.NULL:
            raise MemoryError("OxiDD BCDD operation ran out of memory")
        function = cls.__new__(cls)
        function._func = raw
        return function

    def __del__(self):
        _lib.oxidd_bcdd_unref(self._func)

    @override
    def __eq__(self, other: object) -> bool:
        """Check if ``other`` references the same node in the same manager and
        has the same edge tag

        Since BCDDs are a strong canonical form, ``self`` and ``other`` are
        semantically equal (i.e., represent the same Boolean functions) if and
        only if this method returns ``True``.
        """
        return (
            isinstance(other, BCDDFunction)
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
    def manager(self) -> BCDDManager:
        return BCDDManager._from_raw(_lib.oxidd_bcdd_containing_manager(self._func))

    @override
    def cofactors(self) -> tuple[Self, Self]:
        raw_pair = _lib.oxidd_bcdd_cofactors(self._func)
        return (
            self.__class__._from_raw(raw_pair.first),
            self.__class__._from_raw(raw_pair.second),
        )

    @override
    def cofactor_true(self) -> Self:
        return self.__class__._from_raw(_lib.oxidd_bcdd_cofactor_true(self._func))

    @override
    def cofactor_false(self) -> Self:
        return self.__class__._from_raw(_lib.oxidd_bcdd_cofactor_false(self._func))

    @override
    def level(self) -> Optional[int]:
        val = _lib.oxidd_bcdd_level(self._func)
        return val if val != util._LEVEL_NO_MAX else None

    @override
    def __invert__(self) -> Self:
        return self.__class__._from_raw(_lib.oxidd_bcdd_not(self._func))

    @override
    def __and__(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bcdd_and(self._func, rhs._func))

    @override
    def __or__(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bcdd_or(self._func, rhs._func))

    @override
    def __xor__(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bcdd_xor(self._func, rhs._func))

    @override
    def nand(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bcdd_nand(self._func, rhs._func))

    @override
    def nor(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bcdd_nor(self._func, rhs._func))

    @override
    def equiv(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bcdd_equiv(self._func, rhs._func))

    @override
    def imp(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bcdd_imp(self._func, rhs._func))

    @override
    def imp_strict(self, rhs: Self) -> Self:
        assert (
            self._func._p == rhs._func._p
        ), "`self` and `rhs` must belong to the same manager"
        return self.__class__._from_raw(
            _lib.oxidd_bcdd_imp_strict(self._func, rhs._func)
        )

    @override
    def ite(self, t: Self, e: Self) -> Self:
        assert (
            self._func._p == t._func._p == e._func._p
        ), "`self` and `t` and `e` must belong to the same manager"
        return self.__class__._from_raw(
            _lib.oxidd_bcdd_ite(self._func, t._func, e._func)
        )

    @override
    @classmethod
    def make_substitution(
        cls, pairs: collections.abc.Iterable[tuple[Self, Self]]
    ) -> BCDDSubstitution:
        return BCDDSubstitution(pairs)

    @override
    def substitute(self, substitution: BCDDSubstitution) -> Self:
        return self.__class__._from_raw(
            _lib.oxidd_bcdd_substitute(self._func, substitution._subst)
        )

    @override
    def forall(self, vars: Self) -> Self:
        assert (
            self._func._p == vars._func._p
        ), "`self` and `vars` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bcdd_forall(self._func, vars._func))

    @override
    def exist(self, vars: Self) -> Self:
        assert (
            self._func._p == vars._func._p
        ), "`self` and `vars` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bcdd_exist(self._func, vars._func))

    @override
    def unique(self, vars: Self) -> Self:
        assert (
            self._func._p == vars._func._p
        ), "`self` and `vars` must belong to the same manager"
        return self.__class__._from_raw(_lib.oxidd_bcdd_unique(self._func, vars._func))

    @override
    def apply_forall(self, op: util.BooleanOperator, rhs: Self, vars: Self):
        if not isinstance(op, util.BooleanOperator):
            # If op were an arbitrary integer that is not part of the enum,
            # this would lead to undefined behavior
            raise ValueError("`op` must be a BooleanOperator")
        assert (
            self._func._p == rhs._func._p == vars._func._p
        ), "`self`, `rhs`, and `vars` must belong to the same manager"
        return self.__class__._from_raw(
            _lib.oxidd_bcdd_apply_forall(op, self._func, rhs._func, vars._func)
        )

    @override
    def apply_exist(self, op: util.BooleanOperator, rhs: Self, vars: Self):
        if not isinstance(op, util.BooleanOperator):
            raise ValueError("`op` must be a BooleanOperator")
        assert (
            self._func._p == rhs._func._p == vars._func._p
        ), "`self`, `rhs`, and `vars` must belong to the same manager"
        return self.__class__._from_raw(
            _lib.oxidd_bcdd_apply_exist(op, self._func, rhs._func, vars._func)
        )

    @override
    def apply_unique(self, op: util.BooleanOperator, rhs: Self, vars: Self):
        if not isinstance(op, util.BooleanOperator):
            raise ValueError("`op` must be a BooleanOperator")
        assert (
            self._func._p == rhs._func._p == vars._func._p
        ), "`self`, `rhs`, and `vars` must belong to the same manager"
        return self.__class__._from_raw(
            _lib.oxidd_bcdd_apply_unique(op, self._func, rhs._func, vars._func)
        )

    @override
    def node_count(self) -> int:
        return int(_lib.oxidd_bcdd_node_count(self._func))

    @override
    def satisfiable(self) -> bool:
        return bool(_lib.oxidd_bcdd_satisfiable(self._func))

    @override
    def valid(self) -> bool:
        return bool(_lib.oxidd_bcdd_valid(self._func))

    @override
    def sat_count_float(self, vars: int) -> float:
        return float(_lib.oxidd_bcdd_sat_count_double(self._func, vars))

    @override
    def pick_cube(self) -> Optional[util.Assignment]:
        raw = _lib.oxidd_bcdd_pick_cube(self._func)
        return util.Assignment._from_raw(raw) if raw.len > 0 else None

    @override
    def pick_cube_symbolic(self) -> Self:
        return self.__class__._from_raw(_lib.oxidd_bcdd_pick_cube_symbolic(self._func))

    @override
    def pick_cube_symbolic_set(self, literal_set: Self) -> Self:
        assert (
            self._func._p == literal_set._func._p
        ), "`self` and `literal_set` must belong to the same manager"
        return self.__class__._from_raw(
            _lib.oxidd_bcdd_pick_cube_symbolic_set(self._func, literal_set._func)
        )

    @override
    def eval(self, args: collections.abc.Collection[tuple[Self, bool]]) -> bool:
        n = len(args)
        c_args = util._alloc("oxidd_bcdd_bool_pair_t[]", n)
        for c_arg, (f, v) in zip(c_args, args):
            c_arg.func = f._func
            c_arg.val = v

        return bool(_lib.oxidd_bcdd_eval(self._func, c_args, n))
