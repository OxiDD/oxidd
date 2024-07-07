from collections.abc import Sequence
from typing import Generic, Optional, Protocol, TypeVar

import oxidd
from oxidd.protocols import (
    BooleanFunction,
    BooleanFunctionManager,
    BooleanFunctionQuant,
    FunctionSubst,
)

# spell-checker:ignore nvars,BFQS


class BooleanFunctionQuantSubst(BooleanFunctionQuant, FunctionSubst, Protocol):
    pass


BF = TypeVar("BF", bound=BooleanFunction)
BFQS = TypeVar("BFQS", bound=BooleanFunctionQuantSubst)


def bit_count(x: int) -> int:
    """Count the number of one bits

    To be replaced by int.bit_count() once we require Python 3.10
    """
    b = 0
    while x > 0:
        x &= x - 1
        b += 1
    return b


class AllBooleanFunctions(Generic[BF]):
    """Python translation of ``TestAllBooleanFunctions`` from
    ``crates/oxidd/tests/boolean_function.rs``"""

    _mgr: BooleanFunctionManager[BF]
    _vars: Sequence[BF]
    _var_handles: Sequence[BF]
    #: stores all possible Boolean functions with `len(vars)` variables
    _boolean_functions: list[BF]
    _dd_to_boolean_func: dict[BF, int]

    def __init__(
        self,
        manager: BooleanFunctionManager[BF],
        vars: Sequence[BF],
        var_handles: Sequence[BF],
    ):
        """Initialize the test, generating DDs for all Boolean functions for the
        given variable set. ``vars`` are the Boolean functions representing the
        variables identified by ``var_handles``. For BDDs, the two coincide, but
        not for ZBDDs."""
        assert len(vars) == len(var_handles)

        self._mgr = manager
        self._vars = vars
        self._var_handles = var_handles

        self._boolean_functions = []
        self._dd_to_boolean_func = {}

        nvars = len(vars)
        num_assignments = 1 << nvars
        num_functions = 1 << num_assignments

        for explicit_f in range(num_functions):
            # naïve DD construction from the on-set
            f = self._mgr.false()
            for assignment in range(num_assignments):
                if explicit_f & (1 << assignment) == 0:
                    continue  # not part of the on-set
                cube = self._mgr.true()

                for var in range(nvars):
                    v = self._vars[var]
                    if assignment & (1 << var) == 0:
                        v = ~v
                    cube &= v

                f |= cube

            # check that evaluating the function yields the desired values
            for assignment in range(num_assignments):
                expected = explicit_f & (1 << assignment) != 0
                args = [
                    (vh, (assignment & (1 << var)) != 0)
                    for var, vh in enumerate(var_handles)
                ]
                actual = f.eval(args)
                assert actual == expected
                assert int(f.sat_count_float(nvars)) == bit_count(explicit_f)

            self._boolean_functions.append(f)
            assert f not in self._dd_to_boolean_func
            self._dd_to_boolean_func[f] = explicit_f

    def basic(self):
        """Test basic operations on all Boolean function"""

        nvars = len(self._vars)
        num_assignments = 1 << nvars
        num_functions = 1 << num_assignments
        func_mask = num_functions - 1

        # false & true
        assert self._mgr.false() == self._boolean_functions[0]
        assert self._mgr.true() == self._boolean_functions[-1]

        # vars
        for vi, var in enumerate(self._vars):
            expected = 0
            for assignment in range(num_assignments):
                expected |= ((assignment >> vi) & 1) << assignment
            actual = self._dd_to_boolean_func[var]
            assert actual == expected

        # arity >= 1
        for f_explicit, f in enumerate(self._boolean_functions):
            # not
            expected = ~f_explicit & func_mask
            actual = self._dd_to_boolean_func[~f]
            assert actual == expected

            # arity >= 2
            for g_explicit, g in enumerate(self._boolean_functions):
                # and
                expected = f_explicit & g_explicit
                actual = self._dd_to_boolean_func[f & g]
                assert actual == expected

                # or
                expected = f_explicit | g_explicit
                actual = self._dd_to_boolean_func[f | g]
                assert actual == expected

                # xor
                expected = f_explicit ^ g_explicit
                actual = self._dd_to_boolean_func[f ^ g]
                assert actual == expected

                # equiv
                expected = ~(f_explicit ^ g_explicit) & func_mask
                actual = self._dd_to_boolean_func[f.equiv(g)]
                assert actual == expected

                # nand
                expected = ~(f_explicit & g_explicit) & func_mask
                actual = self._dd_to_boolean_func[f.nand(g)]
                assert actual == expected

                # nor
                expected = ~(f_explicit | g_explicit) & func_mask
                actual = self._dd_to_boolean_func[f.nor(g)]
                assert actual == expected

                # implication
                expected = (~f_explicit | g_explicit) & func_mask
                actual = self._dd_to_boolean_func[f.imp(g)]
                assert actual == expected

                # strict implication
                expected = ~f_explicit & g_explicit
                actual = self._dd_to_boolean_func[f.imp_strict(g)]
                assert actual == expected

                # arity >= 3
                for h_explicit, h in enumerate(self._boolean_functions):
                    # ite
                    expected = (f_explicit & g_explicit) | (~f_explicit & h_explicit)
                    actual = self._dd_to_boolean_func[f.ite(g, h)]
                    assert actual == expected


class AllBooleanFunctionsQuantSubst(AllBooleanFunctions[BFQS]):
    def _subst_rec(self, replacements: list[Optional[int]], current_var: int):
        assert len(replacements) == len(self._vars)
        if current_var < len(self._vars):
            replacements[current_var] = None
            self._subst_rec(replacements, current_var + 1)
            for f in range(0, len(self._boolean_functions)):
                replacements[current_var] = f
                self._subst_rec(replacements, current_var + 1)
        else:
            nvars = len(self._vars)
            num_assignments = 1 << nvars

            subst = self._vars[0].make_substitution(
                (
                    (self._var_handles[i], self._boolean_functions[repl])
                    for i, repl in enumerate(replacements)
                    if repl is not None
                )
            )

            for f_explicit, f in enumerate(self._boolean_functions):
                expected = 0
                # To compute the expected truth table, we first compute a
                # mapped assignment that we look up in the truth table for `f`
                for assignment in range(num_assignments):
                    mapped_assignment = 0
                    for var, repl in enumerate(replacements):
                        val = (
                            # replacement function evaluated for `assignment`
                            repl >> assignment
                            if repl is not None
                            # `var` is set in `assignment`?
                            else assignment >> var
                        ) & 1
                        mapped_assignment |= val << var
                    expected |= ((f_explicit >> mapped_assignment) & 1) << assignment

                actual = self._dd_to_boolean_func[f.substitute(subst)]
                assert actual == expected

    def subst(self):
        """Test all possible substitutions"""
        self._subst_rec([None] * len(self._vars), 0)

    def quant(self):
        """Test quantification operations on all Boolean function"""

        nvars = len(self._vars)
        num_assignments = 1 << nvars
        num_functions = 1 << num_assignments
        func_mask = num_functions - 1

        def var_explicit_func(i: int) -> int:
            f = 0
            for assignment in range(num_assignments):
                f |= ((assignment >> i) & 1) << assignment
            return f

        var_functions = [var_explicit_func(i) for i in range(nvars)]

        # TODO: restrict (once we have it in the Python API)

        def assignment_to_mask(assignment: int, var_set: int) -> int:
            mask = func_mask
            for i in range(nvars):
                if ((var_set >> i) & 1) != 0:
                    continue
                f = var_functions[i]
                mask &= f if ((assignment >> i) & 1) != 0 else ~f
            return mask

        # quantification
        for var_set in range(num_assignments):
            dd_var_set = self._mgr.true()

            for i in range(nvars):
                if (var_set & (1 << i)) != 0:
                    dd_var_set &= self._vars[i]

            # precompute `assignment_to_mask`
            assignment_to_mask_pc = [
                assignment_to_mask(a, var_set) for a in range(num_assignments)
            ]

            for f_explicit, f in enumerate(self._boolean_functions):
                exist_expected = 0
                forall_expected = 0
                unique_expected = 0
                for assignment in range(num_assignments):
                    mask = assignment_to_mask_pc[assignment]
                    bit = 1 << assignment

                    # or of all bits under mask
                    if (f_explicit & mask) != 0:
                        exist_expected |= bit
                    # and of all bits under mask
                    if (f_explicit & mask) == mask:
                        forall_expected |= bit
                    # xor of all bits under mask
                    if (bit_count(f_explicit & mask) & 1) != 0:
                        unique_expected |= bit

                exist_actual = self._dd_to_boolean_func[f.exist(dd_var_set)]
                assert exist_actual == exist_expected

                forall_actual = self._dd_to_boolean_func[f.forall(dd_var_set)]
                assert forall_actual == forall_expected

                unique_actual = self._dd_to_boolean_func[f.unique(dd_var_set)]
                assert unique_actual == unique_expected


def test_bdd_all_boolean_functions_2vars_t1():
    mgr = oxidd.bdd.BDDManager(1024, 1024, 1)
    vars = [mgr.new_var() for _ in range(2)]
    test = AllBooleanFunctionsQuantSubst(mgr, vars, vars)
    test.basic()
    test.subst()
    test.quant()


def test_bcdd_all_boolean_functions_2vars_t1():
    mgr = oxidd.bcdd.BCDDManager(1024, 1024, 1)
    vars = [mgr.new_var() for _ in range(2)]
    test = AllBooleanFunctionsQuantSubst(mgr, vars, vars)
    test.basic()
    test.subst()
    test.quant()


def test_zbdd_all_boolean_functions_2vars_t1():
    mgr = oxidd.zbdd.ZBDDManager(1024, 1024, 1)
    singletons = [mgr.new_singleton() for _ in range(2)]
    vars = [s.var_boolean_function() for s in singletons]
    test = AllBooleanFunctions(mgr, vars, singletons)
    test.basic()


def pick_cube(mgr: BooleanFunctionManager):
    """Only works for B(C)DDs"""
    tt = mgr.true()

    x = mgr.new_var()
    y = mgr.new_var()

    c = tt.pick_cube()
    assert c is not None
    assert len(c) == 2
    assert c[0] is None and c[1] is None
    assert list(c) == [None, None]
    assert list(reversed(c)) == [None, None]
    assert c[:] == [None, None]
    assert None in c
    assert True not in c
    assert False not in c

    c = (~x).pick_cube()
    assert c is not None
    assert len(c) == 2
    assert c[0] is False
    assert c[1] is None
    assert list(c) == [False, None]
    assert list(reversed(c)) == [None, False]
    assert c[:] == [False, None]
    assert None in c
    assert True not in c
    assert False in c

    c = y.pick_cube()
    assert c is not None
    assert len(c) == 2
    assert c[0] is None
    assert c[1] is True
    assert list(c) == [None, True]
    assert list(reversed(c)) == [True, None]
    assert c[:] == [None, True]
    assert None in c
    assert True in c
    assert False not in c

    assert mgr.false().pick_cube() is None


def test_bdd_pick_cube():
    pick_cube(oxidd.bdd.BDDManager(1024, 1024, 1))


def test_bcdd_pick_cube():
    pick_cube(oxidd.bcdd.BCDDManager(1024, 1024, 1))


def ord_hash(mgr):
    assert hash(mgr) == hash(mgr)

    tt = mgr.true()
    ff = mgr.false()

    assert hash(ff) == hash(ff)
    assert ff == ff and not ff != ff  # noqa: SIM202
    assert ff <= ff and ff >= ff and not ff < ff and not ff > ff

    assert hash(tt) == hash(tt)
    assert tt == tt and not tt != tt  # noqa: SIM202
    assert tt <= tt and tt >= tt and not tt < tt and not tt > tt

    assert not tt == ff and tt != ff  # noqa: SIM201
    assert (tt <= ff and tt < ff and ff >= tt and ff > tt) or (
        tt >= ff and tt > ff and ff <= tt and ff < tt
    )


def test_bdd_ord_hash():
    ord_hash(oxidd.bdd.BDDManager(1024, 1024, 1))


def test_bcdd_ord_hash():
    ord_hash(oxidd.bcdd.BCDDManager(1024, 1024, 1))


def test_zbdd_ord_hash():
    ord_hash(oxidd.zbdd.ZBDDManager(1024, 1024, 1))
