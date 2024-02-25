from typing import TypeVar
from oxidd.bdd import BDDManager
from oxidd.traits import BooleanFunctionManager, BooleanFunction

mgr = BDDManager.new(1024, 1024, 1)

tt = mgr.true()
ff = mgr.false()

assert tt == tt
assert ff == ff
assert tt != ff

assert tt & tt == tt
assert tt & ff == ff
assert ff & tt == ff
assert ff & ff == ff

assert tt | tt == tt
assert tt | ff == tt
assert ff | tt == tt
assert ff | ff == ff

assert ff.node_count() == 1
assert tt.node_count() == 1

x1 = mgr.new_var()

assert x1.node_count() == 3

assert x1 & -x1 == ff
assert x1 | -x1 == tt
assert x1 ^ x1 == ff
assert x1 ^ -x1 == tt

M = TypeVar("M", bound=BooleanFunctionManager)


def generic(mgr: M) -> BooleanFunction[M]:
    return mgr.true()


def generic2(f: BooleanFunction[M]) -> BooleanFunction[M]:
    return f ^ f | f.manager.true()


generic(mgr)

print("Success ✅")
