from oxidd.bdd import BDDManager, BDDFunction

mgr = BDDManager(1024, 1024, 1)

tt = BDDFunction.true(mgr)
ff = BDDFunction.false(mgr)

assert tt == tt
assert ff == ff
assert tt != ff

assert (tt & tt) == tt
assert (tt & ff) == ff
assert (ff & tt) == ff
assert (ff & ff) == ff

print("Success :)")
