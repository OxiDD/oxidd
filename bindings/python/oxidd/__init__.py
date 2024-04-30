"""OxiDD

A concurrent decision diagram library.
"""

import importlib.metadata

__all__ = ["bcdd", "bdd", "abc", "zbdd"]
__version__ = importlib.metadata.version("oxidd")

from . import abc, bcdd, bdd, zbdd
