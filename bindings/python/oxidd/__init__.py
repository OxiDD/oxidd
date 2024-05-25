"""OxiDD

A concurrent decision diagram library.
"""

import importlib.metadata

__all__ = ["bcdd", "bdd", "protocols", "zbdd"]
__version__ = importlib.metadata.version("oxidd")

from . import bcdd, bdd, protocols, zbdd
