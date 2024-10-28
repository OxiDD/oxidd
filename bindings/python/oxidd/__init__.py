"""OxiDD

A concurrent decision diagram library.
"""

from __future__ import annotations

import importlib.metadata

__all__ = ["bcdd", "bdd", "protocols", "util", "zbdd"]
__version__ = importlib.metadata.version("oxidd")

from . import bcdd, bdd, protocols, util, zbdd
