from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

# This is `bindings/python/build/util.py`, so we need four times `.parent`
repo_dir = Path(__file__).parent.parent.parent.parent.absolute()


def fatal(msg: str) -> NoReturn:
    """Print an error message and exit"""
    print(msg, file=sys.stderr)
    exit(1)


def which(bin: str) -> str:
    """shutil.which() but exit with a human-readable message if not found"""
    res = shutil.which(bin)
    if res is None:
        fatal(f"Error: could not find {bin}")
    return res


def run(*args: str) -> None:
    """Run the given command and exit with a human-readable error message in
    case of a non-zero exit code"""
    res = subprocess.run(args, cwd=repo_dir, check=False)
    if res.returncode != 0:
        fatal(f"Error: {shlex.join(args)} failed")


#: Python platform tag -> Rust target triple
# spell-checker:ignore armv,gnueabi,musleabihf
linux_targets = {
    # manylinux
    # ---------
    "manylinux2014_x86_64": "x86_64-unknown-linux-gnu",
    "manylinux2014_i686": "i686-unknown-linux-gnu",
    "manylinux2014_aarch64": "aarch64-unknown-linux-gnu",
    "manylinux2014_ppc64le": "powerpc64le-unknown-linux-gnu",
    "manylinux2014_s390x": "s390x-unknown-linux-gnu",
    # not supported by cibuildwheel:
    # "manylinux2014_armv7l": "armv7-unknown-linux-gnueabi",
    # "manylinux2014_ppc64": "powerpc64-unknown-linux-gnu",
    #
    # musllinux
    # ---------
    # Note: Rust's targets require musl 1.2
    "musllinux_1_2_x86_64": "x86_64-unknown-linux-musl",
    "musllinux_1_2_i686": "i686-unknown-linux-musl",
    "musllinux_1_2_aarch64": "aarch64-unknown-linux-musl",
    "musllinux_1_2_armv7l": "armv7-unknown-linux-musleabihf",
    # Currently tier 3 targets, so we cannot install them using rustup
    # "musllinux_1_2_ppc64le": "powerpc64le-unknown-linux-gnu",
    # "musllinux_1_2_s390x": "s390x-unknown-linux-gnu",
}
