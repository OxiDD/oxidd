import os
import shlex
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, List, NoReturn

from cffi import FFI

# spell-checker:ignore cdef,cdefs,liboxidd,PYFFI


def fatal(msg: str) -> NoReturn:
    print(msg, file=sys.stderr)
    exit(1)


if os.name != "posix" and os.name != "nt":
    fatal("Error: only Unix and Windows systems are currently supported")

repo_dir = Path(__file__).parent.parent.parent
target_dir = repo_dir / "target"
include_dir = target_dir / "include"
oxidd_h = include_dir / "oxidd.h"
profile = "release"
lib_dir = target_dir / profile

include_dir.mkdir(parents=True, exist_ok=True)


class LinkMode(Enum):
    """Modes to build OxiDD, set via the environment variable OXIDD_PYFFI_LINK_MODE"""

    STATIC = 0
    """
    Statically link against `liboxidd_ffi.a` or `oxidd_ffi.lib` in the
    `target/<profile>` directory.

    This is the mode we will be using for shipping via PyPI since we do not need
    to mess around with the RPath (which does not exist on Windows anyway). It
    is also the default mode on Windows.
    """

    SHARED_SYSTEM = 1
    """
    Dynamically link against a system-installed `liboxidd.so`, `liboxidd.dylib`,
    or `oxidd.dll`, respectively.

    This mode is useful for packaging in, e.g., Linux distributions. With this
    mode, we do not need to ship the main OxiDD library in both packages,
    `liboxidd` and `python3-oxidd`. We can furthermore decouple updates of the
    two packages.
    """

    SHARED_DEV = 2
    """
    Dynamically link against `liboxidd_ffi.so`, `liboxidd_ffi.dylib`, or
    `oxidd_ffi.dll` in the `target/<profile>` directory.

    This mode is the default for developing on Unix systems. When tuning
    heuristics for instance, a simple `cargo build --release` suffices, no
    `pip install --editable .` is required before re-running the Python script.
    On Windows, setting this mode up requires extra work, since there is no
    RPath like on Unix systems.
    """

    @staticmethod
    def from_env() -> "LinkMode":
        key = "OXIDD_PYFFI_LINK_MODE"
        s = os.environ.get(key, None)
        if s is None:
            # Since Windows does not have an RPath, static linking yields the
            # better developer experience.
            return LinkMode.SHARED_DEV if os.name != "nt" else LinkMode.STATIC
        s = s.lower()
        if s == "static":
            return LinkMode.STATIC
        if s == "shared-system":
            return LinkMode.SHARED_SYSTEM
        if s == "shared-dev":
            return LinkMode.SHARED_DEV
        fatal(
            f"Error: unknown build mode '{s}', supported values for {key} are "
            "`static`, `shared-system`, and `shared-dev`.",
        )


build_mode = LinkMode.from_env()


def which(bin: str) -> str:
    res = shutil.which(bin)
    if res is None:
        fatal(f"Error: could not find {bin}")
    return res


cbindgen_bin = which("cbindgen")


def run(*args: str):
    """Runs the script and checks the return code"""
    res = subprocess.run(args, cwd=repo_dir, check=False)
    if res.returncode != 0:
        fatal(f"Error: {shlex.join(args)} failed")


if build_mode != LinkMode.SHARED_SYSTEM:
    cargo_bin = which("cargo")
    print("building crates/oxidd-ffi ...")
    run(cargo_bin, "build", f"--profile={profile}", "--package=oxidd-ffi")

print("running cbindgen ...")
run(
    cbindgen_bin,
    f"--config={repo_dir / 'crates' / 'oxidd-ffi' / 'cbindgen.toml'}",
    f"--output={oxidd_h}",
    str(repo_dir / "crates" / "oxidd-ffi"),
)


def read_cdefs(header: Path) -> str:
    """
    Remove C macros and include directives from the include header since CFFI cannot
    deal with them.
    """

    res = ""
    with header.open("r") as f:
        lines = iter(f)
        while True:
            line = next(lines, None)
            if line is None:
                break
            if not line:
                continue
            if line[0] == "#":
                if line.startswith("#if"):
                    while True:
                        line = next(lines, None)
                        if line is None:
                            fatal(
                                f"Error parsing {header}: reached end of file while"
                                "searching for #endif",
                            )
                        if line.startswith("#endif"):
                            break
                continue
            res += line
    return res


cdefs = read_cdefs(oxidd_h)

flags: Dict[str, List[str]] = {
    "libraries": []  # for `+=` (MSVC)
}

if build_mode == LinkMode.STATIC:
    flags["include_dirs"] = [str(include_dir)]

    if os.name == "posix":
        flags["extra_link_args"] = [str(lib_dir / "liboxidd_ffi.a")]
    elif os.name == "nt":
        # TODO: This should be derived from
        #       `cargo rustc -q -- --print=native-static-libs`, but without the
        #       `windows.lib` and without duplicates?
        # spell-checker:ignore advapi,ntdll,userenv
        flags["libraries"] += [
            "kernel32",
            "advapi32",
            "bcrypt",
            "ntdll",
            "userenv",
            "ws2_32",
            "msvcrt",
        ]
        flags["extra_link_args"] = [str(lib_dir / "oxidd_ffi.lib")]
elif build_mode == LinkMode.SHARED_SYSTEM:
    if os.name == "posix":
        flags["libraries"] = ["oxidd"]
    elif os.name == "nt":
        flags["libraries"] = ["oxidd.dll"]
elif build_mode == LinkMode.SHARED_DEV:
    flags["include_dirs"] = [str(include_dir)]
    flags["library_dirs"] = [str(lib_dir)]
    if os.name == "posix":
        flags["libraries"] = ["oxidd_ffi"]
        flags["runtime_library_dirs"] = flags["library_dirs"]
    elif os.name == "nt":
        flags["libraries"] = ["oxidd_ffi.dll"]
        # There is no RPath on Windows :(

flags["extra_compile_args"] = ["-O2" if os.name != "nt" else "/O2"]


ffi = FFI()
ffi.set_source("_oxidd", "#include <oxidd.h>\n", ".c", **flags)
ffi.cdef(cdefs)

if __name__ == "__main__":
    ffi.compile(verbose=True)
